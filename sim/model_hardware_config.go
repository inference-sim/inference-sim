package sim

import (
	"fmt"
	"math"
)

// ModelConfig holds model architecture parameters parsed from a HuggingFace config.json.
// Used by the roofline and cross-model latency models for step time estimation.
// Parsing functions are in sim/latency/config.go.
type ModelConfig struct {
	NumLayers              int     `json:"num_hidden_layers"`
	HiddenDim              int     `json:"hidden_size"`
	NumHeads               int     `json:"num_attention_heads"`
	NumKVHeads             int     `json:"num_key_value_heads"`
	VocabSize              int     `json:"vocab_size"`
	BytesPerParam          float64 `json:"bytes_per_param"`
	IntermediateDim        int     `json:"intermediate_size"`
	NumLocalExperts        int     `json:"num_local_experts"`                // 0 = dense model (MoE: number of experts)
	NumExpertsPerTok       int     `json:"num_experts_per_tok"`              // 0 = dense model (MoE: active experts per token)
	MoEExpertFFNDim        int     `json:"moe_intermediate_size"`            // Per-routed-expert FFN dim; 0 = use IntermediateDim (Mixtral convention)
	SharedExpertFFNDim     int     `json:"shared_expert_intermediate_size"`  // Total shared-expert FFN dim; 0 = no shared experts
	InterleaveMoELayerStep int     `json:"interleave_moe_layer_step"`        // Layer interleave pattern: 0 = uniform (all same type), 1 = alternate MoE/dense, 2 = every 3rd layer is MoE, etc. Used for Scout-style hybrid architectures.
	DenseIntermediateDim   int     `json:"intermediate_size_mlp"`            // Dense layer FFN dimension; 0 = use IntermediateDim. For models like Scout where dense layers have different FFN size than MoE expert FFN.
	HiddenAct              string  `json:"hidden_act"`                       // Activation function (e.g. "silu", "gelu", "relu"); used by KV capacity (3-matrix SwiGLU detection), reserved for future roofline per-activation tuning
	WeightBytesPerParam    float64 `json:"weight_bytes_per_param,omitempty"` // Quantized weight precision (bytes/param); 0 = not set, use BytesPerParam. Auto-detected from quantization_config or model name conventions.
	KVBytesPerParam        float64 `json:"kv_bytes_per_param,omitempty"`     // KV-cache storage precision (bytes/param) from --kv-cache-dtype (vLLM CacheConfig.cache_dtype); 0 = "auto", use BytesPerParam. Independent of weight quantization: fp8 KV (1) under bf16 compute (2) halves KV bytes/token → ~2x KV capacity. Used by KV capacity + PD KV-transfer sizing only (NOT step time). See EffectiveKVBytesPerParam.
	HeadDim                int     `json:"head_dim,omitempty"`               // Explicit attention head dimension; 0 = not set, fall back to HiddenDim/NumHeads. Modern MLA/GQA models (GLM-5.2, Qwen3) declare a head_dim that differs from hidden/heads. Used by KV/weight capacity only (NOT step time). See EffectiveHeadDim.
	KVLoraRank             int     `json:"kv_lora_rank,omitempty"`           // MLA compressed-KV latent rank; 0 = not MLA (standard MHA/GQA KV). When > 0 the KV cache stores a compressed latent of KVLoraRank+QKRopeHeadDim scalars per token per layer (DeepSeek-V2/V3, Kimi-K3, GLM-5.2). See KVBytesPerToken.
	QKRopeHeadDim          int     `json:"qk_rope_head_dim,omitempty"`       // MLA decoupled-RoPE key dimension; the second summand of the MLA latent width. Meaningful only when KVLoraRank > 0.
	FirstKDenseReplace     int     `json:"first_k_dense_replace,omitempty"`  // Number of leading layers that are dense (non-MoE) in a MoE model; remaining layers are MoE. 0 = no dense prefix (all layers MoE when IsMoE). Distinct from InterleaveMoELayerStep (every-Nth interleave). Used by weight estimation.
	KVBearingLayers        int     `json:"kv_bearing_layers,omitempty"`      // Number of KV-cache-bearing (full-attention) layers for hybrid-attention models (Kimi-K3: 24 MLA layers of 93; the other 69 are linear-attention KDA layers with O(1)-in-sequence recurrent state and no growing KV). 0 = not a hybrid model → EffectiveKVBearingLayers falls back to NumLayers (every standard-MHA and non-hybrid MLA model, INV-6). Derived from len(linear_attn_config.full_attn_layers). Two consumers (see EffectiveKVBearingLayers): the KV-capacity path (both the MLA and standard MHA/GQA branches of KVBytesPerToken) and the step-time attention terms (#1636, trained-physics + roofline); NOT weights (#1638). See EffectiveKVBearingLayers.
}

// EffectiveHeadDim returns the attention head dimension to use for KV-cache and
// weight-memory calculations. Returns HeadDim when explicitly set (> 0),
// otherwise falls back to HiddenDim/NumHeads (the implicit convention). Modern
// MLA/GQA models declare an explicit head_dim that differs from hidden/heads
// (e.g. GLM-5.2: head_dim=192 while 6144/64=96); using it corrects the KV and
// weight estimates. Returns 0 when the implicit fallback would divide by zero
// (NumHeads == 0), leaving the caller's own validation to reject it.
//
// This is deliberately NOT used by the step-time (trained-physics/roofline)
// latency models, which retain HiddenDim/NumHeads — scoping the change to the
// capacity path keeps step-time golden datasets and INV-BC-DP1 byte-identical.
func (mc ModelConfig) EffectiveHeadDim() int {
	if mc.HeadDim > 0 {
		return mc.HeadDim
	}
	if mc.NumHeads == 0 {
		return 0
	}
	return mc.HiddenDim / mc.NumHeads
}

// EffectiveKVBearingLayers returns the number of layers that store a per-token KV
// cache, for KV-capacity sizing. Returns KVBearingLayers when explicitly set (> 0),
// otherwise falls back to NumLayers. Hybrid-attention models (Kimi-K3) interleave
// full Multi-head Latent Attention layers — which store a compressed KV latent per
// token — with linear-attention (Kimi Delta Attention) layers, which keep a
// fixed-size recurrent + short-conv state independent of sequence length and store
// no growing KV. Sizing the KV cache over KVBearingLayers (K3: 24) rather than all
// NumLayers (93) corrects the per-token KV footprint, and hence the KV block count
// and batch sizes, for such models. Returns NumLayers unchanged for every
// non-hybrid model (KVBearingLayers == 0), so the KV footprint is byte-identical
// there (INV-6).
//
// Two consumers use this accessor. (1) KV capacity (#1635): KVBytesPerToken sizes the
// per-token KV footprint over the full-attention layers only. (2) Step time (#1636):
// the trained-physics and roofline models scope the sequence-length-dependent
// attention cost — the O(context)/O(N²) attention-score compute and the growing-KV
// read/write bandwidth — to the full-attention layers, charging the KDA layers a
// linear-attention (O(N) prefill, O(state) decode) cost instead. The weight-memory
// estimate (computeModelWeightBytes) still retains NumLayers — KDA layers carry
// full-attention weights (#1638, out of scope) — as does EffectiveHeadDim, which stays
// capacity-only. For a non-hybrid model this returns NumLayers, so all three consumers
// are byte-identical to the pre-#1635/#1636 behavior (INV-6/INV-BC-DP1).
func (mc ModelConfig) EffectiveKVBearingLayers() int {
	if mc.KVBearingLayers > 0 {
		// Clamp to NumLayers: the KV-bearing (full-attention) layer count can never
		// exceed the total layer count. A malformed config with KVBearingLayers >
		// NumLayers would otherwise over-count KV worse than the all-layers default,
		// so bound it defensively (mirrors FirstKDenseReplace's [0, numLayers] clamp).
		if mc.NumLayers > 0 && mc.KVBearingLayers > mc.NumLayers {
			return mc.NumLayers
		}
		return mc.KVBearingLayers
	}
	return mc.NumLayers
}

// EffectiveWeightBytesPerParam returns the bytes-per-parameter to use for
// weight memory calculations. Returns WeightBytesPerParam when explicitly set
// (> 0), otherwise falls back to BytesPerParam (the compute/activation dtype).
// This decouples weight bandwidth (often quantized, e.g. 0.5 for W4A16) from
// activation memory (which uses the compute dtype, e.g. 2.0 for bfloat16). KV-cache
// storage precision is likewise decoupled from the compute dtype — see
// EffectiveKVBytesPerParam (#1565).
func (mc ModelConfig) EffectiveWeightBytesPerParam() float64 {
	if mc.WeightBytesPerParam > 0 {
		return mc.WeightBytesPerParam
	}
	return mc.BytesPerParam
}

// EffectiveKVBytesPerParam returns the bytes-per-parameter to use for KV-cache
// capacity (per-token byte width). Returns KVBytesPerParam when explicitly set
// (> 0, e.g. --kv-cache-dtype fp8 → 1.0), otherwise falls back to BytesPerParam
// (the compute/activation dtype). This mirrors EffectiveWeightBytesPerParam on the
// KV axis (#1565): it decouples KV storage precision from both the compute dtype
// and weight quantization — vLLM's --kv-cache-dtype is an engine arg independent of
// weight quant, and fp8 KV under bf16 compute roughly halves KV memory (doubling KV
// block capacity). Returning BytesPerParam when unset keeps the KV footprint
// byte-identical to a build without the flag (INV-6, "auto" default).
//
// Like EffectiveWeightBytesPerParam / EffectiveHeadDim, this is deliberately scoped
// to the capacity path (KVBytesPerToken): the step-time (trained-physics/roofline)
// models retain BytesPerParam for the KV-read bandwidth term, so step-time golden
// datasets and INV-BC-DP1 stay byte-identical.
func (mc ModelConfig) EffectiveKVBytesPerParam() float64 {
	if mc.KVBytesPerParam > 0 {
		return mc.KVBytesPerParam
	}
	return mc.BytesPerParam
}

// MoEMinExperts is the minimum NumLocalExperts for a model to be treated as MoE.
// It is the single source of truth for the MoE-vs-dense boundary across BLIS:
// the detection predicate (IsMoE), the parse-time expert-count resolver
// (latency.HFConfig.ResolveNumExperts), and the KV-capacity MoE branch all key
// off this constant.
//
// Single-expert configs (NumLocalExperts == 1) are dense-equivalent in BLIS. The
// MoE weight/FLOP formulas (sim/latency/kv_capacity.go, roofline.go,
// trained_physics_model.go) read NumLocalExperts as a multiplier/divisor and would
// MISESTIMATE at N=1 (e.g. MoE weight 3·h·f_expert·1 + router ≠ dense 3·h·f_dense).
// The >= 2 threshold keeps N=1 out of those formulas.
//
// This is an intentional, documented divergence from vLLM, whose is_moe is
// get_num_experts() > 0 and which does not reject a 1-expert FusedMoE (no
// num_experts==1 guard exists), so it would construct one rather than fall back to
// dense. On every real model the two thresholds agree — no real HF config has
// NumLocalExperts == 1 — so keeping >= 2 loses no parity and preserves BLIS's
// analytic correctness.
const MoEMinExperts = 2

// IsMoE reports whether the model is a mixture-of-experts model
// (NumLocalExperts >= MoEMinExperts). See MoEMinExperts for the threshold rationale
// and the vLLM divergence note. This is the canonical MoE-detection predicate:
// prefer it over inline NumLocalExperts comparisons at detection sites. Validation
// code that compares the count against other expert quantities (e.g. NumExpertsPerTok)
// legitimately reads the raw field instead.
func (mc ModelConfig) IsMoE() bool {
	return mc.NumLocalExperts >= MoEMinExperts
}

// IsMLA reports whether the model uses Multi-head Latent Attention (a positive
// compressed-KV latent rank), e.g. DeepSeek-V2/V3, Kimi-K3, GLM-5.2 glm_moe_dsa.
// This is the canonical MLA-detection predicate — prefer it over inline
// KVLoraRank comparisons so future MLA-aware paths share one definition (mirrors
// IsMoE). When true, the KV cache stores a compressed latent of
// KVLoraRank+QKRopeHeadDim scalars per token per layer (see KVBytesPerToken).
func (mc ModelConfig) IsMLA() bool {
	return mc.KVLoraRank > 0
}

// HardwareCalib holds GPU hardware calibration parameters.
// Used by the roofline latency model for compute/memory bandwidth estimation.
// Parsing functions are in sim/latency/config.go.
type HardwareCalib struct {
	TFlopsPeak float64 `json:"TFlopsPeak"` // Tera (10^12) FLOP/s for FP16/BF16 compute
	TFlopsFP8  float64 `json:"TFlopsFP8"`  // Tera (10^12) FLOP/s for FP8 compute; 0 = no native FP8 support
	BwPeakTBs  float64 `json:"BwPeakTBs"`  // in TB/s
	MfuPrefill float64 `json:"mfuPrefill"`
	MfuDecode  float64 `json:"mfuDecode"`
	MemoryGiB  float64 `json:"MemoryGiB"` // GPU memory capacity in GiB

	// Interconnect calibration (#1530), used only to price CROSS-NODE collective
	// traffic in the trained-physics step-time model. Both are effective
	// (achievable, not theoretical-peak) per-GPU unidirectional bandwidths in GB/s.
	// Only their RATIO enters the cost model, so the absolute scale cancels — what
	// matters is how much slower the fabric is than the on-node link.
	//
	// Either field left at 0 (or non-finite) means "interconnect uncalibrated":
	// InterconnectBwRatio() then returns 1.0 and cross-node traffic is priced
	// exactly like intra-node traffic, i.e. byte-identical to a pre-#1530 build
	// (INV-6). Adding them to a hardware config is what turns the cross-node
	// penalty on for a spanning placement.
	IntraNodeBwGBps float64 `json:"IntraNodeBwGBps"` // on-node GPU-to-GPU link (NVLink/xGMI, or PCIe on non-NVLink parts)
	InterNodeBwGBps float64 `json:"InterNodeBwGBps"` // per-GPU share of the node's inter-node fabric (InfiniBand/RoCE NIC)

	// InterNodeLatencyUs is the fixed cost of ONE cross-node collective in
	// microseconds — NCCL launch plus fabric round-trip plus the synchronization a
	// hierarchical collective imposes — independent of message size. It is charged per
	// collective that crosses a node boundary, so a step running L layers × 2 comm
	// phases pays it L × 2 times.
	//
	// This is the size-independent half of the cross-node cost, and for the small
	// messages a decode step produces it can exceed the bandwidth half by an order of
	// magnitude. It is nonetheless 0 (uncalibrated ⇒ not charged) in the bundled
	// hardware config, deliberately: BLIS has no measured per-collective latency to
	// ship, and inventing one would put a fabricated constant in front of every
	// multi-node estimate. Supply a measured value here to model it — see #1661.
	//
	// Calibration frame: like the bandwidth half, this rides the learned communication
	// coefficient (β₄ for TP collectives, β_EP for MoE dispatch), so the charge is
	// β·units·InterNodeLatencyUs. Calibrate it in that frame, not as a raw wall-clock
	// number.
	InterNodeLatencyUs float64 `json:"InterNodeLatencyUs"`
}

// ValidateInterconnect checks the optional interconnect calibration (#1530). Declaring
// none of the three fields is valid and inert — that is every hardware config written
// before #1530, and cross-node traffic is then priced at the on-node rate.
//
// It rejects two things, both of which would otherwise be swallowed by the accessors'
// clamps and leave the user believing a fabric was modeled when it was not (R1):
//
//   - a value that was clearly meant to be a bandwidth or a latency but cannot be one
//     (negative, NaN, or infinite);
//   - exactly one of the two BANDWIDTHS set. The cost model uses only their ratio, so a
//     lone bandwidth produces no bandwidth penalty at all. There is no reading of a
//     half-calibrated pair, whereas a latency on its own IS meaningful (a fabric can be
//     modeled as latency-dominated), so the latency is deliberately not paired.
//
// Pure query; the caller decides fatality (cmd/ → logrus.Fatalf, sim/ factory → error).
func (hc HardwareCalib) ValidateInterconnect() error {
	for _, f := range []struct {
		name string
		v    float64
	}{
		{"IntraNodeBwGBps", hc.IntraNodeBwGBps},
		{"InterNodeBwGBps", hc.InterNodeBwGBps},
		{"InterNodeLatencyUs", hc.InterNodeLatencyUs},
	} {
		if f.v == 0 {
			continue // not calibrated — the feature stays inert for this field
		}
		if f.v < 0 || math.IsNaN(f.v) || math.IsInf(f.v, 0) {
			return fmt.Errorf("%s must be a finite positive value (or 0 for \"not calibrated\"), got %v", f.name, f.v)
		}
	}
	if (hc.IntraNodeBwGBps > 0) != (hc.InterNodeBwGBps > 0) {
		return fmt.Errorf("interconnect bandwidth calibration is incomplete "+
			"(IntraNodeBwGBps=%v, InterNodeBwGBps=%v): the cost model uses their ratio, so it needs BOTH "+
			"bandwidths, or neither (which prices cross-node traffic at the on-node rate)",
			hc.IntraNodeBwGBps, hc.InterNodeBwGBps)
	}
	return nil
}

// HasInterconnectCalibration reports whether this GPU declares enough interconnect
// calibration to charge ANY cross-node cost: either a usable bandwidth ratio (the
// size-dependent half) or a positive per-collective latency (the size-independent
// half). When false, a collective that crosses a node boundary is priced exactly as
// if it had not (INV-6) — which callers should surface rather than leave silent (R1).
func (hc HardwareCalib) HasInterconnectCalibration() bool {
	return hc.InterconnectBwRatio() > 1.0 || hc.EffectiveInterNodeLatencyUs() > 0
}

// EffectiveInterNodeLatencyUs returns the per-cross-node-collective latency to
// charge, or 0 when it is unset or unusable (negative, NaN, Inf). Pure query.
func (hc HardwareCalib) EffectiveInterNodeLatencyUs() float64 {
	if hc.InterNodeLatencyUs <= 0 || math.IsNaN(hc.InterNodeLatencyUs) || math.IsInf(hc.InterNodeLatencyUs, 0) {
		return 0
	}
	return hc.InterNodeLatencyUs
}

// InterconnectBwRatio returns how many times slower this GPU's inter-node fabric
// is than its on-node link: IntraNodeBwGBps / InterNodeBwGBps.
//
// Returns exactly 1.0 — "cross-node traffic costs the same as on-node traffic",
// the pre-#1530 behavior — whenever the ratio cannot be trusted or would make
// spanning cheaper than not spanning:
//   - either bandwidth is unset (0), negative, NaN or Inf (uncalibrated hardware);
//   - the computed ratio is below 1 (a fabric declared faster than the on-node
//     link). Clamping instead of honoring a sub-unit ratio matters because the
//     comm coefficient beta_4 is calibrated ON the intra-node link: scaling BELOW
//     that baseline would price a spanning instance faster than a single-node one,
//     which is physically absurd (R20 — degrade to the calibrated baseline, never
//     to a nonsense value);
//   - the division itself overflows to +Inf (reachable only from absurd inputs, e.g.
//     a subnormal inter-node bandwidth against a near-MaxFloat64 on-node one). Every
//     consumer must be able to treat the result as a finite number, so this returns
//     the neutral 1.0 rather than a value that would poison the cost model. Callers
//     that report on the calibration should therefore describe a 1.0 as "no USABLE
//     bandwidths", which covers both the unset and the unusable case.
//
// Pure query, no state (R13).
func (hc HardwareCalib) InterconnectBwRatio() float64 {
	intra, inter := hc.IntraNodeBwGBps, hc.InterNodeBwGBps
	if intra <= 0 || inter <= 0 ||
		math.IsNaN(intra) || math.IsInf(intra, 0) ||
		math.IsNaN(inter) || math.IsInf(inter, 0) {
		return 1.0
	}
	ratio := intra / inter
	// `!(ratio > 1.0)` is false for NaN too — belt-and-braces after the guards above.
	// The explicit Inf test keeps the contract "always finite": both bandwidths can be
	// finite and positive while their quotient still overflows.
	if !(ratio > 1.0) || math.IsInf(ratio, 0) {
		return 1.0
	}
	return ratio
}
