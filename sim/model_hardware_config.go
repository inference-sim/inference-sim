package sim

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
	HeadDim                int     `json:"head_dim,omitempty"`               // Explicit attention head dimension; 0 = not set, fall back to HiddenDim/NumHeads. Modern MLA/GQA models (GLM-5.2, Qwen3) declare a head_dim that differs from hidden/heads. Used by KV/weight capacity only (NOT step time). See EffectiveHeadDim.
	KVLoraRank             int     `json:"kv_lora_rank,omitempty"`           // MLA compressed-KV latent rank; 0 = not MLA (standard MHA/GQA KV). When > 0 the KV cache stores a compressed latent of KVLoraRank+QKRopeHeadDim scalars per token per layer (DeepSeek-V2/V3, Kimi-K3, GLM-5.2). See KVBytesPerToken.
	QKRopeHeadDim          int     `json:"qk_rope_head_dim,omitempty"`       // MLA decoupled-RoPE key dimension; the second summand of the MLA latent width. Meaningful only when KVLoraRank > 0.
	FirstKDenseReplace     int     `json:"first_k_dense_replace,omitempty"`  // Number of leading layers that are dense (non-MoE) in a MoE model; remaining layers are MoE. 0 = no dense prefix (all layers MoE when IsMoE). Distinct from InterleaveMoELayerStep (every-Nth interleave). Used by weight estimation.
	KVBearingLayers        int     `json:"kv_bearing_layers,omitempty"`      // Number of KV-cache-bearing (full-attention) layers for hybrid-attention models (Kimi-K3: 24 MLA layers of 93; the other 69 are linear-attention KDA layers with O(1)-in-sequence recurrent state and no growing KV). 0 = not a hybrid model → EffectiveKVBearingLayers falls back to NumLayers (every standard-MHA and non-hybrid MLA model, INV-6). Derived from len(linear_attn_config.full_attn_layers). Used by the MLA KV-capacity path only (NOT weights or step time). See EffectiveKVBearingLayers.
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
// Like EffectiveHeadDim, this is deliberately scoped to the KV-capacity path: the
// weight-memory estimate (computeModelWeightBytes) and the step-time
// (trained-physics/roofline) models retain NumLayers, since the KDA layers still
// carry weights and compute — only their KV footprint differs. See issue #1635
// (KDA weights: #1638; KDA step time: #1636).
func (mc ModelConfig) EffectiveKVBearingLayers() int {
	if mc.KVBearingLayers > 0 {
		return mc.KVBearingLayers
	}
	return mc.NumLayers
}

// EffectiveWeightBytesPerParam returns the bytes-per-parameter to use for
// weight memory calculations. Returns WeightBytesPerParam when explicitly set
// (> 0), otherwise falls back to BytesPerParam (the compute/activation dtype).
// This decouples weight bandwidth (often quantized, e.g. 0.5 for W4A16) from
// KV cache and activation memory (which use the compute dtype, e.g. 2.0 for bfloat16).
func (mc ModelConfig) EffectiveWeightBytesPerParam() float64 {
	if mc.WeightBytesPerParam > 0 {
		return mc.WeightBytesPerParam
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
}
