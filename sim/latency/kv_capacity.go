package latency

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
)

// KVCapacityParams holds model-architecture parameters that are not part of
// sim.ModelConfig but are needed for KV block capacity estimation.
// These come from the HuggingFace config.json (hidden_act, MoE indicators,
// tie_word_embeddings, per-expert and shared-expert FFN dims).
type KVCapacityParams struct {
	IsMoE              bool
	NumLocalExperts    int
	TieWordEmbeddings  bool
	HiddenAct          string
	MoEExpertFFNDim    int // Per-routed-expert FFN dim; 0 = use IntermediateDim
	SharedExpertFFNDim int // Total shared-expert FFN dim; 0 = no shared experts
}

// NewKVCapacityParams creates a KVCapacityParams. Positional arguments ensure
// that adding a field causes a compiler error at every construction site (R4).
func NewKVCapacityParams(isMoE bool, numLocalExperts int, tieWordEmbeddings bool, hiddenAct string, moeExpertFFNDim int, sharedExpertFFNDim int) KVCapacityParams {
	return KVCapacityParams{
		IsMoE:              isMoE,
		NumLocalExperts:    numLocalExperts,
		TieWordEmbeddings:  tieWordEmbeddings,
		HiddenAct:          hiddenAct,
		MoEExpertFFNDim:    moeExpertFFNDim,
		SharedExpertFFNDim: sharedExpertFFNDim,
	}
}

// hasRoutedExperts reports whether the weight estimator's MoE branch applies — i.e.
// whether the model carries routed (FusedMoE) expert weights at all. It is the single
// predicate for that question (R23): the expert-parallel resolution and the weight
// arithmetic must agree on it, or a group size could be validated against experts the
// weight term does not charge.
func (p KVCapacityParams) hasRoutedExperts() bool {
	return p.IsMoE && p.NumLocalExperts >= sim.MoEMinExperts
}

// Constants matching the llm-d-benchmark capacity_planner.py reference.
const (
	activationMemoryDenseGiB = 5.5
	activationMemoryMoEGiB   = 8.0
	nonTorchMemoryTP1GiB     = 0.15
	nonTorchMemoryTPMultiGiB = 0.6
	gibToBytes               = 1 << 30
)

// swiGLUActivations is the set of activation functions that use the SwiGLU
// 3-matrix MLP pattern (gate + up + down). Empty string is accepted as a
// default fallback. R8: unexported map, accessed only within this file.
var swiGLUActivations = map[string]bool{
	"silu":   true,
	"swiglu": true,
	"geglu":  true,
	// "situ" is Kimi-K3's SiTU-GLU (Sigmoid Tanh Unit), the successor to K2's
	// SwiGLU. It is a 3-matrix gated GLU (gate + up + down) with the identical
	// weight/FLOP shape as SwiGLU — only the pointwise gate nonlinearity differs
	// (parameterized by activation_situ_beta / activation_situ_linear_beta). BLIS's
	// capacity and compute models depend only on the MLP matrix count, not the gate
	// function, so SiTU-GLU is SwiGLU-family here. See issue #1526.
	"situ": true,
	"":     true,
}

// KVBytesPerToken computes the per-GPU KV cache bytes per token for a given
// model config and tensor parallelism degree. This is used for both KV cache
// capacity sizing and PD transfer duration estimation.
//
// The formula is: EffectiveKVBearingLayers × 2 (K+V) × headDim × numKVHeads × EffectiveKVBytesPerParam / TP
// (EffectiveKVBearingLayers equals NumLayers for every non-hybrid model, INV-6; for a
// hybrid-attention model it is the full-attention layer count — #1635.)
//
// Uses EffectiveKVBytesPerParam (the KV-cache storage precision), NOT
// WeightBytesPerParam. By default (KVBytesPerParam == 0, "auto") this equals
// BytesPerParam (the compute/activation dtype) — byte-identical to the pre-#1565
// behavior (INV-6). When --kv-cache-dtype fp8 is set, EffectiveKVBytesPerParam is
// 1.0 while compute stays bf16 (2.0), so KV bytes/token halve (vLLM parity: KV cache
// stored at a different, independent precision from compute and from weight quant).
//
// Returns a float64 so callers can choose when to truncate. CalculateKVBlocks
// multiplies by blockSize before truncating (avoids loss when the per-token
// value is fractional, e.g., INT4 quantization with small head dimensions).
// PD transfer sizing truncates to int64 immediately.
//
// Returns per-GPU bytes (divided by TP), since each GPU stores/transfers its
// own KV shard. When numKVHeads < TP (e.g., GQA with 2 KV heads at TP=4),
// vLLM replicates KV heads per GPU; dividing by TP underestimates per-GPU KV
// bytes in this case. This is a known approximation (optimistic).
//
// When numKVHeads < tp, divisibility is not enforced — the GQA head-replication
// case is accepted. In this case the returned value underestimates the true
// per-GPU bytes (optimistic approximation). When numKVHeads >= tp, numKVHeads
// must be evenly divisible by tp or an error is returned.
func KVBytesPerToken(mc sim.ModelConfig, tp int) (float64, error) {
	// Validations common to both the MLA and standard paths. NumHeads and HiddenDim
	// are validated here (not only on the standard path) because CalculateKVBlocks
	// also calls computeModelWeightBytes for MLA configs, and that weight estimate
	// reads NumHeads/HiddenDim — a degenerate MLA config must error here, not run on
	// garbage. Only the HiddenDim%NumHeads divisibility check is MLA-exempt (below).
	if tp <= 0 {
		return 0, fmt.Errorf("KVBytesPerToken: TP must be > 0, got %d", tp)
	}
	if mc.NumLayers <= 0 {
		return 0, fmt.Errorf("KVBytesPerToken: num_layers must be > 0, got %d", mc.NumLayers)
	}
	if mc.NumHeads <= 0 {
		return 0, fmt.Errorf("KVBytesPerToken: num_attention_heads must be > 0, got %d", mc.NumHeads)
	}
	if mc.HiddenDim <= 0 {
		return 0, fmt.Errorf("KVBytesPerToken: hidden_dim must be > 0, got %d", mc.HiddenDim)
	}
	if mc.BytesPerParam <= 0 || math.IsNaN(mc.BytesPerParam) || math.IsInf(mc.BytesPerParam, 0) {
		return 0, fmt.Errorf("KVBytesPerToken: precision (BytesPerParam) must be a valid positive number, got %v", mc.BytesPerParam)
	}
	// KV-storage precision (#1565) is optional (0 = "auto", fall back to BytesPerParam
	// via EffectiveKVBytesPerParam). When explicitly set (e.g. --kv-cache-dtype fp8) it
	// must be a valid positive number — a hand-built ModelConfig with a negative/NaN/Inf
	// value errors here (R1), mirroring the WeightBytesPerParam guard in CalculateKVBlocks.
	if mc.KVBytesPerParam != 0 && (mc.KVBytesPerParam < 0 || math.IsNaN(mc.KVBytesPerParam) || math.IsInf(mc.KVBytesPerParam, 0)) {
		return 0, fmt.Errorf("KVBytesPerToken: KVBytesPerParam must be positive when set, got %v", mc.KVBytesPerParam)
	}

	// --- MLA (Multi-head Latent Attention) path (F2, #1527) ---
	// See ModelConfig.IsMLA / the field comments: KV is a single compressed latent
	// of (kv_lora_rank + qk_rope_head_dim) per token per layer, replicated across TP
	// ranks (NOT divided by TP), and independent of numKVHeads/headDim. The
	// hidden%heads guard is skipped (the latent path never uses that quotient).
	if mc.IsMLA() {
		latentWidth := mc.KVLoraRank + mc.QKRopeHeadDim
		// Size the compressed latent over the KV-bearing (full-attention) layers.
		// For a hybrid-attention MLA model (Kimi-K3) only the full-attention layers
		// store a per-token KV cache; the linear-attention (KDA) layers keep a
		// fixed-size recurrent state and bear no growing KV (#1635). For a non-hybrid
		// MLA model (DeepSeek-V2/V3, GLM-5.2) EffectiveKVBearingLayers == NumLayers,
		// so this is byte-identical to the pre-#1635 all-layers value (INV-6).
		perTokenKVBytesF := float64(mc.EffectiveKVBearingLayers()) * float64(latentWidth) * mc.EffectiveKVBytesPerParam()
		// Public-API boundary guard: normal configs are validated at parse time
		// (config.go rejects negative shape fields), but a hand-built ModelConfig
		// with a negative QKRopeHeadDim could make latentWidth <= 0.
		if perTokenKVBytesF <= 0 {
			return 0, fmt.Errorf("KVBytesPerToken: MLA latent width must be > 0, got kv_lora_rank=%d + qk_rope_head_dim=%d",
				mc.KVLoraRank, mc.QKRopeHeadDim)
		}
		return perTokenKVBytesF, nil
	}

	// --- Standard MHA/GQA path ---
	// (NumHeads>0 and HiddenDim>0 are already validated above, common to both paths.)
	// The hidden%heads guard applies only to the implicit head-dim path; an explicit
	// head_dim (F1) is used directly and does not require divisibility.
	if mc.HeadDim <= 0 && mc.HiddenDim%mc.NumHeads != 0 {
		return 0, fmt.Errorf("KVBytesPerToken: hidden_dim (%d) must be evenly divisible by num_attention_heads (%d)", mc.HiddenDim, mc.NumHeads)
	}

	numKVHeads := mc.NumKVHeads
	if numKVHeads < 0 {
		return 0, fmt.Errorf("KVBytesPerToken: num_kv_heads must be >= 0, got %d", numKVHeads)
	}
	if numKVHeads == 0 {
		numKVHeads = mc.NumHeads
	}

	if numKVHeads >= tp && numKVHeads%tp != 0 {
		return 0, fmt.Errorf("KVBytesPerToken: num_kv_heads (%d) must be evenly divisible by TP (%d)", numKVHeads, tp)
	}

	// Effective head dim: explicit head_dim (F1) when set, else hidden/heads.
	headDim := mc.EffectiveHeadDim()
	// KV-bearing layer count (#1635): full-attention layers for a hybrid model, and
	// == NumLayers for every non-hybrid model (byte-identical, INV-6). Applied on this
	// standard MHA/GQA branch too — not only the MLA branch — so KVBearingLayers governs
	// both attention paths consistently; a non-MLA hybrid would otherwise populate the
	// field but have it silently ignored here.
	perTokenKVBytesF := float64(mc.EffectiveKVBearingLayers()) * 2.0 * float64(headDim) * float64(numKVHeads) * mc.EffectiveKVBytesPerParam()
	perTokenKVBytesPerGPUF := perTokenKVBytesF / float64(tp)

	if perTokenKVBytesPerGPUF <= 0 {
		return 0, fmt.Errorf("KVBytesPerToken: computed value is %.4f (expected > 0); check BytesPerParam=%.4f, numKVHeads=%d, headDim=%d, tp=%d",
			perTokenKVBytesPerGPUF, mc.BytesPerParam, numKVHeads, headDim, tp)
	}
	return perTokenKVBytesPerGPUF, nil
}

// kvCapacityOptions accumulates optional inputs to CalculateKVBlocks. Zero value
// ⇒ no adapter reservation, so the block count is byte-identical to a pre-LoRA
// build (INV-6). A variadic Option (mirroring latency.Option for NewLatencyModel,
// #1467) keeps the existing positional call sites unchanged.
type kvCapacityOptions struct {
	adapterReservedBytes int64
	expertParallelSize   int
}

// KVCapacityOption customizes CalculateKVBlocks.
type KVCapacityOption func(*kvCapacityOptions)

// WithAdapterReservedBytes reserves a fixed, capacity-based block of GPU HBM for
// resident LoRA adapters, subtracted once at startup beside model weights (the
// static memory model, design D2 / INV-L4). The value is the sim/lora cost model's
// pure AdapterReservedBytes() query (capacity × per-slot footprint); 0 (or the
// option absent) leaves KV capacity unchanged (INV-6 no-op). A negative value is
// rejected by CalculateKVBlocks.
func WithAdapterReservedBytes(bytes int64) KVCapacityOption {
	return func(o *kvCapacityOptions) { o.adapterReservedBytes = bytes }
}

// WithExpertParallelSize declares the expert-parallel (EP) group size: the number of
// GPUs the model's ROUTED-EXPERT weights are sharded across (#1656). Under vLLM's
// --enable-expert-parallel, ep_size = dp_size·tp_size and each rank owns
// num_experts/ep_size WHOLE experts; every other weight (attention, shared experts,
// dense-prefix MLP, the router/gate, embeddings, norms) stays on the TP-sharded path,
// so the EP-ON per-GPU routed footprint this option produces (R/ep) matches vLLM.
//
// The EP-OFF baseline it is differenced against is BLIS's own DP model, NOT vLLM's:
// BLIS models MoE --dp N as N independent single-node engine replicas (#1531), each
// holding the full model tensor-sharded across its TP GPUs, so routed experts are
// charged at R/tp per GPU. vLLM instead flattens TP across DP for MoE layers
// unconditionally (FusedMoEParallelConfig.make → flatten_tp_across_dp_and_pcp: at
// TP=2/DP=2 with EP OFF the MoE tp_size is 4), so real vLLM per-GPU routed bytes are
// R/(tp·dp) in BOTH modes. BLIS's EP-off DP>1 capacity is therefore CONSERVATIVE
// (over-charges) relative to vLLM — a divergence in BLIS's DP model rather than in this
// option, tracked separately; the trained-physics step-time model already uses the
// flattened TP·DP group.
//
// The canonical value is sim.EffectiveEPSize(isMoE, tp, dp, enableExpertParallel) —
// computed from the LOGICAL, user-requested topology. A per-instance config is not a
// safe source: DP-as-placement (#1531) reconfigures each engine replica to DP=1, so a
// config-bound EP collapses to TP and the sharding silently vanishes.
//
// 0 (the option absent), 1 (sim.EffectiveEPSize's "EP off" value), and any value <= tp
// all mean "no sharding beyond the TP group": the pre-#1656 accounting ⇒ byte-identical
// block counts (INV-6). A larger value must satisfy ep <= tp·dp — the EP group cannot
// span more GPUs than the deployment has — and a group wider than the routed-expert
// count is CLAMPED to that count with a warning (the loaded ranks still hold one whole
// expert each; charging the sub-one-expert average would be optimistic).
func WithExpertParallelSize(ep int) KVCapacityOption {
	return func(o *kvCapacityOptions) { o.expertParallelSize = ep }
}

// resolveExpertShardSize resolves the WithExpertParallelSize option into the number of
// GPUs routed-expert weights are sharded across, given the rank's TP degree, the
// deployment's DP degree, and the model's routed-expert count (0 for a model with no
// routed experts). EP off (0 or 1) resolves to tp — the pre-#1656 accounting.
//
// Callers must have validated tp > 0 and dp >= 1 before calling (CalculateKVBlocks does,
// via KVBytesPerToken and its own dp guard), so the bounds below cannot be reported
// against a degenerate topology. int64 arithmetic in the upper bound keeps an absurd
// tp·dp from overflowing into a permissive comparison.
//
// The bounds encode vLLM's flattened MoE group (ep_size = dp_size·tp_size), which is the
// deployment model BLIS represents; they are deliberately not a general law (SGLang's
// independent --ep-size, for instance, satisfies ep_size <= tp_size).
//
// Two bad-input directions, handled differently on purpose — the resolved value is a
// DIVISOR of the routed-expert bytes (charge = R·tp/shard), so a LARGER resolved value
// charges LESS memory and a SMALLER one charges MORE:
//
//   - ep > tp·dp is REJECTED. There is no safe value to substitute: the caller has
//     described a deployment with more EP ranks than GPUs, and honouring it would charge
//     only tp·dp/ep of the routed experts — capacity for memory that does not exist.
//     This bound is load-bearing and is deliberately NOT clamped.
//   - ep > num_routed_experts is CLAMPED DOWN to num_routed_experts. Clamping down
//     shrinks the divisor, so it charges MORE memory than the caller asked for — the
//     conservative direction, and the physically right one (see the clamp site below).
//
// So both directions end up at or above the true footprint; neither can inflate capacity.
func resolveExpertShardSize(ep, tp, dp, numRoutedExperts int) (int, error) {
	if ep < 0 {
		return 0, fmt.Errorf("expert-parallel group size must be >= 0 (0 or 1 means EP off), got %d", ep)
	}
	// Defensive: CalculateKVBlocks guarantees tp > 0 here, but this function divides by tp
	// below (the overflow probe), so a future caller must not be able to reach that with a
	// zero TP.
	if tp < 1 {
		return 0, fmt.Errorf("expert-parallel group size cannot be resolved for TP %d (must be > 0)", tp)
	}
	// Anything at or below tp means "no sharding beyond the TP group" and resolves to the
	// pre-#1656 accounting. This includes the option's off sentinels (0, 1) AND the very
	// common ep == tp case — a DP=1 deployment with --enable-expert-parallel, where the EP
	// group IS the TP group. Those must never fail validation: the weight arithmetic is a
	// strict no-op for them (INV-6), so a bound rejection here would break configurations
	// that size fine, without changing any number. A value in (1, tp) is not a vLLM
	// topology (its EP group is tp·dp ≥ tp), and BLIS's arithmetic cannot express charging
	// MORE than the TP-sharded footprint, so it too resolves to tp rather than pretending
	// to model it.
	if ep <= tp {
		return tp, nil
	}
	// Go's int64 multiply wraps silently, and a wrapped (negative or small) product would
	// turn the upper bound below into a permissive one. Detect it instead of trusting it.
	maxGroup := int64(tp) * int64(dp)
	if maxGroup/int64(tp) != int64(dp) {
		return 0, fmt.Errorf("expert-parallel bound is not computable: TP·DP overflows int64 (tp=%d, dp=%d)", tp, dp)
	}
	if int64(ep) > maxGroup {
		return 0, fmt.Errorf("expert-parallel group size (%d) must be <= TP·DP (%d·%d = %d): the EP group cannot span more GPUs than the deployment has", ep, tp, dp, maxGroup)
	}
	// A group wider than the routed-expert count is a real planning input for wide EP
	// (DeepSeek-class EP320 over 256 experts, and any deployment using --enable-eplb /
	// --num-redundant-experts; vLLM does not reject it either — only --enable-eplb requires
	// an even distribution), so it is CLAMPED, not rejected.
	//
	// Why num_routed_experts is the right clamp: the ranks that hold an expert hold one
	// WHOLE expert, so num_experts is the widest divisor the weight footprint can support.
	// Charging the sub-one-expert average num_experts/ep would be optimistic — capacity for
	// memory that does not exist — while clamping DOWN to num_experts shrinks the divisor
	// and therefore charges MORE memory (fewer KV blocks): the safe direction.
	//
	// Why not reject: rejecting was tried during review of #1656 and is strictly worse. It
	// fired on the INERT ep == tp case (every DP=1 --enable-expert-parallel run, where the
	// block count does not change at all), breaking configurations that had always sized —
	// e.g. --tp 16 --enable-expert-parallel on an 8-expert Mixtral — and for genuine wide EP
	// it replaced the CLI's honest unsupported-topology diagnostic (#1548) with an
	// expert-count fatal, which is the masking the feature exists to remove.
	//
	// Warned, never silent (R1): the caller's requested group, the model's expert count and
	// the substituted divisor are all named on stderr.
	if clamped, wasClamped := ClampExpertShardToExpertCount(ep, numRoutedExperts); wasClamped {
		logrus.Warnf("KV capacity: expert-parallel group size %d exceeds the model's routed-expert count %d; "+
			"charging routed-expert weights over %d GPUs (one whole expert each) instead. Expert redundancy "+
			"(--enable-eplb / --num-redundant-experts) is not modeled.", ep, numRoutedExperts, clamped)
		return clamped, nil
	}
	return ep, nil
}

// ClampExpertShardToExpertCount is the shared "a loaded rank holds one WHOLE expert" rule
// for the routed-expert WEIGHT divisor (#1548), used by both the KV-capacity model and the
// trained-physics step-time model so the two cannot disagree about the same experts (R23 —
// the agreement sim.ModelHardwareConfig.EffectiveExpertShardGroupSize documents).
//
// An expert-parallel group wider than the routed-expert count is a legitimate planning input
// (DeepSeek-class EP320 over 256 experts; anything using --enable-eplb /
// --num-redundant-experts), so it is CLAMPED rather than rejected. num_experts is the widest
// divisor the weight footprint can support: charging the sub-one-expert average
// num_experts/ep would model memory that does not exist, which is the optimistic direction.
// Clamping DOWN shrinks the divisor and so charges MORE weight bytes — the safe direction.
//
// It returns wasClamped so each caller can word its own diagnostic (R1: never silent);
// numRoutedExperts <= 0 (a dense model, or an unpopulated config) is a no-op.
//
// IMPORTANT — this rule holds only for EXPERT-PARALLEL sharding, where each rank owns whole
// experts. It must NOT be applied to the expert-parallel-OFF divisor, where experts are
// TENSOR-sharded and a rank genuinely holds a FRACTION of every expert: num_experts/group
// below 1 full-expert-equivalent is the correct charge there (e.g. 8 experts tensor-sharded
// over a 16-GPU TP·DP group is 0.5 each). Nor to the dispatch/combine collective, which
// really does run over every rank in the group.
func ClampExpertShardToExpertCount(ep, numRoutedExperts int) (int, bool) {
	if numRoutedExperts > 0 && ep > numRoutedExperts {
		return numRoutedExperts, true
	}
	return ep, false
}

// CalculateKVBlocks computes the maximum number of KV cache blocks that fit
// in GPU memory after accounting for model weights, activations, non-PyTorch
// overhead, and (optionally) the static LoRA adapter HBM reservation. The base
// formula matches the llm-d-benchmark capacity_planner.py reference.
//
// Parameters:
//
//   - mc: model architecture (layers, heads, dims, precision)
//
//   - hc: GPU hardware calibration (must include MemoryGiB)
//
//   - tp: tensor parallelism degree (must be > 0)
//
//   - dp: data parallelism degree (must be > 0). For an MoE model with dp > 1 the
//     aggregate usable KV-block count scales by dp: each DP rank is a separate vLLM
//     EngineCore with its own full KV budget on its own GPUs, and requests split
//     disjointly across ranks (vllm@f6ec81c7 v1/engine/core.py:1243-1276). Per-GPU KV
//     bytes are unaffected (sized by attention TP only), so dp multiplies only the
//     final block total. Per-token KV bytes stay EP-mode-independent — EP shards only
//     MoE experts, never attention/KV — but the model's WEIGHT footprint is not: see
//     WithExpertParallelSize (#1656), which shards the routed-expert term across the
//     EP group and so changes how much memory is left over for KV.
//
//     The isMoE gate below is the active correctness guard for dense dp > 1: the CLI
//     also rejects dense dp > 1 and roofline dp > 1 (in resolveLatencyConfig,
//     cmd/root.go; added in #1417), but on the run whole-instance auto-capacity path
//     that rejection fires slightly AFTER this call (same resolver function). So this
//     call must itself be safe: dense → not scaled (gate), roofline MoE → scaled but
//     the result is discarded when the CLI aborts. The gate is load-bearing, not
//     merely redundant.
//
//   - blockSize: tokens per KV cache block (must be > 0)
//
//   - gpuMemoryUtilization: fraction of GPU HBM available for KV cache (must be in (0, 1.0])
//
//   - params: MoE indicators, activation type, embedding tying
//
// Returns the number of blocks, or an error if inputs are invalid or memory
// budget is insufficient.
func CalculateKVBlocks(mc sim.ModelConfig, hc sim.HardwareCalib, tp int, dp int, blockSize int64, gpuMemoryUtilization float64, params KVCapacityParams, options ...KVCapacityOption) (int64, error) {
	var opts kvCapacityOptions
	for _, o := range options {
		o(&opts)
	}

	// --- Input validation (R3, R11) ---
	if gpuMemoryUtilization <= 0 || gpuMemoryUtilization > 1.0 || math.IsNaN(gpuMemoryUtilization) || math.IsInf(gpuMemoryUtilization, 0) {
		return 0, fmt.Errorf("CalculateKVBlocks: gpuMemoryUtilization must be in (0, 1.0], got %v", gpuMemoryUtilization)
	}
	if opts.adapterReservedBytes < 0 {
		return 0, fmt.Errorf("CalculateKVBlocks: adapterReservedBytes must be >= 0, got %d", opts.adapterReservedBytes)
	}
	if blockSize <= 0 {
		return 0, fmt.Errorf("CalculateKVBlocks: block size must be > 0, got %d", blockSize)
	}
	if dp < 1 {
		return 0, fmt.Errorf("CalculateKVBlocks: dp must be >= 1, got %d", dp)
	}
	if mc.IntermediateDim <= 0 {
		return 0, fmt.Errorf("CalculateKVBlocks: intermediate_dim must be > 0, got %d", mc.IntermediateDim)
	}
	if mc.VocabSize <= 0 {
		return 0, fmt.Errorf("CalculateKVBlocks: vocab_size must be > 0, got %d", mc.VocabSize)
	}
	if hc.MemoryGiB <= 0 || math.IsNaN(hc.MemoryGiB) || math.IsInf(hc.MemoryGiB, 0) {
		return 0, fmt.Errorf("CalculateKVBlocks: GPU memory (MemoryGiB) must be a valid positive number, got %v", hc.MemoryGiB)
	}
	// WeightBytesPerParam is optional (0 = not set, fall back to BytesPerParam).
	// When set, it must be a valid positive number.
	if mc.WeightBytesPerParam != 0 {
		if mc.WeightBytesPerParam < 0 || math.IsNaN(mc.WeightBytesPerParam) || math.IsInf(mc.WeightBytesPerParam, 0) {
			return 0, fmt.Errorf("CalculateKVBlocks: WeightBytesPerParam must be positive when set, got %v", mc.WeightBytesPerParam)
		}
	}

	// Only SwiGLU-family activations are supported (3-matrix MLP).
	if !swiGLUActivations[params.HiddenAct] {
		return 0, fmt.Errorf("CalculateKVBlocks: unsupported activation %q; only SwiGLU-family activations (silu, swiglu, geglu, situ) are supported", params.HiddenAct)
	}

	// --- Step 1-2: Per-token KV bytes per GPU ---
	perTokenKVBytesPerGPUF, err := KVBytesPerToken(mc, tp)
	if err != nil {
		return 0, fmt.Errorf("CalculateKVBlocks: %w", err)
	}

	// --- Expert-parallel weight sharding (#1656) ---
	// Resolved after KVBytesPerToken so tp > 0 is already guaranteed (and after the dp
	// guard above), keeping the bound messages meaningful. EP off ⇒ tp ⇒ the weight
	// term below is bit-identical to the pre-#1656 expression (INV-6).
	// routedExpertCount is 0 unless the weight estimator's own MoE branch will run, so the
	// resolver's expert-count clamp uses exactly the predicate the arithmetic uses (R23).
	routedExpertCount := 0
	if params.hasRoutedExperts() {
		routedExpertCount = params.NumLocalExperts
	}
	expertShardSize, epErr := resolveExpertShardSize(opts.expertParallelSize, tp, dp, routedExpertCount)
	if epErr != nil {
		return 0, fmt.Errorf("CalculateKVBlocks: %w", epErr)
	}

	// --- Step 3: Per-block bytes ---
	// Multiply by blockSize before truncating to int64 to avoid loss when the
	// per-token value is fractional (e.g., INT4 quantization with small head dims).
	perBlockBytes := int64(perTokenKVBytesPerGPUF * float64(blockSize))
	if perBlockBytes <= 0 {
		return 0, fmt.Errorf(
			"CalculateKVBlocks: per-block KV bytes is %d (expected > 0); "+
				"perTokenKVBytesPerGPU=%.6f, blockSize=%d — check BytesPerParam and TP",
			perBlockBytes, perTokenKVBytesPerGPUF, blockSize)
	}

	// --- Step 4: Available memory budget (total across all TP GPUs) ---
	// Reference: available_memory = gpu_mem * gpu_mem_util * gpu_count
	totalAvailableGiB := hc.MemoryGiB * gpuMemoryUtilization * float64(tp)

	// Model weights: total model size (distributed across TP GPUs, but sum = total)
	modelWeightBytes := computeModelWeightBytes(mc, params, tp, expertShardSize)
	modelWeightGiB := float64(modelWeightBytes) / float64(gibToBytes)

	// Activation memory: per-replica constant, NOT multiplied by TP. This budget is
	// computed per DP rank; dp scaling (#1420) applies only to the final block count,
	// not to per-rank overhead (each rank has its own GPUs with this same overhead).
	var activationGiB float64
	if params.IsMoE {
		activationGiB = activationMemoryMoEGiB
	} else {
		activationGiB = activationMemoryDenseGiB
	}

	// Non-torch overhead: per-GPU (NCCL buffers, CUDA context) × number of GPUs
	var nonTorchPerGPU float64
	if tp == 1 {
		nonTorchPerGPU = nonTorchMemoryTP1GiB
	} else {
		nonTorchPerGPU = nonTorchMemoryTPMultiGiB
	}
	nonTorchGiB := nonTorchPerGPU * float64(tp)

	// Static LoRA adapter HBM reservation (D2 / INV-L4): a fixed, capacity-based
	// block of memory reserved once beside model weights. Treated EXACTLY like
	// model weights: a per-DP-rank overhead that is NOT multiplied by dp here. Each
	// DP rank is an independent EngineCore on its own GPUs that reserves its own
	// adapter slots, so the per-rank budget subtracts the reservation once; the
	// dp block-count scaling below then aggregates per-rank budgets into the
	// instance total (multiplying the reservation by dp here as well would
	// double-count it — same reasoning that keeps modelWeightGiB per-rank). The
	// reservation is also TP-independent (a total across the rank's TP GPUs, since
	// the adapter A/B matrices are sharded like weights). Zero when no
	// adapters/capacity are configured (INV-6).
	adapterReservedGiB := float64(opts.adapterReservedBytes) / float64(gibToBytes)

	overheadGiB := modelWeightGiB + activationGiB + nonTorchGiB + adapterReservedGiB
	if overheadGiB >= totalAvailableGiB {
		perGPUAvailable := hc.MemoryGiB * gpuMemoryUtilization

		// Calculate minimum TP needed: use TP-independent overhead (weights +
		// activation + adapter reservation) and subtract per-GPU non-torch overhead
		// from available capacity. The static adapter reservation is TP-independent
		// (sharded like weights, total constant), so it raises the minimum TP.
		// For TP>1, use nonTorchMemoryTPMultiGiB (0.6 GiB/GPU) to account for NCCL/CUDA overhead.
		nonTorchPerGPUForMinTP := nonTorchMemoryTPMultiGiB
		perGPUCapacity := perGPUAvailable - nonTorchPerGPUForMinTP

		if perGPUCapacity <= 0 {
			return 0, fmt.Errorf(
				"CalculateKVBlocks: insufficient per-GPU capacity (%.2f GiB available - %.2f GiB non-torch overhead = %.2f GiB). "+
					"Cannot fit model even with increased TP",
				perGPUAvailable, nonTorchPerGPUForMinTP, perGPUCapacity)
		}

		tpIndependentOverhead := modelWeightGiB + activationGiB + adapterReservedGiB
		minTP := int(math.Ceil(tpIndependentOverhead / perGPUCapacity))

		// Under expert parallelism the weight term already reflects routed experts sharded
		// across the EP group, so the minimum-GPU estimate is only meaningful under the
		// assumption the CLI actually satisfies: that the EP group grows with TP at the
		// same DP (ep = TP·DP), which makes the routed term R/DP and hence TP-independent.
		// Under a literally fixed group the per-GPU floor R/ep never shrinks with TP and
		// the number can be unreachable, so name the assumption rather than let the number
		// read as EP-blind. Empty (message byte-identical to the pre-#1656 text) whenever
		// the EP reduction is not active.
		epNote := ""
		if expertShardSize > tp {
			epNote = fmt.Sprintf(" Routed-expert weights are sharded across the expert-parallel group (size %d); "+
				"the minimum-GPU estimate assumes that group grows with TP at the same DP (ep = TP·DP).", expertShardSize)
		}

		return 0, fmt.Errorf(
			"CalculateKVBlocks: model overhead (%.2f GiB = %.2f weights + %.2f activation + %.2f non-torch + %.2f lora-adapter-reservation) "+
				"exceeds available GPU memory (%.2f GiB = %.1f GiB × %.0f%% util × %d GPUs). "+
				"Minimum GPUs required per instance: %d%s",
			overheadGiB, modelWeightGiB, activationGiB, nonTorchGiB, adapterReservedGiB,
			totalAvailableGiB, hc.MemoryGiB, gpuMemoryUtilization*100, tp, minTP, epNote)
	}

	allocatableGiB := totalAvailableGiB - overheadGiB
	allocatableBytes := int64(allocatableGiB * float64(gibToBytes))

	// --- Step 5: Total blocks (per DP rank) ---
	totalBlocks := allocatableBytes / perBlockBytes
	if totalBlocks <= 0 {
		return 0, fmt.Errorf(
			"CalculateKVBlocks: computed 0 blocks (allocatable=%.2f GiB, per_block=%d bytes)",
			allocatableGiB, perBlockBytes)
	}

	// --- Step 6: DP scaling (#1420) ---
	// All sizing above is per DP rank (one EngineCore on its own TP GPUs). For an MoE
	// model with dp > 1, vLLM runs dp independent EngineCores each with this full KV
	// budget and splits requests disjointly across them, so the aggregate usable block
	// count scales by dp. The IsMoE gate is the active guard: it ensures a dense model
	// is never scaled even if dp > 1 reaches here (which can happen on the run
	// whole-instance path, where this call precedes the CLI's dense-dp>1 rejection).
	if params.IsMoE && dp > 1 {
		totalBlocks *= int64(dp)
	}

	return totalBlocks, nil
}

// computeModelWeightBytes estimates the model weight bytes CHARGED AGAINST ONE DP
// RANK's TP-GPU memory budget, using the standard transformer architecture formula.
// Matches capacity_planner.py.
//
// The value is a TOTAL over the rank's tp GPUs: CalculateKVBlocks compares it against
// gpu_mem × util × tp, so the implicit per-GPU charge is (returned bytes)/tp — i.e.
// weights are modeled as tensor-sharded across the TP group.
//
// expertShardSize is the number of GPUs the ROUTED-EXPERT weights are sharded across
// (#1656): tp when expert parallelism is off (experts replicated per DP rank, the
// pre-#1656 accounting), or the EP group size tp·dp when it is on. Because the return
// value is a per-rank total on a tp basis, the routed-expert term is scaled by
// tp/expertShardSize — which makes the per-GPU charge routedBytes/expertShardSize,
// NOT routedBytes/(tp·expertShardSize). Every other term (attention, shared experts,
// dense-prefix MLP, router/gate, embeddings, norms) stays TP-sharded, mirroring vLLM:
// only FusedMoE routed-expert weights are distributed over the EP group.
//
// expertShardSize <= tp (or a model with no routed experts) returns the original
// expression evaluated in the original order, so EP-off results are bit-identical
// (INV-6).
func computeModelWeightBytes(mc sim.ModelConfig, params KVCapacityParams, tp, expertShardSize int) int64 {
	hiddenDim := int64(mc.HiddenDim)
	vocabSize := int64(mc.VocabSize)
	numLayers := int64(mc.NumLayers)
	intermediateDim := int64(mc.IntermediateDim)

	numKVHeads := mc.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = mc.NumHeads
	}
	// Effective head dim: explicit head_dim (F1, #1527) when set, else hidden/heads.
	headDim := int64(mc.EffectiveHeadDim())
	kvDim := int64(numKVHeads) * headDim

	// Embeddings: vocab_size * hidden_dim
	embeddings := vocabSize * hiddenDim

	// Attention per layer: Q proj + K proj + V proj + output proj
	// Q: hidden_dim * hidden_dim
	// K: hidden_dim * kv_dim
	// V: hidden_dim * kv_dim
	// O: hidden_dim * hidden_dim
	attentionPerLayer := hiddenDim*(hiddenDim+2*kvDim) + hiddenDim*hiddenDim

	// MLP per layer: SwiGLU uses 3 matrices (gate, up, down) to match capacity_planner.py.
	// NOTE: roofline step time (mlpMatrixCount in roofline.go) uses 2-matrix convention for
	// FLOPs/bandwidth — see that function's comment for the calibration rationale.
	//
	// Dense MLP term for a pure dense model: gate + up + down at intermediate_size.
	// (Unchanged from the pre-F3 behavior — INV-6.)
	denseMLPPerLayer := 3 * hiddenDim * intermediateDim

	// Total MLP params across all layers. For a MoE model with a dense prefix
	// (first_k_dense_replace = K, #1527 F3), the first K layers use the dense MLP
	// term and the remaining (numLayers - K) use the MoE term — a prefix split,
	// distinct from the every-Nth InterleaveMoELayerStep. K == 0 (or absent) ⇒ all
	// layers MoE ⇒ byte-identical to the pre-F3 all-MoE accounting (INV-6).
	var totalMLPParams int64
	// routedExpertParams is the routed-expert (FusedMoE) share of totalMLPParams — the
	// only term expert parallelism shards differently from tensor parallelism (#1656).
	// It stays 0 for a dense model and for a degenerate sub-threshold expert count, so
	// those models are EP-inert by construction.
	var routedExpertParams int64
	// hasRoutedExperts requires NumLocalExperts >= MoEMinExperts as well as IsMoE, which is
	// a defensive guard rather than a duplicate: NewKVCapacityParams is a public positional
	// constructor, so a caller could pass an inconsistent (IsMoE=true, NumLocalExperts<2)
	// pair. The MoE arithmetic below multiplies by NumLocalExperts, so a degenerate count
	// would silently produce zero/under-weighted MLP bytes — this keeps it on the dense path.
	if params.hasRoutedExperts() {
		// MoE per-layer term: use per-expert FFN dim for routed experts, add shared and gate.
		expertFFNDim := intermediateDim // Mixtral convention: IntermediateDim IS per-expert
		if params.MoEExpertFFNDim > 0 {
			expertFFNDim = int64(params.MoEExpertFFNDim)
		}
		// All routed experts (total model weight, not active)
		routedExpertPerLayer := 3 * hiddenDim * expertFFNDim * int64(params.NumLocalExperts)
		moeMLPPerLayer := routedExpertPerLayer
		// Shared experts
		if params.SharedExpertFFNDim > 0 {
			moeMLPPerLayer += 3 * hiddenDim * int64(params.SharedExpertFFNDim)
		}
		// Router weights: num_local_experts * hidden_dim per layer
		moeMLPPerLayer += int64(params.NumLocalExperts) * hiddenDim

		// Dense-prefix MLP term. Within a hybrid MoE model the dense-prefix layers
		// use DenseIntermediateDim (intermediate_size_mlp) when set, else
		// intermediate_size — mirroring the step-time dense-layer sizing
		// (roofline.go / trained_physics_model.go) so capacity and step-time agree
		// on the dense FFN dim for Scout-style hybrids. Scoped to the MoE branch so a
		// pure dense model's estimate is unchanged (INV-6).
		densePrefixMLPPerLayer := denseMLPPerLayer
		if mc.DenseIntermediateDim > 0 {
			densePrefixMLPPerLayer = 3 * hiddenDim * int64(mc.DenseIntermediateDim)
		}

		// Dense-prefix split (F3). Clamp K to [0, numLayers] so a degenerate
		// first_k_dense_replace >= numLayers yields all-dense with no negative count.
		numDenseLayers := int64(mc.FirstKDenseReplace)
		if numDenseLayers < 0 {
			numDenseLayers = 0
		}
		if numDenseLayers > numLayers {
			numDenseLayers = numLayers
		}
		numMoELayers := numLayers - numDenseLayers
		totalMLPParams = numDenseLayers*densePrefixMLPPerLayer + numMoELayers*moeMLPPerLayer
		// Only the MoE layers carry routed experts; the dense-prefix layers do not.
		routedExpertParams = numMoELayers * routedExpertPerLayer
	} else {
		// Dense model: every layer uses the dense MLP term.
		totalMLPParams = numLayers * denseMLPPerLayer
	}

	// Layer norms: 2 per layer (pre-attention + pre-MLP), each = hidden_dim params
	layerNormsPerLayer := 2 * hiddenDim

	// Attention + layer norms are per-layer for ALL layers (dense and MoE alike);
	// only the MLP term differs by layer type (handled in totalMLPParams above).
	attentionAndNormsAllLayers := numLayers * (attentionPerLayer + layerNormsPerLayer)

	// lm_head: vocab_size * hidden_dim (omitted if tie_word_embeddings)
	var lmHead int64
	if !params.TieWordEmbeddings {
		lmHead = vocabSize * hiddenDim
	}

	// Final layer norm: hidden_dim
	finalNorm := hiddenDim

	totalParams := embeddings + attentionAndNormsAllLayers + totalMLPParams + lmHead + finalNorm

	// Expert-parallel sharding of the routed-expert term (#1656). Guarded so the EP-off
	// path returns the original expression untouched (INV-6), and so a non-MoE model —
	// which has no routed experts to shard — is inert even if a group size is supplied.
	if expertShardSize > tp && routedExpertParams > 0 {
		nonRoutedParams := totalParams - routedExpertParams
		scaledRoutedParams := float64(routedExpertParams) * float64(tp) / float64(expertShardSize)
		return int64((float64(nonRoutedParams) + scaledRoutedParams) * mc.EffectiveWeightBytesPerParam())
	}

	return int64(float64(totalParams) * mc.EffectiveWeightBytesPerParam())
}

// ExtractKVCapacityParamsFromFile reads a HuggingFace config.json file and
// extracts the KVCapacityParams needed for CalculateKVBlocks.
func ExtractKVCapacityParamsFromFile(hfConfigPath string) (KVCapacityParams, error) {
	hf, err := ParseHFConfig(hfConfigPath)
	if err != nil {
		return KVCapacityParams{}, fmt.Errorf("extract KV capacity params: %w", err)
	}
	params, err := ExtractKVCapacityParams(hf)
	if err != nil {
		return KVCapacityParams{}, fmt.Errorf("extract KV capacity params: %w", err)
	}
	return params, nil
}

// ExtractKVCapacityParams extracts KVCapacityParams from a parsed HFConfig.
// MoE detection uses the shared (*HFConfig).ResolveNumExperts (>= MoEMinExperts);
// see that method for the field set and resolution order. Returns an error if MoE
// is detected via activation-count fields (n_shared_experts / num_experts_per_tok,
// and their Kimi-K3 aliases num_shared_experts / num_experts_per_token; #1634)
// without a total expert count — weight estimation requires the count. Shared-expert
// resolution uses moeSharedExpertFields so this path matches GetModelConfigFromHF
// (R23 code-path parity).
func ExtractKVCapacityParams(hf *HFConfig) (KVCapacityParams, error) {
	hiddenAct := hf.MustGetString("hidden_act", "")
	tieWordEmbeddings := false
	if tied, ok := hf.GetBool("tie_word_embeddings"); ok {
		tieWordEmbeddings = tied
	}

	// MoE expert count: resolved via the shared chain (R23 code-path parity with
	// GetModelConfigFromHF). Single-expert models are dense-equivalent and must not
	// enter the MoE weight-estimation path below.
	numLocalExperts := hf.ResolveNumExperts()

	if numLocalExperts >= sim.MoEMinExperts {
		// Extract per-expert and shared expert dims for weight estimation
		moeExpertFFNDim := hf.MustGetInt("moe_intermediate_size", 0)
		var sharedExpertFFNDim int
		if v := hf.MustGetInt("shared_expert_intermediate_size", 0); v > 0 {
			sharedExpertFFNDim = v
		} else if nShared := hf.mustGetIntFallback(0, moeSharedExpertFields...); nShared > 0 {
			perExpert := moeExpertFFNDim
			if perExpert == 0 {
				perExpert = hf.MustGetInt("intermediate_size", 0)
			}
			sharedExpertFFNDim = nShared * perExpert
		}
		return NewKVCapacityParams(true, numLocalExperts, tieWordEmbeddings, hiddenAct, moeExpertFFNDim, sharedExpertFFNDim), nil
	}

	// Activation-count or shared-expert fields: signal MoE but don't provide
	// a reliable total expert count. Without the total count, weight estimation
	// would use dense MLP weights — massively underestimating MoE model size.
	// Return an error so the caller can fall back to --total-kv-blocks.
	signalFields := append(append([]string{}, moeSharedExpertFields...), moeActiveExpertFields...)
	for _, key := range signalFields {
		if v := hf.MustGetInt(key, 0); v > 0 {
			return KVCapacityParams{}, fmt.Errorf(
				"model appears to be MoE (%s=%d) but num_local_experts is missing; "+
					"cannot estimate weight size accurately. Set --total-kv-blocks explicitly", key, v)
		}
	}

	return NewKVCapacityParams(false, 0, tieWordEmbeddings, hiddenAct, 0, 0), nil
}
