package latency

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
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
	"":       true,
}

// KVBytesPerToken computes the per-GPU KV cache bytes per token for a given
// model config and tensor parallelism degree. This is used for both KV cache
// capacity sizing and PD transfer duration estimation.
//
// The formula is: NumLayers × 2 (K+V) × headDim × numKVHeads × BytesPerParam / TP
//
// Uses BytesPerParam (compute/activation dtype), not WeightBytesPerParam, since
// KV cache is stored at compute precision regardless of weight quantization.
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
	// calls computeModelWeightBytes for ALL configs — including MLA — and that
	// weight estimate reads NumHeads/HiddenDim (via EffectiveHeadDim and the
	// attention/embedding terms). A degenerate MLA config that returned a valid KV
	// value here but then silently under-counted (NumHeads==0 → kvDim 0) or wildly
	// inflated (HiddenDim==0 → ~0 weight → huge block count) weight would run on
	// garbage with no error, so these guards must gate the MLA path too. Only the
	// HiddenDim%NumHeads divisibility check is MLA-exempt (the latent path never
	// uses that quotient) and stays on the standard path below.
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

	// --- MLA (Multi-head Latent Attention) path (F2, #1527) ---
	// MLA models (DeepSeek-V2/V3, Kimi-K3, GLM-5.2 `glm_moe_dsa`) store a single
	// compressed latent KV vector per token per layer of width
	// (kv_lora_rank + qk_rope_head_dim) — NOT the standard 2 × headDim × numKVHeads.
	// The latent is shared across all heads (so it does not scale with numKVHeads
	// or headDim) and is REPLICATED across TP ranks rather than sharded, so the
	// per-GPU value is the full latent width and is NOT divided by TP (design D2;
	// matches vLLM's MLA cache, e.g. DeepSeek (512+64)×2 bytes/token/layer). This
	// branch is selected only when KVLoraRank > 0; a standard MHA/GQA config
	// (KVLoraRank == 0) takes the byte-identical standard path below (INV-6). Only
	// the hidden%heads divisibility guard is skipped here (the latent path never
	// uses the hidden/heads quotient); NumHeads>0 and HiddenDim>0 are validated in
	// the common block above so computeModelWeightBytes stays protected for MLA too.
	if mc.IsMLA() {
		if mc.QKRopeHeadDim < 0 {
			return 0, fmt.Errorf("KVBytesPerToken: qk_rope_head_dim must be >= 0, got %d", mc.QKRopeHeadDim)
		}
		latentWidth := mc.KVLoraRank + mc.QKRopeHeadDim
		perTokenKVBytesF := float64(mc.NumLayers) * float64(latentWidth) * mc.BytesPerParam
		if perTokenKVBytesF <= 0 {
			return 0, fmt.Errorf("KVBytesPerToken: MLA computed value is %.4f (expected > 0); check BytesPerParam=%.4f, kv_lora_rank=%d, qk_rope_head_dim=%d",
				perTokenKVBytesF, mc.BytesPerParam, mc.KVLoraRank, mc.QKRopeHeadDim)
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
	perTokenKVBytesF := float64(mc.NumLayers) * 2.0 * float64(headDim) * float64(numKVHeads) * mc.BytesPerParam
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
//     final block total. KV capacity is EP-mode-independent — EP shards only MoE
//     experts, never attention/KV — so there is intentionally no EP parameter.
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
		return 0, fmt.Errorf("CalculateKVBlocks: unsupported activation %q; only SwiGLU-family activations (silu, swiglu, geglu) are supported", params.HiddenAct)
	}

	// --- Step 1-2: Per-token KV bytes per GPU ---
	perTokenKVBytesPerGPUF, err := KVBytesPerToken(mc, tp)
	if err != nil {
		return 0, fmt.Errorf("CalculateKVBlocks: %w", err)
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
	modelWeightBytes := computeModelWeightBytes(mc, params)
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

		return 0, fmt.Errorf(
			"CalculateKVBlocks: model overhead (%.2f GiB = %.2f weights + %.2f activation + %.2f non-torch + %.2f lora-adapter-reservation) "+
				"exceeds available GPU memory (%.2f GiB = %.1f GiB × %.0f%% util × %d GPUs). "+
				"Minimum GPUs required per instance: %d",
			overheadGiB, modelWeightGiB, activationGiB, nonTorchGiB, adapterReservedGiB,
			totalAvailableGiB, hc.MemoryGiB, gpuMemoryUtilization*100, tp, minTP)
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

// computeModelWeightBytes estimates total model weight bytes using the
// standard transformer architecture formula. Matches capacity_planner.py.
func computeModelWeightBytes(mc sim.ModelConfig, params KVCapacityParams) int64 {
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
	// The NumLocalExperts >= MoEMinExperts clause is a defensive guard, not a
	// duplicate of IsMoE: NewKVCapacityParams is a public positional constructor, so
	// a caller could pass an inconsistent (IsMoE=true, NumLocalExperts<2) pair. The
	// MoE arithmetic below multiplies by NumLocalExperts, so a degenerate count would
	// silently produce zero/under-weighted MLP bytes — this keeps it on the dense path.
	if params.IsMoE && params.NumLocalExperts >= sim.MoEMinExperts {
		// MoE per-layer term: use per-expert FFN dim for routed experts, add shared and gate.
		expertFFNDim := intermediateDim // Mixtral convention: IntermediateDim IS per-expert
		if params.MoEExpertFFNDim > 0 {
			expertFFNDim = int64(params.MoEExpertFFNDim)
		}
		// All routed experts (total model weight, not active)
		moeMLPPerLayer := 3 * hiddenDim * expertFFNDim * int64(params.NumLocalExperts)
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
// is detected via activation-count fields (n_shared_experts, num_experts_per_tok)
// without a total expert count — weight estimation requires the count.
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
		} else if nShared := hf.MustGetInt("n_shared_experts", 0); nShared > 0 {
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
	for _, key := range []string{"n_shared_experts", "num_experts_per_tok"} {
		if v := hf.MustGetInt(key, 0); v > 0 {
			return KVCapacityParams{}, fmt.Errorf(
				"model appears to be MoE (%s=%d) but num_local_experts is missing; "+
					"cannot estimate weight size accurately. Set --total-kv-blocks explicitly", key, v)
		}
	}

	return NewKVCapacityParams(false, 0, tieWordEmbeddings, hiddenAct, 0, 0), nil
}
