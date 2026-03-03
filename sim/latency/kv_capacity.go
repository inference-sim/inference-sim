package latency

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// KVCapacityParams holds model-architecture parameters that are not part of
// sim.ModelConfig but are needed for KV block capacity estimation.
// These come from the HuggingFace config.json (hidden_act, MoE indicators,
// tie_word_embeddings).
type KVCapacityParams struct {
	IsMoE             bool
	NumLocalExperts   int
	TieWordEmbeddings bool
	HiddenAct         string
}

// Constants matching the llm-d-benchmark capacity_planner.py reference.
const (
	gpuMemUtil                 = 0.9
	activationMemoryDenseGiB   = 5.5
	activationMemoryMoEGiB     = 8.0
	nonTorchMemoryTP1GiB       = 0.15
	nonTorchMemoryTPMultiGiB   = 0.6
	gibToBytes                 = 1 << 30
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

// CalculateKVBlocks computes the maximum number of KV cache blocks that fit
// in GPU memory after accounting for model weights, activations, and
// non-PyTorch overhead. The formula matches the llm-d-benchmark
// capacity_planner.py reference.
//
// Parameters:
//   - mc: model architecture (layers, heads, dims, precision)
//   - hc: GPU hardware calibration (must include MemoryGiB)
//   - tp: tensor parallelism degree (must be > 0)
//   - blockSize: tokens per KV cache block (must be > 0)
//   - params: MoE indicators, activation type, embedding tying
//
// Returns the number of blocks, or an error if inputs are invalid or memory
// budget is insufficient.
func CalculateKVBlocks(mc sim.ModelConfig, hc sim.HardwareCalib, tp int, blockSize int64, params KVCapacityParams) (int64, error) {
	// --- Input validation (R3, R11) ---
	if tp <= 0 {
		return 0, fmt.Errorf("CalculateKVBlocks: TP must be > 0, got %d", tp)
	}
	if blockSize <= 0 {
		return 0, fmt.Errorf("CalculateKVBlocks: block size must be > 0, got %d", blockSize)
	}
	if mc.NumHeads <= 0 {
		return 0, fmt.Errorf("CalculateKVBlocks: num_attention_heads must be > 0, got %d", mc.NumHeads)
	}
	if mc.NumLayers <= 0 {
		return 0, fmt.Errorf("CalculateKVBlocks: num_layers must be > 0, got %d", mc.NumLayers)
	}
	if mc.HiddenDim <= 0 {
		return 0, fmt.Errorf("CalculateKVBlocks: hidden_dim must be > 0, got %d", mc.HiddenDim)
	}
	if mc.IntermediateDim <= 0 {
		return 0, fmt.Errorf("CalculateKVBlocks: intermediate_dim must be > 0, got %d", mc.IntermediateDim)
	}
	if mc.VocabSize <= 0 {
		return 0, fmt.Errorf("CalculateKVBlocks: vocab_size must be > 0, got %d", mc.VocabSize)
	}
	if mc.BytesPerParam <= 0 || math.IsNaN(mc.BytesPerParam) || math.IsInf(mc.BytesPerParam, 0) {
		return 0, fmt.Errorf("CalculateKVBlocks: precision (BytesPerParam) must be a valid positive number, got %v", mc.BytesPerParam)
	}
	if hc.MemoryGiB <= 0 || math.IsNaN(hc.MemoryGiB) || math.IsInf(hc.MemoryGiB, 0) {
		return 0, fmt.Errorf("CalculateKVBlocks: GPU memory (MemoryGiB) must be a valid positive number, got %v", hc.MemoryGiB)
	}

	// Head dimension must be evenly divisible.
	if mc.HiddenDim%mc.NumHeads != 0 {
		return 0, fmt.Errorf("CalculateKVBlocks: hidden_dim (%d) must be evenly divisible by num_attention_heads (%d)", mc.HiddenDim, mc.NumHeads)
	}

	// Only SwiGLU-family activations are supported (3-matrix MLP).
	if !swiGLUActivations[params.HiddenAct] {
		return 0, fmt.Errorf("CalculateKVBlocks: unsupported activation %q; only SwiGLU-family activations (silu, swiglu, geglu) are supported", params.HiddenAct)
	}

	// Resolve numKVHeads: 0 means MHA (same as NumHeads); negative is invalid.
	numKVHeads := mc.NumKVHeads
	if numKVHeads < 0 {
		return 0, fmt.Errorf("CalculateKVBlocks: num_kv_heads must be >= 0, got %d", numKVHeads)
	}
	if numKVHeads == 0 {
		numKVHeads = mc.NumHeads
	}

	// TP divisibility: if numKVHeads >= tp, they must be evenly divisible.
	// When numKVHeads < tp (e.g., GQA with 2 KV heads, TP=4), vLLM replicates
	// KV heads per GPU. Our formula divides total KV by tp, which underestimates
	// per-GPU KV memory in this case. This is a known approximation — the error
	// is optimistic (overestimates available blocks).
	if numKVHeads >= tp && numKVHeads%tp != 0 {
		return 0, fmt.Errorf("CalculateKVBlocks: num_kv_heads (%d) must be evenly divisible by TP (%d)", numKVHeads, tp)
	}

	headDim := mc.HiddenDim / mc.NumHeads

	// --- Step 1: Per-token KV bytes ---
	// Each layer has a K and V projection, each of size headDim * numKVHeads.
	// Use float64 arithmetic to avoid int64 truncation of fractional BytesPerParam
	// (e.g., INT4 = 0.5 bytes/param would truncate to 0 with int64 cast).
	perTokenKVBytesF := float64(mc.NumLayers) * 2.0 * float64(headDim) * float64(numKVHeads) * mc.BytesPerParam

	// --- Step 2: Per-token KV bytes per GPU (TP sharding) ---
	perTokenKVBytesPerGPUF := perTokenKVBytesF / float64(tp)

	// --- Step 3: Per-block bytes ---
	perBlockBytes := int64(perTokenKVBytesPerGPUF * float64(blockSize))
	if perBlockBytes <= 0 {
		return 0, fmt.Errorf("CalculateKVBlocks: per-block KV bytes is %d (expected > 0); check BytesPerParam=%.4f, numKVHeads=%d, headDim=%d",
			perBlockBytes, mc.BytesPerParam, numKVHeads, headDim)
	}

	// --- Step 4: Available memory budget (total across all TP GPUs) ---
	// Reference: available_memory = gpu_mem * gpu_mem_util * gpu_count
	totalAvailableGiB := hc.MemoryGiB * gpuMemUtil * float64(tp)

	// Model weights: total model size (distributed across TP GPUs, but sum = total)
	modelWeightBytes := computeModelWeightBytes(mc, params)
	modelWeightGiB := float64(modelWeightBytes) / float64(gibToBytes)

	// Activation memory: per-replica constant (dp=1 in BLIS), NOT multiplied by TP
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

	overheadGiB := modelWeightGiB + activationGiB + nonTorchGiB
	if overheadGiB >= totalAvailableGiB {
		return 0, fmt.Errorf(
			"CalculateKVBlocks: model overhead (%.2f GiB = %.2f weights + %.2f activation + %.2f non-torch) "+
				"exceeds available GPU memory (%.2f GiB = %.1f GiB × %.0f%% util × %d GPUs)",
			overheadGiB, modelWeightGiB, activationGiB, nonTorchGiB,
			totalAvailableGiB, hc.MemoryGiB, gpuMemUtil*100, tp)
	}

	allocatableGiB := totalAvailableGiB - overheadGiB
	allocatableBytes := int64(allocatableGiB * float64(gibToBytes))

	// --- Step 5: Total blocks ---
	totalBlocks := allocatableBytes / perBlockBytes
	if totalBlocks <= 0 {
		return 0, fmt.Errorf(
			"CalculateKVBlocks: computed 0 blocks (allocatable=%.2f GiB, per_block=%d bytes)",
			allocatableGiB, perBlockBytes)
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
	headDim := int64(mc.HiddenDim / mc.NumHeads)
	kvDim := int64(numKVHeads) * headDim

	// Embeddings: vocab_size * hidden_dim
	embeddings := vocabSize * hiddenDim

	// Attention per layer: Q proj + K proj + V proj + output proj
	// Q: hidden_dim * hidden_dim
	// K: hidden_dim * kv_dim
	// V: hidden_dim * kv_dim
	// O: hidden_dim * hidden_dim
	attentionPerLayer := hiddenDim*(hiddenDim+2*kvDim) + hiddenDim*hiddenDim

	// MLP per layer: SwiGLU uses 3 matrices (gate, up, down)
	// gate: hidden_dim * intermediate_dim
	// up:   hidden_dim * intermediate_dim
	// down: intermediate_dim * hidden_dim
	mlpPerLayer := 3 * hiddenDim * intermediateDim

	// MoE: multiply MLP by number of local experts, add router weights.
	if params.IsMoE && params.NumLocalExperts > 0 {
		mlpPerLayer *= int64(params.NumLocalExperts)
		// Router weights: num_local_experts * hidden_dim per layer
		mlpPerLayer += int64(params.NumLocalExperts) * hiddenDim
	}

	// Layer norms: 2 per layer (pre-attention + post-attention), each = hidden_dim params
	layerNormsPerLayer := 2 * hiddenDim

	// Per-layer total
	perLayerParams := attentionPerLayer + mlpPerLayer + layerNormsPerLayer

	// lm_head: vocab_size * hidden_dim (omitted if tie_word_embeddings)
	var lmHead int64
	if !params.TieWordEmbeddings {
		lmHead = vocabSize * hiddenDim
	}

	// Final layer norm: hidden_dim
	finalNorm := hiddenDim

	totalParams := embeddings + numLayers*perLayerParams + lmHead + finalNorm
	return int64(float64(totalParams) * mc.BytesPerParam)
}

// ExtractKVCapacityParamsFromFile reads a HuggingFace config.json file and
// extracts the KVCapacityParams needed for CalculateKVBlocks.
func ExtractKVCapacityParamsFromFile(hfConfigPath string) (KVCapacityParams, error) {
	hf, err := parseHFConfig(hfConfigPath)
	if err != nil {
		return KVCapacityParams{}, fmt.Errorf("extract KV capacity params: %w", err)
	}
	return ExtractKVCapacityParams(hf), nil
}

// ExtractKVCapacityParams extracts KVCapacityParams from a parsed HFConfig.
// MoE detection: checks num_local_experts > 1, then falls back to
// n_routed_experts, n_shared_experts, num_experts, num_experts_per_tok.
func ExtractKVCapacityParams(hf *HFConfig) KVCapacityParams {
	var params KVCapacityParams

	// hidden_act
	params.HiddenAct = hf.MustGetString("hidden_act", "")

	// tie_word_embeddings
	if tied, ok := hf.GetBool("tie_word_embeddings"); ok {
		params.TieWordEmbeddings = tied
	}

	// MoE detection: check multiple field names used by different architectures.
	numLocalExperts := hf.MustGetInt("num_local_experts", 0)
	if numLocalExperts > 1 {
		params.IsMoE = true
		params.NumLocalExperts = numLocalExperts
		return params
	}

	// Fallback MoE indicators: fields that represent total expert count can
	// be used directly as NumLocalExperts. Fields like num_experts_per_tok
	// represent activation count (e.g., 2 for Mixtral) and only signal MoE
	// presence — they must NOT be used as the expert multiplier for weights.
	for _, key := range []string{"n_routed_experts", "num_experts"} {
		if v := hf.MustGetInt(key, 0); v > 1 {
			params.IsMoE = true
			params.NumLocalExperts = v
			return params
		}
	}
	// Activation-count or shared-expert fields: signal MoE but don't provide
	// a reliable total expert count. Set IsMoE for activation memory but
	// leave NumLocalExperts at 0 (computeModelWeightBytes skips MoE MLP
	// multiplication when NumLocalExperts == 0).
	for _, key := range []string{"n_shared_experts", "num_experts_per_tok"} {
		if v := hf.MustGetInt(key, 0); v > 1 {
			params.IsMoE = true
			return params
		}
	}

	return params
}
