package latency

import (
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// PrefillRequestConfig describes a single prefill request in a batch step.
type PrefillRequestConfig struct {
	ProgressIndex       int64 `json:"progress_index"`
	NumNewPrefillTokens int   `json:"num_new_prefill_tokens"`
}

// DecodeRequestConfig describes a single decode request in a batch step.
type DecodeRequestConfig struct {
	ProgressIndex      int64 `json:"progress_index"`
	NumNewDecodeTokens int   `json:"num_new_decode_tokens"`
}

// StepConfig describes the requests in a single batch step for roofline estimation.
type StepConfig struct {
	PrefillRequests []PrefillRequestConfig `json:"prefill_requests"`
	DecodeRequests  []DecodeRequestConfig  `json:"decode_requests"`
}

// --- Bento FLOPS Logic ---
//
// Precondition: config must pass ValidateRooflineConfig (NumHeads > 0, and when
// NumLocalExperts > 1, NumExpertsPerTok must be > 0). Violating preconditions
// produces silently incorrect results (zero MLP FLOPs, +Inf from division).
func calculateTransformerFlops(config sim.ModelConfig, sequenceLength int64, newTokens int64, includeAttention, includeMLP bool) map[string]float64 {
	dModel := float64(config.HiddenDim)
	nLayers := float64(config.NumLayers)
	nHeads := float64(config.NumHeads)
	nKVHeads := float64(config.NumKVHeads)
	if nKVHeads == 0 {
		nKVHeads = nHeads
	}
	dHead := dModel / nHeads

	// Qwen2.5 uses specific intermediate dims for SwiGLU
	dFF := 4.0 * dModel
	if config.IntermediateDim > 0 {
		dFF = float64(config.IntermediateDim)
	}

	seqLen := float64(sequenceLength)
	newT := float64(newTokens)
	flops := make(map[string]float64)

	if includeAttention {
		dKV := nKVHeads * dHead

		// 1. Standard GEMMs (Weights)
		qkvFlops := 2 * newT * (dModel*dModel + 2*dModel*dKV)
		projFlops := 2 * newT * dModel * dModel
		flops["gemm_ops"] = (qkvFlops + projFlops) * nLayers

		// SRAM-local ops (FlashAttention)
		effectiveCtx := seqLen
		if newT > 1 {
			effectiveCtx = seqLen + (newT-1)/2.0
		}

		// 2. Attention Score Ops (The TTFT Killer)
		// We treat the QK^T and AV as "GEMM" ops if they are large enough,
		// because they utilize the same execution units as standard GEMMs in FlashAttention.
		attnGemmOps := (4 * nHeads * newT * effectiveCtx * dHead)

		// 3. Vector Ops (Softmax, Masking, RoPE)
		ropeOps := 2 * newT * dModel
		vectorOps := (5 * nHeads * newT * effectiveCtx) + ropeOps

		flops["gemm_ops"] += (attnGemmOps * nLayers)
		flops["sram_ops"] = (vectorOps * nLayers)
	}

	if includeMLP {
		if config.NumLocalExperts > 1 {
			// MoE MLP FLOPs: routed (top_k) + shared + gate
			topK := float64(config.NumExpertsPerTok)
			numExperts := float64(config.NumLocalExperts)

			// Per-expert FFN dim: explicit or fallback to IntermediateDim
			dRouted := dFF
			if config.MoEExpertFFNDim > 0 {
				dRouted = float64(config.MoEExpertFFNDim)
			}

			// Routed expert FLOPs: top_k active experts, 2-matrix MLP (up + down)
			routedFLOPs := 2 * newT * (2 * dModel * dRouted) * topK * nLayers

			// Shared expert FLOPs (always active, every token)
			var sharedFLOPs float64
			if config.SharedExpertFFNDim > 0 {
				dShared := float64(config.SharedExpertFFNDim)
				sharedFLOPs = 2 * newT * (2 * dModel * dShared) * nLayers
			}

			// Gate (router) FLOPs: linear projection → E logits per layer
			gateFLOPs := 2 * newT * dModel * numExperts * nLayers

			flops["gemm_ops"] += routedFLOPs + sharedFLOPs + gateFLOPs
		} else {
			// Dense MLP: 2-matrix (up + down projections), matching llm-optimizer
			flops["gemm_ops"] += 2 * newT * (2 * dModel * dFF) * nLayers
		}
	}

	flops["total"] = flops["gemm_ops"] + flops["sram_ops"]
	return flops
}

// Precondition: same as calculateTransformerFlops — config must pass
// ValidateRooflineConfig. NumHeads > 0 required (division at dHead).
func calculateMemoryAccessBytes(
	config sim.ModelConfig,
	sequenceLength int64,
	newTokens int64,
	includeKVCache bool,
) map[string]float64 {
	dModel := float64(config.HiddenDim)
	nLayers := float64(config.NumLayers)
	nHeads := float64(config.NumHeads)
	nKVHeads := float64(config.NumKVHeads)
	if nKVHeads == 0 {
		nKVHeads = nHeads
	}

	dHead := dModel / nHeads
	seq := float64(sequenceLength)
	newT := float64(newTokens)

	dFF := 4.0 * dModel
	if config.IntermediateDim > 0 {
		dFF = float64(config.IntermediateDim)
	}

	mem := make(map[string]float64)

	// Weights: Loaded exactly once. (Static)
	dKV := nKVHeads * dHead
	attnWeightsPerLayer := dModel*(dModel+2*dKV) + (dModel * dModel)

	var mlpWeightsPerLayer float64
	if config.NumLocalExperts > 1 {
		// MoE ACTIVE weights per step: only top_k expert MLPs are loaded from HBM,
		// matching vLLM's fused_moe kernel behavior. For TOTAL model weights
		// (all E experts, used for GPU memory budgeting), see computeModelWeightBytes
		// in kv_capacity.go.
		topK := float64(config.NumExpertsPerTok)
		numExperts := float64(config.NumLocalExperts)

		dRouted := dFF
		if config.MoEExpertFFNDim > 0 {
			dRouted = float64(config.MoEExpertFFNDim)
		}

		// Active routed expert weights (per step, only top_k loaded), 2-matrix MLP
		routedWeights := topK * (2 * dModel * dRouted)

		// Shared expert weights (always loaded)
		var sharedWeights float64
		if config.SharedExpertFFNDim > 0 {
			sharedWeights = 2 * dModel * float64(config.SharedExpertFFNDim)
		}

		// Gate weights: dModel × numExperts
		gateWeights := dModel * numExperts

		mlpWeightsPerLayer = routedWeights + sharedWeights + gateWeights
	} else {
		// Dense MLP weights: 2-matrix (up + down), matching llm-optimizer
		mlpWeightsPerLayer = 2 * dModel * dFF
	}

	weightsPerLayer := attnWeightsPerLayer + mlpWeightsPerLayer
	mem["model_weights"] = weightsPerLayer * nLayers * config.BytesPerParam

	if includeKVCache {
		// KV Growth: Writing new tokens to HBM.
		kvWritePerNewToken := 2 * nLayers * nKVHeads * dHead * config.BytesPerParam
		mem["kv_cache_growth"] = kvWritePerNewToken * newT

		// KV Access: Only read PAST history.
		// IMPORTANT: For Prefill (newT > 1), the newT tokens attend to each other in SRAM.
		// They do NOT generate HBM read traffic for themselves.
		kvReadPerToken := 2 * nLayers * nKVHeads * dHead * config.BytesPerParam
		mem["kv_cache_access"] = kvReadPerToken * seq
	}

	// Token activations (linear)
	mem["activations_tokens"] = nLayers * dModel * config.BytesPerParam * newT

	// LOGICAL FIX: Remove attention map bytes entirely.
	// FlashAttention fuses this; it never hits HBM.

	// Sort keys before accumulation for deterministic float summation
	// (Go map iteration order is non-deterministic — antipattern #2)
	keys := make([]string, 0, len(mem))
	for k := range mem {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var total float64
	for _, k := range keys {
		total += mem[k]
	}
	mem["total"] = total
	return mem
}

// rooflineStepTime computes step latency using the roofline model.
//
// Models a single forward pass per step (matching vLLM chunked prefill):
// all prefill and decode tokens are processed together, weights loaded once.
// Uses single-crossover roofline: step_time = max(compute_time, memory_time).
// No bandwidth haircut, no overhead terms.
//
// Compute uses phase-specific MFU: prefill tokens at MfuPrefill, decode at MfuDecode,
// reflecting that prefill is compute-bound (large GEMMs) while decode is memory-bound.
//
// Precondition: ValidateRooflineConfig(modelConfig, hwConfig) must return nil
// and tp must be > 0. Callers must validate before first call.
func rooflineStepTime(modelConfig sim.ModelConfig, hwConfig sim.HardwareCalib, stepConfig StepConfig, tp int) int64 {

	tpFactor := float64(tp)
	peakFlops := hwConfig.TFlopsPeak * 1e12
	peakBW := hwConfig.BwPeakTBs * 1e12

	if len(stepConfig.PrefillRequests) == 0 && len(stepConfig.DecodeRequests) == 0 {
		return 0
	}

	var totalComputeS float64
	var totalDynamicBytes float64

	// 1. PREFILL FLOPs + dynamic memory (KV cache, activations)
	for _, req := range stepConfig.PrefillRequests {
		numTokens := int64(req.NumNewPrefillTokens)

		f := calculateTransformerFlops(modelConfig, req.ProgressIndex, numTokens, true, true)
		totalComputeS += f["total"] / tpFactor / (peakFlops * hwConfig.MfuPrefill)

		m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, numTokens, true)
		totalDynamicBytes += (m["total"] - m["model_weights"]) / tpFactor
	}

	// 2. DECODE FLOPs + dynamic memory (KV cache, activations)
	for _, req := range stepConfig.DecodeRequests {
		f := calculateTransformerFlops(modelConfig, req.ProgressIndex, 1, true, true)
		totalComputeS += f["total"] / tpFactor / (peakFlops * hwConfig.MfuDecode)

		m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, 1, true)
		totalDynamicBytes += (m["total"] - m["model_weights"]) / tpFactor
	}

	// 3. WEIGHTS loaded once per step (single forward pass, per Sarathi-Serve/vLLM V1)
	baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
	weightBytes := baseMem["model_weights"] / tpFactor

	totalMemoryS := (weightBytes + totalDynamicBytes) / peakBW

	// 4. ROOFLINE: single crossover
	totalMicros := math.Max(totalComputeS, totalMemoryS) * 1e6

	return int64(math.Round(totalMicros))
}
