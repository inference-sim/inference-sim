package sim

import (
	"math"
)

// --- Bento FLOPS Logic ---
func calculateTransformerFlops(config ModelConfig, sequenceLength int64, newTokens int64, includeAttention, includeMLP bool) map[string]float64 {
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

		// Attention QKV projection GEMMs
		qkvFlops := 2 * newT * (dModel*dModel + 2*dModel*dKV)
		// Attention output projection GEMM
		projFlops := 2 * newT * dModel * dModel
		flops["gemm_ops"] = (qkvFlops + projFlops) * nLayers

		// FlashAttention: Compute-efficient attention with paged blocks
		// Per vllm.md: processes in 512-token partitions in two stages
		effectiveCtx := seqLen
		if newT > 1 {
			// Prefill: self-attention within batch, reduced effective context
			effectiveCtx = seqLen + (newT-1)/2.0
		}

		// QK^T multiplication (matrix ops): 2 * newT * effectiveCtx * dHead per head
		qkMatMul := 2 * nHeads * newT * effectiveCtx * dHead

		// Softmax: exp + sum + div (reduced from 5 to 4 ops per element)
		// Vector ops run on scalar units at lower efficiency
		softmaxOps := 4 * nHeads * newT * effectiveCtx

		// Value projection (matrix ops): 2 * newT * effectiveCtx * dHead per head
		avMatMul := 2 * nHeads * newT * effectiveCtx * dHead

		// Total attention math, run on vector/scalar units (10% efficiency)
		attnMath := qkMatMul + softmaxOps + avMatMul
		flops["sram_ops"] = attnMath * nLayers
	}

	if includeMLP {
		// SwiGLU Gating: Gate, Up, and Down (3 matrices)
		flops["gemm_ops"] += 2 * newT * (3 * dModel * dFF) * nLayers
	}

	flops["total"] = flops["gemm_ops"] + flops["sram_ops"]
	return flops
}

func calculateMemoryAccessBytes(
	config ModelConfig,
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
	weightsPerLayer := dModel*(dModel+2*dKV) + (dModel * dModel) + (3 * dModel * dFF)
	mem["model_weights"] = weightsPerLayer * nLayers * config.BytesPerParam

	if includeKVCache {
		// KV Growth: Writing new tokens to HBM.
		kvWritePerNewToken := 2 * nLayers * nKVHeads * dHead * config.BytesPerParam
		mem["kv_cache_growth"] = kvWritePerNewToken * newT

		// KV Access: Paged attention with 512-token partitions (per vllm.md)
		// CRITICAL: In prefill, new tokens process within partitions in SRAM
		// Most intermediate KV stays in cache, doesn't go to HBM
		kvReadPerToken := 2 * nLayers * nKVHeads * dHead * config.BytesPerParam

		// For decode (newT==1): scattered KV access over full historical sequence
		// Each decode reads ALL historical KV (expensive!)
		if newT == 1 {
			// Decode: scattered KV access via paged attention over full sequence
			mem["kv_cache_access"] = kvReadPerToken * seq * 0.80
		} else {
			// Prefill: sequential KV reads with cache reuse
			mem["kv_cache_access"] = kvReadPerToken * seq * 0.92
		}
	}

	// Token activations: Based on vllm.md paged attention and locality
	// Decode has some SRAM reuse, prefill has batching benefits
	var activationBytes float64
	if newT == 1 {
		// Decode: some activations in SRAM, most hit HBM
		activationBytes = nLayers * dModel * config.BytesPerParam * newT * 0.75
	} else {
		// Prefill: better batching locality and reuse
		activationBytes = nLayers * dModel * config.BytesPerParam * newT * 0.85
	}
	mem["activations_tokens"] = activationBytes

	// LOGICAL FIX: Remove attention map bytes entirely.
	// FlashAttention fuses this; it never hits HBM.

	var total float64
	for _, v := range mem {
		total += v
	}
	mem["total"] = total
	return mem
}

func rooflineStepTime(_ string, modelConfig ModelConfig, hwConfig HardwareCalib, stepConfig StepConfig, tp int) int64 {

	tpFactor := float64(tp)
	peakFlops := hwConfig.TFlopsPeak * 1e12
	peakBW := hwConfig.BwPeakTBs * 1e12
	effBW := peakBW * hwConfig.BwEffConstant
	vectorPeak := peakFlops * hwConfig.VectorPeakFraction // Non-tensor core ops

	// TP scaling efficiency - actual speedup is sublinear due to communication overhead
	// Empirical: TP=2 gives 1.36x speedup, TP=4 gives 1.84x speedup (compute-bound prefill)
	effectiveTpPrefill := math.Pow(tpFactor, hwConfig.TpScalingExponent)
	// Decode TP scaling - memory-bound but with communication overhead at higher TP
	decodeTpExp := hwConfig.DecodeTpScalingExponent
	if decodeTpExp == 0 {
		decodeTpExp = 1.0 // Default to linear scaling if not set
	}
	effectiveTpDecode := math.Pow(tpFactor, decodeTpExp)

	var prefillComputeS, prefillMemoryS float64
	var decodeComputeS, decodeMemoryS float64
	var hasPrefill, hasDecode bool

	// 1. PREFILL PHASE
	if len(stepConfig.PrefillRequests) > 0 {
		hasPrefill = true
		var pGemmFlops, pVectorFlops, pDynamicBytes float64

		for _, req := range stepConfig.PrefillRequests {
			numTokens := int64(req.NumNewPrefillTokens)

			f := calculateTransformerFlops(modelConfig, req.ProgressIndex, numTokens, true, true)
			// Use sublinear TP scaling for compute (prefill is compute-bound)
			pGemmFlops += f["gemm_ops"] / effectiveTpPrefill
			pVectorFlops += f["sram_ops"] / effectiveTpPrefill

			m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, numTokens, true)
			// Memory still scales linearly with TP (each GPU has 1/tp of weights)
			pDynamicBytes += (m["total"] - m["model_weights"]) / tpFactor
		}

		baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
		pWeightBytes := baseMem["model_weights"] / tpFactor

		// Prefill MFU - calibrated to match vLLM measured performance
		adjustedPrefillMFU := hwConfig.MfuPrefill * hwConfig.MfuPrefillMultiplier

		// Reduce effective bandwidth for prefill (memory contention during KV cache operations)
		prefillEffBW := effBW * hwConfig.PrefillBwFactor

		prefillComputeS = (pGemmFlops / (peakFlops * adjustedPrefillMFU)) + (pVectorFlops / vectorPeak)
		prefillMemoryS = (pWeightBytes + pDynamicBytes) / prefillEffBW
	}

	// 2. DECODE PHASE
	if len(stepConfig.DecodeRequests) > 0 {
		hasDecode = true
		var dGemmFlops, dVectorFlops, dDynamicBytes float64

		for _, req := range stepConfig.DecodeRequests {
			f := calculateTransformerFlops(modelConfig, req.ProgressIndex, 1, true, true)
			// Decode TP scaling - applies to both compute and memory
			dGemmFlops += f["gemm_ops"] / effectiveTpDecode
			dVectorFlops += f["sram_ops"] / effectiveTpDecode

			m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, 1, true)
			dDynamicBytes += (m["total"] - m["model_weights"]) / effectiveTpDecode
		}

		baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
		dWeightBytes := baseMem["model_weights"] / effectiveTpDecode

		// Unified MFU for decode across all batch sizes
		adjustedDecodeMFU := hwConfig.MfuDecode * hwConfig.MfuDecodeMultiplier

		// Reduce effective bandwidth for decode (scattered KV cache access)
		decodeEffBW := effBW * hwConfig.DecodeBwFactor

		decodeComputeS = (dGemmFlops / (peakFlops * adjustedDecodeMFU)) + (dVectorFlops / vectorPeak)
		decodeMemoryS = (dWeightBytes + dDynamicBytes) / decodeEffBW
	}

	// 3. COMBINE PHASES
	// Model prefill and decode as partially overlappable execution
	// In a mixed batch, requests can interleave but they also compete for resources
	var stepHardwareS float64

	if hasPrefill && hasDecode {
		// Mixed batch: use weighted average based on token count and bottleneck characteristics
		prefillTimeS := math.Max(prefillComputeS, prefillMemoryS)
		decodeTimeS := math.Max(decodeComputeS, decodeMemoryS)

		prefillTokens := 0.0
		for _, req := range stepConfig.PrefillRequests {
			prefillTokens += float64(req.NumNewPrefillTokens)
		}
		decodeTokens := float64(len(stepConfig.DecodeRequests))
		totalTokens := prefillTokens + decodeTokens

		// Adaptive weighting based on token distribution and expected interleaving
		// In vLLM, prefill and decode can be partially pipelined when both present
		if prefillTokens > decodeTokens*4 && prefillTokens > 100 {
			// Strongly prefill-dominated with significant tokens: apply blending
			// Model some compute overlap between prefill and decode layers
			stepHardwareS = 0.75*prefillTimeS + 0.25*decodeTimeS
		} else if decodeTokens > prefillTokens*2 && decodeTokens > 50 {
			// Decode-dominated: weight towards decode with some prefill overlap
			stepHardwareS = 0.35*prefillTimeS + 0.65*decodeTimeS
		} else {
			// Balanced mix: average to account for parallelization benefits
			prefillWeight := prefillTokens / totalTokens
			decodeWeight := decodeTokens / totalTokens
			stepHardwareS = prefillWeight*prefillTimeS + decodeWeight*decodeTimeS
		}
	} else if hasPrefill {
		stepHardwareS = math.Max(prefillComputeS, prefillMemoryS)
	} else if hasDecode {
		stepHardwareS = math.Max(decodeComputeS, decodeMemoryS)
	}

	// Communication overhead - all-reduce per layer
	commOverheadS := 0.0
	if tp > 1 {
		commOverheadS = (float64(modelConfig.NumLayers) * 2 * hwConfig.AllReduceLatency) / 1e6
	}

	// Batch-size-aware overhead: decode-heavy batches have more scheduling overhead
	// Decode steps are more sensitive to scheduling latency per-token
	numDecode := float64(len(stepConfig.DecodeRequests))
	numPrefill := float64(len(stepConfig.PrefillRequests))
	totalRequests := numDecode + numPrefill
	scaledOverheadMicros := hwConfig.TOverheadMicros
	if totalRequests > 2 {
		// Base scaling with batch size
		baseScale := 0.15 * math.Log2(totalRequests/2.0)
		// Decode-heavy batches get more overhead (ITL is per-token)
		decodeRatio := numDecode / totalRequests
		scaledOverheadMicros *= 1.0 + baseScale*(0.3+2.0*decodeRatio)
	}

	// Add overhead for prefill steps (KV cache allocation, scheduling)
	var prefillOverheadMicros float64
	if numPrefill > 0 && numDecode == 0 {
		// Pure prefill step: add fixed overhead per prefill request
		prefillOverheadMicros = numPrefill * hwConfig.PrefillOverheadMicros
	} else if numPrefill > 0 {
		// Mixed batch: smaller overhead
		prefillOverheadMicros = numPrefill * hwConfig.MixedPrefillOverheadMicros
	}

	// Total time
	totalS := stepHardwareS + commOverheadS + (scaledOverheadMicros / 1e6) + (prefillOverheadMicros / 1e6)
	totalMicros := totalS * 1e6

	return int64(math.Round(totalMicros))
}
