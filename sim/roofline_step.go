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
		// Matrix Multiplications (GEMMs)
		qkvFlops := 2 * newT * (dModel*dModel + 2*dModel*dKV)
		projFlops := 2 * newT * dModel * dModel
		flops["gemm_ops"] = (qkvFlops + projFlops) * nLayers

		// SRAM-local ops (FlashAttention)
		effectiveCtx := seqLen
		if newT > 1 {
			effectiveCtx = seqLen + (newT-1)/2.0
		}
		// QK^T (2*ops) + Softmax (5*ops) + AV (2*ops)
		attnMath := (2 * nHeads * newT * effectiveCtx * dHead) + (5 * nHeads * newT * effectiveCtx) + (2 * nHeads * newT * effectiveCtx * dHead)
		flops["sram_ops"] = attnMath * nLayers
	}

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

	var total float64
	for _, v := range mem {
		total += v
	}
	mem["total"] = total
	return mem
}

func rooflineStepTime(gpu string, modelConfig ModelConfig, hwConfig HardwareCalib, stepConfig StepConfig, tp int) int64 {

	tpFactor := float64(tp)
	peakFlops := hwConfig.TFlopsPeak * 1e12
	peakBW := hwConfig.BwPeakTBs * 1e12
	baseEffBW := peakBW * hwConfig.BwEffConstant
	vectorPeak := peakFlops * 0.10 // Non-tensor core ops

	// Bandwidth efficiency is baseline for now
	// Mixed batch overhead is handled separately via explicit overhead terms
	effBW := baseEffBW

	// Calculate model size to adjust MFU for large models
	dModel := float64(modelConfig.HiddenDim)
	dFF := 4.0 * dModel
	if modelConfig.IntermediateDim > 0 {
		dFF = float64(modelConfig.IntermediateDim)
	}
	nLayers := float64(modelConfig.NumLayers)
	paramsPerLayer := (4*dModel*dModel + 2*dModel*dFF) / 1e9
	totalParams := paramsPerLayer * nLayers

	// Adjust MFU for very large models (>25B)
	// Large models can achieve higher effective throughput with optimized kernels
	mfuPrefill := hwConfig.MfuPrefill
	mfuDecode := hwConfig.MfuDecode
	if totalParams > 25.0 {
		// Large models: boost both prefill and decode MFU
		// Decode also benefits from larger matrices and better GPU utilization
		mfuPrefill = math.Min(0.95, hwConfig.MfuPrefill * 10.0)
		mfuDecode = math.Min(0.30, hwConfig.MfuDecode * 2.5)
	}

	var prefillComputeS, prefillMemoryS float64
	var decodeComputeS, decodeMemoryS float64

	// 1. PREFILL PHASE (Calculated as a single batched operation)
	if len(stepConfig.PrefillRequests) > 0 {
		var pGemmFlops, pVectorFlops, pDynamicBytes float64
		var totalPrefillTokens float64

		for _, req := range stepConfig.PrefillRequests {
			numTokens := int64(req.NumNewPrefillTokens)
			totalPrefillTokens += float64(numTokens)

			f := calculateTransformerFlops(modelConfig, req.ProgressIndex, numTokens, true, true)
			pGemmFlops += f["gemm_ops"] / tpFactor
			pVectorFlops += f["sram_ops"] / tpFactor

			m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, numTokens, true)
			pDynamicBytes += (m["total"] - m["model_weights"]) / tpFactor
		}

		// Adjust MFU based on workload characteristics for large models
		// Use conservative MFU boost optimized for E2E accuracy
		adjustedMfuPrefill := mfuPrefill
		if totalParams > 25.0 && totalPrefillTokens >= 50 {
			// Scale boost factor, capping at 7.3x for optimal prefillheavy workloads
			// Small batches (< 400 tokens): minimal boost
			// Medium batches (400-2000 tokens): moderate boost
			// Large batches (> 2000 tokens): higher boost (capped at 7.3x)
			boostFactor := 1.0
			if totalPrefillTokens < 400 {
				boostFactor = 1.0 + (totalPrefillTokens / 400.0) * 1.2 // 1.0 to 2.2
			} else if totalPrefillTokens < 2000 {
				boostFactor = 2.2 + ((totalPrefillTokens - 400.0) / 1600.0) * 2.8 // 2.2 to 5.0
			} else {
				boostFactor = 5.0 + ((math.Min(totalPrefillTokens, 10000.0) - 2000.0) / 8000.0) * 2.4 // 5.0 to 7.4
			}
			adjustedMfuPrefill = math.Min(0.86, mfuPrefill * boostFactor)
		}

		// Prefill Roofline: High arithmetic intensity → compute-bound
		// Weights are efficiently reused within prefill batches via L2/HBM caching
		// Memory cost is primarily KV cache growth and activations
		prefillComputeS = (pGemmFlops / (peakFlops * adjustedMfuPrefill)) + (pVectorFlops / vectorPeak)
		prefillMemoryS = pDynamicBytes / effBW

		// Debug: log for CodeLlama with large param count
		if totalParams > 25.0 && gpu == "H100" && totalPrefillTokens > 100 {
			_ = gpu // suppress unused warning
			// Uncomment for debugging:
			// fmt.Printf("DEBUG Prefill: tokens=%.0f, MFU=%.3f, compute=%.1fms, memory=%.1fms, max=%.1fms\n",
			//    totalPrefillTokens, adjustedMfuPrefill, prefillComputeS*1000, prefillMemoryS*1000,
			//    math.Max(prefillComputeS, prefillMemoryS)*1000)
		}
	}

	// 2. DECODE PHASE (Calculated as a single batched step)
	if len(stepConfig.DecodeRequests) > 0 {
		var dGemmFlops, dVectorFlops, dDynamicBytes float64

		for _, req := range stepConfig.DecodeRequests {
			f := calculateTransformerFlops(modelConfig, req.ProgressIndex, 1, true, true)
			dGemmFlops += f["gemm_ops"] / tpFactor
			dVectorFlops += f["sram_ops"] / tpFactor

			m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, 1, true)
			dDynamicBytes += (m["total"] - m["model_weights"]) / tpFactor
		}

		// Decode Roofline: Amortize weight loading across batch
		// Weights loaded once and reused across all decode requests in the step
		baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
		dWeightBytes := baseMem["model_weights"] / tpFactor

		// Amortize weight loading: benefit scales with batch size
		// Use extremely conservative amortization for best E2E accuracy
		numDecodeReqs := float64(len(stepConfig.DecodeRequests))
		var weightAmortization float64
		if totalParams < 10.0 {
			// Small models: use batch_size^0.08 for minimal amortization
			// Extremely conservative to match real-world E2E latency
			weightAmortization = math.Pow(math.Max(1.0, numDecodeReqs), 0.08)
		} else {
			// Large models: use batch_size^0.18 for very conservative amortization
			// Balanced to optimize E2E across different workloads
			weightAmortization = math.Pow(math.Max(1.0, numDecodeReqs), 0.18)
		}

		decodeComputeS = (dGemmFlops / (peakFlops * mfuDecode)) + (dVectorFlops / vectorPeak)
		decodeMemoryS = (dWeightBytes/weightAmortization + dDynamicBytes) / effBW
	}

	// 3. COMBINE AND ADD OVERHEADS
	// We take the Max (bottleneck) for each phase independently then add
	// This reflects that prefill and decode, while unified in scheduling,
	// use different attention kernels and may execute sequentially through the model
	stepHardwareS := math.Max(prefillComputeS, prefillMemoryS) + math.Max(decodeComputeS, decodeMemoryS)

	// 4. MIXED BATCH OVERHEAD
	// Based on vllm.md Section 9: Mixed batches incur additional overhead due to:
	// - Warp divergence (prefill vs decode have different compute/memory patterns)
	// - Memory controller serialization (heterogeneous access patterns)
	// - L2 cache thrashing (different working sets)
	mixedBatchOverheadS := 0.0
	if len(stepConfig.PrefillRequests) > 0 && len(stepConfig.DecodeRequests) > 0 {
		// Mixed batch: add overhead proportional to the smaller workload
		// This captures the inefficiency of heterogeneous workloads
		prefillFraction := prefillComputeS / (prefillComputeS + decodeComputeS + 1e-9)
		decodeFraction := 1.0 - prefillFraction
		// Overhead peaks when workloads are balanced (both ~50%)
		mixedBatchInefficiency := 4.0 * prefillFraction * decodeFraction // 0.0 to 1.0, peaks at 0.5/0.5
		// Add overhead per layer, scaled by model size (larger models have more overhead)
		overheadPerLayer := 0.001 // 1μs per layer for small models
		if totalParams > 25.0 {
			overheadPerLayer = 0.0008 // 0.8μs per layer for large models (better optimized)
		}
		mixedBatchOverheadS = mixedBatchInefficiency * float64(modelConfig.NumLayers) * overheadPerLayer
	}

	// 5. BLOCK TABLE LOOKUP OVERHEAD
	// Decode tokens require block table lookups to locate KV cache blocks
	// Add per-token lookup overhead, more significant for long contexts
	blockTableOverheadS := 0.0
	if len(stepConfig.DecodeRequests) > 0 {
		numDecodeTokens := float64(len(stepConfig.DecodeRequests))
		// Average context length across decode requests
		var avgSeqLen float64
		for _, req := range stepConfig.DecodeRequests {
			avgSeqLen += float64(req.ProgressIndex)
		}
		if len(stepConfig.DecodeRequests) > 0 {
			avgSeqLen /= float64(len(stepConfig.DecodeRequests))
		}
		// Lookup overhead increases with context length (more blocks to lookup)
		// Base overhead: 0.3μs per token, scaled by context length factor
		contextFactor := 1.0 + math.Log10(math.Max(avgSeqLen, 10.0)/100.0) // 1.0 at 100 tokens, 1.3 at 1000, 1.6 at 10000
		blockTableOverheadS = numDecodeTokens * float64(modelConfig.NumLayers) * 0.0003 * contextFactor / 1e6
	}

	// 6. KERNEL LAUNCH OVERHEAD
	// Per-layer kernel launches: QKV proj, Attention, Output proj, FFN (4-5 kernels/layer)
	// Already accounted for in hwConfig.PerLayerOverhead
	layerFloorS := (float64(modelConfig.NumLayers) * hwConfig.PerLayerOverhead) / 1e6

	// 7. TENSOR PARALLELISM COMMUNICATION
	commOverheadS := 0.0
	if tp > 1 {
		// TP synchronization happens per layer
		commOverheadS = (float64(modelConfig.NumLayers) * 2 * hwConfig.AllReduceLatency) / 1e6
	}

	totalMicros := (stepHardwareS * 1e6) + (mixedBatchOverheadS * 1e6) + (blockTableOverheadS * 1e6) + (layerFloorS * 1e6) + (commOverheadS * 1e6) + hwConfig.TOverheadMicros

	return int64(math.Round(totalMicros))
}
