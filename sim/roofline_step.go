package sim

import (
	"math"
	"sort"
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
// Precondition: ValidateRooflineConfig(modelConfig, hwConfig) must return nil
// and tp must be > 0. Callers must validate before first call.
func rooflineStepTime(gpu string, modelConfig ModelConfig, hwConfig HardwareCalib, stepConfig StepConfig, tp int) int64 {

	tpFactor := float64(tp)
	peakFlops := hwConfig.TFlopsPeak * 1e12
	peakBW := hwConfig.BwPeakTBs * 1e12
	effBW := peakBW * hwConfig.BwEffConstant
	vectorPeak := peakFlops * 0.10 // Non-tensor core ops

	var prefillComputeS, prefillMemoryS float64
	var decodeComputeS, decodeMemoryS float64

	// 1. PREFILL PHASE (Calculated as a single batched operation)
	if len(stepConfig.PrefillRequests) > 0 {
		var pGemmFlops, pVectorFlops, pDynamicBytes float64

		for _, req := range stepConfig.PrefillRequests {
			numTokens := int64(req.NumNewPrefillTokens)

			f := calculateTransformerFlops(modelConfig, req.ProgressIndex, numTokens, true, true)
			pGemmFlops += f["gemm_ops"] / tpFactor
			pVectorFlops += f["sram_ops"] / tpFactor

			m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, numTokens, true)
			pDynamicBytes += (m["total"] - m["model_weights"]) / tpFactor
		}

		// Prefill Roofline: Weights + KV Cache are loaded once for the whole chunk
		baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
		pWeightBytes := baseMem["model_weights"] / tpFactor

		prefillComputeS = (pGemmFlops / (peakFlops * hwConfig.MfuPrefill)) + (pVectorFlops / vectorPeak)
		prefillMemoryS = (pWeightBytes + pDynamicBytes) / effBW
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

		// Decode Roofline: Every step must reload the weights
		baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
		dWeightBytes := baseMem["model_weights"] / tpFactor

		decodeComputeS = (dGemmFlops / (peakFlops * hwConfig.MfuDecode)) + (dVectorFlops / vectorPeak)
		decodeMemoryS = (dWeightBytes + dDynamicBytes) / effBW
	}

	// 3. COMBINE AND ADD OVERHEADS
	// We take the Max (bottleneck) for each phase independently
	stepHardwareS := math.Max(prefillComputeS, prefillMemoryS) + math.Max(decodeComputeS, decodeMemoryS)

	// Parallelism & Launch Overheads
	layerFloorS := (float64(modelConfig.NumLayers) * hwConfig.PerLayerOverhead) / 1e6

	commOverheadS := 0.0
	if tp > 1 {
		// TP synchronization happens per layer
		commOverheadS = (float64(modelConfig.NumLayers) * 2 * hwConfig.AllReduceLatency) / 1e6
	}

	totalMicros := (stepHardwareS * 1e6) + (layerFloorS * 1e6) + (commOverheadS * 1e6) + hwConfig.TOverheadMicros

	return int64(math.Round(totalMicros))
}
