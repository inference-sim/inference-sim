package sim

import (
	"math"
)

// --- Hardware Data Structures ---

type HardwareCalib struct {
	TFlopsEff        float64
	BwEffTBs         float64
	TOverheadMicros  float64
	perLayerOverhead float64
	mfuPrefill       float64
	mfuDecode        float64
	allReduceLatency float64
}

var HardwareList = map[string]HardwareCalib{
	"H100": {
		TFlopsEff:        989.5,       // Tera (10^12) FLOP/s
		BwEffTBs:         3.35 * 0.72, // in TB/s
		TOverheadMicros:  500.0,       // Per-step Overheads unaccounted for
		perLayerOverhead: 20.0,
		mfuPrefill:       0.65,
		mfuDecode:        0.12,
		allReduceLatency: 20.0,
	},
}

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

func rooflineStepTime(gpu string, modelConfig ModelConfig, stepConfig StepConfig, tp int) int64 {
	hw, ok := HardwareList[gpu]
	if !ok {
		return 0
	}

	tpFactor := float64(tp)
	effFLOPSPeak := (hw.TFlopsEff * 1e12) / tpFactor
	effBWBytes := (hw.BwEffTBs * 1e12) // BW is usually per-GPU, weights/KV are split

	var totalGemmFlops, totalSramFlops, totalDynamicMem float64

	// 1. Static weight memory (loaded once per step)
	// We call with 0,0 to isolate the weight component
	baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
	weightBytes := baseMem["model_weights"] / tpFactor

	// 2. Process Prefills
	// Logical Fix: Prefill compute is often much more efficient than decode
	for _, req := range stepConfig.PrefillRequests {
		newT := int64(req.NumNewPrefillTokens)
		if newT == 0 {
			continue
		}

		f := calculateTransformerFlops(modelConfig, req.ProgressIndex, newT, true, true)
		totalGemmFlops += f["gemm_ops"] / tpFactor
		totalSramFlops += f["sram_ops"] / tpFactor

		// Memory: Subtract weights to get only the dynamic KV/activation traffic
		m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, newT, true)
		totalDynamicMem += (m["total"] - m["model_weights"]) / tpFactor
	}

	// 3. Process Decodes
	for _, req := range stepConfig.DecodeRequests {
		newT := int64(1) // Decode is always 1 token

		f := calculateTransformerFlops(modelConfig, req.ProgressIndex, newT, true, true)
		totalGemmFlops += f["gemm_ops"] / tpFactor
		totalSramFlops += f["sram_ops"] / tpFactor

		m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, newT, true)
		totalDynamicMem += (m["total"] - m["model_weights"]) / tpFactor
	}

	// 4. Determine Workload State
	isPrefillWorkload := len(stepConfig.PrefillRequests) > 0

	// Choose MFU based on the dominant kernel type in the batch
	mfu := hw.mfuDecode
	if isPrefillWorkload {
		mfu = hw.mfuPrefill
	}

	// 5. Compute Time Calculation
	// SRAM ops (Attention math) are nearly 2x more efficient than GEMMs on H100
	tComputeS := (totalGemmFlops / (effFLOPSPeak * mfu)) + (totalSramFlops / (effFLOPSPeak * 0.90))

	// 6. Memory Time Calculation
	tMemoryS := (weightBytes + totalDynamicMem) / effBWBytes

	// 7. Roofline Bottleneck
	// Step time is determined by the slowest component (Max, not Sum)
	hardwareStepTimeS := math.Max(tComputeS, tMemoryS)

	// 8. Layer & Static Overheads
	// perLayerOverhead captures the sequential kernel launch floor (15-25us)
	// TOverheadMicros captures the base engine/scheduler cost (500us)
	layerFloorS := (float64(modelConfig.NumLayers) * hw.perLayerOverhead) / 1e6
	// --- TP COMMUNICATION OVERHEAD ---
	// There are 2 All-Reduces per layer.
	// This latency does NOT scale down with more GPUs; it often increases
	// slightly or stays flat depending on NVLink topology.
	commOverheadS := 0.0
	if tp > 1 {
		commOverheadS = (float64(modelConfig.NumLayers) * 2 * hw.allReduceLatency) / 1e6
	}

	totalMicros := (hardwareStepTimeS * 1e6) + (layerFloorS * 1e6) + (commOverheadS * 1e6) + hw.TOverheadMicros

	return int64(math.Round(totalMicros))
}
