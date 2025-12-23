package sim

import (
	"math"
)

// --- Hardware Data Structures ---

type HardwareCalib struct {
	TFlopsEff       float64
	BwEffBytesS     float64
	TOverheadMicros float64
}

var HardwareList = map[string]HardwareCalib{
	"H100": {
		TFlopsEff:       1979, // Tera (10^12) FLOP/s
		BwEffBytesS:     3.35, // in TB/s
		TOverheadMicros: 0.0,
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
	dFF := 4.0 * dModel
	seqLen := float64(sequenceLength)

	flops := make(map[string]float64)

	if includeAttention {
		dKV := nKVHeads * dHead
		// 1. QKV Projections
		qkvFlops := 2 * seqLen * (dModel*dModel + 2*dModel*dKV)
		flops["attention_qkv"] = qkvFlops * nLayers

		// 2. Attention Scores (QK^T)
		qkFlops := 2 * nHeads * float64(newTokens) * seqLen * dHead
		flops["attention_scores"] = qkFlops * nLayers

		// 3. Softmax
		softmaxFlops := 3 * nHeads * float64(newTokens) * seqLen
		flops["attention_softmax"] = softmaxFlops * nLayers

		// 4. Attention Output (Softmax @ V)
		avFlops := 2 * nHeads * float64(newTokens) * seqLen * dHead
		flops["attention_output"] = avFlops * nLayers

		// 5. Output Projection
		projFlops := 2 * float64(newTokens) * dModel * dModel
		flops["attention_proj"] = projFlops * nLayers
	}

	if includeMLP {
		mlpFlops := 2 * float64(newTokens) * (dModel*dFF + dFF*dModel)
		flops["mlp"] = mlpFlops * nLayers
	}

	var total float64
	for _, v := range flops {
		total += v
	}
	flops["total"] = total
	return flops
}

func calculateMemoryAccessBytes(config ModelConfig, sequenceLength int64, newTokens int64, includeKVCache bool) map[string]float64 {
	dModel := float64(config.HiddenDim)
	nLayers := float64(config.NumLayers)
	nHeads := float64(config.NumHeads)
	nKVHeads := float64(config.NumKVHeads)
	if nKVHeads == 0 {
		nKVHeads = nHeads
	}
	dHead := dModel / nHeads
	dFF := 4.0 * dModel
	seqLen := float64(sequenceLength)

	mem := make(map[string]float64)

	// 1. Model Weights
	dKV := nKVHeads * dHead
	weightsPerLayer := dModel*(dModel+2*dKV) + dModel*dModel + dModel*dFF*2
	mem["model_weights"] = weightsPerLayer * nLayers * config.BytesPerParam

	// 2. KV Cache
	if includeKVCache {
		kvPerToken := 2 * nLayers * nKVHeads * dHead * config.BytesPerParam
		mem["kv_cache"] = kvPerToken * seqLen
	}

	// 3. Activations
	activationsSize := (float64(newTokens)*dModel*config.BytesPerParam + nHeads*seqLen*seqLen*config.BytesPerParam)
	mem["activations"] = activationsSize

	var total float64
	for _, v := range mem {
		total += v
	}
	mem["total"] = total
	return mem
}

func rooflineStepTime(gpu string, modelConfig ModelConfig, stepConfig StepConfig) int64 {
	hw := HardwareList[gpu]
	tComputeS := 0.0
	tMemoryS := 0.0

	for _, req := range stepConfig.PrefillRequests {
		flopsBreakdown := calculateTransformerFlops(modelConfig, req.ProgressIndex, int64(req.NumNewPrefillTokens), true, true)
		memBreakdown := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, int64(req.NumNewPrefillTokens), true)

		tComputeS += flopsBreakdown["total"]
		tMemoryS += memBreakdown["total"]
	}

	for _, req := range stepConfig.DecodeRequests {
		flopsBreakdown := calculateTransformerFlops(modelConfig, 1, int64(req.NumNewDecodeTokens), true, true)
		memBreakdown := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, int64(req.NumNewDecodeTokens), true)

		tComputeS += flopsBreakdown["total"]
		tMemoryS += memBreakdown["total"]
	}

	tComputeMicros := (tComputeS / (hw.TFlopsEff * 1e12)) * 1e6
	tMemoryMicros := (tComputeS / (hw.BwEffBytesS * 1e12)) * 1e6

	return int64(math.Max(tComputeMicros, tMemoryMicros) + hw.TOverheadMicros)
}
