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
}

var HardwareList = map[string]HardwareCalib{
	"H100": {
		TFlopsEff:        989.5,       // Tera (10^12) FLOP/s
		BwEffTBs:         3.35 * 0.78, // in TB/s
		TOverheadMicros:  500.0,       // Per-step Overheads unaccounted for
		perLayerOverhead: 20.0,
		mfuPrefill:       0.55,
		mfuDecode:        0.20,
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
		qkvFlops := 2 * float64(newTokens) * (dModel*dModel + 2*dModel*dKV)
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
	dFF := 4.0 * dModel
	seq := float64(sequenceLength)
	newT := float64(newTokens)

	mem := make(map[string]float64)

	// Weights: Read once per step
	dKV := nKVHeads * dHead
	weightsPerLayer := dModel*(dModel+2*dKV) + dModel*dModel + dModel*dFF*3
	mem["model_weights"] = weightsPerLayer * nLayers * config.BytesPerParam

	if includeKVCache {
		// KV write: New tokens being added to cache
		kvWritePerNewToken := 2 * nLayers * nKVHeads * dHead * config.BytesPerParam
		mem["kv_cache_growth"] = kvWritePerNewToken * newT

		// KV read: Reading history.
		// For Decode: it's exactly 'seq'
		// For Prefill: FlashAttention is used. We only read the KV cache for
		// EXISTING history (seq), not the newTokens themselves (they stay in SRAM).
		kvReadPerToken := 2 * nLayers * nKVHeads * dHead * config.BytesPerParam
		mem["kv_cache_access"] = kvReadPerToken * seq * (math.Max(1, newT))
	}

	// Activations: Only token-wise activations hit HBM
	// Attention maps are NOT written to HBM in FlashAttention/vLLM.
	mem["activations_tokens"] = nLayers * dModel * config.BytesPerParam * newT

	var total float64
	for _, v := range mem {
		total += v
	}
	mem["total"] = total
	return mem
}

func rooflineStepTime(gpu string, modelConfig ModelConfig, stepConfig StepConfig) int64 {
	hw, ok := HardwareList[gpu]
	if !ok {
		return 0
	}

	effFLOPS := hw.TFlopsEff * 1e12
	bwBytesPerSec := hw.BwEffTBs * 1e12

	var totalFlops, prefillFlops, decodeFlops, totalDynamicMem float64

	baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
	weightBytes := baseMem["model_weights"]

	for _, req := range stepConfig.PrefillRequests {
		f := calculateTransformerFlops(modelConfig, req.ProgressIndex, int64(req.NumNewPrefillTokens), true, true)
		prefillFlops += f["total"]
		m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, int64(req.NumNewPrefillTokens), true)
		totalDynamicMem += (m["total"] - m["model_weights"])
	}

	for _, req := range stepConfig.DecodeRequests {
		f := calculateTransformerFlops(modelConfig, req.ProgressIndex, 1, true, true)
		decodeFlops += f["total"]
		m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, 1, true)
		totalDynamicMem += (m["total"] - m["model_weights"])
	}

	totalFlops = prefillFlops + decodeFlops

	// Weighted MFU
	mfu := hw.mfuDecode
	if totalFlops > 0 {
		mfu = (prefillFlops*hw.mfuPrefill + decodeFlops*hw.mfuDecode) / totalFlops
	}

	tComputeS := totalFlops / (effFLOPS * mfu)
	tMemoryS := (weightBytes + totalDynamicMem) / bwBytesPerSec

	// Layer-wise Hardware Floor ---
	// Every layer has a fixed overhead for kernel launches and GPU state transitions.
	layerOverheadS := (float64(modelConfig.NumLayers) * hw.perLayerOverhead) / 1e6

	// Roofline: Max(Compute, Memory) + Layer Overhead
	stepTimeS := math.Max(tComputeS, tMemoryS) + layerOverheadS

	return int64(math.Round(stepTimeS*1e6 + hw.TOverheadMicros))
}
