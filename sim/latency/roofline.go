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

// --- Transformer FLOPs and Memory Access ---

func calculateTransformerFlops(config sim.ModelConfig, sequenceLength int64, newTokens int64, includeAttention, includeMLP bool) map[string]float64 {
	dModel := float64(config.HiddenDim)
	nLayers := float64(config.NumLayers)
	nHeads := float64(config.NumHeads)
	nKVHeads := float64(config.NumKVHeads)
	if nKVHeads == 0 {
		nKVHeads = nHeads
	}
	dHead := dModel / nHeads

	dFF := 4.0 * dModel
	if config.IntermediateDim > 0 {
		dFF = float64(config.IntermediateDim)
	}

	seqLen := float64(sequenceLength)
	newT := float64(newTokens)
	flops := make(map[string]float64)

	if includeAttention {
		dKV := nKVHeads * dHead

		qkvFlops := 2 * newT * (dModel*dModel + 2*dModel*dKV)
		projFlops := 2 * newT * dModel * dModel
		flops["gemm_ops"] = (qkvFlops + projFlops) * nLayers

		effectiveCtx := seqLen
		if newT > 1 {
			effectiveCtx = seqLen + (newT-1)/2.0
		}

		qkMatMul := 2 * nHeads * newT * effectiveCtx * dHead
		softmaxOps := 4 * nHeads * newT * effectiveCtx
		avMatMul := 2 * nHeads * newT * effectiveCtx * dHead
		attnMath := qkMatMul + softmaxOps + avMatMul
		flops["sram_ops"] = attnMath * nLayers
	}

	if includeMLP {
		flops["gemm_ops"] += 2 * newT * (3 * dModel * dFF) * nLayers
	}

	flops["total"] = flops["gemm_ops"] + flops["sram_ops"]
	return flops
}

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

	dKV := nKVHeads * dHead
	weightsPerLayer := dModel*(dModel+2*dKV) + (dModel * dModel) + (3 * dModel * dFF)
	mem["model_weights"] = weightsPerLayer * nLayers * config.BytesPerParam

	if includeKVCache {
		kvWritePerNewToken := 2 * nLayers * nKVHeads * dHead * config.BytesPerParam
		mem["kv_cache_growth"] = kvWritePerNewToken * newT

		kvReadPerToken := 2 * nLayers * nKVHeads * dHead * config.BytesPerParam
		if newT == 1 {
			mem["kv_cache_access"] = kvReadPerToken * seq * 0.80
		} else {
			mem["kv_cache_access"] = kvReadPerToken * seq * 0.92
		}
	}

	var activationBytes float64
	if newT == 1 {
		activationBytes = nLayers * dModel * config.BytesPerParam * newT * 0.75
	} else {
		activationBytes = nLayers * dModel * config.BytesPerParam * newT * 0.85
	}
	mem["activations_tokens"] = activationBytes

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

// --- Helper Functions for GEMM Time Computation ---

// computeGEMMTime calculates time for a single GEMM operation using MFU lookup
// with a memory-bandwidth floor.
//
// At small M (batch size), the compute time shrinks linearly with M while
// the weight matrix [K×N] must still be fully loaded from HBM. The memory
// floor ensures that GEMM time never drops below weight-load time:
//
//	time = max(flops / (peakFlops * mfu), weightBytes / peakBW)
func computeGEMMTime(m, k, n int, peakFlops, peakBW, bytesPerParam float64, mfuDB *sim.MFUDatabase) float64 {
	flops := 2.0 * float64(m) * float64(k) * float64(n)
	mfu := mfuDB.GetGEMMmfu(m, k, n)
	computeTime := flops / (peakFlops * mfu)
	weightBytes := float64(k) * float64(n) * bytesPerParam
	memFloor := weightBytes / peakBW
	return math.Max(computeTime, memFloor)
}

// computeTransformerGEMMTimes calculates total time for all GEMM projections in transformer.
// Includes: QKV projections, O projection, MLP Gate/Up/Down projections.
// tpScaling should be 1/tp to account for TP parallelization.
func computeTransformerGEMMTimes(
	modelConfig sim.ModelConfig,
	batchSize int,
	peakFlops float64,
	peakBW float64,
	mfuDB *sim.MFUDatabase,
	tpScaling float64,
) float64 {
	dModel := modelConfig.HiddenDim
	nLayers := modelConfig.NumLayers
	nHeads := modelConfig.NumHeads
	nKVHeads := modelConfig.NumKVHeads
	if nKVHeads == 0 {
		nKVHeads = nHeads
	}

	headDim := dModel / nHeads
	dKV := nKVHeads * headDim
	bytesPerParam := modelConfig.BytesPerParam

	dFF := 4 * dModel
	if modelConfig.IntermediateDim > 0 {
		dFF = modelConfig.IntermediateDim
	}

	totalTime := 0.0

	for layer := 0; layer < nLayers; layer++ {
		// === Attention GEMMs ===
		qTime := computeGEMMTime(batchSize, dModel, dModel, peakFlops, peakBW, bytesPerParam, mfuDB)
		totalTime += qTime * tpScaling

		kTime := computeGEMMTime(batchSize, dModel, dKV, peakFlops, peakBW, bytesPerParam, mfuDB)
		totalTime += kTime * tpScaling

		vTime := computeGEMMTime(batchSize, dModel, dKV, peakFlops, peakBW, bytesPerParam, mfuDB)
		totalTime += vTime * tpScaling

		oTime := computeGEMMTime(batchSize, dModel, dModel, peakFlops, peakBW, bytesPerParam, mfuDB)
		totalTime += oTime * tpScaling

		// === MLP GEMMs (SwiGLU) ===
		gateTime := computeGEMMTime(batchSize, dModel, dFF, peakFlops, peakBW, bytesPerParam, mfuDB)
		totalTime += gateTime * tpScaling

		upTime := computeGEMMTime(batchSize, dModel, dFF, peakFlops, peakBW, bytesPerParam, mfuDB)
		totalTime += upTime * tpScaling

		downTime := computeGEMMTime(batchSize, dFF, dModel, peakFlops, peakBW, bytesPerParam, mfuDB)
		totalTime += downTime * tpScaling
	}

	return totalTime
}

// --- Attention Core FLOPs Calculation ---

// calculateAttentionCoreFLOPs computes FLOPs for attention core operations.
// Includes: QK^T matmul + attention-value matmul (excludes softmax).
func calculateAttentionCoreFLOPs(
	nHeads int,
	_ int, // nKVHeads unused: Q-head count drives both QK^T and AV FLOPs
	dModel int,
	batchSize int,
	seqLen int64,
) float64 {
	headDim := dModel / nHeads
	effectiveCtx := float64(seqLen)

	qkMatMul := 2.0 * float64(nHeads) * float64(batchSize) * effectiveCtx * float64(headDim)
	avMatMul := 2.0 * float64(nHeads) * float64(batchSize) * effectiveCtx * float64(headDim)

	return qkMatMul + avMatMul
}

// --- Main Roofline Function ---

// rooflineStepTime computes step latency using the roofline model (v2).
// When mfuDB is non-nil, MFU values are looked up from benchmark data.
// When mfuDB is nil, falls back to a minimal compute-only estimate.
// Precondition: ValidateRooflineConfig(modelConfig, hwConfig) must return nil
// and tp must be > 0. Callers must validate before first call.
func rooflineStepTime(
	_ string,
	modelConfig sim.ModelConfig,
	hwConfig sim.HardwareCalib,
	stepConfig StepConfig,
	tp int,
	mfuDB *sim.MFUDatabase,
) int64 {
	tpFactor := float64(tp)
	tpScaling := 1.0 / tpFactor

	peakFlops := hwConfig.TFlopsPeak * 1e12
	peakBW := hwConfig.BwPeakTBs * 1e12
	if hwConfig.BwEfficiencyFactor != 0 {
		peakBW *= hwConfig.BwEfficiencyFactor
	}

	var prefillComputeS, prefillMemoryS float64
	var decodeComputeS, decodeMemoryS float64
	var hasPrefill, hasDecode bool

	// ========================================
	// 1. DECODE PHASE (Aggregate All Requests)
	// ========================================
	if len(stepConfig.DecodeRequests) > 0 {
		hasDecode = true

		totalBatchSize := len(stepConfig.DecodeRequests)

		// === GEMM Projections ===
		gemmTimeS := computeTransformerGEMMTimes(
			modelConfig,
			totalBatchSize,
			peakFlops,
			peakBW,
			mfuDB,
			tpScaling,
		)

		// === Attention Core ===
		// FLOPs-weighted MFU across heterogeneous KV lengths
		var attnCoreFLOPs float64
		var weightedMFUSum float64
		for _, req := range stepConfig.DecodeRequests {
			reqFLOPs := calculateAttentionCoreFLOPs(
				modelConfig.NumHeads,
				modelConfig.NumKVHeads,
				modelConfig.HiddenDim,
				1,
				req.ProgressIndex,
			) * float64(modelConfig.NumLayers)
			attnCoreFLOPs += reqFLOPs
			reqMFU := mfuDB.GetAttnDecodeMFU(totalBatchSize, int(req.ProgressIndex), tp)
			weightedMFUSum += reqFLOPs * reqMFU
		}

		var attnCoreTimeS float64
		if attnCoreFLOPs > 0 {
			effectiveMFU := weightedMFUSum / attnCoreFLOPs
			if effectiveMFU > 0 {
				attnCoreTimeS = attnCoreFLOPs / (peakFlops * effectiveMFU) * tpScaling
			}
		}

		decodeComputeS = gemmTimeS + attnCoreTimeS

		// === Memory Bandwidth ===
		var dDynamicBytes float64
		for _, req := range stepConfig.DecodeRequests {
			m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, 1, true)
			dDynamicBytes += (m["total"] - m["model_weights"]) * tpScaling
		}

		baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
		dWeightBytes := baseMem["model_weights"] * tpScaling

		decodeMemoryS = (dWeightBytes + dDynamicBytes) / peakBW
	}

	// ========================================
	// 2. PREFILL PHASE (Bucket by seq_len)
	// ========================================
	if len(stepConfig.PrefillRequests) > 0 {
		hasPrefill = true

		// Group prefill requests by power-of-2 seq_len bucket
		bucketMap := make(map[int][]PrefillRequestConfig)

		for _, req := range stepConfig.PrefillRequests {
			seqLen := int(req.ProgressIndex + int64(req.NumNewPrefillTokens))

			bucket := 512
			for bucket < seqLen && bucket < 65536 {
				bucket *= 2
			}
			if bucket > 65536 {
				bucket = 65536
			}

			bucketMap[bucket] = append(bucketMap[bucket], req)
		}

		// Sort bucket keys for deterministic iteration (R2)
		bucketKeys := make([]int, 0, len(bucketMap))
		for k := range bucketMap {
			bucketKeys = append(bucketKeys, k)
		}
		sort.Ints(bucketKeys)

		for _, bucketSeqLen := range bucketKeys {
			requests := bucketMap[bucketSeqLen]

			totalPrefillTokens := 0
			for _, req := range requests {
				totalPrefillTokens += req.NumNewPrefillTokens
			}

			// === GEMM Projections ===
			gemmTimeS := computeTransformerGEMMTimes(
				modelConfig,
				totalPrefillTokens,
				peakFlops,
				peakBW,
				mfuDB,
				tpScaling,
			)

			// === Attention Core ===
			var attnCoreFLOPs float64
			for _, req := range requests {
				actualSeqLen := req.ProgressIndex + int64(req.NumNewPrefillTokens)
				attnCoreFLOPs += calculateAttentionCoreFLOPs(
					modelConfig.NumHeads,
					modelConfig.NumKVHeads,
					modelConfig.HiddenDim,
					req.NumNewPrefillTokens,
					actualSeqLen,
				) * float64(modelConfig.NumLayers)
			}

			attnMFU := mfuDB.GetAttnPrefillMFU(bucketSeqLen)

			// The /1.8 factor corrects for causal attention masking.
			attnCoreTimeS := attnCoreFLOPs / 1.8 / (peakFlops * attnMFU) * tpScaling

			prefillComputeS += gemmTimeS + attnCoreTimeS
		}

		// === Memory Bandwidth ===
		var pDynamicBytes float64
		for _, req := range stepConfig.PrefillRequests {
			numTokens := int64(req.NumNewPrefillTokens)
			m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, numTokens, true)
			pDynamicBytes += (m["total"] - m["model_weights"]) * tpScaling
		}

		baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
		pWeightBytes := baseMem["model_weights"] * tpScaling

		prefillMemoryS = (pWeightBytes + pDynamicBytes) / peakBW
	}

	// ========================================
	// 3. COMBINE PHASES
	// ========================================
	var stepHardwareS float64

	if hasPrefill && hasDecode {
		prefillTimeS := math.Max(prefillComputeS, prefillMemoryS)
		decodeTimeS := math.Max(decodeComputeS, decodeMemoryS)
		stepHardwareS = math.Max(prefillTimeS, decodeTimeS)
	} else if hasPrefill {
		stepHardwareS = math.Max(prefillComputeS, prefillMemoryS)
	} else if hasDecode {
		stepHardwareS = math.Max(decodeComputeS, decodeMemoryS)
	}

	// ========================================
	// 4. CPU SCHEDULING OVERHEAD
	// ========================================
	overheadMicros := hwConfig.PerLayerCPUOverhead * float64(modelConfig.NumLayers) / tpFactor

	totalS := stepHardwareS + (overheadMicros / 1e6)
	totalMicros := totalS * 1e6

	return int64(math.Round(totalMicros))
}
