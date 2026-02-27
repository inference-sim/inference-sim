package latency

import (
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// --- Roofline Model Calibration Constants ---
//
// Known approximations: these empirical discount factors were calibrated against
// H100 kernel measurements. They account for HW-level effects (caching, prefetching,
// warp scheduling) that reduce effective memory traffic below theoretical values.
// Exact values may vary by GPU architecture; making them HardwareCalib fields is a
// future improvement tracked as a known limitation.
const (
	// kvCacheAccessDiscountDecode is the fraction of theoretical KV cache read
	// bytes actually transferred during decode (batch_size=1 token per request).
	// Lower than prefill because decode reads are more scattered (single-token
	// attention patterns), reducing cache line utilization.
	kvCacheAccessDiscountDecode = 0.80

	// kvCacheAccessDiscountPrefill is the fraction of theoretical KV cache read
	// bytes actually transferred during prefill (contiguous multi-token reads).
	// Higher than decode because prefill reads are sequential, benefiting from
	// HBM burst mode and L2 prefetching.
	kvCacheAccessDiscountPrefill = 0.92

	// activationDiscountDecode is the fraction of theoretical activation bytes
	// transferred during decode. Decode activations are small (single token),
	// reducing effective bandwidth utilization.
	activationDiscountDecode = 0.75

	// activationDiscountPrefill is the fraction of theoretical activation bytes
	// transferred during prefill. Prefill activations are larger and more
	// sequential, achieving better bandwidth utilization.
	activationDiscountPrefill = 0.85

	// causalAttentionFLOPsReduction is the divisor applied to attention FLOPs
	// to account for causal masking. Causal attention skips the upper triangle
	// of the attention matrix, theoretically halving FLOPs (factor of 2.0).
	// The empirical value of 1.8 accounts for FlashAttention's tile-based
	// execution where partial tiles still execute some masked operations.
	// Known approximation: this value was calibrated on H100; exact value
	// varies by GPU architecture and kernel implementation.
	causalAttentionFLOPsReduction = 1.8

	// minStepTimeMicros is the minimum step time in microseconds.
	// Prevents zero-advance steps that could cause livelock in the DES event loop
	// when both phases are empty and PerLayerCPUOverhead is zero (R19/INV-3).
	minStepTimeMicros = 1.0
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

// calculateTransformerFlops is test-only infrastructure used by roofline_test.go
// for validating FLOPs decomposition. It is NOT called by production rooflineStepTime,
// which computes FLOPs inline via MFU-based lookups from bench_data.
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
			mem["kv_cache_access"] = kvReadPerToken * seq * kvCacheAccessDiscountDecode
		} else {
			mem["kv_cache_access"] = kvReadPerToken * seq * kvCacheAccessDiscountPrefill
		}
	}

	var activationBytes float64
	if newT == 1 {
		activationBytes = nLayers * dModel * config.BytesPerParam * newT * activationDiscountDecode
	} else {
		activationBytes = nLayers * dModel * config.BytesPerParam * newT * activationDiscountPrefill
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
//
// Note on TP scaling: This function computes time for the FULL (un-split) GEMM.
// The caller (computeTransformerGEMMTimes) applies tpScaling = 1/tp AFTER the max().
// This is correct because max(a,b)/tp = max(a/tp, b/tp), so the TP-split compute
// and TP-split memory floor are both correctly represented.
func computeGEMMTime(m, k, n int, peakFlops, peakBW, bytesPerParam float64, mfuDB *sim.MFUDatabase) float64 {
	flops := 2.0 * float64(m) * float64(k) * float64(n)
	mfu := mfuDB.GetGEMMmfu(m, k, n)

	// R11: guard division by peakFlops*mfu and peakBW (validated at factory, defense-in-depth)
	effectiveFlops := peakFlops * mfu
	if effectiveFlops <= 0 {
		effectiveFlops = 1 // floor: prevents Inf; degenerate config yields ~flops seconds
	}
	computeTime := flops / effectiveFlops

	weightBytes := float64(k) * float64(n) * bytesPerParam
	if peakBW <= 0 {
		return computeTime // no valid BW floor; return compute-only
	}
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

	// R11: guard division by nHeads (validated at factory via ValidateRooflineConfig, defense-in-depth)
	if nHeads <= 0 {
		return 0
	}
	headDim := dModel / nHeads
	dKV := nKVHeads * headDim
	bytesPerParam := modelConfig.BytesPerParam

	dFF := 4 * dModel
	if modelConfig.IntermediateDim > 0 {
		dFF = modelConfig.IntermediateDim
	}

	// All layers have identical dimensions; compute one layer and multiply (nLayers × 7 MFU lookups → 7).
	// === Attention GEMMs ===
	qTime := computeGEMMTime(batchSize, dModel, dModel, peakFlops, peakBW, bytesPerParam, mfuDB)
	kTime := computeGEMMTime(batchSize, dModel, dKV, peakFlops, peakBW, bytesPerParam, mfuDB)
	vTime := computeGEMMTime(batchSize, dModel, dKV, peakFlops, peakBW, bytesPerParam, mfuDB)
	oTime := computeGEMMTime(batchSize, dModel, dModel, peakFlops, peakBW, bytesPerParam, mfuDB)

	// === MLP GEMMs (SwiGLU) ===
	gateTime := computeGEMMTime(batchSize, dModel, dFF, peakFlops, peakBW, bytesPerParam, mfuDB)
	upTime := computeGEMMTime(batchSize, dModel, dFF, peakFlops, peakBW, bytesPerParam, mfuDB)
	downTime := computeGEMMTime(batchSize, dFF, dModel, peakFlops, peakBW, bytesPerParam, mfuDB)

	perLayerTime := (qTime + kTime + vTime + oTime + gateTime + upTime + downTime) * tpScaling
	return perLayerTime * float64(nLayers)
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
	// R11: guard division by nHeads (validated at factory via ValidateRooflineConfig, defense-in-depth)
	if nHeads <= 0 {
		return 0
	}
	headDim := dModel / nHeads
	effectiveCtx := float64(seqLen)

	qkMatMul := 2.0 * float64(nHeads) * float64(batchSize) * effectiveCtx * float64(headDim)
	avMatMul := 2.0 * float64(nHeads) * float64(batchSize) * effectiveCtx * float64(headDim)

	return qkMatMul + avMatMul
}

// --- Main Roofline Function ---

// rooflineStepTime computes step latency using the roofline model (v2).
// MFU values are looked up from benchmark data via mfuDB.
// Precondition: mfuDB must be non-nil, ValidateRooflineConfig(modelConfig, hwConfig)
// must return nil, and tp must be > 0. NewLatencyModel enforces these at construction.
func rooflineStepTime(
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
		// Known approximation: decode MFU from bench_data measures standalone attention
		// kernel efficiency. In mixed batches, actual MFU may differ due to kernel scheduling
		// interactions. See hypothesis H16 for decode MFU semantic validation.
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
		// Known approximation: model weights are counted once per phase (decode OR prefill).
		// In mixed batches where max() selects the slower phase, only that phase's weight
		// load is counted. This slightly underestimates total memory traffic but aligns
		// with the max()-based phase combination model.
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

			// Known approximation: The /1.8 factor corrects for causal attention masking.
			// Causal masking skips ~50% of FLOPs (lower triangle), but HW utilization
			// differs from dense attention. The 1.8 constant was empirically calibrated
			// against H100 kernel measurements; exact value varies by GPU architecture.
			attnCoreTimeS := attnCoreFLOPs / causalAttentionFLOPsReduction / (peakFlops * attnMFU) * tpScaling

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
		// Known approximation: mixed batches use max(prefill, decode) rather than sum.
		// This models chunked-prefill scheduling where prefill and decode overlap on the
		// GPU pipeline. In practice, overlap is partial; max() slightly underestimates
		// mixed-batch latency. See hypothesis H27 for empirical validation.
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
	// Known approximation: dividing by tpFactor models the assumption that CPU
	// scheduling work is distributed across TP ranks. This is an approximation —
	// standard TP splits each layer's tensors across GPUs (not distributing layers),
	// so CPU overhead per layer is arguably constant regardless of TP. However,
	// empirical H2b calibration showed better MAPE with the division, suggesting
	// vLLM's per-rank scheduling does scale with TP degree in practice.
	overheadMicros := hwConfig.PerLayerCPUOverhead * float64(modelConfig.NumLayers) / tpFactor

	totalS := stepHardwareS + (overheadMicros / 1e6)
	totalMicros := totalS * 1e6

	// Enforce minimum step time to prevent zero-advance livelock (R19/INV-3).
	if totalMicros < minStepTimeMicros {
		totalMicros = minStepTimeMicros
	}

	return int64(math.Round(totalMicros))
}
