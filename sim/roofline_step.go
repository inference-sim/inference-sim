package sim

import (
	"math"
	"sort"
)

// --- Transformer FLOPs and Memory Access ---

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
// Formula: time = flops / (peakFlops * mfu)
func computeGEMMTime(m, k, n int, peakFlops float64, mfuDB *MFUDatabase) float64 {
	// GEMM FLOPs: 2 * m * k * n (multiply-add)
	flops := 2.0 * float64(m) * float64(k) * float64(n)

	// Lookup MFU for this GEMM shape
	mfu := mfuDB.GetGEMMmfu(m, k, n)

	// Compute time in seconds
	return flops / (peakFlops * mfu)
}

// computeTransformerGEMMTimes calculates total time for all GEMM projections in transformer.
// Includes: QKV projections, O projection, MLP Gate/Up/Down projections.
//
// TP is handled by dimension-splitting: each GPU executes GEMMs with TP-adjusted
// output dimensions (nHeads/tp, dFF/tp), matching the actual kernel shapes. This
// ensures the MFU database returns values for the real GEMM executed on each GPU,
// rather than looking up MFU for the full (larger) shape and scaling by 1/tp.
//
// fuseQKV controls whether Q/K/V are looked up as separate GEMMs (false) or as a
// single fused QKV GEMM (true), matching vLLM's qkv_proj execution pattern.
func computeTransformerGEMMTimes(
	modelConfig ModelConfig,
	batchSize int,
	peakFlops float64,
	mfuDB *MFUDatabase,
	tp int,
	fuseQKV bool,
) float64 {
	dModel := modelConfig.HiddenDim
	nLayers := modelConfig.NumLayers
	nHeads := modelConfig.NumHeads
	nKVHeads := modelConfig.NumKVHeads
	if nKVHeads == 0 {
		nKVHeads = nHeads
	}

	headDim := dModel / nHeads

	// TP-adjusted dimensions: each GPU handles a shard of the heads/FFN
	nHeadsPerGPU := nHeads / tp
	nKVHeadsPerGPU := nKVHeads / tp
	if nKVHeadsPerGPU == 0 {
		nKVHeadsPerGPU = 1 // At least 1 KV head per GPU
	}
	dKVPerGPU := nKVHeadsPerGPU * headDim

	// MLP dimensions (SwiGLU), TP-adjusted
	dFF := 4 * dModel
	if modelConfig.IntermediateDim > 0 {
		dFF = modelConfig.IntermediateDim
	}
	dFFPerGPU := dFF / tp

	totalTime := 0.0

	for layer := 0; layer < nLayers; layer++ {
		// === Attention GEMMs (TP-adjusted output dimensions) ===

		if fuseQKV {
			// Fused QKV projection per GPU:
			// [batch_size, dModel] @ [dModel, (nHeadsPerGPU + 2*nKVHeadsPerGPU)*headDim]
			dQKVPerGPU := (nHeadsPerGPU + 2*nKVHeadsPerGPU) * headDim
			qkvTime := computeGEMMTime(batchSize, dModel, dQKVPerGPU, peakFlops, mfuDB)
			totalTime += qkvTime
		} else {
			// Split Q/K/V projections per GPU:
			// Q: [batch_size, dModel] @ [dModel, nHeadsPerGPU*headDim]
			dQPerGPU := nHeadsPerGPU * headDim
			qTime := computeGEMMTime(batchSize, dModel, dQPerGPU, peakFlops, mfuDB)
			totalTime += qTime

			// K: [batch_size, dModel] @ [dModel, dKVPerGPU]
			kTime := computeGEMMTime(batchSize, dModel, dKVPerGPU, peakFlops, mfuDB)
			totalTime += kTime

			// V: [batch_size, dModel] @ [dModel, dKVPerGPU]
			vTime := computeGEMMTime(batchSize, dModel, dKVPerGPU, peakFlops, mfuDB)
			totalTime += vTime
		}

		// O projection per GPU: [batch_size, nHeadsPerGPU*headDim] @ [nHeadsPerGPU*headDim, dModel]
		dOIn := nHeadsPerGPU * headDim
		oTime := computeGEMMTime(batchSize, dOIn, dModel, peakFlops, mfuDB)
		totalTime += oTime

		// === MLP GEMMs (SwiGLU, TP-adjusted) ===

		// Gate projection per GPU: [batch_size, dModel] @ [dModel, dFFPerGPU]
		gateTime := computeGEMMTime(batchSize, dModel, dFFPerGPU, peakFlops, mfuDB)
		totalTime += gateTime

		// Up projection per GPU: [batch_size, dModel] @ [dModel, dFFPerGPU]
		upTime := computeGEMMTime(batchSize, dModel, dFFPerGPU, peakFlops, mfuDB)
		totalTime += upTime

		// Down projection per GPU: [batch_size, dFFPerGPU] @ [dFFPerGPU, dModel]
		downTime := computeGEMMTime(batchSize, dFFPerGPU, dModel, peakFlops, mfuDB)
		totalTime += downTime
	}

	return totalTime
}

// --- Attention Core FLOPs Calculation ---

// calculateAttentionCoreFLOPs computes FLOPs for attention core operations.
// Includes: QK^T matmul + attention-value matmul (excludes softmax).
//
// Softmax is excluded because the MFU benchmark data (bench_data/) computes
// MFU as: mfu = attn_core_gflops / (peak_tflops * measured_time), where
// attn_core_gflops only counts QK^T + AV matmuls (see InferSim's
// flops/flops.py:get_mha_gflops and kernel_benchmark/flashinfer_mha_decode.py).
// Including softmax FLOPs in the numerator while using MFU values computed
// without softmax would systematically overestimate attention compute time.
func calculateAttentionCoreFLOPs(
	nHeads int,
	nKVHeads int,
	dModel int,
	batchSize int,
	seqLen int64,
) float64 {
	if nKVHeads == 0 {
		nKVHeads = nHeads
	}

	headDim := dModel / nHeads
	effectiveCtx := float64(seqLen)

	// QK^T matmul: 2 * nHeads * batchSize * effectiveCtx * headDim
	qkMatMul := 2.0 * float64(nHeads) * float64(batchSize) * effectiveCtx * float64(headDim)

	// Attention-Value matmul: 2 * nHeads * batchSize * effectiveCtx * headDim
	avMatMul := 2.0 * float64(nHeads) * float64(batchSize) * effectiveCtx * float64(headDim)

	return qkMatMul + avMatMul
}

// --- Main Roofline Function ---

func rooflineStepTime(
	_ string,
	modelConfig ModelConfig,
	hwConfig HardwareCalib,
	stepConfig StepConfig,
	tp int,
	mfuDB *MFUDatabase,
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

		// Aggregate batch size and find max kv_len
		totalBatchSize := len(stepConfig.DecodeRequests)
		maxKVLen := int64(0)

		for _, req := range stepConfig.DecodeRequests {
			if req.ProgressIndex > maxKVLen {
				maxKVLen = req.ProgressIndex
			}
		}

		// === GEMM Projections ===
		// Single aggregated lookup for all decode requests
		gemmTimeS := computeTransformerGEMMTimes(
			modelConfig,
			totalBatchSize,
			peakFlops,
			mfuDB,
			tp,
			hwConfig.FuseQKV,
		)

		// === Attention Core ===
		if hwConfig.PerComponentRoofline {
			// Per-component roofline: apply max(attn_compute, kv_memory) per layer,
			// then sum across layers. This captures bottleneck transitions where
			// attention is compute-bound at some KV lengths but memory-bound at others.
			// The aggregate approach does max(totalCompute, totalMemory) once, which
			// masks these per-layer crossovers.
			nKVHeads := modelConfig.NumKVHeads
			if nKVHeads == 0 {
				nKVHeads = modelConfig.NumHeads
			}
			headDim := modelConfig.HiddenDim / modelConfig.NumHeads

			// Attention core FLOPs for ONE layer (all decode requests)
			attnCoreFLOPs1Layer := calculateAttentionCoreFLOPs(
				modelConfig.NumHeads,
				modelConfig.NumKVHeads,
				modelConfig.HiddenDim,
				totalBatchSize,
				maxKVLen,
			)
			attnMFU := mfuDB.GetAttnDecodeMFU(totalBatchSize, int(maxKVLen), tp)
			attnComputePerLayerS := attnCoreFLOPs1Layer / (peakFlops * attnMFU) * tpScaling

			// Per-layer KV cache read bytes (decode: scattered access, 0.80 factor)
			// Sum across all decode requests for per-layer KV bytes
			var kvReadBytesPerLayer float64
			for _, req := range stepConfig.DecodeRequests {
				kvReadBytesPerLayer += 2 * float64(nKVHeads) * float64(headDim) *
					modelConfig.BytesPerParam * float64(req.ProgressIndex) * 0.80
			}
			kvMemoryPerLayerS := kvReadBytesPerLayer * tpScaling / peakBW

			// Per-layer roofline: max(attn_compute, kv_memory) per layer, summed
			nLayers := float64(modelConfig.NumLayers)
			perLayerTimeS := math.Max(attnComputePerLayerS, kvMemoryPerLayerS)
			totalAttnTimeS := perLayerTimeS * nLayers

			decodeComputeS = gemmTimeS + totalAttnTimeS
		} else {
			// Aggregate roofline (default): compute attention time across all layers
			// as a single monolithic term.
			attnCoreFLOPs := calculateAttentionCoreFLOPs(
				modelConfig.NumHeads,
				modelConfig.NumKVHeads,
				modelConfig.HiddenDim,
				totalBatchSize,
				maxKVLen,
			) * float64(modelConfig.NumLayers)

			attnMFU := mfuDB.GetAttnDecodeMFU(totalBatchSize, int(maxKVLen), tp)
			attnCoreTimeS := attnCoreFLOPs / (peakFlops * attnMFU) * tpScaling

			decodeComputeS = gemmTimeS + attnCoreTimeS
		}

		// === Memory Bandwidth ===
		if hwConfig.PerComponentRoofline {
			// Per-component mode: memory path = weight loading only.
			// KV cache access is already accounted for in the per-layer
			// max(attn_compute, kv_memory) above, so including it here
			// would double-count. This makes the per-component model less
			// conservative than aggregate when KV access dominates.
			baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
			dWeightBytes := baseMem["model_weights"] * tpScaling
			decodeMemoryS = dWeightBytes / peakBW
		} else {
			// Aggregate mode: memory path = weights + all dynamic bytes
			// (KV access, KV growth, activations).
			var dDynamicBytes float64
			for _, req := range stepConfig.DecodeRequests {
				m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, 1, true)
				dDynamicBytes += (m["total"] - m["model_weights"]) * tpScaling
			}

			baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
			dWeightBytes := baseMem["model_weights"] * tpScaling

			decodeMemoryS = (dWeightBytes + dDynamicBytes) / peakBW
		}
	}

	// ========================================
	// 2. PREFILL PHASE (Bucket by seq_len)
	// ========================================
	if len(stepConfig.PrefillRequests) > 0 {
		hasPrefill = true

		// Group prefill requests by seq_len bucket
		// Use power-of-2 buckets: 512, 1024, 2048, 4096, 8192, etc.
		bucketMap := make(map[int][]PrefillRequestConfig)

		for _, req := range stepConfig.PrefillRequests {
			seqLen := int(req.ProgressIndex + int64(req.NumNewPrefillTokens))

			// Find next power-of-2 bucket
			bucket := 512
			for bucket < seqLen && bucket < 65536 {
				bucket *= 2
			}
			if bucket > 65536 {
				bucket = 65536 // Cap at max bucket
			}

			bucketMap[bucket] = append(bucketMap[bucket], req)
		}

		// Sort bucket keys for deterministic iteration (R2)
		bucketKeys := make([]int, 0, len(bucketMap))
		for k := range bucketMap {
			bucketKeys = append(bucketKeys, k)
		}
		sort.Ints(bucketKeys)

		// Process each bucket independently
		for _, bucketSeqLen := range bucketKeys {
			requests := bucketMap[bucketSeqLen]
			batchSize := len(requests)

			// === GEMM Projections ===
			// Use bucket batch size for GEMM lookups
			gemmTimeS := computeTransformerGEMMTimes(
				modelConfig,
				batchSize,
				peakFlops,
				mfuDB,
				tp,
				hwConfig.FuseQKV,
			)

			if hwConfig.PerComponentRoofline {
				// Per-component roofline for prefill: per-layer attention roofline
				nKVHeads := modelConfig.NumKVHeads
				if nKVHeads == 0 {
					nKVHeads = modelConfig.NumHeads
				}
				headDim := modelConfig.HiddenDim / modelConfig.NumHeads

				// Attention core FLOPs for ONE layer (this bucket)
				attnCoreFLOPs1Layer := calculateAttentionCoreFLOPs(
					modelConfig.NumHeads,
					modelConfig.NumKVHeads,
					modelConfig.HiddenDim,
					batchSize,
					int64(bucketSeqLen),
				)
				attnMFU := mfuDB.GetAttnPrefillMFU(bucketSeqLen)
				attnComputePerLayerS := attnCoreFLOPs1Layer / 1.8 / (peakFlops * attnMFU) * tpScaling

				// Per-layer KV cache read bytes (prefill: sequential access, 0.92 factor)
				kvReadBytesPerLayer := 2 * float64(nKVHeads) * float64(headDim) *
					modelConfig.BytesPerParam * float64(bucketSeqLen) * float64(batchSize) * 0.92
				kvMemoryPerLayerS := kvReadBytesPerLayer * tpScaling / peakBW

				// Per-layer roofline: max(attn_compute, kv_memory) per layer, summed
				nLayers := float64(modelConfig.NumLayers)
				perLayerTimeS := math.Max(attnComputePerLayerS, kvMemoryPerLayerS)
				totalAttnTimeS := perLayerTimeS * nLayers

				prefillComputeS += gemmTimeS + totalAttnTimeS

			} else {
				// Aggregate roofline (default)

				// === Attention Core ===
				attnCoreFLOPs := calculateAttentionCoreFLOPs(
					modelConfig.NumHeads,
					modelConfig.NumKVHeads,
					modelConfig.HiddenDim,
					batchSize,
					int64(bucketSeqLen),
				) * float64(modelConfig.NumLayers)

				attnMFU := mfuDB.GetAttnPrefillMFU(bucketSeqLen)
				attnCoreTimeS := attnCoreFLOPs / 1.8 / (peakFlops * attnMFU) * tpScaling

				prefillComputeS += gemmTimeS + attnCoreTimeS
			}
		}

		// === Memory Bandwidth ===
		if hwConfig.PerComponentRoofline {
			// Per-component mode: memory path = weight loading only.
			// KV cache access handled per-layer above (same reasoning as decode).
			baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
			pWeightBytes := baseMem["model_weights"] * tpScaling
			prefillMemoryS = pWeightBytes / peakBW
		} else {
			// Aggregate mode: memory path = weights + all dynamic bytes.
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
	}

	// ========================================
	// 3. COMBINE PHASES
	// ========================================
	var stepHardwareS float64

	if hasPrefill && hasDecode {
		switch hwConfig.MixedBatchMode {
		case "additive":
			// Additive model: matches vLLM chunked-prefill execution.
			// (1) One fused GEMM over the concatenated prefill+decode batch
			// (2) Separate FlashAttention kernels for prefill and decode
			// (3) Single model weight load (not once per phase)
			totalBatchSize := len(stepConfig.PrefillRequests) + len(stepConfig.DecodeRequests)
			fusedGEMMTime := computeTransformerGEMMTimes(
				modelConfig, totalBatchSize, peakFlops, mfuDB, tp, hwConfig.FuseQKV,
			)

			// Attention times are already computed separately above
			prefillAttnTime := prefillComputeS - computeTransformerGEMMTimes(
				modelConfig, len(stepConfig.PrefillRequests), peakFlops, mfuDB, tp, hwConfig.FuseQKV,
			)
			decodeAttnTime := decodeComputeS - computeTransformerGEMMTimes(
				modelConfig, len(stepConfig.DecodeRequests), peakFlops, mfuDB, tp, hwConfig.FuseQKV,
			)
			if prefillAttnTime < 0 {
				prefillAttnTime = 0
			}
			if decodeAttnTime < 0 {
				decodeAttnTime = 0
			}

			computeS := fusedGEMMTime + prefillAttnTime + decodeAttnTime

			// Memory: single weight load + combined dynamic access
			baseMem := calculateMemoryAccessBytes(modelConfig, 0, 0, false)
			weightBytes := baseMem["model_weights"] * tpScaling

			var dynamicBytes float64
			for _, req := range stepConfig.PrefillRequests {
				numTokens := int64(req.NumNewPrefillTokens)
				m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, numTokens, true)
				dynamicBytes += (m["total"] - m["model_weights"]) * tpScaling
			}
			for _, req := range stepConfig.DecodeRequests {
				m := calculateMemoryAccessBytes(modelConfig, req.ProgressIndex, 1, true)
				dynamicBytes += (m["total"] - m["model_weights"]) * tpScaling
			}

			memoryS := (weightBytes + dynamicBytes) / peakBW
			stepHardwareS = math.Max(computeS, memoryS)

		case "smooth-wa":
			// Smooth weighted-average: token-proportional without branch thresholds.
			// Isolates branch removal from the additive structural change.
			prefillTimeS := math.Max(prefillComputeS, prefillMemoryS)
			decodeTimeS := math.Max(decodeComputeS, decodeMemoryS)

			prefillTokens := 0.0
			for _, req := range stepConfig.PrefillRequests {
				prefillTokens += float64(req.NumNewPrefillTokens)
			}
			decodeTokens := float64(len(stepConfig.DecodeRequests))
			totalTokens := prefillTokens + decodeTokens

			prefillWeight := prefillTokens / totalTokens
			decodeWeight := decodeTokens / totalTokens
			stepHardwareS = prefillWeight*prefillTimeS + decodeWeight*decodeTimeS

		default:
			// Default: current branching weighted average (baseline)
			prefillTimeS := math.Max(prefillComputeS, prefillMemoryS)
			decodeTimeS := math.Max(decodeComputeS, decodeMemoryS)

			prefillTokens := 0.0
			for _, req := range stepConfig.PrefillRequests {
				prefillTokens += float64(req.NumNewPrefillTokens)
			}
			decodeTokens := float64(len(stepConfig.DecodeRequests))
			totalTokens := prefillTokens + decodeTokens

			// Adaptive weighting based on token distribution
			if prefillTokens > decodeTokens*4 && prefillTokens > 100 {
				// Prefill-dominated: apply blending
				stepHardwareS = 0.75*prefillTimeS + 0.25*decodeTimeS
			} else if decodeTokens > prefillTokens*2 && decodeTokens > 50 {
				// Decode-dominated: weight towards decode
				stepHardwareS = 0.35*prefillTimeS + 0.65*decodeTimeS
			} else {
				// Balanced mix: weighted average
				prefillWeight := prefillTokens / totalTokens
				decodeWeight := decodeTokens / totalTokens
				stepHardwareS = prefillWeight*prefillTimeS + decodeWeight*decodeTimeS
			}
		}
	} else if hasPrefill {
		stepHardwareS = math.Max(prefillComputeS, prefillMemoryS)
	} else if hasDecode {
		stepHardwareS = math.Max(decodeComputeS, decodeMemoryS)
	}

	// ========================================
	// 4. CPU SCHEDULING OVERHEAD
	// ========================================

	// Per-step CPU overhead: perLayerOverhead × (num_layers / tp)
	overheadMicros := hwConfig.PerLayerCPUOverhead * float64(modelConfig.NumLayers) / tpFactor

	// Total time
	totalS := stepHardwareS + (overheadMicros / 1e6)
	totalMicros := totalS * 1e6

	return int64(math.Round(totalMicros))
}
