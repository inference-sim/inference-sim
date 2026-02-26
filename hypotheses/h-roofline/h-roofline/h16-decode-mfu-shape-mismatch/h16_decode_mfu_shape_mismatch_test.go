//go:build ignore

package sim

import (
	"fmt"
	"math"
	"testing"
)

// =============================================================================
// H16: Decode Attention MFU Shape Mismatch
//
// Hypothesis: In heterogeneous decode batches, using maxKVLen for the attention
// MFU lookup while using per-request actual KV lengths for FLOPs systematically
// underestimates decode attention time, because the MFU at maxKVLen is higher
// than the effective per-request MFU at shorter KV lengths.
//
// Background: After the H28 fix, the roofline model correctly sums per-request
// attention FLOPs with actual KV lengths. However, it still uses a SINGLE MFU
// lookup at (totalBatchSize, maxKVLen) for the entire batch:
//
//   attnMFU := mfuDB.GetAttnDecodeMFU(totalBatchSize, maxKVLen, tp)
//   attnCoreTimeS := attnCoreFLOPs / (peakFlops * attnMFU) * tpScaling
//
// If MFU is monotonically increasing with KV length (because longer sequences
// have higher arithmetic intensity and better GPU utilization), then the MFU
// at maxKVLen is HIGHER than the MFU at shorter KV lengths. This means:
//   - The denominator (peakFlops * attnMFU) is too large
//   - The computed attention time is too small (underestimation)
//
// The alternative approach computes attention time per-request, each with its
// own MFU at (1, req.kvLen), and sums the individual times. This would give
// a larger total time because short-KV requests have lower MFU (lower
// efficiency) and thus take disproportionately longer per FLOP.
//
// Refuted if: The decode attention time computed with the single maxKVLen MFU
// lookup is within 5% of the per-request MFU-weighted attention time for
// heterogeneous batches spanning a 10x+ KV length range.
//
// Independent variable: batch composition (KV length heterogeneity)
// Controlled variables: model config, hardware config, TP=1
// Dependent variable: ratio of current method / per-request method attention time
// =============================================================================

// h16BatchSpec defines a heterogeneous decode batch for the experiment.
type h16BatchSpec struct {
	name   string
	kvLens []int64
}

// h16Scenarios returns the decode batch compositions for the experiment.
// These span a range of KV heterogeneity from mild (2x range) to extreme (64x range).
func h16Scenarios() []h16BatchSpec {
	return []h16BatchSpec{
		// --- Homogeneous baselines (expect ratio ~1.0) ---
		{"homo_4x128", []int64{128, 128, 128, 128}},
		{"homo_4x1024", []int64{1024, 1024, 1024, 1024}},
		{"homo_4x4096", []int64{4096, 4096, 4096, 4096}},

		// --- Mild heterogeneity (2-4x range) ---
		{"mild_2x_range", []int64{512, 512, 1024, 1024}},
		{"mild_4x_range", []int64{256, 512, 512, 1024}},

		// --- Moderate heterogeneity (8-10x range) ---
		{"mod_8x_range", []int64{128, 256, 512, 1024}},
		{"mod_10x_short_heavy", []int64{128, 128, 128, 1024}},
		{"mod_10x_long_heavy", []int64{128, 1024, 1024, 1024}},

		// --- High heterogeneity (32x range) ---
		{"high_32x_4req", []int64{128, 128, 128, 4096}},
		{"high_32x_5req", []int64{256, 512, 1024, 4096, 8192}},

		// --- Extreme heterogeneity (64x range) ---
		{"extreme_64x_4req", []int64{128, 128, 128, 8192}},
		{"extreme_64x_8req", []int64{128, 128, 256, 256, 512, 1024, 4096, 8192}},
		{"extreme_64x_16req", []int64{
			128, 128, 128, 128, 256, 256, 256, 512,
			512, 1024, 1024, 2048, 2048, 4096, 4096, 8192,
		}},

		// --- Pathological: one long request + many short ---
		{"pathological_1long_7short", []int64{8192, 128, 128, 128, 128, 128, 128, 128}},
		{"pathological_1long_15short", []int64{
			8192, 128, 128, 128, 128, 128, 128, 128,
			128, 128, 128, 128, 128, 128, 128, 128,
		}},
	}
}

// TestH16_MFUShapeMismatch is the core experiment. For each batch, it computes
// the decode attention time two ways:
//
//	(a) Current method: sum per-request FLOPs, single MFU at (totalBS, maxKVLen)
//	(b) Per-request method: each request gets its own MFU at (1, req.kvLen),
//	    compute per-request time, sum individual times
//
// Reports the ratio (current / per-request) for each batch.
func TestH16_MFUShapeMismatch(t *testing.T) {
	mfuDB := loadTestMFUDatabase(t)
	mc := testModelConfig()
	hc := testHardwareCalib()

	peakFlops := hc.TFlopsPeak * 1e12
	tp := 1
	tpScaling := 1.0 / float64(tp)

	scenarios := h16Scenarios()

	// Print structured output for analyze.py
	fmt.Println("H16_RESULTS_START")
	fmt.Printf("%-30s | %5s | %8s | %8s | %6s | %14s | %14s | %14s | %10s\n",
		"scenario", "bs", "maxKV", "meanKV", "range",
		"currentTimeS", "perReqTimeS", "ratio",
		"mfuMaxKV")
	fmt.Println("---")

	var totalScenarios, withinThreshold int

	for _, sc := range scenarios {
		bs := len(sc.kvLens)

		// Compute maxKVLen and meanKVLen
		var maxKV int64
		var sumKV int64
		for _, kv := range sc.kvLens {
			if kv > maxKV {
				maxKV = kv
			}
			sumKV += kv
		}
		meanKV := float64(sumKV) / float64(bs)

		// Find actual min for range computation
		minKV := sc.kvLens[0]
		for _, kv := range sc.kvLens {
			if kv < minKV {
				minKV = kv
			}
		}
		kvRange := float64(maxKV) / float64(minKV)

		// === Method A: Current method ===
		// Sum per-request attention FLOPs with actual KV lengths
		var attnCoreFLOPs float64
		for _, kv := range sc.kvLens {
			attnCoreFLOPs += calculateAttentionCoreFLOPs(
				mc.NumHeads, mc.NumKVHeads, mc.HiddenDim, 1, kv,
			) * float64(mc.NumLayers)
		}
		// Single MFU lookup at (totalBS, maxKVLen)
		attnMFU := mfuDB.GetAttnDecodeMFU(bs, int(maxKV), tp)
		currentTimeS := attnCoreFLOPs / (peakFlops * attnMFU) * tpScaling

		// === Method B: Per-request method ===
		// Each request gets its own MFU at (1, req.kvLen), compute time, sum
		var perReqTimeS float64
		for _, kv := range sc.kvLens {
			reqFLOPs := calculateAttentionCoreFLOPs(
				mc.NumHeads, mc.NumKVHeads, mc.HiddenDim, 1, kv,
			) * float64(mc.NumLayers)
			reqMFU := mfuDB.GetAttnDecodeMFU(1, int(kv), tp)
			reqTimeS := reqFLOPs / (peakFlops * reqMFU) * tpScaling
			perReqTimeS += reqTimeS
		}

		ratio := currentTimeS / perReqTimeS

		totalScenarios++
		if math.Abs(ratio-1.0) < 0.05 {
			withinThreshold++
		}

		fmt.Printf("%-30s | %5d | %8d | %8.0f | %5.0fx | %14.9f | %14.9f | %14.6f | %10.6f\n",
			sc.name, bs, maxKV, meanKV, kvRange,
			currentTimeS, perReqTimeS, ratio, attnMFU)

		// Log for test output parsing
		t.Logf("H16 scenario=%s bs=%d maxKV=%d meanKV=%.0f range=%.0fx currentTime=%.9f perReqTime=%.9f ratio=%.6f",
			sc.name, bs, maxKV, meanKV, kvRange, currentTimeS, perReqTimeS, ratio)
	}

	fmt.Println("H16_RESULTS_END")

	// Summary
	fmt.Println()
	fmt.Printf("H16_SUMMARY: %d/%d scenarios within 5%% threshold (ratio in [0.95, 1.05])\n",
		withinThreshold, totalScenarios)
}

// TestH16_MFUMonotonicity verifies the underlying assumption of the hypothesis:
// that MFU is monotonically increasing with KV length for decode attention.
// If MFU is NOT monotone, the direction of the bias is uncertain.
func TestH16_MFUMonotonicity(t *testing.T) {
	mfuDB := loadTestMFUDatabase(t)
	tp := 1

	kvLens := []int{64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}
	batchSizes := []int{1, 4, 8, 16, 32}

	fmt.Println("H16_MFU_MONOTONICITY_START")
	fmt.Printf("%-8s", "KVLen")
	for _, bs := range batchSizes {
		fmt.Printf(" | BS=%-8d", bs)
	}
	fmt.Println()
	fmt.Println("---")

	for _, kv := range kvLens {
		fmt.Printf("%-8d", kv)
		for _, bs := range batchSizes {
			mfu := mfuDB.GetAttnDecodeMFU(bs, kv, tp)
			fmt.Printf(" | %-11.6f", mfu)
		}
		fmt.Println()
	}
	fmt.Println("H16_MFU_MONOTONICITY_END")

	// Check monotonicity along KV axis for each batch size
	fmt.Println()
	fmt.Println("H16_MONOTONICITY_CHECK:")
	var totalChecks, violations int
	for _, bs := range batchSizes {
		prevMFU := 0.0
		prevKV := 0
		bsViolations := 0
		for _, kv := range kvLens {
			mfu := mfuDB.GetAttnDecodeMFU(bs, kv, tp)
			if prevKV > 0 && mfu < prevMFU-1e-9 {
				bsViolations++
				t.Logf("MFU monotonicity violation: bs=%d, kv=%d (mfu=%.6f) < kv=%d (mfu=%.6f)",
					bs, kv, mfu, prevKV, prevMFU)
			}
			if prevKV > 0 {
				totalChecks++
			}
			prevMFU = mfu
			prevKV = kv
		}
		violations += bsViolations
		status := "MONOTONE"
		if bsViolations > 0 {
			status = fmt.Sprintf("NON-MONOTONE (%d violations)", bsViolations)
		}
		fmt.Printf("  BS=%-4d: %s\n", bs, status)
	}
	fmt.Printf("  Total: %d violations / %d checks\n", violations, totalChecks)
}

// TestH16_PerRequestMFUProfile shows the MFU value at each request's KV length
// and how it compares to the batch-level maxKVLen MFU. This directly illustrates
// the mismatch: short-KV requests have lower MFU, meaning they are less
// efficient and take more time per FLOP than the batch-level MFU implies.
func TestH16_PerRequestMFUProfile(t *testing.T) {
	mfuDB := loadTestMFUDatabase(t)
	tp := 1

	// Use a few representative batches
	batches := []h16BatchSpec{
		{"mild_2x", []int64{512, 512, 1024, 1024}},
		{"high_32x", []int64{128, 128, 128, 4096}},
		{"extreme_64x", []int64{128, 128, 256, 256, 512, 1024, 4096, 8192}},
		{"pathological", []int64{8192, 128, 128, 128, 128, 128, 128, 128}},
	}

	fmt.Println("H16_MFU_PROFILE_START")

	for _, batch := range batches {
		bs := len(batch.kvLens)
		var maxKV int64
		for _, kv := range batch.kvLens {
			if kv > maxKV {
				maxKV = kv
			}
		}

		batchMFU := mfuDB.GetAttnDecodeMFU(bs, int(maxKV), tp)

		fmt.Printf("\n--- %s (bs=%d, maxKV=%d, batchMFU=%.6f) ---\n",
			batch.name, bs, maxKV, batchMFU)
		fmt.Printf("%-6s | %8s | %10s | %10s | %10s\n",
			"ReqIdx", "KVLen", "PerReqMFU", "BatchMFU", "MFURatio")

		for i, kv := range batch.kvLens {
			perReqMFU := mfuDB.GetAttnDecodeMFU(1, int(kv), tp)
			mfuRatio := perReqMFU / batchMFU

			fmt.Printf("%-6d | %8d | %10.6f | %10.6f | %10.4f\n",
				i, kv, perReqMFU, batchMFU, mfuRatio)
		}
	}

	fmt.Println("H16_MFU_PROFILE_END")
}

// TestH16_StepTimeImpact computes the full decode step time (not just attention)
// using both methods, to show whether the MFU shape mismatch is material in the
// context of GEMM costs, memory bandwidth, and CPU overhead.
func TestH16_StepTimeImpact(t *testing.T) {
	mfuDB := loadTestMFUDatabase(t)
	mc := testModelConfig()
	hc := testHardwareCalib()
	tp := 1

	scenarios := h16Scenarios()

	fmt.Println("H16_STEP_TIME_START")
	fmt.Printf("%-30s | %5s | %8s | %14s | %14s | %10s | %12s\n",
		"scenario", "bs", "maxKV",
		"baselineStepUS", "adjustedStepUS", "stepRatio", "attnFraction%")
	fmt.Println("---")

	peakFlops := hc.TFlopsPeak * 1e12
	peakBW := hc.BwPeakTBs * 1e12
	if hc.BwEfficiencyFactor != 0 {
		peakBW *= hc.BwEfficiencyFactor
	}
	tpScaling := 1.0 / float64(tp)

	for _, sc := range scenarios {
		bs := len(sc.kvLens)

		var maxKV int64
		for _, kv := range sc.kvLens {
			if kv > maxKV {
				maxKV = kv
			}
		}

		// Build decode requests
		decodeReqs := make([]DecodeRequestConfig, bs)
		for i, kv := range sc.kvLens {
			decodeReqs[i] = DecodeRequestConfig{
				ProgressIndex:      kv,
				NumNewDecodeTokens: 1,
			}
		}

		// === Baseline: current roofline (single MFU at maxKVLen) ===
		step := StepConfig{DecodeRequests: decodeReqs}
		baselineUS := rooflineStepTime("", mc, hc, step, tp, mfuDB)

		// === Adjusted: per-request MFU attention time ===
		// Recompute attention time with per-request MFU, keeping everything else the same.

		// GEMM time (same for both methods)
		gemmTimeS := computeTransformerGEMMTimes(mc, bs, peakFlops, peakBW, mfuDB, tpScaling)

		// Per-request attention time with individual MFU lookups
		var adjustedAttnTimeS float64
		for _, kv := range sc.kvLens {
			reqFLOPs := calculateAttentionCoreFLOPs(
				mc.NumHeads, mc.NumKVHeads, mc.HiddenDim, 1, kv,
			) * float64(mc.NumLayers)
			reqMFU := mfuDB.GetAttnDecodeMFU(1, int(kv), tp)
			adjustedAttnTimeS += reqFLOPs / (peakFlops * reqMFU) * tpScaling
		}

		adjustedComputeS := gemmTimeS + adjustedAttnTimeS

		// Memory time (same for both methods)
		var dDynamicBytes float64
		for _, req := range decodeReqs {
			m := calculateMemoryAccessBytes(mc, req.ProgressIndex, 1, true)
			dDynamicBytes += (m["total"] - m["model_weights"]) * tpScaling
		}
		baseMem := calculateMemoryAccessBytes(mc, 0, 0, false)
		dWeightBytes := baseMem["model_weights"] * tpScaling
		memTimeS := (dWeightBytes + dDynamicBytes) / peakBW

		// CPU overhead
		overheadS := (hc.PerLayerCPUOverhead * float64(mc.NumLayers) / float64(tp)) / 1e6

		// Adjusted step time = max(adjustedCompute, memory) + overhead
		adjustedStepS := math.Max(adjustedComputeS, memTimeS) + overheadS
		adjustedStepUS := int64(math.Round(adjustedStepS * 1e6))

		stepRatio := float64(baselineUS) / float64(adjustedStepUS)

		// Compute attention fraction of baseline step time
		baselineStepS := float64(baselineUS) / 1e6
		baselineHardwareS := baselineStepS - overheadS
		// Current method attention time
		var attnCoreFLOPs float64
		for _, kv := range sc.kvLens {
			attnCoreFLOPs += calculateAttentionCoreFLOPs(
				mc.NumHeads, mc.NumKVHeads, mc.HiddenDim, 1, kv,
			) * float64(mc.NumLayers)
		}
		attnMFU := mfuDB.GetAttnDecodeMFU(bs, int(maxKV), tp)
		currentAttnTimeS := attnCoreFLOPs / (peakFlops * attnMFU) * tpScaling
		attnFraction := currentAttnTimeS / baselineHardwareS * 100.0
		if attnFraction > 100.0 {
			attnFraction = 100.0 // cap for display
		}

		fmt.Printf("%-30s | %5d | %8d | %14d | %14d | %10.6f | %11.2f%%\n",
			sc.name, bs, maxKV, baselineUS, adjustedStepUS, stepRatio, attnFraction)

		t.Logf("H16_STEP scenario=%s baseline_us=%d adjusted_us=%d ratio=%.6f attn_frac=%.2f%%",
			sc.name, baselineUS, adjustedStepUS, stepRatio, attnFraction)
	}

	fmt.Println("H16_STEP_TIME_END")
}

// TestH16_KVRangeVsRatio sweeps KV range factor systematically to show the
// relationship between heterogeneity and mismatch ratio. Uses a fixed batch
// structure (1 long-KV anchor + 3 short-KV requests) and varies the ratio.
func TestH16_KVRangeVsRatio(t *testing.T) {
	mfuDB := loadTestMFUDatabase(t)
	mc := testModelConfig()
	hc := testHardwareCalib()

	peakFlops := hc.TFlopsPeak * 1e12
	tp := 1
	tpScaling := 1.0 / float64(tp)

	// Fixed anchor KV length, vary short KV length
	anchorKV := int64(8192)
	shortKVs := []int64{8192, 4096, 2048, 1024, 512, 256, 128}

	fmt.Println("H16_RANGE_SWEEP_START")
	fmt.Printf("%-10s | %8s | %8s | %6s | %14s | %14s | %14s\n",
		"shortKV", "maxKV", "meanKV", "range",
		"currentTimeS", "perReqTimeS", "ratio")
	fmt.Println("---")

	for _, shortKV := range shortKVs {
		kvLens := []int64{anchorKV, shortKV, shortKV, shortKV}
		bs := len(kvLens)

		meanKV := float64(anchorKV+3*shortKV) / 4.0
		kvRange := float64(anchorKV) / float64(shortKV)

		// Method A: current (single MFU at maxKVLen)
		var attnCoreFLOPs float64
		for _, kv := range kvLens {
			attnCoreFLOPs += calculateAttentionCoreFLOPs(
				mc.NumHeads, mc.NumKVHeads, mc.HiddenDim, 1, kv,
			) * float64(mc.NumLayers)
		}
		attnMFU := mfuDB.GetAttnDecodeMFU(bs, int(anchorKV), tp)
		currentTimeS := attnCoreFLOPs / (peakFlops * attnMFU) * tpScaling

		// Method B: per-request MFU
		var perReqTimeS float64
		for _, kv := range kvLens {
			reqFLOPs := calculateAttentionCoreFLOPs(
				mc.NumHeads, mc.NumKVHeads, mc.HiddenDim, 1, kv,
			) * float64(mc.NumLayers)
			reqMFU := mfuDB.GetAttnDecodeMFU(1, int(kv), tp)
			reqTimeS := reqFLOPs / (peakFlops * reqMFU) * tpScaling
			perReqTimeS += reqTimeS
		}

		ratio := currentTimeS / perReqTimeS

		fmt.Printf("%-10d | %8d | %8.0f | %5.0fx | %14.9f | %14.9f | %14.6f\n",
			shortKV, anchorKV, meanKV, kvRange,
			currentTimeS, perReqTimeS, ratio)
	}

	fmt.Println("H16_RANGE_SWEEP_END")
}
