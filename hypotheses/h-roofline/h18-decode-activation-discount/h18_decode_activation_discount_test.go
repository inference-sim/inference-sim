//go:build ignore

package latency

import (
	"fmt"
	"math"
	"testing"
)

// =============================================================================
// H18: Decode Activation Memory Factor Is Inconsequential
//
// Hypothesis: The decode activation memory factor (0.75) is inconsequential
// because activation bytes constitute less than 0.5% of total memory traffic
// across all evaluation operating points (bs=1..256, kvLen=128..8192), so
// replacing 0.75 with any value in [0.5, 1.5] changes predicted step time
// by less than 0.05%.
//
// Refuted if: There exists an operating point where changing the decode
// activation factor from 0.75 to 1.00 shifts predicted decode step time
// by more than 0.1%.
//
// The 0.75 factor is hardcoded in calculateMemoryAccessBytes (roofline_step.go:99):
//   activationBytes = nLayers * dModel * BytesPerParam * newT * 0.75
//
// This test cannot modify the hardcoded constant, so it reverse-engineers
// the activation_tokens bytes at 0.75 and scales to other factors analytically.
// =============================================================================

// TestH18_DecodeActivationDiscountNegligible validates the hypothesis that the
// decode activation memory factor (0.75) has negligible impact on predicted
// step time across all evaluation operating points.
//
// Experiment phases:
//  1. Activation fraction analysis: what fraction of total dynamic memory
//     bytes is activation_tokens at each (bs, kvLen)?
//  2. Full roofline step time comparison: step time at factor=0.75 vs 1.00
//  3. Multi-factor sweep: step time at factors [0.50, 0.75, 1.00, 1.50]
//  4. Max delta across all grid points
func TestH18_DecodeActivationDiscountNegligible(t *testing.T) {
	mc := testModelConfig() // Llama-3.1-8B-like
	hc := testHardwareCalib()

	// Grid of operating points covering the eval suite
	batchSizes := []int{1, 4, 8, 16, 32, 64, 128, 256}
	kvLens := []int64{128, 256, 512, 1024, 2048, 4096, 8192}

	// Activation discount factors to test
	activationFactors := []float64{0.50, 0.75, 1.00, 1.50}

	// ========================================================================
	// Phase 1: Activation Fraction Analysis
	// ========================================================================
	// Show what fraction of total decode dynamic memory bytes is activations.
	fmt.Println("H18_ACTIVATION_FRACTION_START")
	fmt.Printf("%-6s | %-8s | %14s | %14s | %14s | %10s\n",
		"bs", "kvLen", "activ_bytes", "dynamic_bytes", "total_bytes", "actFrac%")
	fmt.Println("---")

	var maxActivFraction float64

	for _, bs := range batchSizes {
		for _, kvLen := range kvLens {
			// Sum per-request activation and dynamic bytes for the batch
			var totalActivBytes, totalDynamicBytes float64
			for i := 0; i < bs; i++ {
				m := calculateMemoryAccessBytes(mc, kvLen, 1, true)
				totalActivBytes += m["activations_tokens"]
				totalDynamicBytes += m["total"] - m["model_weights"]
			}

			baseMem := calculateMemoryAccessBytes(mc, 0, 0, false)
			weightBytes := baseMem["model_weights"]
			totalBytes := weightBytes + totalDynamicBytes

			activFraction := totalActivBytes / totalDynamicBytes * 100.0
			if activFraction > maxActivFraction {
				maxActivFraction = activFraction
			}

			fmt.Printf("%-6d | %-8d | %14.0f | %14.0f | %14.0f | %9.4f%%\n",
				bs, kvLen, totalActivBytes, totalDynamicBytes, totalBytes, activFraction)
		}
	}
	fmt.Println("H18_ACTIVATION_FRACTION_END")
	fmt.Printf("H18_MAX_ACTIVATION_FRACTION=%.6f%%\n", maxActivFraction)

	// ========================================================================
	// Phase 2: Full Roofline Step Time Comparison (0.75 vs 1.00)
	// ========================================================================
	// Requires MFU database -- skip if bench_data not available
	mfuDB := loadTestMFUDatabase(t)

	fmt.Println()
	fmt.Println("H18_STEPTIME_START")
	fmt.Printf("%-6s | %-8s | %12s | %12s | %12s | %12s | %12s | %10s | %6s\n",
		"bs", "kvLen",
		"computeS", "memS_075", "memS_100",
		"step_075", "step_100",
		"stepDelta%", "regime")
	fmt.Println("---")

	tp := 1
	var maxStepDelta float64
	var maxStepDeltaBS int
	var maxStepDeltaKV int64

	for _, bs := range batchSizes {
		for _, kvLen := range kvLens {
			// Build decode requests: uniform batch at this (bs, kvLen) point
			decodeReqs := make([]DecodeRequestConfig, bs)
			for i := 0; i < bs; i++ {
				decodeReqs[i] = DecodeRequestConfig{
					ProgressIndex:      kvLen,
					NumNewDecodeTokens: 1,
				}
			}

			step := StepConfig{DecodeRequests: decodeReqs}

			// Get baseline step time at hardcoded factor=0.75
			stepTime075 := rooflineStepTime("", mc, hc, step, tp, mfuDB)

			// Compute memory times analytically for factor scaling
			tpScaling := 1.0 / float64(tp)
			peakBWLocal := hc.BwPeakTBs * 1e12
			if hc.BwEfficiencyFactor != 0 {
				peakBWLocal *= hc.BwEfficiencyFactor
			}

			// Accumulate per-request dynamic bytes and activation bytes at 0.75
			var dynamicBytes075, activBytes075 float64
			for _, req := range decodeReqs {
				m := calculateMemoryAccessBytes(mc, req.ProgressIndex, 1, true)
				dynamicBytes075 += (m["total"] - m["model_weights"]) * tpScaling
				activBytes075 += m["activations_tokens"] * tpScaling
			}

			baseMem := calculateMemoryAccessBytes(mc, 0, 0, false)
			weightBytes := baseMem["model_weights"] * tpScaling

			memTime075 := (weightBytes + dynamicBytes075) / peakBWLocal

			// Adjust for factor=1.00:
			// activations_tokens at 0.75 = nLayers * dModel * BytesPerParam * 1 * 0.75
			// activations_tokens at 1.00 = nLayers * dModel * BytesPerParam * 1 * 1.00
			// Scale: activBytes_at_X = activBytes_at_075 * (X / 0.75)
			activBytes100 := activBytes075 * (1.00 / 0.75)
			adjustedDynamic := dynamicBytes075 - activBytes075 + activBytes100
			memTime100 := (weightBytes + adjustedDynamic) / peakBWLocal

			// Derive compute time from step time
			overheadS := hc.PerLayerCPUOverhead * float64(mc.NumLayers) / float64(tp) / 1e6
			stepTime075_s := float64(stepTime075) / 1e6
			hardwareTime075 := stepTime075_s - overheadS

			isMemBound := hardwareTime075 <= memTime075+1e-9

			var computeS float64
			if isMemBound {
				computeS = hardwareTime075 // memory-bound: computeS <= memTime075
			} else {
				computeS = hardwareTime075 // compute-bound: computeS = hardwareTime075
			}

			// Estimate step time at factor=1.00
			var stepTime100_s float64
			if isMemBound {
				// Memory-bound: step time scales with memory time
				stepTime100_s = memTime100 + overheadS
			} else {
				// Compute-bound: step time unchanged unless memTime100 > computeS
				stepTime100_s = math.Max(computeS, memTime100) + overheadS
			}

			stepTime100 := int64(math.Round(stepTime100_s * 1e6))
			stepDelta := float64(stepTime100-stepTime075) / float64(stepTime075) * 100.0

			if math.Abs(stepDelta) > math.Abs(maxStepDelta) {
				maxStepDelta = stepDelta
				maxStepDeltaBS = bs
				maxStepDeltaKV = kvLen
			}

			regime := "MEM"
			if !isMemBound {
				regime = "CMP"
			}

			fmt.Printf("%-6d | %-8d | %12.6f | %12.6f | %12.6f | %12d | %12d | %9.4f%% | %6s\n",
				bs, kvLen,
				computeS, memTime075, memTime100,
				stepTime075, stepTime100,
				stepDelta, regime)
		}
	}
	fmt.Println("H18_STEPTIME_END")
	fmt.Printf("H18_MAX_STEP_DELTA_075_VS_100=%.6f%%\n", maxStepDelta)
	fmt.Printf("H18_MAX_STEP_DELTA_BS=%d\n", maxStepDeltaBS)
	fmt.Printf("H18_MAX_STEP_DELTA_KV=%d\n", maxStepDeltaKV)

	// ========================================================================
	// Phase 3: Multi-Factor Sweep
	// ========================================================================
	// Show step time at all 4 activation factors for representative operating points.
	fmt.Println()
	fmt.Println("H18_FACTOR_SWEEP_START")
	fmt.Printf("%-6s | %-8s | %8s | %12s | %12s | %10s\n",
		"bs", "kvLen", "factor", "memTimeS", "estStepTime", "deltaVs075%")
	fmt.Println("---")

	// Sweep across all grid points and all factors
	var globalMaxDelta float64

	for _, bs := range batchSizes {
		for _, kvLen := range kvLens {
			decodeReqs := make([]DecodeRequestConfig, bs)
			for i := 0; i < bs; i++ {
				decodeReqs[i] = DecodeRequestConfig{
					ProgressIndex:      kvLen,
					NumNewDecodeTokens: 1,
				}
			}

			step := StepConfig{DecodeRequests: decodeReqs}
			stepTime075 := rooflineStepTime("", mc, hc, step, tp, mfuDB)

			// Compute per-request activation and dynamic bytes at 0.75
			var activBytes075, dynamicBytes075 float64
			for _, req := range decodeReqs {
				m := calculateMemoryAccessBytes(mc, req.ProgressIndex, 1, true)
				activBytes075 += m["activations_tokens"] * (1.0 / float64(tp))
				dynamicBytes075 += (m["total"] - m["model_weights"]) * (1.0 / float64(tp))
			}
			baseMem := calculateMemoryAccessBytes(mc, 0, 0, false)
			wBytes := baseMem["model_weights"] * (1.0 / float64(tp))

			peakBWLocal := hc.BwPeakTBs * 1e12
			if hc.BwEfficiencyFactor != 0 {
				peakBWLocal *= hc.BwEfficiencyFactor
			}

			overheadS := hc.PerLayerCPUOverhead * float64(mc.NumLayers) / float64(tp) / 1e6
			hardwareTime075 := float64(stepTime075)/1e6 - overheadS
			memTime075 := (wBytes + dynamicBytes075) / peakBWLocal
			isMemBound := hardwareTime075 <= memTime075+1e-9

			for _, factor := range activationFactors {
				activBytesF := activBytes075 * (factor / 0.75)
				adjustedDynamic := dynamicBytes075 - activBytes075 + activBytesF
				memTimeF := (wBytes + adjustedDynamic) / peakBWLocal

				var estStepTimeS float64
				if isMemBound {
					estStepTimeS = memTimeF + overheadS
				} else {
					estStepTimeS = math.Max(hardwareTime075, memTimeF) + overheadS
				}
				estStepTimeUs := int64(math.Round(estStepTimeS * 1e6))
				deltaVs075 := float64(estStepTimeUs-stepTime075) / float64(stepTime075) * 100.0

				if math.Abs(deltaVs075) > math.Abs(globalMaxDelta) {
					globalMaxDelta = deltaVs075
				}

				fmt.Printf("%-6d | %-8d | %8.2f | %12.6f | %12d | %9.4f%%\n",
					bs, kvLen, factor, memTimeF, estStepTimeUs, deltaVs075)
			}
		}
	}
	fmt.Println("H18_FACTOR_SWEEP_END")
	fmt.Printf("H18_GLOBAL_MAX_DELTA=%.6f%%\n", globalMaxDelta)

	// ========================================================================
	// Phase 4: Summary and Verdict
	// ========================================================================
	fmt.Println()
	fmt.Println("H18_VERDICT_START")
	fmt.Printf("max_activation_fraction_of_dynamic=%.6f%%\n", maxActivFraction)
	fmt.Printf("max_step_delta_075_vs_100=%.6f%%\n", maxStepDelta)
	fmt.Printf("max_step_delta_075_vs_100_bs=%d\n", maxStepDeltaBS)
	fmt.Printf("max_step_delta_075_vs_100_kv=%d\n", maxStepDeltaKV)
	fmt.Printf("global_max_delta_any_factor=%.6f%%\n", globalMaxDelta)

	if math.Abs(maxStepDelta) < 0.1 {
		fmt.Println("verdict=CONFIRMED")
		fmt.Println("reason=changing activation factor from 0.75 to 1.00 shifts step time by less than 0.1% at all operating points")
	} else {
		fmt.Println("verdict=REFUTED")
		fmt.Printf("reason=activation factor change shifts step time by %.4f%% at bs=%d kvLen=%d, exceeding 0.1%% threshold\n",
			maxStepDelta, maxStepDeltaBS, maxStepDeltaKV)
	}
	fmt.Println("H18_VERDICT_END")

	// ========================================================================
	// Invariants
	// ========================================================================

	// Invariant 1: Increasing activation factor must monotonically increase
	// total memory bytes (activation is always additive to total).
	for _, bs := range []int{1, 16, 64, 256} {
		for _, kvLen := range []int64{128, 2048, 8192} {
			m := calculateMemoryAccessBytes(mc, kvLen, 1, true)
			activBase := m["activations_tokens"] * float64(bs)

			var prevMemBytes float64
			for i, factor := range activationFactors {
				activF := activBase * (factor / 0.75)
				dynamicPerReq := m["total"] - m["model_weights"]
				totalDynamic := (dynamicPerReq*float64(bs) - activBase + activF)
				if i > 0 && totalDynamic < prevMemBytes-1e-6 {
					t.Errorf("monotonicity violation: bs=%d kvLen=%d factor=%.2f memBytes (%.0f) < factor=%.2f memBytes (%.0f)",
						bs, kvLen, factor, totalDynamic, activationFactors[i-1], prevMemBytes)
				}
				prevMemBytes = totalDynamic
			}
		}
	}

	// Invariant 2: At factor=0.75 (baseline), the analytical reconstruction
	// must match the real roofline step time exactly (no drift from reverse-engineering).
	for _, bs := range []int{1, 8, 64} {
		for _, kvLen := range []int64{512, 2048} {
			decodeReqs := make([]DecodeRequestConfig, bs)
			for i := 0; i < bs; i++ {
				decodeReqs[i] = DecodeRequestConfig{
					ProgressIndex:      kvLen,
					NumNewDecodeTokens: 1,
				}
			}
			step := StepConfig{DecodeRequests: decodeReqs}
			actual := rooflineStepTime("", mc, hc, step, tp, mfuDB)

			// Reconstruct at factor=0.75 (should be identity)
			var activBytes075, dynamicBytes075 float64
			for _, req := range decodeReqs {
				mLocal := calculateMemoryAccessBytes(mc, req.ProgressIndex, 1, true)
				activBytes075 += mLocal["activations_tokens"] * (1.0 / float64(tp))
				dynamicBytes075 += (mLocal["total"] - mLocal["model_weights"]) * (1.0 / float64(tp))
			}
			baseMemLocal := calculateMemoryAccessBytes(mc, 0, 0, false)
			wBytesLocal := baseMemLocal["model_weights"] * (1.0 / float64(tp))

			peakBWLocal := hc.BwPeakTBs * 1e12
			if hc.BwEfficiencyFactor != 0 {
				peakBWLocal *= hc.BwEfficiencyFactor
			}

			// At factor=0.75, activBytes stays the same, so reconstruction = identity
			activBytesRecon := activBytes075 * 1.0 // identity: factor cancels
			adjustedDynamic := dynamicBytes075 - activBytes075 + activBytesRecon
			memTimeRecon := (wBytesLocal + adjustedDynamic) / peakBWLocal

			overheadSLocal := hc.PerLayerCPUOverhead * float64(mc.NumLayers) / float64(tp) / 1e6
			hardwareTime := float64(actual)/1e6 - overheadSLocal
			isMemBound := hardwareTime <= memTimeRecon+1e-9

			var estStepTimeS float64
			if isMemBound {
				estStepTimeS = memTimeRecon + overheadSLocal
			} else {
				estStepTimeS = math.Max(hardwareTime, memTimeRecon) + overheadSLocal
			}
			estStepTimeUs := int64(math.Round(estStepTimeS * 1e6))

			if estStepTimeUs != actual {
				t.Errorf("reconstruction drift at bs=%d kvLen=%d: actual=%d, reconstructed=%d (diff=%d us)",
					bs, kvLen, actual, estStepTimeUs, estStepTimeUs-actual)
			}
		}
	}
}
