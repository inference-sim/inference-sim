package sim

import (
	"fmt"
	"math"
	"testing"
)

// =============================================================================
// H34: Decode GEMM Time Underestimate Below MFU Grid Minimum (M < 8)
//
// Hypothesis: For decode steps at batch sizes 1-7 (below the GEMM MFU grid
// minimum M=8), computeTransformerGEMMTimes predicts GEMM time proportional
// to batch size (ratio bs/8 relative to bs=8), whereas actual GPU GEMM
// kernels at these small M values exhibit a near-constant memory-bound latency
// floor, causing up to 8x underestimate at bs=1.
//
// The mechanism: GetGEMMmfu clamps m <= mPoints[0].m to the first grid point's
// MFU value. For all bs in [1..8], the MFU is identical (clamped to the grid
// minimum, typically M=8). Since GEMM time = 2*m*k*n / (peakFlops * mfu),
// the time is strictly proportional to m (batch size), giving ratio bs/8.
//
// In reality, small-M GEMM kernels are memory-bandwidth bound (loading the
// full weight matrix regardless of M), so actual latency has a near-constant
// floor that doesn't scale down linearly with M.
//
// Refuted if: The predicted ratio computeTransformerGEMMTimes(bs=1) /
// computeTransformerGEMMTimes(bs=8) exceeds 0.25 (i.e., underestimate < 4x).
//
// Independent variable: batch size (1, 2, 4, 8, 16, 32)
// Controlled variables: model config, hardware config, TP=1, tpScaling=1.0
// Dependent variable: GEMM time (seconds), ratio relative to bs=8
// =============================================================================

// TestH34_GEMMTimeScalingAcrossBatchSizes measures how computeTransformerGEMMTimes
// scales across batch sizes, with particular focus on bs < 8 (below MFU grid minimum).
func TestH34_GEMMTimeScalingAcrossBatchSizes(t *testing.T) {
	mfuDB := loadTestMFUDatabase(t)
	hwCalib := testHardwareCalib()
	peakFlops := hwCalib.TFlopsPeak * 1e12

	batchSizes := []int{1, 2, 4, 8, 16, 32}
	tpScaling := 1.0

	fmt.Println("=== H34: GEMM Time Scaling Across Batch Sizes ===")
	fmt.Println()

	for _, ms := range evalSuiteModels() {
		mc := ms.toModelConfig()

		fmt.Printf("--- %s (hidden=%d, layers=%d) ---\n", ms.Name, mc.HiddenDim, mc.NumLayers)
		fmt.Printf("%-10s %14s %14s %10s %12s\n",
			"BatchSize", "GEMMTime(s)", "GEMMTime(us)", "Ratio/bs8", "Expected")
		fmt.Println("-------------------------------------------------------------")

		// Collect GEMM times
		gemmTimes := make(map[int]float64)
		for _, bs := range batchSizes {
			gemmTime := computeTransformerGEMMTimes(mc, bs, peakFlops, mfuDB, tpScaling)
			gemmTimes[bs] = gemmTime
		}

		bs8Time := gemmTimes[8]
		if bs8Time <= 0 {
			t.Errorf("%s: bs=8 GEMM time is zero or negative (%.6e)", ms.Name, bs8Time)
			continue
		}

		for _, bs := range batchSizes {
			gemmTime := gemmTimes[bs]
			ratio := gemmTime / bs8Time
			expected := float64(bs) / 8.0

			fmt.Printf("%-10d %14.6e %14.1f %10.4f %12.4f\n",
				bs, gemmTime, gemmTime*1e6, ratio, expected)

			t.Logf("H34_GEMM_SCALING model=%s bs=%d gemm_time_s=%.6e ratio_vs_bs8=%.6f expected_linear=%.6f",
				ms.Name, bs, gemmTime, ratio, expected)
		}

		// === Key check: ratio at bs=1 vs bs=8 ===
		ratio1 := gemmTimes[1] / bs8Time
		fmt.Printf("\n  KEY RATIO: gemmTime(bs=1) / gemmTime(bs=8) = %.6f\n", ratio1)
		fmt.Printf("  LINEAR EXPECTATION: 1/8 = 0.125000\n")
		fmt.Printf("  REFUTATION THRESHOLD: > 0.25 (underestimate < 4x)\n")

		if ratio1 > 0.25 {
			fmt.Printf("  VERDICT: REFUTED for %s (ratio %.4f > 0.25)\n", ms.Name, ratio1)
		} else {
			fmt.Printf("  VERDICT: CONFIRMED for %s (ratio %.4f <= 0.25, ~%.1fx underestimate at bs=1)\n",
				ms.Name, ratio1, 1.0/ratio1)
		}
		fmt.Println()

		// Emit structured output for analyze.py
		fmt.Printf("H34_KEY_RATIO model=%s ratio_bs1_vs_bs8=%.6f refuted=%t\n",
			ms.Name, ratio1, ratio1 > 0.25)
	}
}

// TestH34_GEMMTimeScaling_Llama8B is a focused test using just the standard
// Llama-3.1-8B config, for faster execution when bench_data is available.
func TestH34_GEMMTimeScaling_Llama8B(t *testing.T) {
	mfuDB := loadTestMFUDatabase(t)
	hwCalib := testHardwareCalib()
	peakFlops := hwCalib.TFlopsPeak * 1e12

	mc := testModelConfig() // Llama-3.1-8B
	batchSizes := []int{1, 2, 4, 8, 16, 32}
	tpScaling := 1.0

	fmt.Println("=== H34: GEMM Time Scaling — Llama-3.1-8B ===")
	fmt.Printf("%-10s %14s %10s %12s %10s\n",
		"BatchSize", "GEMMTime(us)", "Ratio/bs8", "LinearExp", "Delta%")
	fmt.Println("-------------------------------------------------------------------")

	gemmTimes := make(map[int]float64)
	for _, bs := range batchSizes {
		gemmTimes[bs] = computeTransformerGEMMTimes(mc, bs, peakFlops, mfuDB, tpScaling)
	}

	bs8Time := gemmTimes[8]
	if bs8Time <= 0 {
		t.Fatalf("bs=8 GEMM time is zero or negative: %.6e", bs8Time)
	}

	for _, bs := range batchSizes {
		ratio := gemmTimes[bs] / bs8Time
		expected := float64(bs) / 8.0
		deltaPct := 0.0
		if expected > 0 {
			deltaPct = (ratio - expected) / expected * 100.0
		}

		fmt.Printf("%-10d %14.1f %10.4f %12.4f %9.2f%%\n",
			bs, gemmTimes[bs]*1e6, ratio, expected, deltaPct)
	}

	// Refutation check
	ratio1 := gemmTimes[1] / bs8Time
	fmt.Printf("\nH34_REFUTATION_CHECK: ratio_bs1_vs_bs8=%.6f threshold=0.25 refuted=%t\n",
		ratio1, ratio1 > 0.25)

	if math.Abs(ratio1-0.125) < 0.01 {
		t.Logf("CONFIRMED: bs=1 GEMM time is ~1/8 of bs=8 (ratio=%.4f), confirming linear proportionality", ratio1)
	}
}

// TestH34_PerGEMMBreakdown shows the per-projection GEMM time breakdown
// (Q, K, V, O, Gate, Up, Down) for each batch size, demonstrating that
// every individual GEMM has the same scaling behavior.
func TestH34_PerGEMMBreakdown(t *testing.T) {
	mfuDB := loadTestMFUDatabase(t)
	hwCalib := testHardwareCalib()
	peakFlops := hwCalib.TFlopsPeak * 1e12

	mc := testModelConfig() // Llama-3.1-8B
	batchSizes := []int{1, 2, 4, 8, 16, 32}

	nKVHeads := mc.NumKVHeads
	if nKVHeads == 0 {
		nKVHeads = mc.NumHeads
	}
	headDim := mc.HiddenDim / mc.NumHeads
	dKV := nKVHeads * headDim
	dFF := 4 * mc.HiddenDim
	if mc.IntermediateDim > 0 {
		dFF = mc.IntermediateDim
	}

	// Define all GEMM projections: (name, M_is_bs, K, N)
	type gemmSpec struct {
		name string
		k    int
		n    int
	}
	gemms := []gemmSpec{
		{"Q_proj", mc.HiddenDim, mc.HiddenDim},
		{"K_proj", mc.HiddenDim, dKV},
		{"V_proj", mc.HiddenDim, dKV},
		{"O_proj", mc.HiddenDim, mc.HiddenDim},
		{"Gate_proj", mc.HiddenDim, dFF},
		{"Up_proj", mc.HiddenDim, dFF},
		{"Down_proj", dFF, mc.HiddenDim},
	}

	fmt.Println("=== H34: Per-GEMM Breakdown (Single Layer, Llama-3.1-8B) ===")
	fmt.Println()

	// Header
	fmt.Printf("%-12s", "GEMM")
	for _, bs := range batchSizes {
		fmt.Printf(" %10s", fmt.Sprintf("bs=%d(us)", bs))
	}
	fmt.Printf(" %12s\n", "Ratio bs1/8")
	fmt.Println("-----------------------------------------------------------------------------------------------")

	fmt.Println("H34_PER_GEMM_START")

	for _, g := range gemms {
		fmt.Printf("%-12s", g.name)

		times := make(map[int]float64)
		for _, bs := range batchSizes {
			gTime := computeGEMMTime(bs, g.k, g.n, peakFlops, mfuDB)
			times[bs] = gTime
			fmt.Printf(" %10.3f", gTime*1e6)
		}

		ratio := 0.0
		if times[8] > 0 {
			ratio = times[1] / times[8]
		}
		fmt.Printf(" %12.6f\n", ratio)

		t.Logf("H34_PER_GEMM name=%s k=%d n=%d ratio_bs1_vs_bs8=%.6f", g.name, g.k, g.n, ratio)
	}

	fmt.Println("H34_PER_GEMM_END")
	fmt.Println()

	// Also show MFU values for each batch size at a representative GEMM shape
	fmt.Println("=== H34: GEMM MFU Values at Different Batch Sizes ===")
	fmt.Printf("%-12s", "GEMM")
	for _, bs := range batchSizes {
		fmt.Printf(" %10s", fmt.Sprintf("bs=%d", bs))
	}
	fmt.Println()
	fmt.Println("-----------------------------------------------------------------------------------------------")

	fmt.Println("H34_MFU_VALUES_START")
	for _, g := range gemms {
		fmt.Printf("%-12s", g.name)
		for _, bs := range batchSizes {
			mfu := mfuDB.GetGEMMmfu(bs, g.k, g.n)
			fmt.Printf(" %10.6f", mfu)
		}
		fmt.Println()
	}
	fmt.Println("H34_MFU_VALUES_END")
}

// TestH34_MFUClampingMechanism directly verifies that GetGEMMmfu returns the
// same MFU value for all batch sizes below the grid minimum, confirming the
// clamping mechanism that causes the linear proportionality.
func TestH34_MFUClampingMechanism(t *testing.T) {
	mfuDB := loadTestMFUDatabase(t)

	mc := testModelConfig()
	dFF := mc.IntermediateDim

	// Representative GEMM shapes from the transformer
	type gemmShape struct {
		name string
		k    int
		n    int
	}
	shapes := []gemmShape{
		{"Q_proj", mc.HiddenDim, mc.HiddenDim},
		{"Gate_proj", mc.HiddenDim, dFF},
		{"Down_proj", dFF, mc.HiddenDim},
	}

	batchSizes := []int{1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64}

	fmt.Println("=== H34: MFU Clamping Verification ===")
	fmt.Println()
	fmt.Println("If MFU is identical for bs=1..7 but different at bs=8+,")
	fmt.Println("this confirms the sub-grid clamping mechanism.")
	fmt.Println()

	fmt.Println("H34_CLAMPING_START")

	for _, shape := range shapes {
		fmt.Printf("--- %s (k=%d, n=%d) ---\n", shape.name, shape.k, shape.n)
		fmt.Printf("%-10s %12s %10s\n", "BatchSize", "MFU", "SameAsBs1")
		fmt.Println("----------------------------------")

		mfuBs1 := mfuDB.GetGEMMmfu(1, shape.k, shape.n)
		var clampedCount, totalSubgrid int

		for _, bs := range batchSizes {
			mfu := mfuDB.GetGEMMmfu(bs, shape.k, shape.n)
			sameAsBs1 := math.Abs(mfu-mfuBs1) < 1e-12

			marker := ""
			if sameAsBs1 {
				marker = "SAME"
			} else {
				marker = "DIFFERENT"
			}
			fmt.Printf("%-10d %12.6f %10s\n", bs, mfu, marker)

			if bs < 8 {
				totalSubgrid++
				if sameAsBs1 {
					clampedCount++
				}
			}
		}

		fmt.Printf("  Sub-grid (bs<8) clamped: %d / %d\n\n", clampedCount, totalSubgrid)

		if clampedCount == totalSubgrid {
			t.Logf("H34_CLAMPING %s: ALL sub-grid batch sizes clamped to same MFU (%.6f)", shape.name, mfuBs1)
		} else {
			t.Logf("H34_CLAMPING %s: %d/%d sub-grid batch sizes clamped (interpolation present)", shape.name, clampedCount, totalSubgrid)
		}
	}

	fmt.Println("H34_CLAMPING_END")
}

// TestH34_RegimeAnalysis checks whether small-batch decode steps are
// compute-bound or memory-bound under the current roofline model,
// and how the GEMM underestimate affects the regime classification.
func TestH34_RegimeAnalysis(t *testing.T) {
	mfuDB := loadTestMFUDatabase(t)
	hwCalib := testHardwareCalib()
	peakFlops := hwCalib.TFlopsPeak * 1e12
	peakBW := hwCalib.BwPeakTBs * 1e12
	if hwCalib.BwEfficiencyFactor != 0 {
		peakBW *= hwCalib.BwEfficiencyFactor
	}

	mc := testModelConfig()
	batchSizes := []int{1, 2, 4, 8, 16, 32}
	kvLen := int64(1024)
	tp := 1
	tpScaling := 1.0

	fmt.Println("=== H34: Compute vs Memory Regime at Small Batch Sizes ===")
	fmt.Printf("Model: Llama-3.1-8B, KVLen=%d, TP=%d\n\n", kvLen, tp)
	fmt.Printf("%-10s %12s %12s %12s %12s %8s\n",
		"BatchSize", "GEMM(us)", "Attn(us)", "Memory(us)", "Step(us)", "Regime")
	fmt.Println("--------------------------------------------------------------------------")

	fmt.Println("H34_REGIME_START")

	for _, bs := range batchSizes {
		// GEMM compute time
		gemmTimeS := computeTransformerGEMMTimes(mc, bs, peakFlops, mfuDB, tpScaling)

		// Attention compute time
		decodeReqs := make([]DecodeRequestConfig, bs)
		for i := 0; i < bs; i++ {
			decodeReqs[i] = DecodeRequestConfig{
				ProgressIndex:      kvLen,
				NumNewDecodeTokens: 1,
			}
		}

		var attnCoreFLOPs float64
		for _, req := range decodeReqs {
			attnCoreFLOPs += calculateAttentionCoreFLOPs(
				mc.NumHeads, mc.NumKVHeads, mc.HiddenDim, 1, req.ProgressIndex,
			) * float64(mc.NumLayers)
		}
		attnMFU := mfuDB.GetAttnDecodeMFU(bs, int(kvLen), tp)
		attnCoreTimeS := attnCoreFLOPs / (peakFlops * attnMFU) * tpScaling

		decodeComputeS := gemmTimeS + attnCoreTimeS

		// Memory time
		var dDynamicBytes float64
		for _, req := range decodeReqs {
			m := calculateMemoryAccessBytes(mc, req.ProgressIndex, 1, true)
			dDynamicBytes += (m["total"] - m["model_weights"]) * tpScaling
		}
		baseMem := calculateMemoryAccessBytes(mc, 0, 0, false)
		dWeightBytes := baseMem["model_weights"] * tpScaling
		decodeMemoryS := (dWeightBytes + dDynamicBytes) / peakBW

		// Step time via roofline
		step := StepConfig{DecodeRequests: decodeReqs}
		stepTimeUS := rooflineStepTime("", mc, hwCalib, step, tp, mfuDB)

		regime := "MEM"
		if decodeComputeS > decodeMemoryS {
			regime = "COMP"
		}

		fmt.Printf("%-10d %12.1f %12.1f %12.1f %12d %8s\n",
			bs, gemmTimeS*1e6, attnCoreTimeS*1e6, decodeMemoryS*1e6, stepTimeUS, regime)

		t.Logf("H34_REGIME bs=%d gemm_us=%.1f attn_us=%.1f memory_us=%.1f step_us=%d regime=%s",
			bs, gemmTimeS*1e6, attnCoreTimeS*1e6, decodeMemoryS*1e6, stepTimeUS, regime)
	}

	fmt.Println("H34_REGIME_END")
	fmt.Println()

	// Impact analysis: If GEMM at bs=1 had the same time as bs=8 (constant floor),
	// how would that change the step time?
	fmt.Println("=== H34: Impact of Constant GEMM Floor (bs=1 gets bs=8 GEMM time) ===")
	fmt.Printf("%-10s %12s %12s %12s %10s\n",
		"BatchSize", "Current(us)", "Floor(us)", "Delta(us)", "Increase%")
	fmt.Println("--------------------------------------------------------")

	gemmTimeBS8 := computeTransformerGEMMTimes(mc, 8, peakFlops, mfuDB, tpScaling)

	for _, bs := range []int{1, 2, 4} {
		decodeReqs := make([]DecodeRequestConfig, bs)
		for i := 0; i < bs; i++ {
			decodeReqs[i] = DecodeRequestConfig{
				ProgressIndex:      kvLen,
				NumNewDecodeTokens: 1,
			}
		}

		step := StepConfig{DecodeRequests: decodeReqs}
		currentUS := rooflineStepTime("", mc, hwCalib, step, tp, mfuDB)

		// Compute what step time would be with GEMM floor = bs=8 GEMM time
		gemmTimeCurrent := computeTransformerGEMMTimes(mc, bs, peakFlops, mfuDB, tpScaling)
		gemmDelta := gemmTimeBS8 - gemmTimeCurrent // additional GEMM time if we had floor

		// Recompute: add gemmDelta to compute time
		var attnCoreFLOPs float64
		for _, req := range decodeReqs {
			attnCoreFLOPs += calculateAttentionCoreFLOPs(
				mc.NumHeads, mc.NumKVHeads, mc.HiddenDim, 1, req.ProgressIndex,
			) * float64(mc.NumLayers)
		}
		attnMFU := mfuDB.GetAttnDecodeMFU(bs, int(kvLen), tp)
		attnCoreTimeS := attnCoreFLOPs / (peakFlops * attnMFU) * tpScaling

		newComputeS := gemmTimeBS8 + attnCoreTimeS

		var dDynamicBytes float64
		for _, req := range decodeReqs {
			m := calculateMemoryAccessBytes(mc, req.ProgressIndex, 1, true)
			dDynamicBytes += (m["total"] - m["model_weights"]) * tpScaling
		}
		baseMem := calculateMemoryAccessBytes(mc, 0, 0, false)
		dWeightBytes := baseMem["model_weights"] * tpScaling
		peakBWLocal := hwCalib.BwPeakTBs * 1e12
		if hwCalib.BwEfficiencyFactor != 0 {
			peakBWLocal *= hwCalib.BwEfficiencyFactor
		}
		decodeMemoryS := (dWeightBytes + dDynamicBytes) / peakBWLocal

		overheadS := hwCalib.PerLayerCPUOverhead * float64(mc.NumLayers) / float64(tp) / 1e6
		newStepTimeS := math.Max(newComputeS, decodeMemoryS) + overheadS
		newStepTimeUS := int64(math.Round(newStepTimeS * 1e6))

		deltaUS := newStepTimeUS - currentUS
		increasePct := 0.0
		if currentUS > 0 {
			increasePct = float64(deltaUS) / float64(currentUS) * 100.0
		}

		fmt.Printf("%-10d %12d %12d %12d %9.2f%%\n",
			bs, currentUS, newStepTimeUS, deltaUS, increasePct)

		t.Logf("H34_FLOOR_IMPACT bs=%d current_us=%d floor_us=%d delta_us=%d increase_pct=%.2f gemm_delta_us=%.1f",
			bs, currentUS, newStepTimeUS, deltaUS, increasePct, gemmDelta*1e6)
	}
}

// TestH34_MultiModelSummary produces a concise summary table across all eval
// suite models showing the bs=1/bs=8 GEMM time ratio (the key hypothesis metric).
func TestH34_MultiModelSummary(t *testing.T) {
	mfuDB := loadTestMFUDatabase(t)
	hwCalib := testHardwareCalib()
	peakFlops := hwCalib.TFlopsPeak * 1e12
	tpScaling := 1.0

	fmt.Println("=== H34: Multi-Model bs=1/bs=8 GEMM Time Ratio Summary ===")
	fmt.Println()
	fmt.Printf("%-18s %14s %14s %10s %12s %10s\n",
		"Model", "GEMM_bs1(us)", "GEMM_bs8(us)", "Ratio", "Underest_X", "Verdict")
	fmt.Println("------------------------------------------------------------------------------------")

	fmt.Println("H34_SUMMARY_START")

	var confirmedCount, refutedCount int

	for _, ms := range evalSuiteModels() {
		mc := ms.toModelConfig()

		gemmBS1 := computeTransformerGEMMTimes(mc, 1, peakFlops, mfuDB, tpScaling)
		gemmBS8 := computeTransformerGEMMTimes(mc, 8, peakFlops, mfuDB, tpScaling)

		ratio := 0.0
		underest := 0.0
		if gemmBS8 > 0 {
			ratio = gemmBS1 / gemmBS8
			underest = 1.0 / ratio
		}

		verdict := "CONFIRMED"
		if ratio > 0.25 {
			verdict = "REFUTED"
			refutedCount++
		} else {
			confirmedCount++
		}

		fmt.Printf("%-18s %14.1f %14.1f %10.4f %12.1f %10s\n",
			ms.Name, gemmBS1*1e6, gemmBS8*1e6, ratio, underest, verdict)
	}

	fmt.Println("H34_SUMMARY_END")
	fmt.Println()
	fmt.Printf("H34_OVERALL: confirmed=%d refuted=%d total=%d\n",
		confirmedCount, refutedCount, confirmedCount+refutedCount)

	if confirmedCount > 0 && refutedCount == 0 {
		fmt.Println("H34_VERDICT=CONFIRMED")
		t.Logf("HYPOTHESIS CONFIRMED: All %d models show GEMM time ratio bs1/bs8 <= 0.25", confirmedCount)
	} else if refutedCount > 0 && confirmedCount == 0 {
		fmt.Println("H34_VERDICT=REFUTED")
		t.Logf("HYPOTHESIS REFUTED: All %d models show GEMM time ratio bs1/bs8 > 0.25", refutedCount)
	} else {
		fmt.Println("H34_VERDICT=MIXED")
		t.Logf("MIXED: %d confirmed, %d refuted", confirmedCount, refutedCount)
	}
}
