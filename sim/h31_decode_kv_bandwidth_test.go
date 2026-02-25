package sim

import (
	"fmt"
	"math"
	"testing"
)

// TestH31_DecodeKVBandwidthDiscount validates the hypothesis that the 0.80
// discount factor on decode KV cache access bytes underestimates actual memory
// traffic. The experiment:
//   1. Computes memory access bytes at the current 0.80 discount
//   2. Derives what the bytes WOULD be at discount factors 0.90, 0.95, 1.00
//   3. Computes the fractional change in decode memory time and total step time
//   4. Reports whether memory-bound decode steps are significantly affected
//
// The 0.80 factor is hardcoded in calculateMemoryAccessBytes (roofline_step.go:91):
//   mem["kv_cache_access"] = kvReadPerToken * seq * 0.80
//
// This test cannot modify the hardcoded constant, so it reverse-engineers
// the kv_cache_access at 0.80 and scales to other discount factors analytically.
func TestH31_DecodeKVBandwidthDiscount(t *testing.T) {
	mc := testModelConfig() // Llama-3.1-8B-like
	hc := testHardwareCalib()

	// Effective peak BW after efficiency factor
	peakBW := hc.BwPeakTBs * 1e12
	if hc.BwEfficiencyFactor != 0 {
		peakBW *= hc.BwEfficiencyFactor
	}

	// Discount factors to test
	discountFactors := []float64{0.80, 0.90, 0.95, 1.00}

	// Representative decode scenarios: (batchSize, kvLen) pairs
	// spanning the eval suite operating points
	type decodeScenario struct {
		name      string
		batchSize int
		kvLens    []int64 // one kvLen per request in the batch
	}

	scenarios := []decodeScenario{
		{"single_short", 1, []int64{128}},
		{"single_medium", 1, []int64{512}},
		{"single_long", 1, []int64{2048}},
		{"single_vlong", 1, []int64{4096}},
		{"batch4_uniform", 4, []int64{512, 512, 512, 512}},
		{"batch8_uniform", 8, []int64{256, 256, 256, 256, 256, 256, 256, 256}},
		{"batch16_mixed", 16, []int64{64, 128, 256, 512, 512, 1024, 1024, 2048, 2048, 2048, 4096, 4096, 4096, 4096, 4096, 4096}},
		{"batch4_long", 4, []int64{2048, 2048, 4096, 4096}},
		{"batch1_extreme", 1, []int64{8192}},
	}

	// Print header for structured output (parsed by analyze.py)
	fmt.Println("H31_RESULTS_START")
	fmt.Printf("%-25s | %5s | %8s | %8s | %8s | %8s | %10s | %10s | %10s | %10s | %10s\n",
		"scenario", "bs", "maxKV", "meanKV",
		"d0.80", "d1.00",
		"memBytes80", "memBytes100",
		"memTimeS80", "memTimeS100",
		"pctIncrease")
	fmt.Println("---")

	for _, sc := range scenarios {
		// Build decode requests
		decodeReqs := make([]DecodeRequestConfig, len(sc.kvLens))
		for i, kv := range sc.kvLens {
			decodeReqs[i] = DecodeRequestConfig{
				ProgressIndex:      kv,
				NumNewDecodeTokens: 1,
			}
		}

		// Compute memory bytes at the hardcoded 0.80 discount using the real function
		var dynamicBytes080 float64
		for _, req := range decodeReqs {
			m := calculateMemoryAccessBytes(mc, req.ProgressIndex, 1, true)
			dynamicBytes080 += m["total"] - m["model_weights"]
		}

		baseMem := calculateMemoryAccessBytes(mc, 0, 0, false)
		weightBytes := baseMem["model_weights"]

		// Reverse-engineer kv_cache_access at 0.80 to compute other discount factors.
		// From the code:
		//   kv_cache_access = kvReadPerToken * seq * 0.80
		// So kv_cache_access_at_X = kv_cache_access_at_080 * (X / 0.80)
		//
		// We need the kv_cache_access portion. Extract it from the per-request results.
		var kvCacheAccess080 float64
		for _, req := range decodeReqs {
			m := calculateMemoryAccessBytes(mc, req.ProgressIndex, 1, true)
			kvCacheAccess080 += m["kv_cache_access"]
		}

		// Compute max and mean KV len
		var maxKV, sumKV int64
		for _, kv := range sc.kvLens {
			if kv > maxKV {
				maxKV = kv
			}
			sumKV += kv
		}
		meanKV := float64(sumKV) / float64(len(sc.kvLens))

		// For each discount factor, compute adjusted memory bytes and time
		type result struct {
			discount     float64
			totalMemByte float64
			memTimeS     float64
		}

		results := make([]result, len(discountFactors))
		for i, d := range discountFactors {
			// Scale kv_cache_access from 0.80 to d
			kvCacheAccessD := kvCacheAccess080 * (d / 0.80)
			// Recompute dynamic bytes: replace old kv_cache_access with new
			adjustedDynamic := dynamicBytes080 - kvCacheAccess080 + kvCacheAccessD
			totalMemBytesD := weightBytes + adjustedDynamic
			memTimeS := totalMemBytesD / peakBW

			results[i] = result{
				discount:     d,
				totalMemByte: totalMemBytesD,
				memTimeS:     memTimeS,
			}
		}

		// Print the 0.80 vs 1.00 comparison row
		r080 := results[0]  // discount=0.80
		r100 := results[3]  // discount=1.00
		pctIncrease := (r100.memTimeS - r080.memTimeS) / r080.memTimeS * 100.0

		fmt.Printf("%-25s | %5d | %8d | %8.0f | %8.2f | %8.2f | %10.0f | %10.0f | %10.6f | %10.6f | %9.2f%%\n",
			sc.name, len(sc.kvLens), maxKV, meanKV,
			0.80, 1.00,
			r080.totalMemByte, r100.totalMemByte,
			r080.memTimeS, r100.memTimeS,
			pctIncrease)

		// Invariant: increasing discount must monotonically increase memory bytes
		for i := 1; i < len(results); i++ {
			if results[i].totalMemByte < results[i-1].totalMemByte {
				t.Errorf("%s: discount %.2f memBytes (%.0f) < discount %.2f memBytes (%.0f) — violates monotonicity",
					sc.name, results[i].discount, results[i].totalMemByte,
					results[i-1].discount, results[i-1].totalMemByte)
			}
		}
	}
	fmt.Println("H31_RESULTS_END")

	// === Phase 2: Full roofline step time comparison ===
	// This requires MFU database — skip if bench_data not available
	mfuDB := loadTestMFUDatabase(t)

	fmt.Println()
	fmt.Println("H31_STEPTIME_START")
	fmt.Printf("%-25s | %5s | %8s | %12s | %12s | %12s | %12s | %10s | %10s\n",
		"scenario", "bs", "maxKV",
		"computeS", "memS_080", "memS_100",
		"stepTime_080", "stepTime_100",
		"stepDelta%")
	fmt.Println("---")

	tp := 1

	for _, sc := range scenarios {
		decodeReqs := make([]DecodeRequestConfig, len(sc.kvLens))
		for i, kv := range sc.kvLens {
			decodeReqs[i] = DecodeRequestConfig{
				ProgressIndex:      kv,
				NumNewDecodeTokens: 1,
			}
		}

		step := StepConfig{DecodeRequests: decodeReqs}

		// Get baseline step time at 0.80 (current hardcoded)
		stepTime080 := rooflineStepTime("", mc, hc, step, tp, mfuDB)

		// To estimate step time at discount=1.00, we compute:
		// 1. The decode compute time (unchanged)
		// 2. The decode memory time at 0.80 and 1.00
		// 3. The overhead (unchanged)
		// Then total = max(compute, memory_at_1.00) + overhead

		tpFactor := float64(tp)
		tpScaling := 1.0 / tpFactor
		peakBWLocal := hc.BwPeakTBs * 1e12
		if hc.BwEfficiencyFactor != 0 {
			peakBWLocal *= hc.BwEfficiencyFactor
		}

		// Compute decode memory time at 0.80 and 1.00
		var dynamicBytes080 float64
		var kvCacheAccess080 float64
		for _, req := range decodeReqs {
			m := calculateMemoryAccessBytes(mc, req.ProgressIndex, 1, true)
			dynamicBytes080 += (m["total"] - m["model_weights"]) * tpScaling
			kvCacheAccess080 += m["kv_cache_access"] * tpScaling
		}
		baseMemLocal := calculateMemoryAccessBytes(mc, 0, 0, false)
		weightBytesLocal := baseMemLocal["model_weights"] * tpScaling

		memTime080 := (weightBytesLocal + dynamicBytes080) / peakBWLocal

		// Adjust for discount=1.00
		kvCacheAccess100 := kvCacheAccess080 * (1.00 / 0.80)
		adjustedDynamic := dynamicBytes080 - kvCacheAccess080 + kvCacheAccess100
		memTime100 := (weightBytesLocal + adjustedDynamic) / peakBWLocal

		// Compute decode compute time by running the roofline and backing out memory/overhead
		overheadMicros := hc.PerLayerCPUOverhead * float64(mc.NumLayers) / tpFactor
		overheadS := overheadMicros / 1e6

		// stepTime080_us = max(computeS, memTime080) + overheadS in micros
		// We can derive computeS from the step time:
		// stepTime080_s = max(computeS, memTime080) + overheadS
		stepTime080_s := float64(stepTime080) / 1e6
		hardwareTime080 := stepTime080_s - overheadS
		// If hardwareTime080 == memTime080, then step was memory-bound
		// If hardwareTime080 > memTime080, step was compute-bound and computeS = hardwareTime080
		var computeS float64
		if hardwareTime080 > memTime080+1e-9 {
			computeS = hardwareTime080 // compute-bound
		} else {
			computeS = hardwareTime080 // memory-bound: computeS <= memTime080
		}

		// Estimate step time at discount=1.00
		stepTime100_s := math.Max(computeS, memTime100) + overheadS
		// But if originally compute-bound, computeS was the binding constraint,
		// and memTime100 might still be below computeS.
		// If originally memory-bound, computeS < memTime080, so computeS < memTime100.
		// We can reconstruct computeS properly:
		// hardwareTime080 = max(computeS_actual, memTime080)
		// If memory-bound: hardwareTime080 = memTime080, computeS_actual unknown (below memTime080)
		// If compute-bound: hardwareTime080 = computeS_actual
		//
		// For memory-bound case, we know computeS_actual < memTime080 < memTime100,
		// so stepTime100 = memTime100 + overheadS

		isMemBound := math.Abs(hardwareTime080-memTime080) < 1e-9 ||
			hardwareTime080 < memTime080+1e-9

		if isMemBound {
			// Memory-bound: step time scales with memory time
			stepTime100_s = memTime100 + overheadS
		}
		// else compute-bound: step time unchanged unless memTime100 > computeS

		stepTime100 := int64(math.Round(stepTime100_s * 1e6))

		stepDelta := float64(stepTime100-stepTime080) / float64(stepTime080) * 100.0

		var maxKV int64
		for _, kv := range sc.kvLens {
			if kv > maxKV {
				maxKV = kv
			}
		}

		regime := "MEM"
		if !isMemBound {
			regime = "CMP"
		}

		fmt.Printf("%-25s | %5d | %8d | %12.6f | %12.6f | %12.6f | %12d | %12d | %9.2f%% [%s]\n",
			sc.name, len(sc.kvLens), maxKV,
			computeS, memTime080, memTime100,
			stepTime080, stepTime100,
			stepDelta, regime)
	}
	fmt.Println("H31_STEPTIME_END")

	// === Phase 3: KV fraction analysis ===
	// Show what fraction of total decode memory bytes is kv_cache_access
	fmt.Println()
	fmt.Println("H31_KV_FRACTION_START")
	fmt.Printf("%-25s | %5s | %8s | %12s | %12s | %8s | %12s\n",
		"scenario", "bs", "maxKV",
		"kvAccess080", "totalDynamic",
		"kvFrac%",
		"kvIncrease%")
	fmt.Println("---")

	for _, sc := range scenarios {
		decodeReqs := make([]DecodeRequestConfig, len(sc.kvLens))
		for i, kv := range sc.kvLens {
			decodeReqs[i] = DecodeRequestConfig{
				ProgressIndex:      kv,
				NumNewDecodeTokens: 1,
			}
		}

		var kvAccess080, totalDynamic float64
		for _, req := range decodeReqs {
			m := calculateMemoryAccessBytes(mc, req.ProgressIndex, 1, true)
			kvAccess080 += m["kv_cache_access"]
			totalDynamic += m["total"] - m["model_weights"]
		}

		kvFraction := kvAccess080 / totalDynamic * 100.0
		// Increase from 0.80 -> 1.00: the kv portion increases by (1.00/0.80 - 1) = 25%
		// Impact on total dynamic: kvFraction * 25%
		totalDynamicIncrease := kvFraction / 100.0 * 25.0

		var maxKV int64
		for _, kv := range sc.kvLens {
			if kv > maxKV {
				maxKV = kv
			}
		}

		fmt.Printf("%-25s | %5d | %8d | %12.0f | %12.0f | %7.1f%% | %11.2f%%\n",
			sc.name, len(sc.kvLens), maxKV,
			kvAccess080, totalDynamic,
			kvFraction, totalDynamicIncrease)
	}
	fmt.Println("H31_KV_FRACTION_END")

	// === Phase 4: Discount factor sweep ===
	// Show step time at all 4 discount factors for a representative scenario
	fmt.Println()
	fmt.Println("H31_DISCOUNT_SWEEP_START")
	fmt.Printf("%-25s | %8s | %12s | %12s | %10s\n",
		"scenario", "discount", "memTimeS", "estStepTime", "deltaVs080%")
	fmt.Println("---")

	sweepScenarios := []struct {
		name  string
		kvLen int64
	}{
		{"single_128", 128},
		{"single_512", 512},
		{"single_2048", 2048},
		{"single_4096", 4096},
	}

	for _, ss := range sweepScenarios {
		decodeReqs := []DecodeRequestConfig{{ProgressIndex: ss.kvLen, NumNewDecodeTokens: 1}}
		step := StepConfig{DecodeRequests: decodeReqs}
		stepTime080 := rooflineStepTime("", mc, hc, step, tp, mfuDB)

		m := calculateMemoryAccessBytes(mc, ss.kvLen, 1, true)
		kvAccess080 := m["kv_cache_access"]
		dynamic080 := m["total"] - m["model_weights"]
		baseMem := calculateMemoryAccessBytes(mc, 0, 0, false)
		wBytes := baseMem["model_weights"]

		tpScaling := 1.0 / float64(tp)
		peakBWLocal := hc.BwPeakTBs * 1e12
		if hc.BwEfficiencyFactor != 0 {
			peakBWLocal *= hc.BwEfficiencyFactor
		}

		for _, d := range discountFactors {
			kvAccessD := kvAccess080 * (d / 0.80)
			adjustedDynamic := (dynamic080 - kvAccess080 + kvAccessD) * tpScaling
			memTimeD := (wBytes*tpScaling + adjustedDynamic) / peakBWLocal

			overheadS := hc.PerLayerCPUOverhead * float64(mc.NumLayers) / float64(tp) / 1e6
			hardwareTime080 := float64(stepTime080)/1e6 - overheadS
			memTime080 := (wBytes*tpScaling + dynamic080*tpScaling) / peakBWLocal
			isMemBound := hardwareTime080 <= memTime080+1e-9

			var estStepTimeS float64
			if isMemBound {
				estStepTimeS = memTimeD + overheadS
			} else {
				estStepTimeS = math.Max(hardwareTime080, memTimeD) + overheadS
			}
			estStepTimeUs := int64(math.Round(estStepTimeS * 1e6))
			deltaVs080 := float64(estStepTimeUs-stepTime080) / float64(stepTime080) * 100.0

			fmt.Printf("%-25s | %8.2f | %12.6f | %12d | %9.2f%%\n",
				ss.name, d, memTimeD, estStepTimeUs, deltaVs080)
		}
	}
	fmt.Println("H31_DISCOUNT_SWEEP_END")
}
