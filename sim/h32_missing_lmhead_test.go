package sim

import (
	"fmt"
	"math"
	"testing"
)

// =============================================================================
// H32: Missing LM Head GEMM — Quantify impact of omitted LM head projection
// =============================================================================
//
// The roofline model's computeTransformerGEMMTimes computes GEMM times only for
// per-layer transformer projections (Q, K, V, O, Gate, Up, Down) but omits the
// final LM head projection: [batch_size, hidden_dim] x [hidden_dim, vocab_size].
// Similarly, calculateMemoryAccessBytes accounts for per-layer model weights but
// excludes the LM head weight matrix (hidden_dim * vocab_size * bytes_per_param).
//
// This test quantifies the gap to determine if adding the LM head would
// materially change step time predictions.
//
// Independent variable: with vs without LM head GEMM + weight bytes
// Controlled variables: model config, hardware config, batch composition
// Dependent variable: decode step time change (percentage increase)

// modelSpec holds model parameters for the experiment.
type modelSpec struct {
	Name            string
	NumLayers       int
	HiddenDim       int
	NumHeads        int
	NumKVHeads      int
	VocabSize       int
	IntermediateDim int
	BytesPerParam   float64
}

// evalSuiteModels returns ModelConfigs for all models in the evaluation suite.
func evalSuiteModels() []modelSpec {
	return []modelSpec{
		{
			Name:            "llama-2-7b",
			NumLayers:       32,
			HiddenDim:       4096,
			NumHeads:        32,
			NumKVHeads:      32,
			VocabSize:       32000,
			IntermediateDim: 11008,
			BytesPerParam:   2, // float16
		},
		{
			Name:            "llama-2-70b",
			NumLayers:       80,
			HiddenDim:       8192,
			NumHeads:        64,
			NumKVHeads:      8,
			VocabSize:       32000,
			IntermediateDim: 28672,
			BytesPerParam:   2, // float16
		},
		{
			Name:            "llama-3.1-8b",
			NumLayers:       32,
			HiddenDim:       4096,
			NumHeads:        32,
			NumKVHeads:      8,
			VocabSize:       128256,
			IntermediateDim: 14336,
			BytesPerParam:   2, // bfloat16
		},
		{
			Name:            "codellama-34b",
			NumLayers:       48,
			HiddenDim:       8192,
			NumHeads:        64,
			NumKVHeads:      8,
			VocabSize:       32000,
			IntermediateDim: 22016,
			BytesPerParam:   2, // bfloat16
		},
		{
			Name:            "qwen3-14b",
			NumLayers:       40,
			HiddenDim:       5120,
			NumHeads:        40,
			NumKVHeads:      8,
			VocabSize:       151936,
			IntermediateDim: 17408,
			BytesPerParam:   2, // bfloat16
		},
		{
			Name:            "qwen2.5-7b",
			NumLayers:       28,
			HiddenDim:       3584,
			NumHeads:        28,
			NumKVHeads:      4,
			VocabSize:       152064,
			IntermediateDim: 18944,
			BytesPerParam:   2, // bfloat16
		},
	}
}

func (ms modelSpec) toModelConfig() ModelConfig {
	return ModelConfig{
		NumLayers:       ms.NumLayers,
		HiddenDim:       ms.HiddenDim,
		NumHeads:        ms.NumHeads,
		NumKVHeads:      ms.NumKVHeads,
		VocabSize:       ms.VocabSize,
		IntermediateDim: ms.IntermediateDim,
		BytesPerParam:   ms.BytesPerParam,
	}
}

// TestH32_LMHeadWeightFraction computes the LM head weight matrix as a
// fraction of the total transformer model weights for each model.
// This is a pure analytical calculation (no MFU database needed).
func TestH32_LMHeadWeightFraction(t *testing.T) {
	fmt.Println("=== H32: LM Head Weight Fraction of Total Model Weights ===")
	fmt.Println()
	fmt.Printf("%-18s %10s %10s %12s %14s %14s %10s\n",
		"Model", "Hidden", "Vocab", "LMHead(MB)", "TransWts(MB)", "Total(MB)", "LMHead%")
	fmt.Println("----------------------------------------------------------------------------------------------")

	for _, ms := range evalSuiteModels() {
		mc := ms.toModelConfig()

		// LM head weight bytes: hidden_dim * vocab_size * bytes_per_param
		lmHeadBytes := float64(mc.HiddenDim) * float64(mc.VocabSize) * mc.BytesPerParam

		// Transformer per-layer weights (same formula as calculateMemoryAccessBytes)
		nKVHeads := mc.NumKVHeads
		if nKVHeads == 0 {
			nKVHeads = mc.NumHeads
		}
		dHead := mc.HiddenDim / mc.NumHeads
		dKV := nKVHeads * dHead
		dFF := 4 * mc.HiddenDim
		if mc.IntermediateDim > 0 {
			dFF = mc.IntermediateDim
		}

		weightsPerLayer := float64(mc.HiddenDim*(mc.HiddenDim+2*dKV)) +
			float64(mc.HiddenDim*mc.HiddenDim) +
			float64(3*mc.HiddenDim*dFF)
		transformerWeightBytes := weightsPerLayer * float64(mc.NumLayers) * mc.BytesPerParam

		totalBytes := transformerWeightBytes + lmHeadBytes
		lmHeadPct := (lmHeadBytes / totalBytes) * 100.0

		lmHeadMB := lmHeadBytes / (1024 * 1024)
		transformerMB := transformerWeightBytes / (1024 * 1024)
		totalMB := totalBytes / (1024 * 1024)

		fmt.Printf("%-18s %10d %10d %12.1f %14.1f %14.1f %9.2f%%\n",
			ms.Name, mc.HiddenDim, mc.VocabSize, lmHeadMB, transformerMB, totalMB, lmHeadPct)

		// Record for analysis parsing
		t.Logf("H32_WEIGHT_FRACTION model=%s lmhead_bytes=%.0f transformer_bytes=%.0f lmhead_pct=%.4f",
			ms.Name, lmHeadBytes, transformerWeightBytes, lmHeadPct)
	}
	fmt.Println()
}

// TestH32_LMHeadGEMMFLOPs computes the LM head GEMM FLOPs as a fraction of
// total per-step transformer GEMM FLOPs across batch sizes.
func TestH32_LMHeadGEMMFLOPs(t *testing.T) {
	batchSizes := []int{1, 4, 8, 16, 32}

	fmt.Println("=== H32: LM Head GEMM FLOPs as Fraction of Transformer GEMM FLOPs ===")
	fmt.Println()

	for _, ms := range evalSuiteModels() {
		mc := ms.toModelConfig()

		nKVHeads := mc.NumKVHeads
		if nKVHeads == 0 {
			nKVHeads = mc.NumHeads
		}
		dHead := mc.HiddenDim / mc.NumHeads
		dKV := nKVHeads * dHead
		dFF := 4 * mc.HiddenDim
		if mc.IntermediateDim > 0 {
			dFF = mc.IntermediateDim
		}

		fmt.Printf("--- %s (hidden=%d, vocab=%d, layers=%d) ---\n",
			ms.Name, mc.HiddenDim, mc.VocabSize, mc.NumLayers)
		fmt.Printf("%-10s %16s %16s %16s %10s\n",
			"BatchSize", "LMHeadFLOPs", "TransGEMMFLOPs", "TotalFLOPs", "LMHead%")

		for _, bs := range batchSizes {
			// LM head GEMM: 2 * batch_size * hidden_dim * vocab_size
			lmHeadFLOPs := 2.0 * float64(bs) * float64(mc.HiddenDim) * float64(mc.VocabSize)

			// Per-layer transformer GEMM FLOPs (Q, K, V, O, Gate, Up, Down)
			// Q: 2 * bs * dModel * dModel
			// K: 2 * bs * dModel * dKV
			// V: 2 * bs * dModel * dKV
			// O: 2 * bs * dModel * dModel
			// Gate: 2 * bs * dModel * dFF
			// Up: 2 * bs * dModel * dFF
			// Down: 2 * bs * dFF * dModel
			perLayerGEMM := 2.0 * float64(bs) * (float64(mc.HiddenDim*mc.HiddenDim) + // Q
				float64(mc.HiddenDim*dKV) + // K
				float64(mc.HiddenDim*dKV) + // V
				float64(mc.HiddenDim*mc.HiddenDim) + // O
				float64(mc.HiddenDim*dFF) + // Gate
				float64(mc.HiddenDim*dFF) + // Up
				float64(dFF*mc.HiddenDim)) // Down

			transformerGEMMFLOPs := perLayerGEMM * float64(mc.NumLayers)
			totalFLOPs := transformerGEMMFLOPs + lmHeadFLOPs
			lmHeadPct := (lmHeadFLOPs / totalFLOPs) * 100.0

			fmt.Printf("%-10d %16.3e %16.3e %16.3e %9.2f%%\n",
				bs, lmHeadFLOPs, transformerGEMMFLOPs, totalFLOPs, lmHeadPct)

			t.Logf("H32_GEMM_FLOPS model=%s bs=%d lmhead_flops=%.0f transformer_flops=%.0f lmhead_pct=%.4f",
				ms.Name, bs, lmHeadFLOPs, transformerGEMMFLOPs, lmHeadPct)
		}
		fmt.Println()
	}
}

// TestH32_LMHeadGEMMTime computes the actual time impact of adding the LM head
// GEMM using the MFU database. This is the key experiment: it uses the same
// MFU lookup mechanism as the production code to estimate timing impact.
func TestH32_LMHeadGEMMTime(t *testing.T) {
	mfuDB := loadTestMFUDatabase(t)

	hwCalib := testHardwareCalib()
	peakFlops := hwCalib.TFlopsPeak * 1e12
	peakBW := hwCalib.BwPeakTBs * 1e12
	if hwCalib.BwEfficiencyFactor != 0 {
		peakBW *= hwCalib.BwEfficiencyFactor
	}

	batchSizes := []int{1, 4, 8, 16, 32}

	fmt.Println("=== H32: LM Head GEMM Time Impact (using MFU database) ===")
	fmt.Println()

	for _, ms := range evalSuiteModels() {
		mc := ms.toModelConfig()
		tp := 1

		fmt.Printf("--- %s (hidden=%d, vocab=%d, layers=%d, TP=%d) ---\n",
			ms.Name, mc.HiddenDim, mc.VocabSize, mc.NumLayers, tp)
		fmt.Printf("%-10s %14s %14s %14s %10s\n",
			"BatchSize", "TransGEMM(us)", "LMHead(us)", "Total(us)", "LMHead%")

		for _, bs := range batchSizes {
			tpScaling := 1.0 / float64(tp)

			// Current transformer-only GEMM time
			transformerTime := computeTransformerGEMMTimes(mc, bs, peakFlops, peakBW, mfuDB, tpScaling)

			// LM head GEMM time: [bs, hidden_dim] x [hidden_dim, vocab_size]
			// Note: with TP, the vocab_size dimension is split across GPUs,
			// but we apply tpScaling the same way as transformer GEMMs.
			lmHeadTime := computeGEMMTime(bs, mc.HiddenDim, mc.VocabSize, peakFlops, peakBW, mc.BytesPerParam, mfuDB) * tpScaling

			totalTime := transformerTime + lmHeadTime
			lmHeadPct := (lmHeadTime / totalTime) * 100.0

			// Convert to microseconds for readability
			transformerUS := transformerTime * 1e6
			lmHeadUS := lmHeadTime * 1e6
			totalUS := totalTime * 1e6

			fmt.Printf("%-10d %14.1f %14.1f %14.1f %9.2f%%\n",
				bs, transformerUS, lmHeadUS, totalUS, lmHeadPct)

			t.Logf("H32_GEMM_TIME model=%s bs=%d transformer_us=%.1f lmhead_us=%.1f lmhead_pct=%.4f",
				ms.Name, bs, transformerUS, lmHeadUS, lmHeadPct)
		}
		fmt.Println()
	}
}

// TestH32_LMHeadMemoryBandwidthImpact computes the memory bandwidth impact
// of adding the LM head weights to the weight loading phase.
func TestH32_LMHeadMemoryBandwidthImpact(t *testing.T) {
	hwCalib := testHardwareCalib()
	peakBW := hwCalib.BwPeakTBs * 1e12
	if hwCalib.BwEfficiencyFactor != 0 {
		peakBW *= hwCalib.BwEfficiencyFactor
	}

	fmt.Println("=== H32: LM Head Memory Bandwidth Impact ===")
	fmt.Println()
	fmt.Printf("%-18s %14s %14s %14s %10s %14s\n",
		"Model", "TransWts(MB)", "LMHead(MB)", "Total(MB)", "LMHead%", "ExtraTime(us)")
	fmt.Println("--------------------------------------------------------------------------------------------")

	for _, ms := range evalSuiteModels() {
		mc := ms.toModelConfig()

		// Current transformer weight bytes (from calculateMemoryAccessBytes)
		baseMem := calculateMemoryAccessBytes(mc, 0, 0, false)
		transformerWtBytes := baseMem["model_weights"]

		// LM head weight bytes
		lmHeadWtBytes := float64(mc.HiddenDim) * float64(mc.VocabSize) * mc.BytesPerParam

		totalWtBytes := transformerWtBytes + lmHeadWtBytes
		lmHeadPct := (lmHeadWtBytes / totalWtBytes) * 100.0

		// Extra time to load LM head weights at effective bandwidth
		// TP=1 for baseline comparison
		extraTimeS := lmHeadWtBytes / peakBW
		extraTimeUS := extraTimeS * 1e6

		transformerMB := transformerWtBytes / (1024 * 1024)
		lmHeadMB := lmHeadWtBytes / (1024 * 1024)
		totalMB := totalWtBytes / (1024 * 1024)

		fmt.Printf("%-18s %14.1f %14.1f %14.1f %9.2f%% %14.1f\n",
			ms.Name, transformerMB, lmHeadMB, totalMB, lmHeadPct, extraTimeUS)

		t.Logf("H32_MEM_BW model=%s transformer_wt_bytes=%.0f lmhead_wt_bytes=%.0f lmhead_pct=%.4f extra_time_us=%.1f",
			ms.Name, transformerWtBytes, lmHeadWtBytes, lmHeadPct, extraTimeUS)
	}
	fmt.Println()
}

// TestH32_DecodeStepTimeComparison computes the full decode step time with and
// without LM head contributions (GEMM time + memory bandwidth), using the
// actual roofline model infrastructure.
func TestH32_DecodeStepTimeComparison(t *testing.T) {
	mfuDB := loadTestMFUDatabase(t)

	hwCalib := testHardwareCalib()
	peakFlops := hwCalib.TFlopsPeak * 1e12
	peakBW := hwCalib.BwPeakTBs * 1e12
	if hwCalib.BwEfficiencyFactor != 0 {
		peakBW *= hwCalib.BwEfficiencyFactor
	}

	// Representative decode scenarios
	type scenario struct {
		name      string
		batchSize int
		kvLen     int64
		tp        int
	}

	scenarios := []scenario{
		{"bs1-kv512-tp1", 1, 512, 1},
		{"bs4-kv512-tp1", 4, 512, 1},
		{"bs8-kv1024-tp1", 8, 1024, 1},
		{"bs16-kv1024-tp1", 16, 1024, 1},
		{"bs32-kv2048-tp1", 32, 2048, 1},
		{"bs8-kv512-tp2", 8, 512, 2},
		{"bs16-kv1024-tp2", 16, 1024, 2},
	}

	fmt.Println("=== H32: Full Decode Step Time With vs Without LM Head ===")
	fmt.Println()

	for _, ms := range evalSuiteModels() {
		mc := ms.toModelConfig()

		fmt.Printf("--- %s ---\n", ms.Name)
		fmt.Printf("%-22s %14s %14s %14s %10s\n",
			"Scenario", "Baseline(us)", "WithLMH(us)", "Delta(us)", "Increase%")

		for _, sc := range scenarios {
			tpScaling := 1.0 / float64(sc.tp)

			// Build decode requests (all with the same KV length for simplicity)
			decodeReqs := make([]DecodeRequestConfig, sc.batchSize)
			for i := 0; i < sc.batchSize; i++ {
				decodeReqs[i] = DecodeRequestConfig{
					ProgressIndex:      sc.kvLen,
					NumNewDecodeTokens: 1,
				}
			}

			// --- Baseline: current roofline (no LM head) ---
			step := StepConfig{DecodeRequests: decodeReqs}
			baselineUS := rooflineStepTime("", mc, hwCalib, step, sc.tp, mfuDB)

			// --- With LM head: compute additional GEMM time and memory bytes ---
			// Additional GEMM time
			lmHeadGEMMTimeS := computeGEMMTime(sc.batchSize, mc.HiddenDim, mc.VocabSize, peakFlops, peakBW, mc.BytesPerParam, mfuDB) * tpScaling

			// Additional memory: LM head weights
			lmHeadWtBytes := float64(mc.HiddenDim) * float64(mc.VocabSize) * mc.BytesPerParam * tpScaling
			lmHeadMemTimeS := lmHeadWtBytes / peakBW

			// The roofline model uses max(compute, memory) for the decode phase.
			// We need to recompute the decode phase with the additional LM head terms.
			// Current decode compute = gemmTime + attnCoreTime
			// Current decode memory  = (weightBytes + dynamicBytes) / peakBW
			// With LM head:
			//   decode compute += lmHeadGEMMTime
			//   decode memory  += lmHeadWtBytes / peakBW

			// Get current decode compute and memory components
			gemmTimeS := computeTransformerGEMMTimes(mc, sc.batchSize, peakFlops, peakBW, mfuDB, tpScaling)

			var attnCoreFLOPs float64
			for _, req := range decodeReqs {
				attnCoreFLOPs += calculateAttentionCoreFLOPs(
					mc.NumHeads, mc.NumKVHeads, mc.HiddenDim, 1, req.ProgressIndex,
				) * float64(mc.NumLayers)
			}
			attnMFU := mfuDB.GetAttnDecodeMFU(sc.batchSize, int(sc.kvLen), sc.tp)
			attnCoreTimeS := attnCoreFLOPs / (peakFlops * attnMFU) * tpScaling

			decodeComputeBaseline := gemmTimeS + attnCoreTimeS
			decodeComputeWithLMH := decodeComputeBaseline + lmHeadGEMMTimeS

			// Memory
			var dDynamicBytes float64
			for _, req := range decodeReqs {
				m := calculateMemoryAccessBytes(mc, req.ProgressIndex, 1, true)
				dDynamicBytes += (m["total"] - m["model_weights"]) * tpScaling
			}
			baseMem := calculateMemoryAccessBytes(mc, 0, 0, false)
			dWeightBytes := baseMem["model_weights"] * tpScaling

			decodeMemBaseline := (dWeightBytes + dDynamicBytes) / peakBW
			decodeMemWithLMH := decodeMemBaseline + lmHeadMemTimeS

			// CPU overhead (same for both)
			overheadS := (hwCalib.PerLayerCPUOverhead * float64(mc.NumLayers) / float64(sc.tp)) / 1e6

			// math.Max(compute, memory) + overhead
			baselineCalc := math.Max(decodeComputeBaseline, decodeMemBaseline) + overheadS
			withLMHCalc := math.Max(decodeComputeWithLMH, decodeMemWithLMH) + overheadS

			baselineCalcUS := baselineCalc * 1e6
			withLMHCalcUS := withLMHCalc * 1e6
			deltaUS := withLMHCalcUS - baselineCalcUS
			increasePct := 0.0
			if baselineCalcUS > 0 {
				increasePct = (deltaUS / baselineCalcUS) * 100.0
			}

			fmt.Printf("%-22s %14.1f %14.1f %14.1f %9.2f%%\n",
				sc.name, baselineCalcUS, withLMHCalcUS, deltaUS, increasePct)

			t.Logf("H32_STEP_TIME model=%s scenario=%s baseline_us=%.1f withlmh_us=%.1f delta_us=%.1f increase_pct=%.4f baseline_roofline_us=%d",
				ms.Name, sc.name, baselineCalcUS, withLMHCalcUS, deltaUS, increasePct, baselineUS)
		}
		fmt.Println()
	}
}

// TestH32_ComputeVsMemoryRegime determines whether the LM head addition
// shifts the bottleneck from memory-bound to compute-bound (or vice versa).
func TestH32_ComputeVsMemoryRegime(t *testing.T) {
	mfuDB := loadTestMFUDatabase(t)

	hwCalib := testHardwareCalib()
	peakFlops := hwCalib.TFlopsPeak * 1e12
	peakBW := hwCalib.BwPeakTBs * 1e12
	if hwCalib.BwEfficiencyFactor != 0 {
		peakBW *= hwCalib.BwEfficiencyFactor
	}

	batchSizes := []int{1, 8, 32}
	kvLen := int64(1024)
	tp := 1
	tpScaling := 1.0 / float64(tp)

	fmt.Println("=== H32: Compute vs Memory Regime Analysis ===")
	fmt.Println()

	for _, ms := range evalSuiteModels() {
		mc := ms.toModelConfig()

		fmt.Printf("--- %s (kvLen=%d, TP=%d) ---\n", ms.Name, kvLen, tp)
		fmt.Printf("%-10s %12s %12s %8s %12s %12s %8s\n",
			"BatchSize",
			"Comp_Base", "Mem_Base", "Regime",
			"Comp_LMH", "Mem_LMH", "Regime")

		for _, bs := range batchSizes {
			decodeReqs := make([]DecodeRequestConfig, bs)
			for i := 0; i < bs; i++ {
				decodeReqs[i] = DecodeRequestConfig{ProgressIndex: kvLen, NumNewDecodeTokens: 1}
			}

			// Baseline compute
			gemmTimeS := computeTransformerGEMMTimes(mc, bs, peakFlops, peakBW, mfuDB, tpScaling)
			var attnCoreFLOPs float64
			for _, req := range decodeReqs {
				attnCoreFLOPs += calculateAttentionCoreFLOPs(
					mc.NumHeads, mc.NumKVHeads, mc.HiddenDim, 1, req.ProgressIndex,
				) * float64(mc.NumLayers)
			}
			attnMFU := mfuDB.GetAttnDecodeMFU(bs, int(kvLen), tp)
			attnCoreTimeS := attnCoreFLOPs / (peakFlops * attnMFU) * tpScaling
			computeBase := gemmTimeS + attnCoreTimeS

			// Baseline memory
			var dDynamicBytes float64
			for _, req := range decodeReqs {
				m := calculateMemoryAccessBytes(mc, req.ProgressIndex, 1, true)
				dDynamicBytes += (m["total"] - m["model_weights"]) * tpScaling
			}
			baseMem := calculateMemoryAccessBytes(mc, 0, 0, false)
			dWeightBytes := baseMem["model_weights"] * tpScaling
			memBase := (dWeightBytes + dDynamicBytes) / peakBW

			// With LM head
			lmHeadGEMMTimeS := computeGEMMTime(bs, mc.HiddenDim, mc.VocabSize, peakFlops, peakBW, mc.BytesPerParam, mfuDB) * tpScaling
			computeLMH := computeBase + lmHeadGEMMTimeS

			lmHeadWtBytes := float64(mc.HiddenDim) * float64(mc.VocabSize) * mc.BytesPerParam * tpScaling
			memLMH := memBase + lmHeadWtBytes/peakBW

			regimeBase := "MEM"
			if computeBase > memBase {
				regimeBase = "COMP"
			}
			regimeLMH := "MEM"
			if computeLMH > memLMH {
				regimeLMH = "COMP"
			}

			fmt.Printf("%-10d %12.1f %12.1f %8s %12.1f %12.1f %8s\n",
				bs,
				computeBase*1e6, memBase*1e6, regimeBase,
				computeLMH*1e6, memLMH*1e6, regimeLMH)

			t.Logf("H32_REGIME model=%s bs=%d compute_base_us=%.1f mem_base_us=%.1f regime_base=%s compute_lmh_us=%.1f mem_lmh_us=%.1f regime_lmh=%s",
				ms.Name, bs, computeBase*1e6, memBase*1e6, regimeBase, computeLMH*1e6, memLMH*1e6, regimeLMH)
		}
		fmt.Println()
	}
}

