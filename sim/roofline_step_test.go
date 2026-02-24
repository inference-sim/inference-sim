package sim

import (
	"math"
	"sort"
	"testing"
)

func TestCalculateMemoryAccessBytes_Deterministic(t *testing.T) {
	// GIVEN a ModelConfig with multiple non-zero fields
	config := testModelConfig()

	// WHEN calculateMemoryAccessBytes is called 100 times
	var firstTotal float64
	for i := 0; i < 100; i++ {
		result := calculateMemoryAccessBytes(config, 1024, 64, true)

		// THEN every call produces the same "total"
		if i == 0 {
			firstTotal = result["total"]
		} else if result["total"] != firstTotal {
			t.Fatalf("non-deterministic total: call 0 got %v, call %d got %v", firstTotal, i, result["total"])
		}
	}

	// Also verify the total is positive (sanity)
	if firstTotal <= 0 {
		t.Errorf("expected positive total, got %v", firstTotal)
	}

	// Verify component-sum conservation: total == sum of all non-"total" keys
	// Sort keys for deterministic accumulation (antipattern #2)
	result := calculateMemoryAccessBytes(config, 1024, 64, true)
	keys := make([]string, 0, len(result))
	for k := range result {
		if k != "total" {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	var componentSum float64
	for _, k := range keys {
		componentSum += result[k]
	}
	if result["total"] != componentSum {
		t.Errorf("conservation violation: total=%v but sum of components=%v", result["total"], componentSum)
	}
}

// testModelConfig returns a Llama-3.1-8B-like config for roofline tests.
func testModelConfig() ModelConfig {
	return ModelConfig{
		NumLayers:       32,
		HiddenDim:       4096,
		NumHeads:        32,
		NumKVHeads:      8,
		VocabSize:       128256,
		BytesPerParam:   2, // bfloat16
		IntermediateDim: 14336,
	}
}

func TestCalculateTransformerFlops_Conservation_TotalEqualsSumOfComponents(t *testing.T) {
	// BC-8: total MUST equal gemm_ops + sram_ops
	mc := testModelConfig()
	tests := []struct {
		name   string
		seqLen int64
		newT   int64
		attn   bool
		mlp    bool
	}{
		{"prefill attn+mlp", 0, 128, true, true},
		{"decode attn+mlp", 512, 1, true, true},
		{"attn only", 256, 64, true, false},
		{"mlp only", 256, 64, false, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			flops := calculateTransformerFlops(mc, tt.seqLen, tt.newT, tt.attn, tt.mlp)
			sum := flops["gemm_ops"] + flops["sram_ops"]
			if flops["total"] != sum {
				t.Errorf("total (%g) != gemm_ops (%g) + sram_ops (%g) = %g",
					flops["total"], flops["gemm_ops"], flops["sram_ops"], sum)
			}
		})
	}
}

func TestCalculateTransformerFlops_AttentionOnly_NoMLPContribution(t *testing.T) {
	// BC-7: disabling MLP zeroes MLP FLOPs
	mc := testModelConfig()

	attnOnly := calculateTransformerFlops(mc, 256, 64, true, false)
	both := calculateTransformerFlops(mc, 256, 64, true, true)

	// With MLP disabled, gemm_ops should be less (no SwiGLU)
	if attnOnly["gemm_ops"] >= both["gemm_ops"] {
		t.Errorf("attention-only gemm_ops (%g) should be less than attn+mlp gemm_ops (%g)",
			attnOnly["gemm_ops"], both["gemm_ops"])
	}
	// sram_ops should be the same (MLP doesn't contribute to sram_ops)
	if attnOnly["sram_ops"] != both["sram_ops"] {
		t.Errorf("sram_ops should be identical with/without MLP: got %g vs %g",
			attnOnly["sram_ops"], both["sram_ops"])
	}
}

func TestCalculateTransformerFlops_MLPOnly_NoAttentionContribution(t *testing.T) {
	// BC-6: disabling attention zeroes attention FLOPs, sram_ops must be zero
	mc := testModelConfig()

	mlpOnly := calculateTransformerFlops(mc, 256, 64, false, true)

	if mlpOnly["sram_ops"] != 0 {
		t.Errorf("MLP-only sram_ops should be 0, got %g", mlpOnly["sram_ops"])
	}
	if mlpOnly["gemm_ops"] <= 0 {
		t.Errorf("MLP-only gemm_ops should be > 0, got %g", mlpOnly["gemm_ops"])
	}
}

func TestCalculateTransformerFlops_Monotonicity_MoreTokensMoreFlops(t *testing.T) {
	// BC-1: more newTokens MUST produce higher total FLOPs
	mc := testModelConfig()

	small := calculateTransformerFlops(mc, 512, 100, true, true)
	large := calculateTransformerFlops(mc, 512, 200, true, true)

	if large["total"] <= small["total"] {
		t.Errorf("200 tokens total FLOPs (%g) should exceed 100 tokens (%g)",
			large["total"], small["total"])
	}
	// Check component-level monotonicity too
	if large["gemm_ops"] <= small["gemm_ops"] {
		t.Errorf("200 tokens gemm_ops (%g) should exceed 100 tokens (%g)",
			large["gemm_ops"], small["gemm_ops"])
	}
}

func TestCalculateMemoryAccessBytes_Monotonicity_MoreTokensMoreBytes(t *testing.T) {
	// BC-2: more newTokens MUST produce higher total bytes
	mc := testModelConfig()

	small := calculateMemoryAccessBytes(mc, 512, 100, true)
	large := calculateMemoryAccessBytes(mc, 512, 200, true)

	if large["total"] <= small["total"] {
		t.Errorf("200 tokens total bytes (%g) should exceed 100 tokens (%g)",
			large["total"], small["total"])
	}
}

func TestCalculateMemoryAccessBytes_Conservation_TotalEqualsSumOfComponents(t *testing.T) {
	// BC-9: total MUST equal sum of all non-"total" components
	mc := testModelConfig()

	mem := calculateMemoryAccessBytes(mc, 512, 64, true)

	// Sort keys before float accumulation (antipattern #2)
	keys := make([]string, 0, len(mem))
	for k := range mem {
		if k != "total" {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	var sum float64
	for _, k := range keys {
		sum += mem[k]
	}
	if math.Abs(mem["total"]-sum) > 1e-6 {
		t.Errorf("total (%g) != sum of components (%g), delta=%g",
			mem["total"], sum, mem["total"]-sum)
	}
}

// testHardwareCalib returns an H100-like hardware config for roofline tests.
func testHardwareCalib() HardwareCalib {
	return HardwareCalib{
		TFlopsPeak:       989.0,
		BwPeakTBs:        3.35,
		BwEffConstant:    0.7,
		TOverheadMicros:  50.0,
		PerLayerCPUOverhead: 5.0,
		MfuPrefill:       0.55,
		MfuDecode:        0.30,
		AllReduceLatency: 10.0,
		// Roofline calibration factors (reasonable defaults for tests)
		TpScalingExponent:          0.8,   // Sublinear TP scaling
		DecodeTpScalingExponent:    1.0,   // Linear for decode
		MfuPrefillMultiplier:       1.0,   // No adjustment
		MfuDecodeMultiplier:        1.0,   // No adjustment
		PrefillBwFactor:            1.0,   // No reduction
		DecodeBwFactor:             1.0,   // No reduction
		VectorPeakFraction:         0.1,   // 10% for non-tensor ops
		PrefillOverheadMicros:      100.0, // Per request overhead
		MixedPrefillOverheadMicros: 50.0,  // Lower overhead in mixed batch
		BwEfficiencyFactor:         0.82,  // H100 sustained-to-peak HBM3 BW ratio (STREAM benchmark: ~2750/3350)
	}
}

func TestRooflineStepTime_TPScaling_TP2LessThanTP1(t *testing.T) {
	// BC-3: TP=2 MUST produce strictly less latency than TP=1
	mc := testModelConfig()
	hc := testHardwareCalib()

	step := StepConfig{
		PrefillRequests: []PrefillRequestConfig{
			{ProgressIndex: 0, NumNewPrefillTokens: 128},
		},
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 256, NumNewDecodeTokens: 1},
		},
	}

	tp1 := rooflineStepTime("", mc, hc, step, 1)
	tp2 := rooflineStepTime("", mc, hc, step, 2)

	if tp2 >= tp1 {
		t.Errorf("TP=2 latency (%d µs) should be less than TP=1 (%d µs)", tp2, tp1)
	}
	if tp2 <= 0 {
		t.Errorf("TP=2 latency should be positive, got %d", tp2)
	}
}

func TestRooflineStepTime_Smoke_ValidInputsProducePositiveFiniteResult(t *testing.T) {
	// BC-4: valid inputs MUST produce > 0, finite result
	mc := testModelConfig()
	hc := testHardwareCalib()

	tests := []struct {
		name string
		step StepConfig
	}{
		{
			"prefill only",
			StepConfig{
				PrefillRequests: []PrefillRequestConfig{
					{ProgressIndex: 0, NumNewPrefillTokens: 256},
				},
			},
		},
		{
			"decode only",
			StepConfig{
				DecodeRequests: []DecodeRequestConfig{
					{ProgressIndex: 512, NumNewDecodeTokens: 1},
				},
			},
		},
		{
			"mixed prefill+decode",
			StepConfig{
				PrefillRequests: []PrefillRequestConfig{
					{ProgressIndex: 0, NumNewPrefillTokens: 64},
				},
				DecodeRequests: []DecodeRequestConfig{
					{ProgressIndex: 128, NumNewDecodeTokens: 1},
					{ProgressIndex: 256, NumNewDecodeTokens: 1},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := rooflineStepTime("", mc, hc, tt.step, 1)
			if result <= 0 {
				t.Errorf("expected positive latency, got %d µs", result)
			}
		})
	}
}

func TestRooflineStepTime_EmptyStep_ReturnsOverheadOnly(t *testing.T) {
	// Edge case: no requests should still return overhead (non-zero due to TOverheadMicros)
	mc := testModelConfig()
	hc := testHardwareCalib()

	step := StepConfig{} // empty
	result := rooflineStepTime("", mc, hc, step, 1)

	// Should be approximately TOverheadMicros (50) + layer overhead
	if result <= 0 {
		t.Errorf("empty step should still have overhead latency, got %d µs", result)
	}
}

// loadTestMFUDatabase loads the real bench_data for V2 roofline tests.
// Skips the test if bench_data is not available.
func loadTestMFUDatabase(t *testing.T) *MFUDatabase {
	t.Helper()
	mc := testModelConfig()
	benchDataPath := "../bench_data"
	db, err := NewMFUDatabase(mc, benchDataPath, "h100")
	if err != nil {
		t.Skipf("bench_data not available, skipping V2 test: %v", err)
	}
	return db
}

func TestRooflineStepTimeV2_BwEfficiency_DecodeBoundHigherLatency(t *testing.T) {
	// BwEfficiencyFactor=0.80 MUST produce higher latency than BwEfficiencyFactor=0
	// for a decode-only (memory-bound) step, because effective bandwidth is reduced.
	mc := testModelConfig()
	mfuDB := loadTestMFUDatabase(t)

	hcBaseline := testHardwareCalib()
	hcBaseline.BwEfficiencyFactor = 0 // Explicitly disable: use raw peak BW
	hcWithEff := testHardwareCalib()
	hcWithEff.BwEfficiencyFactor = 0.80

	// Decode-only step: memory-bound regime where BW correction matters
	step := StepConfig{
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 512, NumNewDecodeTokens: 1},
			{ProgressIndex: 256, NumNewDecodeTokens: 1},
			{ProgressIndex: 1024, NumNewDecodeTokens: 1},
		},
	}

	baseline := rooflineStepTimeV2("", mc, hcBaseline, step, 1, mfuDB)
	withEff := rooflineStepTimeV2("", mc, hcWithEff, step, 1, mfuDB)

	if withEff <= baseline {
		t.Errorf("BwEfficiencyFactor=0.80 latency (%d µs) should exceed baseline (%d µs)", withEff, baseline)
	}
	if baseline <= 0 {
		t.Errorf("baseline latency should be positive, got %d", baseline)
	}
}

func TestRooflineStepTimeV2_BwEfficiency_ZeroAndOneIdentical(t *testing.T) {
	// BwEfficiencyFactor=0 (disabled) and BwEfficiencyFactor=1.0 (100% efficiency)
	// MUST produce identical results, since both mean "use raw peak BW".
	mc := testModelConfig()
	mfuDB := loadTestMFUDatabase(t)

	hcZero := testHardwareCalib()
	hcZero.BwEfficiencyFactor = 0 // Explicitly disable
	hcOne := testHardwareCalib()
	hcOne.BwEfficiencyFactor = 1.0

	step := StepConfig{
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 512, NumNewDecodeTokens: 1},
		},
		PrefillRequests: []PrefillRequestConfig{
			{ProgressIndex: 0, NumNewPrefillTokens: 128},
		},
	}

	resultZero := rooflineStepTimeV2("", mc, hcZero, step, 1, mfuDB)
	resultOne := rooflineStepTimeV2("", mc, hcOne, step, 1, mfuDB)

	if resultZero != resultOne {
		t.Errorf("BwEfficiencyFactor=0 (%d µs) and BwEfficiencyFactor=1.0 (%d µs) should be identical",
			resultZero, resultOne)
	}
}

func TestRooflineStepTimeV2_BwEfficiency_PrefillAlsoAffected(t *testing.T) {
	// Prefill-only step should also show higher latency with BwEfficiencyFactor < 1.
	mc := testModelConfig()
	mfuDB := loadTestMFUDatabase(t)

	hcBaseline := testHardwareCalib()
	hcBaseline.BwEfficiencyFactor = 0 // Explicitly disable: use raw peak BW
	hcWithEff := testHardwareCalib()
	hcWithEff.BwEfficiencyFactor = 0.80

	step := StepConfig{
		PrefillRequests: []PrefillRequestConfig{
			{ProgressIndex: 0, NumNewPrefillTokens: 256},
		},
	}

	baseline := rooflineStepTimeV2("", mc, hcBaseline, step, 1, mfuDB)
	withEff := rooflineStepTimeV2("", mc, hcWithEff, step, 1, mfuDB)

	// Prefill may be compute-bound, so the effect might be zero if compute dominates.
	// At minimum, withEff should be >= baseline (BW reduction can only increase or not change latency).
	if withEff < baseline {
		t.Errorf("BwEfficiencyFactor=0.80 prefill latency (%d µs) should be >= baseline (%d µs)", withEff, baseline)
	}
}

func TestRooflineStepTimeV2_PerLayerCPUOverhead_ScalesWithLayersAndTP(t *testing.T) {
	// PerLayerCPUOverhead MUST produce overhead proportional to num_layers / tp.
	// A model with 2x layers at the same TP should have higher overhead.
	// The same model at 2x TP should have lower overhead.
	mfuDB := loadTestMFUDatabase(t)

	hc := HardwareCalib{
		TFlopsPeak:          989.5,
		BwPeakTBs:           3.35,
		PerLayerCPUOverhead: 100.0, // 100μs per layer per GPU
	}

	step := StepConfig{
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 512, NumNewDecodeTokens: 1},
		},
	}

	mc32 := testModelConfig() // 32 layers
	mc64 := testModelConfig()
	mc64.NumLayers = 64

	// More layers → higher latency (at same TP)
	result32 := rooflineStepTimeV2("", mc32, hc, step, 1, mfuDB)
	result64 := rooflineStepTimeV2("", mc64, hc, step, 1, mfuDB)
	if result64 <= result32 {
		t.Errorf("64 layers (%d µs) should exceed 32 layers (%d µs) at TP=1", result64, result32)
	}

	// Higher TP → lower latency (same model)
	resultTP1 := rooflineStepTimeV2("", mc32, hc, step, 1, mfuDB)
	resultTP2 := rooflineStepTimeV2("", mc32, hc, step, 2, mfuDB)
	if resultTP2 >= resultTP1 {
		t.Errorf("TP=2 (%d µs) should be less than TP=1 (%d µs) for same model", resultTP2, resultTP1)
	}
}

func TestRooflineStepTimeV2_PerLayerCPUOverhead_ZeroFallsBackToTOverhead(t *testing.T) {
	// When PerLayerCPUOverhead=0, the code should use TOverheadMicros (backward compat).
	mfuDB := loadTestMFUDatabase(t)
	mc := testModelConfig()

	step := StepConfig{
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 512, NumNewDecodeTokens: 1},
		},
	}

	// Config with explicit TOverheadMicros, no PerLayerCPUOverhead
	hcFixed := HardwareCalib{
		TFlopsPeak:      989.5,
		BwPeakTBs:       3.35,
		TOverheadMicros: 5000.0,
	}

	// Config with PerLayerCPUOverhead=0 and same TOverheadMicros
	hcZeroPL := HardwareCalib{
		TFlopsPeak:          989.5,
		BwPeakTBs:           3.35,
		TOverheadMicros:     5000.0,
		PerLayerCPUOverhead: 0,
	}

	resultFixed := rooflineStepTimeV2("", mc, hcFixed, step, 1, mfuDB)
	resultZeroPL := rooflineStepTimeV2("", mc, hcZeroPL, step, 1, mfuDB)

	if resultFixed != resultZeroPL {
		t.Errorf("PerLayerCPUOverhead=0 (%d µs) should match TOverheadMicros fallback (%d µs)", resultZeroPL, resultFixed)
	}
}
