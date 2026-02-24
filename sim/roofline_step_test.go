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
		TFlopsPeak:          989.0,
		BwPeakTBs:           3.35,
		BwEfficiencyFactor:  0.82,
		PerLayerCPUOverhead: 100.0,
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

func TestRooflineStepTime_BwEfficiency_DecodeBoundHigherLatency(t *testing.T) {
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

	baseline := rooflineStepTime("", mc, hcBaseline, step, 1, mfuDB)
	withEff := rooflineStepTime("", mc, hcWithEff, step, 1, mfuDB)

	if withEff <= baseline {
		t.Errorf("BwEfficiencyFactor=0.80 latency (%d µs) should exceed baseline (%d µs)", withEff, baseline)
	}
	if baseline <= 0 {
		t.Errorf("baseline latency should be positive, got %d", baseline)
	}
}

func TestRooflineStepTime_BwEfficiency_ZeroAndOneIdentical(t *testing.T) {
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

	resultZero := rooflineStepTime("", mc, hcZero, step, 1, mfuDB)
	resultOne := rooflineStepTime("", mc, hcOne, step, 1, mfuDB)

	if resultZero != resultOne {
		t.Errorf("BwEfficiencyFactor=0 (%d µs) and BwEfficiencyFactor=1.0 (%d µs) should be identical",
			resultZero, resultOne)
	}
}

func TestRooflineStepTime_BwEfficiency_PrefillAlsoAffected(t *testing.T) {
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

	baseline := rooflineStepTime("", mc, hcBaseline, step, 1, mfuDB)
	withEff := rooflineStepTime("", mc, hcWithEff, step, 1, mfuDB)

	// Prefill may be compute-bound, so the effect might be zero if compute dominates.
	// At minimum, withEff should be >= baseline (BW reduction can only increase or not change latency).
	if withEff < baseline {
		t.Errorf("BwEfficiencyFactor=0.80 prefill latency (%d µs) should be >= baseline (%d µs)", withEff, baseline)
	}
}

func TestRooflineStepTime_PerLayerCPUOverhead_ScalesWithLayersAndTP(t *testing.T) {
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
	result32 := rooflineStepTime("", mc32, hc, step, 1, mfuDB)
	result64 := rooflineStepTime("", mc64, hc, step, 1, mfuDB)
	if result64 <= result32 {
		t.Errorf("64 layers (%d µs) should exceed 32 layers (%d µs) at TP=1", result64, result32)
	}

	// Higher TP → lower latency (same model)
	resultTP1 := rooflineStepTime("", mc32, hc, step, 1, mfuDB)
	resultTP2 := rooflineStepTime("", mc32, hc, step, 2, mfuDB)
	if resultTP2 >= resultTP1 {
		t.Errorf("TP=2 (%d µs) should be less than TP=1 (%d µs) for same model", resultTP2, resultTP1)
	}
}

func TestRooflineStepTime_PerLayerCPUOverhead_ZeroMeansNoOverhead(t *testing.T) {
	// When PerLayerCPUOverhead=0, no CPU overhead is added.
	// The result should equal pure hardware time (compute/memory).
	mfuDB := loadTestMFUDatabase(t)
	mc := testModelConfig()

	step := StepConfig{
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 512, NumNewDecodeTokens: 1},
		},
	}

	hcZero := HardwareCalib{
		TFlopsPeak: 989.5,
		BwPeakTBs:  3.35,
	}

	hcWithOverhead := HardwareCalib{
		TFlopsPeak:          989.5,
		BwPeakTBs:           3.35,
		PerLayerCPUOverhead: 100.0,
	}

	resultZero := rooflineStepTime("", mc, hcZero, step, 1, mfuDB)
	resultWith := rooflineStepTime("", mc, hcWithOverhead, step, 1, mfuDB)

	if resultWith <= resultZero {
		t.Errorf("PerLayerCPUOverhead=100 (%d µs) should exceed zero-overhead (%d µs)", resultWith, resultZero)
	}
}
