package latency

import (
	"math"
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestCalculateMemoryAccessBytes_Deterministic(t *testing.T) {
	config := testModelConfig()

	var firstTotal float64
	for i := 0; i < 100; i++ {
		result := calculateMemoryAccessBytes(config, 1024, 64, true)

		if i == 0 {
			firstTotal = result["total"]
		} else if result["total"] != firstTotal {
			t.Fatalf("non-deterministic total: call 0 got %v, call %d got %v", firstTotal, i, result["total"])
		}
	}

	if firstTotal <= 0 {
		t.Errorf("expected positive total, got %v", firstTotal)
	}

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
func testModelConfig() sim.ModelConfig {
	return sim.ModelConfig{
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
	mc := testModelConfig()

	attnOnly := calculateTransformerFlops(mc, 256, 64, true, false)
	both := calculateTransformerFlops(mc, 256, 64, true, true)

	if attnOnly["gemm_ops"] >= both["gemm_ops"] {
		t.Errorf("attention-only gemm_ops (%g) should be less than attn+mlp gemm_ops (%g)",
			attnOnly["gemm_ops"], both["gemm_ops"])
	}
	if attnOnly["sram_ops"] != both["sram_ops"] {
		t.Errorf("sram_ops should be identical with/without MLP: got %g vs %g",
			attnOnly["sram_ops"], both["sram_ops"])
	}
}

func TestCalculateTransformerFlops_MLPOnly_NoAttentionContribution(t *testing.T) {
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
	mc := testModelConfig()

	small := calculateTransformerFlops(mc, 512, 100, true, true)
	large := calculateTransformerFlops(mc, 512, 200, true, true)

	if large["total"] <= small["total"] {
		t.Errorf("200 tokens total FLOPs (%g) should exceed 100 tokens (%g)",
			large["total"], small["total"])
	}
	if large["gemm_ops"] <= small["gemm_ops"] {
		t.Errorf("200 tokens gemm_ops (%g) should exceed 100 tokens (%g)",
			large["gemm_ops"], small["gemm_ops"])
	}
}

func TestCalculateMemoryAccessBytes_Monotonicity_MoreTokensMoreBytes(t *testing.T) {
	mc := testModelConfig()

	small := calculateMemoryAccessBytes(mc, 512, 100, true)
	large := calculateMemoryAccessBytes(mc, 512, 200, true)

	if large["total"] <= small["total"] {
		t.Errorf("200 tokens total bytes (%g) should exceed 100 tokens (%g)",
			large["total"], small["total"])
	}
}

func TestCalculateMemoryAccessBytes_Conservation_TotalEqualsSumOfComponents(t *testing.T) {
	mc := testModelConfig()

	mem := calculateMemoryAccessBytes(mc, 512, 64, true)

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
func testHardwareCalib() sim.HardwareCalib {
	return sim.HardwareCalib{
		TFlopsPeak:          989.0,
		BwPeakTBs:           3.35,
		BwEfficiencyFactor:  0.82,
		PerLayerCPUOverhead: 100.0,
	}
}

// loadTestMFUDatabase loads the real bench_data for V2 roofline tests.
// Skips the test if bench_data is not available.
func loadTestMFUDatabase(t *testing.T) *sim.MFUDatabase {
	t.Helper()
	mc := testModelConfig()
	benchDataPath := "../../bench_data"
	db, err := sim.NewMFUDatabase(mc, benchDataPath, "h100")
	if err != nil {
		t.Skipf("bench_data not available, skipping V2 test: %v", err)
	}
	return db
}

func TestRooflineStepTime_BwEfficiency_DecodeBoundHigherLatency(t *testing.T) {
	mc := testModelConfig()
	mfuDB := loadTestMFUDatabase(t)

	hcBaseline := testHardwareCalib()
	hcBaseline.BwEfficiencyFactor = 0
	hcWithEff := testHardwareCalib()
	hcWithEff.BwEfficiencyFactor = 0.80

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
	mc := testModelConfig()
	mfuDB := loadTestMFUDatabase(t)

	hcZero := testHardwareCalib()
	hcZero.BwEfficiencyFactor = 0
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
	mc := testModelConfig()
	mfuDB := loadTestMFUDatabase(t)

	hcBaseline := testHardwareCalib()
	hcBaseline.BwEfficiencyFactor = 0
	hcWithEff := testHardwareCalib()
	hcWithEff.BwEfficiencyFactor = 0.80

	step := StepConfig{
		PrefillRequests: []PrefillRequestConfig{
			{ProgressIndex: 0, NumNewPrefillTokens: 256},
		},
	}

	baseline := rooflineStepTime("", mc, hcBaseline, step, 1, mfuDB)
	withEff := rooflineStepTime("", mc, hcWithEff, step, 1, mfuDB)

	if withEff < baseline {
		t.Errorf("BwEfficiencyFactor=0.80 prefill latency (%d µs) should be >= baseline (%d µs)", withEff, baseline)
	}
}

func TestRooflineStepTime_PerLayerCPUOverhead_ScalesWithLayersAndTP(t *testing.T) {
	mfuDB := loadTestMFUDatabase(t)

	hc := sim.HardwareCalib{
		TFlopsPeak:          989.5,
		BwPeakTBs:           3.35,
		PerLayerCPUOverhead: 100.0,
	}

	step := StepConfig{
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 512, NumNewDecodeTokens: 1},
		},
	}

	mc32 := testModelConfig() // 32 layers
	mc64 := testModelConfig()
	mc64.NumLayers = 64

	result32 := rooflineStepTime("", mc32, hc, step, 1, mfuDB)
	result64 := rooflineStepTime("", mc64, hc, step, 1, mfuDB)
	if result64 <= result32 {
		t.Errorf("64 layers (%d µs) should exceed 32 layers (%d µs) at TP=1", result64, result32)
	}

	resultTP1 := rooflineStepTime("", mc32, hc, step, 1, mfuDB)
	resultTP2 := rooflineStepTime("", mc32, hc, step, 2, mfuDB)
	if resultTP2 >= resultTP1 {
		t.Errorf("TP=2 (%d µs) should be less than TP=1 (%d µs) for same model", resultTP2, resultTP1)
	}
}

func TestRooflineStepTime_PerLayerCPUOverhead_ZeroMeansNoOverhead(t *testing.T) {
	mfuDB := loadTestMFUDatabase(t)
	mc := testModelConfig()

	step := StepConfig{
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 512, NumNewDecodeTokens: 1},
		},
	}

	hcZero := sim.HardwareCalib{
		TFlopsPeak: 989.5,
		BwPeakTBs:  3.35,
	}

	hcWithOverhead := sim.HardwareCalib{
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
