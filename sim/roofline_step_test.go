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
		PerLayerOverhead: 5.0,
		MfuPrefill:       0.55,
		MfuDecode:        0.30,
		AllReduceLatency: 10.0,
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

	tp1 := rooflineStepTime("H100", mc, hc, step, 1)
	tp2 := rooflineStepTime("H100", mc, hc, step, 2)

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
			result := rooflineStepTime("H100", mc, hc, tt.step, 1)
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
	result := rooflineStepTime("H100", mc, hc, step, 1)

	// Should be approximately TOverheadMicros (50) + layer overhead
	if result <= 0 {
		t.Errorf("empty step should still have overhead latency, got %d µs", result)
	}
}
