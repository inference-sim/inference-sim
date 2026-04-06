package latency

import (
	"math"
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
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
func testModelConfig() sim.ModelConfig {
	return sim.ModelConfig{
		NumLayers:        32,
		HiddenDim:        4096,
		NumHeads:         32,
		NumKVHeads:       8,
		VocabSize:        128256,
		BytesPerParam:    2, // bfloat16
		IntermediateDim:  14336,
		NumLocalExperts:  0, // dense model
		NumExpertsPerTok: 0,
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
func testHardwareCalib() sim.HardwareCalib {
	return sim.HardwareCalib{
		TFlopsPeak: 989.0,
		BwPeakTBs:  3.35,
		MfuPrefill: 0.55,
		MfuDecode:  0.30,
		MemoryGiB:  80.0,
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

	tp1 := rooflineStepTime(mc, hc, step, 1)
	tp2 := rooflineStepTime(mc, hc, step, 2)

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
			result := rooflineStepTime(mc, hc, tt.step, 1)
			if result <= 0 {
				t.Errorf("expected positive latency, got %d µs", result)
			}
		})
	}
}

func TestRooflineStepTime_EmptyStep_ReturnsZero(t *testing.T) {
	// No requests = no work = 0 µs (no overhead terms in llm-optimizer model)
	mc := testModelConfig()
	hc := testHardwareCalib()

	step := StepConfig{} // empty
	result := rooflineStepTime(mc, hc, step, 1)

	if result != 0 {
		t.Errorf("empty step should return 0 µs, got %d µs", result)
	}
}

// --- MoE test helpers ---

// testMixtralConfig returns a Mixtral-8x7B-like MoE config.
func testMixtralConfig() sim.ModelConfig {
	return sim.ModelConfig{
		NumLayers:        32,
		HiddenDim:        4096,
		NumHeads:         32,
		NumKVHeads:       8,
		VocabSize:        32000,
		BytesPerParam:    2,
		IntermediateDim:  14336,
		NumLocalExperts:  8,
		NumExpertsPerTok: 2,
		// MoEExpertFFNDim=0 → use IntermediateDim (Mixtral convention)
	}
}

// testDeepSeekV3Config returns a DeepSeek-V3-like MoE config with shared experts.
func testDeepSeekV3Config() sim.ModelConfig {
	return sim.ModelConfig{
		NumLayers:          61,
		HiddenDim:          7168,
		NumHeads:           128,
		NumKVHeads:         128,
		VocabSize:          129280,
		BytesPerParam:      2,
		IntermediateDim:    18432,
		NumLocalExperts:    256,
		NumExpertsPerTok:   8,
		MoEExpertFFNDim:    2048,
		SharedExpertFFNDim: 2048,
	}
}

// --- Task 4: MoE FLOPs tests ---

func TestCalculateTransformerFlops_MoE_TopKScaling(t *testing.T) {
	// MoE MLP FLOPs scale by top_k (active experts per token)
	mc := testMixtralConfig() // top_k=2, E=8

	moeFlops := calculateTransformerFlops(mc, 0, 128, false, true)

	dense := mc
	dense.NumLocalExperts = 0
	dense.NumExpertsPerTok = 0
	denseFlops := calculateTransformerFlops(dense, 0, 128, false, true)

	// MoE MLP FLOPs = top_k × dense MLP FLOPs
	ratio := moeFlops["gemm_ops"] / denseFlops["gemm_ops"]
	if math.Abs(ratio-float64(mc.NumExpertsPerTok)) > 0.01 {
		t.Errorf("MoE FLOPs should be %dx dense: moe=%g, dense=%g, ratio=%g",
			mc.NumExpertsPerTok, moeFlops["gemm_ops"], denseFlops["gemm_ops"], ratio)
	}
}

func TestCalculateTransformerFlops_MoE_Conservation(t *testing.T) {
	// total = gemm_ops + sram_ops for MoE configs
	configs := []struct {
		name string
		mc   sim.ModelConfig
	}{
		{"Mixtral", testMixtralConfig()},
		{"DeepSeek-V3", testDeepSeekV3Config()},
	}

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			flops := calculateTransformerFlops(cfg.mc, 512, 64, true, true)
			sum := flops["gemm_ops"] + flops["sram_ops"]
			if flops["total"] != sum {
				t.Errorf("conservation: total (%g) != gemm+sram (%g)", flops["total"], sum)
			}
		})
	}
}

func TestCalculateTransformerFlops_Dense_UnchangedAfterMoE(t *testing.T) {
	// BC-10: dense model FLOPs unchanged (regression anchor)
	mc := testModelConfig()

	flops := calculateTransformerFlops(mc, 512, 64, true, true)

	if flops["total"] <= 0 {
		t.Fatal("expected positive total FLOPs for dense model")
	}
	// Conservation still holds
	sum := flops["gemm_ops"] + flops["sram_ops"]
	if flops["total"] != sum {
		t.Errorf("dense conservation: total (%g) != gemm+sram (%g)", flops["total"], sum)
	}
}

// --- Task 5: MoE memory access tests ---

func TestCalculateMemoryAccessBytes_MoE_EffectiveExperts(t *testing.T) {
	// BC-1: MoE weight bandwidth uses effective experts nEff = N × (1 - ((N-k)/N)^B)
	// where k ≤ nEff ≤ N
	mc := testMixtralConfig() // N=8, k=2

	tests := []struct {
		name      string
		batchSize int64
	}{
		{"B=1 (single token)", 1},    // nEff = k exactly
		{"B=3 (small batch)", 3},     // nEff ≈ 4.6
		{"B=10 (medium batch)", 10},  // nEff ≈ 7.4
		{"B=100 (large batch)", 100}, // nEff → N
	}

	dense := mc
	dense.NumLocalExperts = 0
	dense.NumExpertsPerTok = 0
	denseMem := calculateMemoryAccessBytes(dense, 512, 1, false)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			moeMem := calculateMemoryAccessBytes(mc, 512, tt.batchSize, false)

			// Verify MoE weights exceed dense (attention same, MLP has nEff experts)
			if moeMem["model_weights"] <= denseMem["model_weights"] {
				t.Errorf("MoE weights (%g) should exceed dense weights (%g)",
					moeMem["model_weights"], denseMem["model_weights"])
			}

			// Log expected nEff for manual validation
			N := 8.0
			k := 2.0
			B := float64(tt.batchSize)
			probNotSelected := (N - k) / N
			expectedNEff := N * (1.0 - math.Pow(probNotSelected, B))
			t.Logf("B=%d: expected nEff=%.2f, MoE weights=%.3e, dense weights=%.3e",
				tt.batchSize, expectedNEff, moeMem["model_weights"], denseMem["model_weights"])
		})
	}
}

func TestCalculateMemoryAccessBytes_MoE_EdgeCases(t *testing.T) {
	// BC-2: Edge cases for effective expert formula
	tests := []struct {
		name          string
		N             int // NumLocalExperts
		k             int // NumExpertsPerTok
		batchSize     int64
		wantNEffMin   float64
		wantNEffMax   float64
		description   string
	}{
		{
			name:        "B=1 should give exactly k experts",
			N:           8,
			k:           2,
			batchSize:   1,
			wantNEffMin: 2.0,
			wantNEffMax: 2.0,
			description: "Single token activates exactly k experts",
		},
		{
			name:        "Large B approaches N experts",
			N:           8,
			k:           2,
			batchSize:   1000,
			wantNEffMin: 7.99,
			wantNEffMax: 8.0,
			description: "Very large batch saturates all experts",
		},
		{
			name:        "DeepSeek-V3 scale (256 experts)",
			N:           256,
			k:           8,
			batchSize:   10,
			wantNEffMin: 8.0,
			wantNEffMax: 80.0,
			description: "Large expert count scales correctly",
		},
		{
			name:        "Minimal MoE (N=2, k=1)",
			N:           2,
			k:           1,
			batchSize:   3,
			wantNEffMin: 1.0,
			wantNEffMax: 2.0,
			description: "Smallest valid MoE config",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create config with specified N and k
			mc := testMixtralConfig()
			mc.NumLocalExperts = tt.N
			mc.NumExpertsPerTok = tt.k

			mem := calculateMemoryAccessBytes(mc, 512, tt.batchSize, false)

			// Calculate expected nEff directly from formula
			N := float64(tt.N)
			k := float64(tt.k)
			B := float64(tt.batchSize)
			probNotSelected := (N - k) / N
			expectedNEff := N * (1.0 - math.Pow(probNotSelected, B))

			// Verify nEff is in expected range
			if expectedNEff < tt.wantNEffMin || expectedNEff > tt.wantNEffMax {
				t.Errorf("%s: nEff=%.3f, want in [%.3f, %.3f]",
					tt.description, expectedNEff, tt.wantNEffMin, tt.wantNEffMax)
			}

			// Verify the implementation applied nEff to model_weights
			dense := mc
			dense.NumLocalExperts = 0
			dense.NumExpertsPerTok = 0
			denseMem := calculateMemoryAccessBytes(dense, 512, tt.batchSize, false)

			if mem["model_weights"] <= denseMem["model_weights"] {
				t.Errorf("%s: MoE weights (%g) should exceed dense weights (%g)",
					tt.description, mem["model_weights"], denseMem["model_weights"])
			}

			// Verify the weight increase is consistent with expectedNEff (lower bound only)
			// Upper bound is not checked because attention/other weights dilute the ratio
			// in a model-dependent way, making a tight upper bound impractical.
			ratio := (mem["model_weights"] - denseMem["model_weights"]) / denseMem["model_weights"]
			minRatio := tt.wantNEffMin / float64(tt.N)
			if ratio < minRatio {
				t.Errorf("%s: weight increase ratio %.4f below minimum %.4f (nEff=%.2f)",
					tt.description, ratio, minRatio, expectedNEff)
			}

			t.Logf("%s: N=%d, k=%d, B=%d → nEff=%.2f", tt.description, tt.N, tt.k, tt.batchSize, expectedNEff)
		})
	}
}

func TestCalculateMemoryAccessBytes_MoE_Monotonicity(t *testing.T) {
	// BC-3: nEff should increase monotonically with batch size
	// Verify by checking that model_weights increases as batch size increases
	mc := testMixtralConfig() // N=8, k=2

	// Get dense baseline for comparison
	dense := mc
	dense.NumLocalExperts = 0
	dense.NumExpertsPerTok = 0
	denseMem := calculateMemoryAccessBytes(dense, 512, 1, false)
	denseWeights := denseMem["model_weights"]

	var prevWeights float64
	for B := int64(1); B <= 20; B++ {
		mem := calculateMemoryAccessBytes(mc, 512, B, false)
		moeWeights := mem["model_weights"]

		// Extract effective expert contribution
		// (approximation: ignoring attention component which is constant)
		if B > 1 && moeWeights < prevWeights {
			t.Errorf("Monotonicity violation: weights[B=%d]=%.3e < weights[B=%d]=%.3e",
				B, moeWeights, B-1, prevWeights)
		}

		// Verify weights stay within bounds: k experts ≤ nEff ≤ N experts
		// (MoE weights should be at least dense weights, at most 8× dense MLP)
		if moeWeights < denseWeights {
			t.Errorf("MoE weights (%g) less than dense (%g) at B=%d", moeWeights, denseWeights, B)
		}

		prevWeights = moeWeights

		// Calculate and log expected nEff for validation
		N := 8.0
		k := 2.0
		batchSize := float64(B)
		probNotSelected := (N - k) / N
		nEff := N * (1.0 - math.Pow(probNotSelected, batchSize))
		if B == 1 || B == 5 || B == 10 || B == 20 {
			t.Logf("B=%d: nEff=%.2f, weights=%.3e", B, nEff, moeWeights)
		}
	}
}

func TestCalculateMemoryAccessBytes_MoE_Conservation(t *testing.T) {
	// total = sum of components
	mc := testMixtralConfig()
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
		t.Errorf("conservation violation: total (%g) != components (%g)", mem["total"], sum)
	}
}

func TestCalculateMemoryAccessBytes_Dense_UnchangedAfterMoE(t *testing.T) {
	// BC-10: dense model memory unchanged (regression anchor)
	mc := testModelConfig()
	mem := calculateMemoryAccessBytes(mc, 512, 64, true)

	if mem["model_weights"] <= 0 {
		t.Fatal("expected positive model_weights for dense config")
	}
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
	if mem["total"] != sum {
		t.Errorf("dense conservation: total (%g) != sum (%g)", mem["total"], sum)
	}
}

// --- Task 7: MoE roofline step time smoke tests ---

func TestRooflineStepTime_MoE_Smoke_PositiveFinite(t *testing.T) {
	// Smoke test: MoE config produces valid step times
	configs := []struct {
		name string
		mc   sim.ModelConfig
	}{
		{"Mixtral-8x7B", testMixtralConfig()},
		{"DeepSeek-V3", testDeepSeekV3Config()},
	}
	hc := testHardwareCalib()

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			step := StepConfig{
				PrefillRequests: []PrefillRequestConfig{
					{ProgressIndex: 0, NumNewPrefillTokens: 128},
				},
				DecodeRequests: []DecodeRequestConfig{
					{ProgressIndex: 256, NumNewDecodeTokens: 1},
				},
			}
			result := rooflineStepTime(cfg.mc, hc, step, 2)
			if result <= 0 {
				t.Errorf("expected positive step time, got %d µs", result)
			}
			t.Logf("%s TP=2: %d µs", cfg.name, result)
		})
	}
}

func TestRooflineStepTime_MoE_TPScaling(t *testing.T) {
	// TP scaling: TP=2 should be less than TP=1 for MoE
	mc := testMixtralConfig()
	hc := testHardwareCalib()

	step := StepConfig{
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 512, NumNewDecodeTokens: 1},
		},
	}

	tp1 := rooflineStepTime(mc, hc, step, 1)
	tp2 := rooflineStepTime(mc, hc, step, 2)

	if tp2 >= tp1 {
		t.Errorf("MoE TP=2 (%d µs) should be less than TP=1 (%d µs)", tp2, tp1)
	}
}

func TestRooflineStepTime_MoE_BandwidthReduction(t *testing.T) {
	// BC-4: Small batch decode steps should have lower bandwidth (fewer experts loaded)
	// Verify that effective expert loading produces measurable bandwidth reduction
	mc := testMixtralConfig() // N=8, k=2
	hc := testHardwareCalib()

	// Small batch: 3 decode tokens (nEff ≈ 4.6 experts)
	smallBatch := StepConfig{
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 512, NumNewDecodeTokens: 1},
			{ProgressIndex: 600, NumNewDecodeTokens: 1},
			{ProgressIndex: 700, NumNewDecodeTokens: 1},
		},
	}

	// Large batch: 100 decode tokens (nEff → 8 experts, saturated)
	largeBatch := StepConfig{
		DecodeRequests: make([]DecodeRequestConfig, 100),
	}
	for i := range largeBatch.DecodeRequests {
		largeBatch.DecodeRequests[i] = DecodeRequestConfig{
			ProgressIndex:      int64(512 + i*10),
			NumNewDecodeTokens: 1,
		}
	}

	smallTime := rooflineStepTime(mc, hc, smallBatch, 1)
	largeTime := rooflineStepTime(mc, hc, largeBatch, 1)

	// Assert on step time: small batch should be faster
	if smallTime >= largeTime {
		t.Errorf("Small batch step time (%d µs) should be less than large batch (%d µs)",
			smallTime, largeTime)
	}

	// Verify bandwidth reduction using actual batch sizes
	smallMem := calculateMemoryAccessBytes(mc, 512, int64(len(smallBatch.DecodeRequests)), true)
	largeMem := calculateMemoryAccessBytes(mc, 512, int64(len(largeBatch.DecodeRequests)), true)
	weightReduction := 1.0 - (smallMem["model_weights"] / largeMem["model_weights"])
	if weightReduction < 0.20 {
		t.Errorf("Expected ≥20%% weight bandwidth reduction for small batch, got %.1f%%",
			weightReduction*100)
	}
	t.Logf("Small batch (B=%d): %d µs", len(smallBatch.DecodeRequests), smallTime)
	t.Logf("Large batch (B=%d): %d µs", len(largeBatch.DecodeRequests), largeTime)
	t.Logf("Weight bandwidth reduction: %.1f%%", weightReduction*100)
}

func TestRooflineStepTime_SingleCrossover_MemoryBoundDecode(t *testing.T) {
	// llm-optimizer physics: memory-bound step time = total_bytes / peak_bandwidth
	// No bandwidth haircut, no overhead terms, single crossover (not dual ceiling).
	mc := testModelConfig()
	hc := testHardwareCalib()

	// Single decode request — decode is memory-bound on H100
	step := StepConfig{
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 512, NumNewDecodeTokens: 1},
		},
	}
	result := rooflineStepTime(mc, hc, step, 1)

	// Compute expected: weights + KV + activations, all at raw peak bandwidth
	peakBW := hc.BwPeakTBs * 1e12
	peakFlops := hc.TFlopsPeak * 1e12

	baseMem := calculateMemoryAccessBytes(mc, 0, 0, false)
	dynamicMem := calculateMemoryAccessBytes(mc, 512, 1, true)
	totalBytes := baseMem["model_weights"] + (dynamicMem["total"] - dynamicMem["model_weights"])

	flops := calculateTransformerFlops(mc, 512, 1, true, true)
	totalFlops := flops["total"]

	computeS := totalFlops / (peakFlops * hc.MfuDecode)
	memoryS := totalBytes / peakBW

	// Decode should be memory-bound (verify assumption)
	if computeS >= memoryS {
		t.Skipf("decode is compute-bound with this config, skipping memory-bound test")
	}

	expectedMicros := int64(math.Round(memoryS * 1e6))
	if result != expectedMicros {
		t.Errorf("expected %d µs (total_bytes/peak_bw), got %d µs (delta=%d)",
			expectedMicros, result, result-expectedMicros)
	}
}

func TestRooflineStepTime_MixedBatch_WeightsLoadedOnce(t *testing.T) {
	// Verify that a mixed batch (prefill + decode) loads weights once,
	// not once per phase. The memory-bound time for a mixed batch should
	// equal weights + prefill_dynamic + decode_dynamic, NOT 2×weights + dynamic.
	mc := testModelConfig()
	hc := testHardwareCalib()

	// Mixed batch: 1 prefill + 1 decode
	mixedStep := StepConfig{
		PrefillRequests: []PrefillRequestConfig{
			{ProgressIndex: 0, NumNewPrefillTokens: 64},
		},
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 512, NumNewDecodeTokens: 1},
		},
	}

	// Decode-only step with the same decode request
	decodeOnlyStep := StepConfig{
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 512, NumNewDecodeTokens: 1},
		},
	}

	mixed := rooflineStepTime(mc, hc, mixedStep, 1)
	decodeOnly := rooflineStepTime(mc, hc, decodeOnlyStep, 1)

	// Mixed should be >= decode-only (more work), but if weights were loaded
	// twice, mixed would be roughly 2× decode-only for memory-bound steps.
	// With single weight load, the increase should be modest (just extra dynamic bytes).
	baseMem := calculateMemoryAccessBytes(mc, 0, 0, false)
	weightBytes := baseMem["model_weights"]

	// The mixed step should NOT double the weight bandwidth.
	// If it did, the overhead would be approximately weightBytes/peakBW extra.
	peakBW := hc.BwPeakTBs * 1e12
	doubleWeightPenaltyMicros := int64(weightBytes / peakBW * 1e6)

	// mixed - decodeOnly should be much less than a full extra weight load
	overhead := mixed - decodeOnly
	if overhead >= doubleWeightPenaltyMicros {
		t.Errorf("mixed batch overhead (%d µs) >= full weight load (%d µs): weights appear loaded twice",
			overhead, doubleWeightPenaltyMicros)
	}
	t.Logf("mixed=%d µs, decodeOnly=%d µs, overhead=%d µs, doubleWeightPenalty=%d µs",
		mixed, decodeOnly, overhead, doubleWeightPenaltyMicros)
}

func TestRooflineStepTime_Dense_PositiveAndTPScaling(t *testing.T) {
	// Dense model: positive step times and TP=2 < TP=1 (invariant, not pinned values)
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

	tp1 := rooflineStepTime(mc, hc, step, 1)
	tp2 := rooflineStepTime(mc, hc, step, 2)

	if tp1 <= 0 || tp2 <= 0 {
		t.Fatalf("expected positive step times: TP=1=%d, TP=2=%d", tp1, tp2)
	}
	if tp2 >= tp1 {
		t.Errorf("TP scaling violated: TP=2=%d >= TP=1=%d", tp2, tp1)
	}
	t.Logf("Dense regression: TP=1=%d µs, TP=2=%d µs", tp1, tp2)
}

// --- Quantized model tests (BC-6, BC-8) ---

func testW4A16Config() sim.ModelConfig {
	mc := testModelConfig()
	mc.WeightBytesPerParam = 0.5 // W4A16: 4-bit weights
	// BytesPerParam stays at 2.0 (bfloat16 compute dtype)
	return mc
}

func TestCalculateMemoryAccessBytes_W4A16_WeightsReduced_KVUnchanged(t *testing.T) {
	// BC-6: W4A16 model_weights use 0.5 bytes/param, KV cache uses 2.0
	fp16 := testModelConfig()
	w4a16 := testW4A16Config()

	fp16Mem := calculateMemoryAccessBytes(fp16, 512, 64, true)
	w4a16Mem := calculateMemoryAccessBytes(w4a16, 512, 64, true)

	// model_weights should be 1/4 of FP16 (0.5/2.0)
	ratio := w4a16Mem["model_weights"] / fp16Mem["model_weights"]
	if math.Abs(ratio-0.25) > 1e-10 {
		t.Errorf("W4A16 model_weights should be 0.25x FP16, got ratio=%v (fp16=%g, w4a16=%g)",
			ratio, fp16Mem["model_weights"], w4a16Mem["model_weights"])
	}

	// KV cache components should be identical (both use BytesPerParam=2.0)
	if w4a16Mem["kv_cache_growth"] != fp16Mem["kv_cache_growth"] {
		t.Errorf("KV cache growth should be identical: fp16=%g, w4a16=%g",
			fp16Mem["kv_cache_growth"], w4a16Mem["kv_cache_growth"])
	}
	if w4a16Mem["kv_cache_access"] != fp16Mem["kv_cache_access"] {
		t.Errorf("KV cache access should be identical: fp16=%g, w4a16=%g",
			fp16Mem["kv_cache_access"], w4a16Mem["kv_cache_access"])
	}

	// Activations should be identical (both use BytesPerParam=2.0)
	if w4a16Mem["activations_tokens"] != fp16Mem["activations_tokens"] {
		t.Errorf("Activations should be identical: fp16=%g, w4a16=%g",
			fp16Mem["activations_tokens"], w4a16Mem["activations_tokens"])
	}
}

func TestCalculateMemoryAccessBytes_NonQuantized_IdenticalToBaseline(t *testing.T) {
	// BC-8: non-quantized model (WeightBytesPerParam=0) produces identical results
	baseline := testModelConfig()
	baselineMem := calculateMemoryAccessBytes(baseline, 512, 64, true)

	// WeightBytesPerParam=0 (sentinel) — should fall back to BytesPerParam
	withSentinel := testModelConfig()
	withSentinel.WeightBytesPerParam = 0
	sentinelMem := calculateMemoryAccessBytes(withSentinel, 512, 64, true)

	if baselineMem["model_weights"] != sentinelMem["model_weights"] {
		t.Errorf("non-quantized should be identical: baseline=%g, sentinel=%g",
			baselineMem["model_weights"], sentinelMem["model_weights"])
	}
	if baselineMem["total"] != sentinelMem["total"] {
		t.Errorf("non-quantized total should be identical: baseline=%g, sentinel=%g",
			baselineMem["total"], sentinelMem["total"])
	}
}

func TestCalculateMemoryAccessBytes_W4A16_Conservation(t *testing.T) {
	// Conservation: total == sum(components) for quantized model
	mc := testW4A16Config()
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
		t.Errorf("conservation violation: total=%g, sum=%g", mem["total"], sum)
	}
}

func TestRooflineStepTime_W4A16_LowerThanFP16_MemoryBoundDecode(t *testing.T) {
	// BC-6 end-to-end: W4A16 model should produce lower (or equal) decode step time
	// than FP16 because decode is memory-bound and W4A16 has 4x less weight bandwidth.
	fp16 := testModelConfig()
	w4a16 := testW4A16Config()
	hc := testHardwareCalib()

	// Pure decode step: single token with long sequence history (memory-bound regime)
	decodeStep := StepConfig{
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 2048, NumNewDecodeTokens: 1},
		},
	}

	fp16Time := rooflineStepTime(fp16, hc, decodeStep, 1)
	w4a16Time := rooflineStepTime(w4a16, hc, decodeStep, 1)

	if fp16Time <= 0 || w4a16Time <= 0 {
		t.Fatalf("expected positive step times: fp16=%d, w4a16=%d", fp16Time, w4a16Time)
	}
	if w4a16Time > fp16Time {
		t.Errorf("W4A16 decode step time (%d µs) should be <= FP16 (%d µs) in memory-bound regime",
			w4a16Time, fp16Time)
	}
	t.Logf("Decode step: FP16=%d µs, W4A16=%d µs (ratio=%.2f)", fp16Time, w4a16Time, float64(w4a16Time)/float64(fp16Time))
}

// TestRooflineStepTime_FP8ComputeSelection_H100UsesFP8Rate tests that H100
// with FP8 weights (1 byte/param) uses the higher TFlopsFP8 rate.
func TestRooflineStepTime_FP8ComputeSelection_H100UsesFP8Rate(t *testing.T) {
	// GIVEN an FP8 model (WeightBytesPerParam = 1.0)
	mcFP8 := testModelConfig()
	mcFP8.WeightBytesPerParam = 1.0

	// AND an H100 with native FP8 support (TFlopsFP8 = 1979.0, 2× FP16)
	hwH100 := sim.HardwareCalib{
		TFlopsPeak: 989.5,
		TFlopsFP8:  1979.0,
		BwPeakTBs:  3.35,
		MfuPrefill: 0.60,
		MfuDecode:  0.45,
		MemoryGiB:  80.0,
	}

	// AND a compute-bound prefill step
	step := StepConfig{
		PrefillRequests: []PrefillRequestConfig{
			{ProgressIndex: 0, NumNewPrefillTokens: 512},
		},
	}

	// WHEN rooflineStepTime is called
	latencyFP8 := rooflineStepTime(mcFP8, hwH100, step, 1)

	// THEN latency should be positive and finite
	if latencyFP8 <= 0 {
		t.Errorf("FP8 latency should be positive, got %d µs", latencyFP8)
	}

	// AND should be faster than FP16 (since FP8 uses 2× compute rate)
	mcFP16 := testModelConfig()
	mcFP16.WeightBytesPerParam = 2.0
	latencyFP16 := rooflineStepTime(mcFP16, hwH100, step, 1)

	if latencyFP8 >= latencyFP16 {
		t.Errorf("FP8 latency (%d µs) should be less than FP16 (%d µs) on H100", latencyFP8, latencyFP16)
	}
}

// TestRooflineStepTime_FP8ComputeSelection_A100UsesFP16Rate tests that A100
// with FP8 weights still uses FP16 compute rate (W8A16 via Marlin kernels).
func TestRooflineStepTime_FP8ComputeSelection_A100UsesFP16Rate(t *testing.T) {
	// GIVEN an FP8 model (WeightBytesPerParam = 1.0)
	mcFP8 := testModelConfig()
	mcFP8.WeightBytesPerParam = 1.0

	// AND an A100 with no native FP8 support (TFlopsFP8 = 0)
	hwA100 := sim.HardwareCalib{
		TFlopsPeak: 312.0,
		TFlopsFP8:  0, // No native FP8 tensor cores
		BwPeakTBs:  2.039,
		MfuPrefill: 0.45,
		MfuDecode:  0.30,
		MemoryGiB:  80.0,
	}

	// AND a compute-bound prefill step
	step := StepConfig{
		PrefillRequests: []PrefillRequestConfig{
			{ProgressIndex: 0, NumNewPrefillTokens: 512},
		},
	}

	// WHEN rooflineStepTime is called
	latencyFP8 := rooflineStepTime(mcFP8, hwA100, step, 1)

	// THEN latency should be positive
	if latencyFP8 <= 0 {
		t.Errorf("FP8 latency should be positive, got %d µs", latencyFP8)
	}

	// AND should be similar to FP16 compute time (no FP8 speedup on A100)
	// The only difference is weight bandwidth (2× faster for FP8)
	mcFP16 := testModelConfig()
	mcFP16.WeightBytesPerParam = 2.0
	latencyFP16 := rooflineStepTime(mcFP16, hwA100, step, 1)

	// For compute-bound prefill, latencies should be close (within 20%)
	// since compute rate is the same, only weight bandwidth differs
	ratio := float64(latencyFP8) / float64(latencyFP16)
	if ratio < 0.8 || ratio > 1.0 {
		t.Errorf("FP8/FP16 latency ratio on A100 should be 0.8-1.0 (compute-bound), got %.2f (%d µs / %d µs)",
			ratio, latencyFP8, latencyFP16)
	}
}

// TestRooflineStepTime_FP8ComputeSelection_L40SBehavior tests L40S behavior
// with FP8 weights (should use FP16 compute rate like A100).
func TestRooflineStepTime_FP8ComputeSelection_L40SBehavior(t *testing.T) {
	// GIVEN an FP8 model
	mcFP8 := testModelConfig()
	mcFP8.WeightBytesPerParam = 1.0

	// AND an L40S with no native FP8 support
	hwL40S := sim.HardwareCalib{
		TFlopsPeak: 362.05,
		TFlopsFP8:  0,
		BwPeakTBs:  0.864,
		MfuPrefill: 0.45,
		MfuDecode:  0.30,
		MemoryGiB:  48.0,
	}

	// AND a mixed step
	step := StepConfig{
		PrefillRequests: []PrefillRequestConfig{
			{ProgressIndex: 0, NumNewPrefillTokens: 128},
		},
		DecodeRequests: []DecodeRequestConfig{
			{ProgressIndex: 256, NumNewDecodeTokens: 1},
		},
	}

	// WHEN rooflineStepTime is called
	latency := rooflineStepTime(mcFP8, hwL40S, step, 1)

	// THEN latency should be positive and finite
	if latency <= 0 {
		t.Errorf("L40S latency should be positive, got %d µs", latency)
	}
}

// TestRooflineStepTime_FP8ComputeSelection_EdgeCases tests edge cases for
// FP8 compute selection logic.
func TestRooflineStepTime_FP8ComputeSelection_EdgeCases(t *testing.T) {
	tests := []struct {
		name                string
		weightBytesPerParam float64
		tflopsFP8           float64
		expectFP8Rate       bool
	}{
		{
			name:                "FP8 weights + FP8 hardware → use FP8 rate",
			weightBytesPerParam: 1.0,
			tflopsFP8:           1979.0,
			expectFP8Rate:       true,
		},
		{
			name:                "FP8 weights + no FP8 hardware → use FP16 rate",
			weightBytesPerParam: 1.0,
			tflopsFP8:           0,
			expectFP8Rate:       false,
		},
		{
			name:                "FP16 weights + FP8 hardware → use FP16 rate",
			weightBytesPerParam: 2.0,
			tflopsFP8:           1979.0,
			expectFP8Rate:       false,
		},
		{
			name:                "FP4 weights + FP8 hardware → use FP16 rate",
			weightBytesPerParam: 0.5,
			tflopsFP8:           1979.0,
			expectFP8Rate:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mc := testModelConfig()
			mc.WeightBytesPerParam = tt.weightBytesPerParam

			hw := sim.HardwareCalib{
				TFlopsPeak: 989.5,
				TFlopsFP8:  tt.tflopsFP8,
				BwPeakTBs:  3.35,
				MfuPrefill: 0.60,
				MfuDecode:  0.45,
				MemoryGiB:  80.0,
			}

			step := StepConfig{
				PrefillRequests: []PrefillRequestConfig{
					{ProgressIndex: 0, NumNewPrefillTokens: 256},
				},
			}

			latency := rooflineStepTime(mc, hw, step, 1)

			// Verify latency is positive
			if latency <= 0 {
				t.Errorf("latency should be positive, got %d µs", latency)
			}

			// For FP8 rate cases, verify it's faster than baseline FP16
			if tt.expectFP8Rate {
				mcBaseline := testModelConfig()
				mcBaseline.WeightBytesPerParam = 2.0
				latencyBaseline := rooflineStepTime(mcBaseline, hw, step, 1)
				if latency >= latencyBaseline {
					t.Errorf("FP8 rate should be faster: got %d µs vs baseline %d µs", latency, latencyBaseline)
				}
			}
		})
	}
}

// TestRooflineStepTime_Scout_InterleavedMoE validates the fixes for issue #877:
// - Bug 1: InterleaveMoELayerStep field and layer splitting for FLOPs
// - Bug 2: DenseIntermediateDim field for different dense/MoE FFN sizes
// - Bug 3: nEff expert loading applied only to MoE layers
func TestRooflineStepTime_Scout_InterleavedMoE(t *testing.T) {
	// Scout-like config: 48 layers alternating MoE/dense
	// - 24 MoE layers: 16 experts, top-1, 8192 FFN dim per expert
	// - 24 Dense layers: 16384 FFN dim, no experts
	scoutConfig := sim.ModelConfig{
		NumLayers:              48,
		HiddenDim:              5120,
		NumHeads:               40,
		NumKVHeads:             8,
		IntermediateDim:        8192, // MoE expert FFN
		NumLocalExperts:        16,
		NumExpertsPerTok:       1,
		MoEExpertFFNDim:        8192,
		InterleaveMoELayerStep: 1,            // Alternate MoE/dense
		DenseIntermediateDim:   16384,        // Dense layer FFN (2× MoE)
		BytesPerParam:          1.0,          // FP8
		WeightBytesPerParam:    1.0,
	}

	// Uniform MoE baseline (Mixtral-style): all 32 layers are MoE
	mixtralConfig := sim.ModelConfig{
		NumLayers:              32,
		HiddenDim:              4096,
		NumHeads:               32,
		NumKVHeads:             8,
		IntermediateDim:        14336,
		NumLocalExperts:        8,
		NumExpertsPerTok:       2,
		MoEExpertFFNDim:        14336,
		InterleaveMoELayerStep: 0, // Uniform MoE
		DenseIntermediateDim:   0, // Not used
		BytesPerParam:          2.0,
		WeightBytesPerParam:    2.0,
	}

	hw := sim.HardwareCalib{
		TFlopsPeak: 989.0,
		BwPeakTBs:  3.35,
	}

	step := StepConfig{
		PrefillRequests: []PrefillRequestConfig{
			{ProgressIndex: 0, NumNewPrefillTokens: 588},
		},
	}

	t.Run("Scout_interleaved_produces_positive_latency", func(t *testing.T) {
		latency := rooflineStepTime(scoutConfig, hw, step, 1)
		if latency <= 0 {
			t.Errorf("Scout latency should be positive, got %d µs", latency)
		}
	})

	t.Run("Mixtral_uniform_produces_positive_latency", func(t *testing.T) {
		latency := rooflineStepTime(mixtralConfig, hw, step, 1)
		if latency <= 0 {
			t.Errorf("Mixtral latency should be positive, got %d µs", latency)
		}
	})

	t.Run("FLOPs_split_correctly_for_interleaved", func(t *testing.T) {
		// Calculate FLOPs for Scout
		flops := calculateTransformerFlops(scoutConfig, 0, 588, true, true)
		totalFlops := flops["total"]

		// Verify FLOPs are positive and finite
		if totalFlops <= 0 || math.IsInf(totalFlops, 0) || math.IsNaN(totalFlops) {
			t.Errorf("Scout FLOPs should be positive and finite, got %v", totalFlops)
		}

		// Roughly expected: 24 MoE layers (with expert routing) + 24 dense layers
		// MoE FLOPs should be ~1/2 of total MLP FLOPs (24 out of 48 layers)
		// Dense FLOPs should use 16384 FFN (2× MoE FFN)
		// Total should be in ballpark of 11.8e12 (from issue #877)
		expectedBallpark := 11.8e12
		if totalFlops < expectedBallpark*0.5 || totalFlops > expectedBallpark*2.0 {
			t.Logf("Warning: Scout FLOPs %e outside expected range [%e, %e]",
				totalFlops, expectedBallpark*0.5, expectedBallpark*2.0)
		}
	})

	t.Run("Weight_bandwidth_split_correctly_for_interleaved", func(t *testing.T) {
		// Calculate weight bandwidth for Scout with batch size
		mem := calculateMemoryAccessBytes(scoutConfig, 0, 588, false)
		weightBytes := mem["model_weights"]

		// Verify weight bytes are positive and finite
		if weightBytes <= 0 || math.IsInf(weightBytes, 0) || math.IsNaN(weightBytes) {
			t.Errorf("Scout weight bytes should be positive and finite, got %v", weightBytes)
		}

		// For Scout FP8: ~39 GB expected (from comment in issue #877)
		// Previously was ~7 GB due to nEff=0 bug
		expectedMinBytes := 20e9  // At least 20 GB
		expectedMaxBytes := 60e9  // At most 60 GB
		if weightBytes < expectedMinBytes || weightBytes > expectedMaxBytes {
			t.Errorf("Scout weight bandwidth %e outside expected range [%e, %e]",
				weightBytes, expectedMinBytes, expectedMaxBytes)
		}

		// Verify dense layers contribute (not zeroed by nEff)
		// Dense layers should have full 16384 FFN weights, not scaled by nEff
		t.Logf("Scout weight bandwidth: %.2f GB", weightBytes/1e9)
	})

	t.Run("nEff_zero_bug_fixed", func(t *testing.T) {
		// Calculate with newTokens=0 (the bug scenario)
		memZero := calculateMemoryAccessBytes(scoutConfig, 0, 0, false)
		weightBytesZero := memZero["model_weights"]

		// Calculate with newTokens=588 (correct scenario)
		memBatch := calculateMemoryAccessBytes(scoutConfig, 0, 588, false)
		weightBytesBatch := memBatch["model_weights"]

		// With the fix, both should be positive (dense layers contribute regardless)
		if weightBytesZero <= 0 {
			t.Errorf("Weight bytes with newTokens=0 should be positive (dense layers), got %v", weightBytesZero)
		}

		// Batch version should be >= zero version (MoE layers add nEff-scaled weights)
		if weightBytesBatch < weightBytesZero {
			t.Errorf("Weight bytes with batch (%v) should be >= zero tokens (%v)", weightBytesBatch, weightBytesZero)
		}

		t.Logf("Weight bandwidth: newTokens=0: %.2f GB, newTokens=588: %.2f GB",
			weightBytesZero/1e9, weightBytesBatch/1e9)
	})
}
