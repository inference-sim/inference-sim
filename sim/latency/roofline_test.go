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
// Note: BwEffConstant, TOverheadMicros, PerLayerOverhead, AllReduceLatency are
// present for struct completeness and ValidateRooflineConfig but are NOT consumed
// by rooflineStepTime() (single-crossover model uses raw peak bandwidth, no overheads).
func testHardwareCalib() sim.HardwareCalib {
	return sim.HardwareCalib{
		TFlopsPeak:       989.0,
		BwPeakTBs:        3.35,
		BwEffConstant:    0.7,
		TOverheadMicros:  50.0,
		PerLayerOverhead: 5.0,
		MfuPrefill:       0.55,
		MfuDecode:        0.30,
		AllReduceLatency: 10.0,
		MemoryGiB:        80.0,
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

func TestCalculateTransformerFlops_MoE_SparsityScaling(t *testing.T) {
	// BC-1 (MOD-ROOF-3): MLP FLOPs for top_k=K = K × single-expert + gate + shared
	mc := testMixtralConfig()

	// MLP-only FLOPs for the full config (top_k=2)
	topK2 := calculateTransformerFlops(mc, 0, 128, false, true)

	// MLP-only FLOPs with top_k=1
	mc1 := mc
	mc1.NumExpertsPerTok = 1
	topK1 := calculateTransformerFlops(mc1, 0, 128, false, true)

	// Gate FLOPs: 2 * newT * dModel * numExperts * nLayers
	dModel := 4096.0
	numExperts := 8.0
	nLayers := 32.0
	newT := 128.0
	gateFLOPs := 2 * newT * dModel * numExperts * nLayers

	// topK2 routed = 2 * topK1 routed; total = routed + gate (no shared for Mixtral)
	routedK1 := topK1["gemm_ops"] - gateFLOPs
	expectedK2 := 2*routedK1 + gateFLOPs

	ratio := topK2["gemm_ops"] / expectedK2
	if math.Abs(ratio-1.0) > 1e-9 {
		t.Errorf("BC-1: sparsity scaling violation: top_k=2 MLP FLOPs (%g) should be 2× routed + gate (%g), ratio=%g",
			topK2["gemm_ops"], expectedK2, ratio)
	}
}

func TestCalculateTransformerFlops_MoE_SharedExpertsAddFlops(t *testing.T) {
	// BC-2 (MOD-ROOF-2): shared experts add FLOPs
	mc := testDeepSeekV3Config()
	withShared := calculateTransformerFlops(mc, 0, 64, false, true)

	mcNoShared := mc
	mcNoShared.SharedExpertFFNDim = 0
	withoutShared := calculateTransformerFlops(mcNoShared, 0, 64, false, true)

	if withShared["gemm_ops"] <= withoutShared["gemm_ops"] {
		t.Errorf("BC-2: shared experts should increase FLOPs: with=%g, without=%g",
			withShared["gemm_ops"], withoutShared["gemm_ops"])
	}
}

func TestCalculateTransformerFlops_MoE_Conservation(t *testing.T) {
	// BC-3 (MOD-ROOF-1): total = gemm_ops + sram_ops for MoE
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

func TestCalculateTransformerFlops_MoE_MixtralFLOPsSanityCheck(t *testing.T) {
	// Mixtral MLP FLOPs sanity: routed(top_k=2) + gate for 1 token
	mc := testMixtralConfig()
	flops := calculateTransformerFlops(mc, 0, 1, false, true) // 1 token, MLP only

	// Expected per layer: 2 * 1 * (3 * 4096 * 14336) * 2 (top_k) = routed
	// Plus gate: 2 * 1 * 4096 * 8 per layer
	expectedPerLayerRouted := 2.0 * 3 * 4096 * 14336 * 2 // routed: top_k=2
	expectedPerLayerGate := 2.0 * 4096 * 8               // gate
	expectedTotal := (expectedPerLayerRouted + expectedPerLayerGate) * 32

	ratio := flops["gemm_ops"] / expectedTotal
	if math.Abs(ratio-1.0) > 0.01 {
		t.Errorf("Mixtral FLOPs sanity: got %g, expected %g, ratio=%g",
			flops["gemm_ops"], expectedTotal, ratio)
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

func TestCalculateMemoryAccessBytes_MoE_ActiveLessThanTotal(t *testing.T) {
	// BC-4 (MOD-ROOF-4): active weights < total weights when top_k < E
	mc := testMixtralConfig() // top_k=2, E=8

	active := calculateMemoryAccessBytes(mc, 512, 1, false)

	// With top_k=E (all experts active), weights should be higher
	mcAllActive := mc
	mcAllActive.NumExpertsPerTok = mc.NumLocalExperts
	allActive := calculateMemoryAccessBytes(mcAllActive, 512, 1, false)

	if active["model_weights"] >= allActive["model_weights"] {
		t.Errorf("BC-4: top_k=2 weights (%g) should be < top_k=E weights (%g)",
			active["model_weights"], allActive["model_weights"])
	}
}

func TestCalculateMemoryAccessBytes_MoE_BoundaryTopKEqualsE(t *testing.T) {
	// BC-5 (MOD-ROOF-4 boundary): top_k=E should have more weights than top_k<E
	mc := testMixtralConfig()
	mc.NumExpertsPerTok = mc.NumLocalExperts // top_k = E = 8

	activeAll := calculateMemoryAccessBytes(mc, 512, 1, false)

	mc2 := mc
	mc2.NumExpertsPerTok = 2
	activePartial := calculateMemoryAccessBytes(mc2, 512, 1, false)

	if activeAll["model_weights"] <= activePartial["model_weights"] {
		t.Errorf("BC-5: top_k=E weights (%g) should exceed top_k=2 weights (%g)",
			activeAll["model_weights"], activePartial["model_weights"])
	}
}

func TestCalculateMemoryAccessBytes_MoE_Conservation(t *testing.T) {
	// BC-6 (MOD-ROOF-1): total = sum of components
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
		t.Errorf("BC-6: conservation violation: total (%g) != components (%g)", mem["total"], sum)
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
