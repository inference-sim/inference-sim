package latency

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
)

// --- Test helpers ---

var trainingFittedBetas = []float64{
	0.7726491335309499,  // β₁: prefill roofline correction
	1.127489556719325,   // β₂: decode roofline correction
	1.0559901872766853,  // β₃: weight loading correction
	0.0,                 // β₄: TP communication (zeroed)
	43.500541908701074,  // β₅: per-layer overhead (µs/layer)
	48.80613214319187,   // β₆: per-request scheduling (µs/req)
	0.0,                 // β₇: per-step overhead (zeroed)
}

var trainingFittedAlphas = []float64{
	9315.338771116985,  // α₀: API processing overhead
	1849.5902371340574, // α₁: post-decode fixed
	1.7079389122469397, // α₂: per-output-token
}

func makePrefillRequest(inputLen int, newTokens int) *sim.Request {
	return &sim.Request{
		InputTokens:   make([]int, inputLen),
		ProgressIndex: 0,
		NumNewTokens:  newTokens,
	}
}

func makeDecodeRequest(inputLen int, outputSoFar int) *sim.Request {
	return &sim.Request{
		InputTokens:   make([]int, inputLen),
		OutputTokens:  make([]int, outputSoFar),
		ProgressIndex: int64(inputLen + outputSoFar),
		NumNewTokens:  1,
	}
}

// newLlama7bModel constructs a TrainedRooflineLatencyModel for Llama-2-7b / H100 / TP=1.
func newLlama7bModel() *TrainedRooflineLatencyModel {
	return &TrainedRooflineLatencyModel{
		betaCoeffs:  trainingFittedBetas,
		alphaCoeffs: trainingFittedAlphas,
		numLayers:   32,
		hiddenDim:   4096,
		numHeads:    32,
		headDim:     128,
		dKV:         4096, // 32 * 128 (MHA: kv_heads=H)
		dFF:         11008,
		kEff:        1,
		numExperts:  0,
		isMoE:       false,
		tp:          1,
		flopsPeakUs: 989.5e6,
		bwHbmUs:     3.35e6,
	}
}

// --- BC-6: Clock safety ---

func TestTrainedRoofline_EmptyBatch_ReturnsOne(t *testing.T) {
	model := newLlama7bModel()
	assert.Equal(t, int64(1), model.StepTime(nil))
	assert.Equal(t, int64(1), model.StepTime([]*sim.Request{}))
}

// --- BC-3: Step-time formula produces positive results ---

func TestTrainedRoofline_PrefillOnly_PositiveStepTime(t *testing.T) {
	model := newLlama7bModel()
	batch := []*sim.Request{makePrefillRequest(512, 512)}
	stepTime := model.StepTime(batch)
	assert.Greater(t, stepTime, int64(1), "prefill step time should be > 1 µs")
}

func TestTrainedRoofline_DecodeOnly_PositiveStepTime(t *testing.T) {
	model := newLlama7bModel()
	batch := []*sim.Request{makeDecodeRequest(512, 100)}
	stepTime := model.StepTime(batch)
	assert.Greater(t, stepTime, int64(1), "decode step time should be > 1 µs")
}

func TestTrainedRoofline_MixedBatch_PositiveStepTime(t *testing.T) {
	model := newLlama7bModel()
	batch := []*sim.Request{
		makePrefillRequest(256, 256),
		makeDecodeRequest(512, 100),
	}
	stepTime := model.StepTime(batch)
	assert.Greater(t, stepTime, int64(1), "mixed batch step time should be > 1 µs")
}

// --- BC-7: QueueingTime = α₀ only ---

func TestTrainedRoofline_QueueingTime_IsAlpha0(t *testing.T) {
	model := &TrainedRooflineLatencyModel{
		alphaCoeffs: []float64{9315.0, 1850.0, 1.71},
	}
	// Constant regardless of input length
	req512 := makePrefillRequest(512, 512)
	req1024 := makePrefillRequest(1024, 1024)
	assert.Equal(t, int64(9315), model.QueueingTime(req512))
	assert.Equal(t, int64(9315), model.QueueingTime(req1024),
		"QueueingTime must be constant (α₀ only), independent of input length")
}

// --- BC-8: OutputTokenProcessingTime = α₂ ---

func TestTrainedRoofline_OutputTokenProcessingTime_IsAlpha2(t *testing.T) {
	model := &TrainedRooflineLatencyModel{
		alphaCoeffs: []float64{9315.0, 1850.0, 1.71},
	}
	assert.Equal(t, int64(1), model.OutputTokenProcessingTime())
}

// --- BC-15: PostDecodeFixedOverhead = α₁ ---

func TestTrainedRoofline_PostDecodeFixedOverhead_IsAlpha1(t *testing.T) {
	model := &TrainedRooflineLatencyModel{
		alphaCoeffs: []float64{9315.0, 1850.0, 1.71},
	}
	assert.Equal(t, int64(1850), model.PostDecodeFixedOverhead())
}

// --- BC-9: MoE-aware weight loading uses effective experts ---

func TestTrainedRoofline_MoE_WeightLoading_UsesEffectiveExperts(t *testing.T) {
	// β₃=1.0 (only weight loading nonzero) to isolate the effect.
	// MoE N=8, k=2: small batch → nEff=2, large batch → nEff=8.
	denseBetas := []float64{0, 0, 1.0, 0, 0, 0, 0}
	alphas := []float64{0, 0, 0}

	model := &TrainedRooflineLatencyModel{
		betaCoeffs:  denseBetas,
		alphaCoeffs: alphas,
		numLayers:   32,
		hiddenDim:   4096,
		numHeads:    32,
		headDim:     128,
		dKV:         1024, // 8 * 128 (GQA: kv_heads=8)
		dFF:         14336,
		kEff:        2,
		numExperts:  8,
		isMoE:       true,
		tp:          1,
		flopsPeakUs: 989.5e6,
		bwHbmUs:     3.35e6,
	}

	smallBatch := []*sim.Request{makeDecodeRequest(100, 10)}
	largeBatch := make([]*sim.Request, 10)
	for i := range largeBatch {
		largeBatch[i] = makeDecodeRequest(100, 10)
	}

	smallTime := model.StepTime(smallBatch)
	largeTime := model.StepTime(largeBatch)

	assert.Greater(t, largeTime, smallTime,
		"larger batch should load more MoE experts → higher weight loading time")
}

// --- BC-11: No MFU scaling — regression anchor with hand-computed values ---

func TestTrainedRoofline_NoMfuScaling_RegressionAnchor(t *testing.T) {
	// Architecture: 1 layer, d=128, H=1, kv_heads=1, d_h=128, d_kv=128, d_ff=1024, TP=1
	// 1 prefill token, s_i=1, t_i=1.
	//
	// Hand-computed FLOPs:
	// FLOPs_proj = 1 * 2 * 1 * 128 * (256 + 256) / 1 = 131072
	// FLOPs_attn = 1 * 4 * 1 * 1 * (1 + 0.5) * 128 = 768
	// FLOPs_ffn  = 1 * 1 * 1 * 6 * 128 * 1024 / 1 = 786432
	// Total = 918272
	// T_pf_compute = 918272 / 1e6 = 0.918272 µs
	// β₁=1.0 → step time = 0.918 → int64 = 0 → floored to 1
	//
	// If MFU (0.45) were applied: 918272 / 0.45e6 = 2.04 → int64 = 2
	betas := []float64{1.0, 0, 0, 0, 0, 0, 0}
	alphas := []float64{0, 0, 0}

	model := &TrainedRooflineLatencyModel{
		betaCoeffs:  betas,
		alphaCoeffs: alphas,
		numLayers:   1,
		hiddenDim:   128,
		numHeads:    1,
		headDim:     128,
		dKV:         128,
		dFF:         1024,
		kEff:        1,
		tp:          1,
		flopsPeakUs: 1.0e6,  // 1 TFLOP/s → easy calculation
		bwHbmUs:     1e12,   // very high BW → compute-bound
	}

	req := makePrefillRequest(1, 1)
	st := model.StepTime([]*sim.Request{req})

	// Without MFU: ~0.92 µs → floored to 1
	// With MFU (0.45): ~2.04 µs → int64 = 2
	assert.Equal(t, int64(1), st,
		"step time should be 1 (no MFU scaling); if MFU were applied it would be 2")
}

// --- Existing backends: PostDecodeFixedOverhead returns 0 ---

func TestExistingBackends_PostDecodeFixedOverhead_ReturnsZero(t *testing.T) {
	bb := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 200},
	}
	assert.Equal(t, int64(0), bb.PostDecodeFixedOverhead())

	rf := &RooflineLatencyModel{
		alphaCoeffs: []float64{100, 1, 200},
	}
	assert.Equal(t, int64(0), rf.PostDecodeFixedOverhead())

	cm := &CrossModelLatencyModel{
		alphaCoeffs: []float64{100, 1, 200},
	}
	assert.Equal(t, int64(0), cm.PostDecodeFixedOverhead())
}

// --- Benchmark: verify zero allocations in hot path ---

func BenchmarkTrainedRoofline_StepTime(b *testing.B) {
	model := newLlama7bModel()
	batch := make([]*sim.Request, 8)
	for i := range batch {
		if i < 2 {
			batch[i] = makePrefillRequest(512, 512)
		} else {
			batch[i] = makeDecodeRequest(512, 100)
		}
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = model.StepTime(batch)
	}
}

// --- Regression: verify non-negative for degenerate but valid inputs ---

func TestTrainedRoofline_StepTime_NeverNegative(t *testing.T) {
	model := newLlama7bModel()
	cases := []struct {
		name  string
		batch []*sim.Request
	}{
		{"nil", nil},
		{"empty", []*sim.Request{}},
		{"single prefill 1 token", []*sim.Request{makePrefillRequest(1, 1)}},
		{"single decode short context", []*sim.Request{makeDecodeRequest(10, 1)}},
		{"large batch", func() []*sim.Request {
			b := make([]*sim.Request, 256)
			for i := range b {
				b[i] = makeDecodeRequest(512, 200)
			}
			return b
		}()},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			st := model.StepTime(tc.batch)
			assert.GreaterOrEqual(t, st, int64(1),
				"StepTime must be >= 1 for all inputs (INV-3)")
		})
	}
}

// --- Verify OutputTokenProcessingTime rounds alpha2 correctly ---

func TestTrainedRoofline_OutputTokenProcessingTime_WithRealCoeffs(t *testing.T) {
	model := &TrainedRooflineLatencyModel{
		alphaCoeffs: trainingFittedAlphas,
	}
	// α₂ = 1.7079 → int64(1.7079) = 1 (truncation)
	assert.Equal(t, int64(1), model.OutputTokenProcessingTime())
}

// --- Verify PostDecodeFixedOverhead with real coefficients ---

func TestTrainedRoofline_PostDecodeFixedOverhead_WithRealCoeffs(t *testing.T) {
	model := &TrainedRooflineLatencyModel{
		alphaCoeffs: trainingFittedAlphas,
	}
	// α₁ = 1849.5902 → int64(1849.5902) = 1849
	assert.Equal(t, int64(1849), model.PostDecodeFixedOverhead())
}

// --- BC-2: Factory construction ---

func TestNewLatencyModel_TrainedRoofline_ReturnsModel(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs(trainingFittedBetas, trainingFittedAlphas)
	hw := sim.ModelHardwareConfig{
		Backend: "trained-roofline",
		ModelConfig: sim.ModelConfig{
			NumLayers: 32, HiddenDim: 4096, NumHeads: 32, NumKVHeads: 32,
			IntermediateDim: 11008, BytesPerParam: 2,
		},
		HWConfig: sim.HardwareCalib{TFlopsPeak: 989.5, BwPeakTBs: 3.35, MfuPrefill: 0.45, MfuDecode: 0.30, MemoryGiB: 80.0},
		TP:      1,
	}
	model, err := NewLatencyModel(coeffs, hw)
	assert.NoError(t, err)
	assert.NotNil(t, model)

	// Verify it produces positive step times
	batch := []*sim.Request{makePrefillRequest(512, 512)}
	assert.Greater(t, model.StepTime(batch), int64(1))
}

// --- BC-13: Coefficient length validation ---

func TestNewLatencyModel_TrainedRoofline_TooFewBetas_Error(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs([]float64{1, 2, 3, 4, 5, 6}, trainingFittedAlphas)
	hw := sim.ModelHardwareConfig{
		Backend: "trained-roofline",
		ModelConfig: sim.ModelConfig{
			NumLayers: 32, HiddenDim: 4096, NumHeads: 32, NumKVHeads: 32,
			IntermediateDim: 11008, BytesPerParam: 2,
		},
		HWConfig: sim.HardwareCalib{TFlopsPeak: 989.5, BwPeakTBs: 3.35, MfuPrefill: 0.45, MfuDecode: 0.30, MemoryGiB: 80.0},
		TP:      1,
	}
	_, err := NewLatencyModel(coeffs, hw)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "7 elements")
}

// --- BC-14: Config validation ---

func TestNewLatencyModel_TrainedRoofline_InvalidConfig_Error(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs(trainingFittedBetas, trainingFittedAlphas)

	baseHW := func() sim.ModelHardwareConfig {
		return sim.ModelHardwareConfig{
			Backend: "trained-roofline",
			ModelConfig: sim.ModelConfig{
				NumLayers: 32, HiddenDim: 4096, NumHeads: 32, NumKVHeads: 32,
				IntermediateDim: 11008, BytesPerParam: 2,
			},
			HWConfig: sim.HardwareCalib{TFlopsPeak: 989.5, BwPeakTBs: 3.35, MfuPrefill: 0.45, MfuDecode: 0.30, MemoryGiB: 80.0},
			TP:      1,
		}
	}

	tests := []struct {
		name   string
		modify func(*sim.ModelHardwareConfig)
		errMsg string
	}{
		{"zero TP", func(hw *sim.ModelHardwareConfig) { hw.TP = 0 }, "TP > 0"},
		{"zero NumLayers", func(hw *sim.ModelHardwareConfig) { hw.ModelConfig.NumLayers = 0 }, "NumLayers > 0"},
		{"zero NumHeads", func(hw *sim.ModelHardwareConfig) { hw.ModelConfig.NumHeads = 0 }, "NumHeads > 0"},
		{"zero HiddenDim", func(hw *sim.ModelHardwareConfig) { hw.ModelConfig.HiddenDim = 0 }, "HiddenDim > 0"},
		{"zero IntermediateDim", func(hw *sim.ModelHardwareConfig) { hw.ModelConfig.IntermediateDim = 0 }, "IntermediateDim > 0"},
		{"zero TFlopsPeak", func(hw *sim.ModelHardwareConfig) { hw.HWConfig.TFlopsPeak = 0 }, "TFlopsPeak"},
		{"zero BwPeakTBs", func(hw *sim.ModelHardwareConfig) { hw.HWConfig.BwPeakTBs = 0 }, "BwPeakTBs"},
		{"NumHeads not divisible by TP", func(hw *sim.ModelHardwareConfig) { hw.TP = 3 }, "divisible"},
		{"NumKVHeads not divisible by TP", func(hw *sim.ModelHardwareConfig) { hw.ModelConfig.NumKVHeads = 5; hw.TP = 2 }, "divisible"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			hw := baseHW()
			tc.modify(&hw)
			_, err := NewLatencyModel(coeffs, hw)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), tc.errMsg)
		})
	}
}

// --- BC-4: Prefill monotonicity ---

func TestTrainedRoofline_PrefillMonotonicity(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs(trainingFittedBetas, trainingFittedAlphas)
	hw := sim.ModelHardwareConfig{
		Backend: "trained-roofline",
		ModelConfig: sim.ModelConfig{
			NumLayers: 32, HiddenDim: 4096, NumHeads: 32, NumKVHeads: 32,
			IntermediateDim: 11008, BytesPerParam: 2,
		},
		HWConfig: sim.HardwareCalib{TFlopsPeak: 989.5, BwPeakTBs: 3.35, MfuPrefill: 0.45, MfuDecode: 0.30, MemoryGiB: 80.0},
		TP:      1,
	}
	model, err := NewLatencyModel(coeffs, hw)
	assert.NoError(t, err)

	tokenCounts := []int{64, 128, 256, 512, 1024}
	var prevTime int64
	for _, n := range tokenCounts {
		batch := []*sim.Request{makePrefillRequest(n, n)}
		st := model.StepTime(batch)
		assert.GreaterOrEqual(t, st, prevTime,
			"prefill step time should be non-decreasing: %d tokens -> %d us (prev %d us)", n, st, prevTime)
		prevTime = st
	}
}

// --- BC-5: Decode monotonicity ---

func TestTrainedRoofline_DecodeMonotonicity(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs(trainingFittedBetas, trainingFittedAlphas)
	hw := sim.ModelHardwareConfig{
		Backend: "trained-roofline",
		ModelConfig: sim.ModelConfig{
			NumLayers: 32, HiddenDim: 4096, NumHeads: 32, NumKVHeads: 32,
			IntermediateDim: 11008, BytesPerParam: 2,
		},
		HWConfig: sim.HardwareCalib{TFlopsPeak: 989.5, BwPeakTBs: 3.35, MfuPrefill: 0.45, MfuDecode: 0.30, MemoryGiB: 80.0},
		TP:      1,
	}
	model, err := NewLatencyModel(coeffs, hw)
	assert.NoError(t, err)

	var prevTime int64
	for nReqs := 1; nReqs <= 16; nReqs *= 2 {
		batch := make([]*sim.Request, nReqs)
		for i := range batch {
			batch[i] = makeDecodeRequest(512, 100)
		}
		st := model.StepTime(batch)
		assert.GreaterOrEqual(t, st, prevTime,
			"decode step time should be non-decreasing: %d reqs -> %d us (prev %d us)", nReqs, st, prevTime)
		prevTime = st
	}
}

// --- Verify no NaN/Inf propagation for extreme but valid inputs ---

func TestTrainedRoofline_StepTime_LargeContext_NoOverflow(t *testing.T) {
	model := newLlama7bModel()
	// Very large context (100K tokens) — tests float64 arithmetic doesn't overflow
	batch := []*sim.Request{makeDecodeRequest(100000, 50000)}
	st := model.StepTime(batch)
	assert.GreaterOrEqual(t, st, int64(1))
	// Step time should be a reasonable value (not overflowed to 1 from max(1, negative))
	assert.Greater(t, st, int64(100), "150K context should produce >100 µs step time")
}
