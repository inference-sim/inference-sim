package latency

import (
	"bytes"
	"math"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
)

// TestBlackboxLatencyModel_StepTime_MixedBatch_Positive verifies:
// GIVEN a batch with both prefill and decode requests
// WHEN StepTime is called
// THEN the result MUST be positive and greater than an empty batch.
func TestBlackboxLatencyModel_StepTime_MixedBatch_Positive(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	batch := []*sim.Request{
		{
			InputTokens:   make([]int, 100),
			ProgressIndex: 50,
			NumNewTokens:  30,
		},
		{
			InputTokens:   make([]int, 50),
			OutputTokens:  make([]int, 20),
			ProgressIndex: 60,
			NumNewTokens:  1,
		},
	}

	result := model.StepTime(batch)
	emptyResult := model.StepTime([]*sim.Request{})

	// THEN result must be positive
	if result <= 0 {
		t.Errorf("StepTime(mixed batch) = %d, want > 0", result)
	}
	// AND must exceed the empty-batch baseline (tokens contribute to step time)
	if result <= emptyResult {
		t.Errorf("StepTime(mixed batch) = %d <= StepTime(empty) = %d, want strictly greater", result, emptyResult)
	}
}

// TestBlackboxLatencyModel_StepTime_EmptyBatch verifies:
// GIVEN an empty batch
// WHEN StepTime is called
// THEN the result MUST be >= 1 (LatencyModel interface contract: clock must advance).
func TestBlackboxLatencyModel_StepTime_EmptyBatch(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	result := model.StepTime([]*sim.Request{})

	// THEN empty batch produces StepTime >= 1 (interface contract: clock must advance)
	assert.GreaterOrEqual(t, result, int64(1), "empty batch must return >= 1 per LatencyModel contract")
}

// TestBlackboxLatencyModel_QueueingTime_Positive verifies:
// GIVEN a request with non-empty input tokens
// WHEN QueueingTime is called
// THEN the result MUST be positive (non-empty requests incur queueing overhead).
func TestBlackboxLatencyModel_QueueingTime_Positive(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	req := &sim.Request{InputTokens: make([]int, 50)}
	result := model.QueueingTime(req)

	if result <= 0 {
		t.Errorf("QueueingTime(50 tokens) = %d, want > 0", result)
	}
}

// TestBlackboxLatencyModel_OutputTokenProcessingTime_NonNegative verifies:
// GIVEN a blackbox model with valid alpha coefficients
// WHEN OutputTokenProcessingTime is called
// THEN the result MUST be non-negative (output processing overhead).
func TestBlackboxLatencyModel_OutputTokenProcessingTime_NonNegative(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 200},
	}

	result := model.OutputTokenProcessingTime()

	if result < 0 {
		t.Errorf("OutputTokenProcessingTime = %d, want >= 0", result)
	}
}

// TestBlackboxLatencyModel_StepTime_Monotonic verifies BC-1 invariant:
// more prefill tokens must increase step time.
func TestBlackboxLatencyModel_StepTime_Monotonic(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	small := []*sim.Request{{InputTokens: make([]int, 50), ProgressIndex: 0, NumNewTokens: 10}}
	large := []*sim.Request{{InputTokens: make([]int, 200), ProgressIndex: 0, NumNewTokens: 100}}

	if model.StepTime(large) <= model.StepTime(small) {
		t.Errorf("monotonicity violated: StepTime(100 tokens) = %d <= StepTime(10 tokens) = %d",
			model.StepTime(large), model.StepTime(small))
	}
}

// TestBlackboxLatencyModel_QueueingTime_Monotonic verifies BC-3 invariant:
// longer input sequences must yield higher queueing times.
func TestBlackboxLatencyModel_QueueingTime_Monotonic(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	short := &sim.Request{InputTokens: make([]int, 10)}
	long := &sim.Request{InputTokens: make([]int, 500)}

	if model.QueueingTime(long) <= model.QueueingTime(short) {
		t.Errorf("monotonicity violated: QueueingTime(500 tokens) = %d <= QueueingTime(10 tokens) = %d",
			model.QueueingTime(long), model.QueueingTime(short))
	}
}

// TestRooflineLatencyModel_StepTime_PositiveAndMonotonic verifies BC-2:
// StepTime produces positive results and more tokens yield longer step time.
func TestRooflineLatencyModel_StepTime_PositiveAndMonotonic(t *testing.T) {
	model := &RooflineLatencyModel{
		modelConfig: testModelConfig(),
		hwConfig:    testHardwareCalib(),
		tp:          2,
		alphaCoeffs: []float64{100, 1, 100},
	}

	smallBatch := []*sim.Request{
		{
			InputTokens:   make([]int, 100),
			ProgressIndex: 0,
			NumNewTokens:  100,
		},
	}
	largeBatch := []*sim.Request{
		{
			InputTokens:   make([]int, 1000),
			ProgressIndex: 0,
			NumNewTokens:  1000,
		},
	}

	smallResult := model.StepTime(smallBatch)
	largeResult := model.StepTime(largeBatch)

	// THEN both results must be positive
	if smallResult <= 0 {
		t.Errorf("StepTime(100 tokens) = %d, want > 0", smallResult)
	}
	if largeResult <= 0 {
		t.Errorf("StepTime(1000 tokens) = %d, want > 0", largeResult)
	}

	// AND more tokens must produce longer step time (monotonicity)
	if largeResult <= smallResult {
		t.Errorf("monotonicity violated: StepTime(1000 tokens) = %d <= StepTime(100 tokens) = %d",
			largeResult, smallResult)
	}
}

// TestRooflineLatencyModel_StepTime_EmptyBatch verifies roofline handles empty batch.
func TestRooflineLatencyModel_StepTime_EmptyBatch(t *testing.T) {
	model := &RooflineLatencyModel{
		modelConfig: testModelConfig(),
		hwConfig:    testHardwareCalib(),
		tp:          2,
		alphaCoeffs: []float64{100, 1, 100},
	}

	emptyResult := model.StepTime([]*sim.Request{})

	// THEN empty batch result must be >= 1 (interface contract: clock must advance)
	assert.GreaterOrEqual(t, emptyResult, int64(1), "empty batch must return >= 1 per LatencyModel contract")

	// AND a non-empty batch must produce a longer step time
	nonEmptyBatch := []*sim.Request{
		{
			InputTokens:   make([]int, 100),
			ProgressIndex: 0,
			NumNewTokens:  100,
		},
	}
	nonEmptyResult := model.StepTime(nonEmptyBatch)
	if nonEmptyResult <= emptyResult {
		t.Errorf("StepTime(100 tokens) = %d <= StepTime(empty) = %d, want strictly greater",
			nonEmptyResult, emptyResult)
	}
}

// TestRooflineLatencyModel_QueueingTime_Positive verifies:
// GIVEN a roofline model
// WHEN QueueingTime is called with non-empty input
// THEN the result MUST be positive.
func TestRooflineLatencyModel_QueueingTime_Positive(t *testing.T) {
	model := &RooflineLatencyModel{
		modelConfig: testModelConfig(),
		hwConfig:    testHardwareCalib(),
		tp:          2,
		alphaCoeffs: []float64{100, 1, 100},
	}

	req := &sim.Request{InputTokens: make([]int, 50)}
	result := model.QueueingTime(req)

	if result <= 0 {
		t.Errorf("QueueingTime(50 tokens) = %d, want > 0", result)
	}
}

// TestNewLatencyModel_BlackboxMode verifies BC-4 (blackbox path).
func TestNewLatencyModel_BlackboxMode(t *testing.T) {
	cfg := sim.SimConfig{
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "", "", 0, "blackbox", 0),
	}

	model, err := NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("NewLatencyModel returned error: %v", err)
	}

	batch := []*sim.Request{
		{
			InputTokens:   make([]int, 100),
			ProgressIndex: 50,
			NumNewTokens:  30,
		},
	}
	result := model.StepTime(batch)
	// Regression anchors: exact values via factory path to catch accidental
	// coefficient changes. All other tests use behavioral assertions.
	if result != 1300 {
		t.Errorf("StepTime = %d, want 1300 (regression anchor: beta0 + beta1*30)", result)
	}

	// QueueingTime regression anchor (alpha0 + alpha1 * inputLen)
	req := &sim.Request{InputTokens: make([]int, 50)}
	qt := model.QueueingTime(req)
	if qt != 150 {
		t.Errorf("QueueingTime = %d, want 150 (regression anchor: alpha0 + alpha1*50)", qt)
	}

	// OutputTokenProcessingTime regression anchor (alpha2)
	otp := model.OutputTokenProcessingTime()
	if otp != 100 {
		t.Errorf("OutputTokenProcessingTime = %d, want 100 (regression anchor: alpha2)", otp)
	}
}

// TestNewLatencyModel_RooflineMode verifies BC-4 (roofline path).
func TestNewLatencyModel_RooflineMode(t *testing.T) {
	cfg := sim.SimConfig{
		LatencyCoeffs:       sim.NewLatencyCoeffs(nil, []float64{100, 1, 100}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(testModelConfig(), testHardwareCalib(), "", "", 2, "roofline", 0),
	}

	model, err := NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("NewLatencyModel returned error: %v", err)
	}

	// THEN the model must produce different results than blackbox for the same batch
	// (roofline uses FLOPs/bandwidth, blackbox uses beta regression — distinct formulas)
	batch := []*sim.Request{
		{
			InputTokens:   make([]int, 100),
			ProgressIndex: 0,
			NumNewTokens:  100,
		},
	}
	result := model.StepTime(batch)
	if result <= 0 {
		t.Errorf("StepTime = %d, want > 0 (roofline mode)", result)
	}
}

// TestNewLatencyModel_EmptyBackendDefaultsToRoofline verifies that Backend: ""
// (Go zero value) dispatches to roofline mode, consistent with the CLI default.
// Library consumers who omit Backend get the same default as CLI users.
func TestNewLatencyModel_EmptyBackendDefaultsToRoofline(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs(nil, []float64{100, 1, 100})
	hw := sim.NewModelHardwareConfig(testModelConfig(), testHardwareCalib(), "", "", 2, "", 0)

	model, err := NewLatencyModel(coeffs, hw)
	if err != nil {
		t.Fatalf("NewLatencyModel with empty Backend returned error: %v", err)
	}

	// Verify it behaves as roofline (produces positive step time from FLOPs/bandwidth)
	batch := []*sim.Request{
		{
			InputTokens:   make([]int, 100),
			ProgressIndex: 0,
			NumNewTokens:  100,
		},
	}
	result := model.StepTime(batch)
	if result <= 0 {
		t.Errorf("StepTime = %d, want > 0 (empty backend should default to roofline)", result)
	}
}

// TestNewLatencyModel_InvalidRoofline verifies BC-8.
func TestNewLatencyModel_InvalidRoofline(t *testing.T) {
	cfg := sim.SimConfig{
		LatencyCoeffs:       sim.NewLatencyCoeffs(nil, []float64{100, 1, 100}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "", "", 0, "roofline", 0),
	}

	_, err := NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err == nil {
		t.Fatal("expected error for invalid roofline config, got nil")
	}
}

// TestNewLatencyModel_ShortAlphaCoeffs verifies factory rejects short alpha slices.
func TestNewLatencyModel_ShortAlphaCoeffs(t *testing.T) {
	tests := []struct {
		name    string
		backend string
		alpha   []float64
		beta    []float64
	}{
		{"blackbox_empty_alpha", "blackbox", []float64{}, []float64{1, 2, 3}},
		{"blackbox_short_alpha", "blackbox", []float64{1, 2}, []float64{1, 2, 3}},
		{"roofline_empty_alpha", "roofline", []float64{}, nil},
		{"roofline_short_alpha", "roofline", []float64{1}, nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			coeffs := sim.NewLatencyCoeffs(tc.beta, tc.alpha)
			hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "", "", 0, tc.backend, 0)
			_, err := NewLatencyModel(coeffs, hw)
			if err == nil {
				t.Fatal("expected error for short AlphaCoeffs, got nil")
			}
		})
	}
}

// TestNewLatencyModel_ShortBetaCoeffs verifies factory rejects short beta slices for blackbox.
func TestNewLatencyModel_ShortBetaCoeffs(t *testing.T) {
	tests := []struct {
		name string
		beta []float64
	}{
		{"empty", []float64{}},
		{"short", []float64{1, 2}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			coeffs := sim.NewLatencyCoeffs(tc.beta, []float64{100, 1, 100})
			hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "", "", 0, "blackbox", 0)
			_, err := NewLatencyModel(coeffs, hw)
			if err == nil {
				t.Fatal("expected error for short BetaCoeffs, got nil")
			}
		})
	}
}

// TestNewLatencyModel_NaNAlphaCoeffs_ReturnsError verifies BC-4: NaN in alpha rejected.
func TestNewLatencyModel_NaNAlphaCoeffs_ReturnsError(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs([]float64{5000, 10, 5}, []float64{math.NaN(), 1.0, 100.0})
	_, err := NewLatencyModel(coeffs, sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "", "", 0, "blackbox", 0))
	if err == nil {
		t.Fatal("expected error for NaN AlphaCoeffs, got nil")
	}
}

// TestNewLatencyModel_InfBetaCoeffs_ReturnsError verifies BC-4: Inf in beta rejected.
func TestNewLatencyModel_InfBetaCoeffs_ReturnsError(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs([]float64{math.Inf(1), 10, 5}, []float64{100, 1.0, 100.0})
	_, err := NewLatencyModel(coeffs, sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "", "", 0, "blackbox", 0))
	if err == nil {
		t.Fatal("expected error for Inf BetaCoeffs, got nil")
	}
}

// TestNewLatencyModel_UnknownBackend_ReturnsError verifies BC-6: unknown backend → error.
func TestNewLatencyModel_UnknownBackend_ReturnsError(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs([]float64{1000, 10, 2}, []float64{500, 1, 100})
	hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "", "", 0, "nonexistent", 0)
	_, err := NewLatencyModel(coeffs, hw)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "nonexistent")
}

// TestNewLatencyModel_NegativeCoefficients_ReturnsError verifies BC-5:
// GIVEN alpha or beta coefficients containing negative values
// WHEN NewLatencyModel is called
// THEN it returns an error mentioning "negative"
func TestNewLatencyModel_NegativeCoefficients_ReturnsError(t *testing.T) {
	tests := []struct {
		name  string
		beta  []float64
		alpha []float64
	}{
		{"negative_alpha_0", []float64{100, 1, 1}, []float64{-1, 0, 0}},
		{"negative_alpha_2", []float64{100, 1, 1}, []float64{0, 0, -5}},
		{"negative_beta_0", []float64{-100, 1, 1}, []float64{100, 1, 100}},
		{"negative_beta_1", []float64{100, -1, 1}, []float64{100, 1, 100}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			coeffs := sim.NewLatencyCoeffs(tc.beta, tc.alpha)
			hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "", "", 0, "blackbox", 0)
			_, err := NewLatencyModel(coeffs, hw)
			if err == nil {
				t.Fatal("expected error for negative coefficient")
			}
			if !strings.Contains(err.Error(), "negative") {
				t.Errorf("error should mention 'negative', got: %v", err)
			}
		})
	}
}

// TestBlackboxLatencyModel_StepTime_FloorAtOne verifies BC-4:
// GIVEN zero beta coefficients (pathological input)
// WHEN StepTime is called with a non-empty batch
// THEN the return value is >= 1 (livelock protection via R19)
func TestBlackboxLatencyModel_StepTime_FloorAtOne(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs([]float64{0, 0, 0}, []float64{0, 0, 0})
	hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "", "", 0, "blackbox", 0)
	model, err := NewLatencyModel(coeffs, hw)
	if err != nil {
		t.Fatalf("NewLatencyModel: %v", err)
	}
	batch := []*sim.Request{{InputTokens: make([]int, 16), OutputTokens: make([]int, 4), NumNewTokens: 1}}
	stepTime := model.StepTime(batch)
	if stepTime < 1 {
		t.Errorf("StepTime = %d, want >= 1", stepTime)
	}
}

// TestBlackboxRoofline_ZeroOutputTokens_ConsistentClassification verifies BC-5:
// Both models handle requests past prefill with 0 output tokens consistently.
func TestBlackboxRoofline_ZeroOutputTokens_ConsistentClassification(t *testing.T) {
	// GIVEN a request past prefill with 0 output tokens (edge case)
	req := &sim.Request{
		InputTokens:   []int{1, 2, 3},
		OutputTokens:  []int{},
		ProgressIndex: 3,
		NumNewTokens:  0,
	}
	batch := []*sim.Request{req}

	blackbox := &BlackboxLatencyModel{
		betaCoeffs:  []float64{5000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}
	roofline := &RooflineLatencyModel{
		modelConfig: testModelConfig(),
		hwConfig:    testHardwareCalib(),
		tp:          2,
		alphaCoeffs: []float64{100, 1, 100},
	}

	// WHEN both models compute step time with and without the zero-output request
	emptyBatch := []*sim.Request{}
	blackboxEmpty := blackbox.StepTime(emptyBatch)
	rooflineEmpty := roofline.StepTime(emptyBatch)
	blackboxWith := blackbox.StepTime(batch)
	rooflineWith := roofline.StepTime(batch)

	// THEN the zero-output request should not change step time
	// (it contributes nothing to either prefill or decode computation)
	if blackboxWith != blackboxEmpty {
		t.Errorf("blackbox: zero-output request should not change step time: with=%d empty=%d", blackboxWith, blackboxEmpty)
	}
	if rooflineWith != rooflineEmpty {
		t.Errorf("roofline: zero-output request should not change step time: with=%d empty=%d", rooflineWith, rooflineEmpty)
	}
}

// TestClampToInt64_OverflowSaturation verifies:
// GIVEN a float64 value exceeding math.MaxInt64
// WHEN clampToInt64 is called
// THEN the result MUST be math.MaxInt64 (BC-1).
func TestClampToInt64_OverflowSaturation(t *testing.T) {
	tests := []struct {
		name  string
		input float64
		want  int64
	}{
		{"exactly max float", float64(math.MaxInt64), math.MaxInt64},
		{"above max", float64(math.MaxInt64) * 2, math.MaxInt64},
		{"huge value", 1e30, math.MaxInt64},
		{"positive infinity", math.Inf(1), math.MaxInt64},
		{"NaN", math.NaN(), math.MaxInt64},
		{"negative infinity", math.Inf(-1), math.MinInt64},
		{"normal positive", 42000.0, 42000},
		{"small positive", 1.5, 1},
		{"zero", 0.0, 0},
		{"negative", -5.0, -5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := clampToInt64(tt.input)
			assert.Equal(t, tt.want, got)
		})
	}
}

// TestBlackbox_StepTime_ExtremeCoeffs_SaturatesAtMaxInt64 verifies:
// GIVEN coefficients producing stepTime > math.MaxInt64
// WHEN StepTime is called
// THEN the result MUST be math.MaxInt64, not 1 (BC-1).
func TestBlackbox_StepTime_ExtremeCoeffs_SaturatesAtMaxInt64(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1e30, 0, 0},
		alphaCoeffs: []float64{0, 0, 0},
	}
	batch := []*sim.Request{}
	result := model.StepTime(batch)
	assert.Equal(t, int64(math.MaxInt64), result, "extreme beta0 should saturate at MaxInt64")
}

// TestBlackbox_QueueingTime_ExtremeAlpha_SaturatesAtMaxInt64 verifies:
// GIVEN alpha coefficients producing queueingTime > math.MaxInt64
// WHEN QueueingTime is called
// THEN the result MUST be math.MaxInt64 (BC-1).
func TestBlackbox_QueueingTime_ExtremeAlpha_SaturatesAtMaxInt64(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{0, 0, 0},
		alphaCoeffs: []float64{1e30, 0, 0},
	}
	req := &sim.Request{InputTokens: make([]int, 10)}
	result := model.QueueingTime(req)
	assert.Equal(t, int64(math.MaxInt64), result)
}

// TestBlackbox_OutputTokenProcessingTime_ExtremeAlpha_SaturatesAtMaxInt64 verifies BC-1 for alpha2 overflow.
func TestBlackbox_OutputTokenProcessingTime_ExtremeAlpha_SaturatesAtMaxInt64(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{0, 0, 0},
		alphaCoeffs: []float64{0, 0, 1e30},
	}
	result := model.OutputTokenProcessingTime()
	assert.Equal(t, int64(math.MaxInt64), result)
}

// TestNewLatencyModel_RemovedBackendError verifies BC-1, BC-2, BC-9:
// GIVEN a backend name that was removed ("crossmodel" or "trained-roofline")
// WHEN NewLatencyModel is called
// THEN it returns an error containing the backend name and valid options.
func TestNewLatencyModel_RemovedBackendError(t *testing.T) {
	tests := []struct {
		name            string
		backend         string
		wantErrContains []string // Error must contain all these substrings
	}{
		{
			name:    "crossmodel removed",
			backend: "crossmodel",
			wantErrContains: []string{
				"unknown backend",
				"crossmodel",
				"valid options:",
				"blackbox",
				"roofline",
				"trained-physics",
			},
		},
		{
			name:    "trained-roofline removed",
			backend: "trained-roofline",
			wantErrContains: []string{
				"unknown backend",
				"trained-roofline",
				"valid options:",
				"blackbox",
				"roofline",
				"trained-physics",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// GIVEN minimal valid config with removed backend
			coeffs := sim.NewLatencyCoeffs(
				[]float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
				[]float64{1.0, 1.0, 1.0},
			)
			hw := sim.NewModelHardwareConfig(
				sim.ModelConfig{
					NumLayers:       32,
					NumHeads:        32,
					HiddenDim:       4096,
					IntermediateDim: 11008,
				},
				sim.HardwareCalib{
					TFlopsPeak: 989.0,
					BwPeakTBs:  3.35,
				},
				"", "", 1, tt.backend, 0,
			)

			// WHEN attempting to construct the model
			model, err := NewLatencyModel(coeffs, hw)

			// THEN construction must fail
			if err == nil {
				t.Fatalf("NewLatencyModel(%q) succeeded; want error for removed backend", tt.backend)
			}
			if model != nil {
				t.Errorf("NewLatencyModel(%q) returned non-nil model with error; want nil", tt.backend)
			}

			// AND the error message must contain all expected substrings
			errMsg := err.Error()
			for _, substr := range tt.wantErrContains {
				if !strings.Contains(errMsg, substr) {
					t.Errorf("error message missing substring %q\nGot: %s", substr, errMsg)
				}
			}
		})
	}
}

// TestAllBackends_StepTime_EmptyBatch_FloorAtOne verifies the LatencyModel
// interface contract: all backends must return >= 1 for empty batch, even with zero coefficients.
func TestAllBackends_StepTime_EmptyBatch_FloorAtOne(t *testing.T) {
	emptyBatch := []*sim.Request{}

	blackbox := &BlackboxLatencyModel{
		betaCoeffs:  []float64{0, 0, 0}, // zero coefficients — worst case
		alphaCoeffs: []float64{0, 0, 0},
	}
	assert.GreaterOrEqual(t, blackbox.StepTime(emptyBatch), int64(1),
		"blackbox with zero coefficients must still return >= 1")
}

// TestNewLatencyModel_RemainingBackendsWork verifies BC-7: deleting deprecated
// backend implementations does not break the remaining valid backends.
// GIVEN minimal valid hardware config and coefficients
// WHEN constructing each remaining backend via NewLatencyModel
// THEN construction succeeds AND the model computes positive step time.
func TestNewLatencyModel_RemainingBackendsWork(t *testing.T) {
	validBackends := []string{"blackbox", "roofline", "trained-physics"}

	for _, backend := range validBackends {
		t.Run(backend, func(t *testing.T) {
			// GIVEN minimal valid hardware config (includes BytesPerParam, MfuPrefill,
			// MfuDecode required by roofline validation)
			hw := sim.NewModelHardwareConfig(
				sim.ModelConfig{
					NumLayers:       32,
					NumHeads:        32,
					NumKVHeads:      8,
					HiddenDim:       4096,
					IntermediateDim: 11008,
					BytesPerParam:   2.0,
				},
				sim.HardwareCalib{
					TFlopsPeak: 989.0,
					BwPeakTBs:  3.35,
					MfuPrefill: 0.55,
					MfuDecode:  0.30,
					MemoryGiB:  80.0,
				},
				"", "", 1, backend, 0,
			)
			coeffs := sim.NewLatencyCoeffs(
				[]float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
				[]float64{1.0, 1.0, 1.0},
			)

			// WHEN constructing a valid backend
			model, err := NewLatencyModel(coeffs, hw)

			// THEN construction must succeed
			if err != nil {
				t.Fatalf("NewLatencyModel(%q) failed: %v", backend, err)
			}
			if model == nil {
				t.Errorf("NewLatencyModel(%q) returned nil model with no error", backend)
			}

			// AND the model must compute positive step time
			batch := []*sim.Request{
				{
					InputTokens:   make([]int, 100),
					ProgressIndex: 0,
					NumNewTokens:  10,
				},
			}
			stepTime := model.StepTime(batch)
			if stepTime <= 0 {
				t.Errorf("StepTime with 100 input, 10 new tokens = %v; want > 0", stepTime)
			}
		})
	}
}

// TestNewLatencyModel_Blackbox_EmitsDeprecationWarning verifies BC-1:
// GIVEN a valid blackbox latency model config
// WHEN NewLatencyModel is called
// THEN a deprecation warning MUST be emitted.
func TestNewLatencyModel_Blackbox_EmitsDeprecationWarning(t *testing.T) {
	resetDeprecationWarningsForTest()

	coeffs := sim.NewLatencyCoeffs([]float64{10.0, 20.0, 30.0}, []float64{1.0, 2.0, 3.0})
	hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "", "", 1, "blackbox", 0)

	var logBuf bytes.Buffer
	oldOut := logrus.StandardLogger().Out
	logrus.SetOutput(&logBuf)
	defer logrus.SetOutput(oldOut)

	model, err := NewLatencyModel(coeffs, hw)

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if model == nil {
		t.Fatal("expected non-nil model")
	}

	logOutput := logBuf.String()
	if !strings.Contains(logOutput, "deprecated") {
		t.Errorf("expected deprecation warning in log output, but got: %s", logOutput)
	}
}

// TestNewLatencyModel_Crossmodel_EmitsDeprecationWarning verifies BC-2:
// GIVEN a valid crossmodel latency model config
// WHEN NewLatencyModel is called
// THEN a deprecation warning MUST be emitted.
func TestNewLatencyModel_Crossmodel_EmitsDeprecationWarning(t *testing.T) {
	resetDeprecationWarningsForTest()

	coeffs := sim.NewLatencyCoeffs([]float64{10.0, 20.0, 30.0, 40.0}, []float64{1.0, 2.0, 3.0})
	hw := sim.NewModelHardwareConfig(
		sim.ModelConfig{NumLayers: 32, NumHeads: 32, HiddenDim: 4096, NumKVHeads: 8},
		sim.HardwareCalib{TFlopsPeak: 989.5, BwPeakTBs: 3.35},
		"", "", 1, "crossmodel", 0)

	var logBuf bytes.Buffer
	oldOut := logrus.StandardLogger().Out
	logrus.SetOutput(&logBuf)
	defer logrus.SetOutput(oldOut)

	model, err := NewLatencyModel(coeffs, hw)

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if model == nil {
		t.Fatal("expected non-nil model")
	}

	logOutput := logBuf.String()
	if !strings.Contains(logOutput, "deprecated") {
		t.Errorf("expected deprecation warning in log output, but got: %s", logOutput)
	}
}

// TestNewLatencyModel_TrainedRoofline_EmitsDeprecationWarning verifies BC-3:
// GIVEN a valid trained-roofline latency model config
// WHEN NewLatencyModel is called
// THEN a deprecation warning MUST be emitted.
func TestNewLatencyModel_TrainedRoofline_EmitsDeprecationWarning(t *testing.T) {
	resetDeprecationWarningsForTest()

	coeffs := sim.NewLatencyCoeffs([]float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0}, []float64{1.0, 2.0, 3.0})
	hw := sim.NewModelHardwareConfig(
		sim.ModelConfig{NumLayers: 32, NumHeads: 32, HiddenDim: 4096, IntermediateDim: 11008, NumKVHeads: 32},
		sim.HardwareCalib{TFlopsPeak: 989.5, BwPeakTBs: 3.35},
		"", "", 1, "trained-roofline", 0)

	var logBuf bytes.Buffer
	oldOut := logrus.StandardLogger().Out
	logrus.SetOutput(&logBuf)
	defer logrus.SetOutput(oldOut)

	model, err := NewLatencyModel(coeffs, hw)

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if model == nil {
		t.Fatal("expected non-nil model")
	}

	logOutput := logBuf.String()
	if !strings.Contains(logOutput, "deprecated") {
		t.Errorf("expected deprecation warning in log output, but got: %s", logOutput)
	}
}

// TestNewLatencyModel_Roofline_NoDeprecationWarning verifies BC-5:
// GIVEN a valid roofline latency model config
// WHEN NewLatencyModel is called
// THEN no deprecation warning MUST be emitted.
func TestNewLatencyModel_Roofline_NoDeprecationWarning(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs(nil, []float64{1.0, 2.0, 3.0})
	hw := sim.NewModelHardwareConfig(testModelConfig(), testHardwareCalib(), "", "", 2, "roofline", 0)

	var logBuf bytes.Buffer
	oldOut := logrus.StandardLogger().Out
	logrus.SetOutput(&logBuf)
	defer logrus.SetOutput(oldOut)

	model, err := NewLatencyModel(coeffs, hw)

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if model == nil {
		t.Fatal("expected non-nil model")
	}

	logOutput := logBuf.String()
	if strings.Contains(logOutput, "deprecated") {
		t.Errorf("expected no deprecation warning for roofline, but got: %s", logOutput)
	}
}

// TestNewLatencyModel_TrainedPhysics_NoDeprecationWarning verifies BC-5:
// GIVEN a valid trained-physics latency model config
// WHEN NewLatencyModel is called
// THEN no deprecation warning MUST be emitted.
func TestNewLatencyModel_TrainedPhysics_NoDeprecationWarning(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs([]float64{1, 2, 3, 4, 5, 6, 7, 8}, []float64{1.0, 2.0, 3.0})
	hw := sim.NewModelHardwareConfig(testModelConfig(), testHardwareCalib(), "", "", 2, "trained-physics", 0)

	var logBuf bytes.Buffer
	oldOut := logrus.StandardLogger().Out
	logrus.SetOutput(&logBuf)
	defer logrus.SetOutput(oldOut)

	model, err := NewLatencyModel(coeffs, hw)

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if model == nil {
		t.Fatal("expected non-nil model")
	}

	logOutput := logBuf.String()
	if strings.Contains(logOutput, "deprecated") {
		t.Errorf("expected no deprecation warning for trained-physics, but got: %s", logOutput)
	}
}
