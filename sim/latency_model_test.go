package sim

import (
	"math"
	"testing"
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

	batch := []*Request{
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
	emptyResult := model.StepTime([]*Request{})

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
// THEN the result MUST be non-negative (overhead-only baseline).
func TestBlackboxLatencyModel_StepTime_EmptyBatch(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	result := model.StepTime([]*Request{})

	// THEN overhead-only baseline is non-negative
	if result < 0 {
		t.Errorf("StepTime(empty) = %d, want >= 0", result)
	}
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

	req := &Request{InputTokens: make([]int, 50)}
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

// TestBlackboxLatencyModel_PlaceholderOverheads verifies placeholders return 0.
func TestBlackboxLatencyModel_PlaceholderOverheads(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	if model.SchedulingProcessingTime() != 0 {
		t.Errorf("SchedulingProcessingTime = %d, want 0", model.SchedulingProcessingTime())
	}
	if model.PreemptionProcessingTime() != 0 {
		t.Errorf("PreemptionProcessingTime = %d, want 0", model.PreemptionProcessingTime())
	}
}

// TestBlackboxLatencyModel_StepTime_Monotonic verifies BC-1 invariant:
// more prefill tokens must increase step time.
func TestBlackboxLatencyModel_StepTime_Monotonic(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	small := []*Request{{InputTokens: make([]int, 50), ProgressIndex: 0, NumNewTokens: 10}}
	large := []*Request{{InputTokens: make([]int, 200), ProgressIndex: 0, NumNewTokens: 100}}

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

	short := &Request{InputTokens: make([]int, 10)}
	long := &Request{InputTokens: make([]int, 500)}

	if model.QueueingTime(long) <= model.QueueingTime(short) {
		t.Errorf("monotonicity violated: QueueingTime(500 tokens) = %d <= QueueingTime(10 tokens) = %d",
			model.QueueingTime(long), model.QueueingTime(short))
	}
}

// TestRooflineLatencyModel_StepTime_PositiveAndMonotonic verifies BC-2:
// StepTime produces positive results and more tokens yield longer step time.
func TestRooflineLatencyModel_StepTime_PositiveAndMonotonic(t *testing.T) {
	mfuDB := loadTestMFUDatabase(t)
	model := &RooflineLatencyModel{
		modelConfig: testModelConfig(),
		hwConfig:    testHardwareCalib(),
		tp:          2,
		mfuDB:       mfuDB,
		alphaCoeffs: []float64{100, 1, 100},
	}

	smallBatch := []*Request{
		{
			InputTokens:   make([]int, 100),
			ProgressIndex: 0,
			NumNewTokens:  100,
		},
	}
	largeBatch := []*Request{
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
	mfuDB := loadTestMFUDatabase(t)
	model := &RooflineLatencyModel{
		modelConfig: testModelConfig(),
		hwConfig:    testHardwareCalib(),
		tp:          2,
		mfuDB:       mfuDB,
		alphaCoeffs: []float64{100, 1, 100},
	}

	emptyResult := model.StepTime([]*Request{})

	// THEN empty batch result must be non-negative (overhead only)
	if emptyResult < 0 {
		t.Errorf("StepTime(empty) = %d, want >= 0", emptyResult)
	}

	// AND a non-empty batch must produce a longer step time
	nonEmptyBatch := []*Request{
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
	mfuDB := loadTestMFUDatabase(t)
	model := &RooflineLatencyModel{
		modelConfig: testModelConfig(),
		hwConfig:    testHardwareCalib(),
		tp:          2,
		mfuDB:       mfuDB,
		alphaCoeffs: []float64{100, 1, 100},
	}

	req := &Request{InputTokens: make([]int, 50)}
	result := model.QueueingTime(req)

	if result <= 0 {
		t.Errorf("QueueingTime(50 tokens) = %d, want > 0", result)
	}
}

// TestNewLatencyModel_BlackboxMode verifies BC-4 (blackbox path).
func TestNewLatencyModel_BlackboxMode(t *testing.T) {
	cfg := SimConfig{
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, false),
	}

	model, err := NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("NewLatencyModel returned error: %v", err)
	}

	batch := []*Request{
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
	req := &Request{InputTokens: make([]int, 50)}
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
	mfuDB := loadTestMFUDatabase(t)
	hw := NewModelHardwareConfig(testModelConfig(), testHardwareCalib(), "", "", 2, true)
	hw.MFUDatabase = mfuDB
	cfg := SimConfig{
		LatencyCoeffs:       NewLatencyCoeffs(nil, []float64{100, 1, 100}),
		ModelHardwareConfig: hw,
	}

	model, err := NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("NewLatencyModel returned error: %v", err)
	}

	batch := []*Request{
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

// TestNewLatencyModel_InvalidRoofline verifies BC-8.
func TestNewLatencyModel_InvalidRoofline(t *testing.T) {
	cfg := SimConfig{
		LatencyCoeffs:       NewLatencyCoeffs(nil, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, true),
	}

	_, err := NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err == nil {
		t.Fatal("expected error for invalid roofline config, got nil")
	}
}

// TestNewLatencyModel_ShortAlphaCoeffs verifies factory rejects short alpha slices.
func TestNewLatencyModel_ShortAlphaCoeffs(t *testing.T) {
	tests := []struct {
		name     string
		roofline bool
		alpha    []float64
		beta     []float64
	}{
		{"blackbox_empty_alpha", false, []float64{}, []float64{1, 2, 3}},
		{"blackbox_short_alpha", false, []float64{1, 2}, []float64{1, 2, 3}},
		{"roofline_empty_alpha", true, []float64{}, nil},
		{"roofline_short_alpha", true, []float64{1}, nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			coeffs := NewLatencyCoeffs(tc.beta, tc.alpha)
			hw := NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, tc.roofline)
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
			coeffs := NewLatencyCoeffs(tc.beta, []float64{100, 1, 100})
			hw := NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, false)
			_, err := NewLatencyModel(coeffs, hw)
			if err == nil {
				t.Fatal("expected error for short BetaCoeffs, got nil")
			}
		})
	}
}

// TestNewLatencyModel_NaNAlphaCoeffs_ReturnsError verifies BC-4: NaN in alpha rejected.
func TestNewLatencyModel_NaNAlphaCoeffs_ReturnsError(t *testing.T) {
	coeffs := NewLatencyCoeffs([]float64{5000, 10, 5}, []float64{math.NaN(), 1.0, 100.0})
	_, err := NewLatencyModel(coeffs, NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, false))
	if err == nil {
		t.Fatal("expected error for NaN AlphaCoeffs, got nil")
	}
}

// TestNewLatencyModel_InfBetaCoeffs_ReturnsError verifies BC-4: Inf in beta rejected.
func TestNewLatencyModel_InfBetaCoeffs_ReturnsError(t *testing.T) {
	coeffs := NewLatencyCoeffs([]float64{math.Inf(1), 10, 5}, []float64{100, 1.0, 100.0})
	_, err := NewLatencyModel(coeffs, NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, false))
	if err == nil {
		t.Fatal("expected error for Inf BetaCoeffs, got nil")
	}
}

// TestBlackboxRoofline_ZeroOutputTokens_ConsistentClassification verifies BC-5:
// Both models handle requests past prefill with 0 output tokens consistently.
func TestBlackboxRoofline_ZeroOutputTokens_ConsistentClassification(t *testing.T) {
	// GIVEN a request past prefill with 0 output tokens (edge case)
	req := &Request{
		InputTokens:   []int{1, 2, 3},
		OutputTokens:  []int{},
		ProgressIndex: 3,
		NumNewTokens:  0,
	}
	batch := []*Request{req}

	blackbox := &BlackboxLatencyModel{
		betaCoeffs:  []float64{5000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}
	mfuDB := loadTestMFUDatabase(t)
	roofline := &RooflineLatencyModel{
		modelConfig: testModelConfig(),
		hwConfig:    testHardwareCalib(),
		tp:          2,
		mfuDB:       mfuDB,
		alphaCoeffs: []float64{100, 1, 100},
	}

	// WHEN both models compute step time with and without the zero-output request
	emptyBatch := []*Request{}
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
