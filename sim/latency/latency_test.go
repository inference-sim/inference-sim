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
// THEN the result MUST be beta0 (fixed overhead), clamped to at least 1.
func TestBlackboxLatencyModel_StepTime_EmptyBatch(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{500, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	result := model.StepTime([]*sim.Request{})

	// THEN result must be at least 1 (clampToInt64 floor)
	if result < 1 {
		t.Errorf("StepTime(empty batch) = %d, want >= 1", result)
	}
	// AND must approximately equal beta0 (fixed overhead)
	expected := int64(500)
	if result != expected {
		t.Errorf("StepTime(empty batch) = %d, want %d (beta0)", result, expected)
	}
}

// TestBlackboxLatencyModel_QueueingTime_ScalesWithInputLength verifies:
// GIVEN requests with varying input lengths
// WHEN QueueingTime is called
// THEN longer inputs MUST have higher queueing time (alpha1 > 0).
func TestBlackboxLatencyModel_QueueingTime_ScalesWithInputLength(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 2, 50}, // alpha1 = 2 (positive scaling)
	}

	shortReq := &sim.Request{InputTokens: make([]int, 10)}
	longReq := &sim.Request{InputTokens: make([]int, 100)}

	shortTime := model.QueueingTime(shortReq)
	longTime := model.QueueingTime(longReq)

	// THEN longer input must have higher queueing time
	if longTime <= shortTime {
		t.Errorf("QueueingTime(long input) = %d <= QueueingTime(short input) = %d, want strictly greater", longTime, shortTime)
	}
}

// TestBlackboxLatencyModel_OutputTokenProcessingTime_ReturnsAlpha2 verifies:
// GIVEN a BlackboxLatencyModel with alphaCoeffs
// WHEN OutputTokenProcessingTime is called
// THEN it MUST return alpha2 (post-processing overhead per token).
func TestBlackboxLatencyModel_OutputTokenProcessingTime_ReturnsAlpha2(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 2, 75},
	}

	result := model.OutputTokenProcessingTime()
	expected := int64(75) // alpha2

	if result != expected {
		t.Errorf("OutputTokenProcessingTime() = %d, want %d (alpha2)", result, expected)
	}
}

// TestBlackboxLatencyModel_PostDecodeFixedOverhead_ReturnsZero verifies:
// GIVEN a BlackboxLatencyModel
// WHEN PostDecodeFixedOverhead is called
// THEN it MUST return 0 (blackbox model has no per-step decode overhead).
func TestBlackboxLatencyModel_PostDecodeFixedOverhead_ReturnsZero(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 2, 75},
	}

	result := model.PostDecodeFixedOverhead()

	if result != 0 {
		t.Errorf("PostDecodeFixedOverhead() = %d, want 0", result)
	}
}

// TestNewLatencyModel_Blackbox_Success verifies:
// GIVEN valid blackbox coefficients (AlphaCoeffs[3], BetaCoeffs[3])
// WHEN NewLatencyModel is called with Backend="blackbox"
// THEN a BlackboxLatencyModel MUST be returned with no error.
func TestNewLatencyModel_Blackbox_Success(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: []float64{100, 2, 75},
		BetaCoeffs:  []float64{1000, 10, 5},
	}
	hw := sim.ModelHardwareConfig{
		Backend: "blackbox",
		TP:      1,
		ModelConfig: sim.ModelConfig{
			NumLayers: 32,
			NumHeads:  32,
			HiddenDim: 4096,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.5,
			BwPeakTBs:  3.35,
		},
	}

	model, err := NewLatencyModel(coeffs, hw)

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if model == nil {
		t.Fatal("expected non-nil model")
	}
	// Verify behavioral contract: model computes valid step times
	testBatch := []*sim.Request{
		{
			InputTokens:   make([]int, 50),
			ProgressIndex: 0,
			NumNewTokens:  20,
		},
	}
	stepTime := model.StepTime(testBatch)
	if stepTime < 1 {
		t.Errorf("StepTime returned %d, expected >= 1", stepTime)
	}
}

// TestNewLatencyModel_Blackbox_InsufficientBetaCoeffs verifies:
// GIVEN BetaCoeffs with < 3 elements
// WHEN NewLatencyModel is called with Backend="blackbox"
// THEN an error MUST be returned.
func TestNewLatencyModel_Blackbox_InsufficientBetaCoeffs(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: []float64{100, 2, 75},
		BetaCoeffs:  []float64{1000, 10}, // only 2 elements
	}
	hw := sim.ModelHardwareConfig{
		Backend: "blackbox",
		TP:      1,
	}

	model, err := NewLatencyModel(coeffs, hw)

	if err == nil {
		t.Fatal("expected error for insufficient BetaCoeffs, got nil")
	}
	if model != nil {
		t.Errorf("expected nil model on error, got %T", model)
	}
	if !strings.Contains(err.Error(), "BetaCoeffs requires at least 3 elements") {
		t.Errorf("expected error message about BetaCoeffs, got: %v", err)
	}
}

// TestNewLatencyModel_Blackbox_NaNInBetaCoeffs verifies:
// GIVEN BetaCoeffs containing NaN
// WHEN NewLatencyModel is called with Backend="blackbox"
// THEN an error MUST be returned.
func TestNewLatencyModel_Blackbox_NaNInBetaCoeffs(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: []float64{100, 2, 75},
		BetaCoeffs:  []float64{1000, math.NaN(), 5},
	}
	hw := sim.ModelHardwareConfig{
		Backend: "blackbox",
		TP:      1,
	}

	model, err := NewLatencyModel(coeffs, hw)

	if err == nil {
		t.Fatal("expected error for NaN in BetaCoeffs, got nil")
	}
	if model != nil {
		t.Errorf("expected nil model on error, got %T", model)
	}
	if !strings.Contains(err.Error(), "NaN") && !strings.Contains(err.Error(), "BetaCoeffs") {
		t.Errorf("expected error message about NaN in BetaCoeffs, got: %v", err)
	}
}

// TestRooflineLatencyModel_StepTime_MixedBatch_Positive verifies:
// GIVEN a valid roofline model with a batch containing both prefill and decode
// WHEN StepTime is called
// THEN the result MUST be positive.
func TestRooflineLatencyModel_StepTime_MixedBatch_Positive(t *testing.T) {
	modelConfig := sim.ModelConfig{
		NumLayers:       32,
		NumHeads:        32,
		NumKVHeads:      8,
		HiddenDim:       4096,
		IntermediateDim: 11008,
	}
	hwConfig := sim.HardwareCalib{
		TFlopsPeak: 989.5,
		BwPeakTBs:  3.35,
	}
	model := &RooflineLatencyModel{
		modelConfig: modelConfig,
		hwConfig:    hwConfig,
		tp:          1,
		alphaCoeffs: []float64{100, 1, 50},
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

	if result <= 0 {
		t.Errorf("StepTime(mixed batch) = %d, want > 0", result)
	}
}

// TestRooflineLatencyModel_QueueingTime_ScalesWithInputLength verifies:
// GIVEN a roofline model with positive alpha1
// WHEN QueueingTime is called with varying input lengths
// THEN longer inputs MUST have higher queueing time.
func TestRooflineLatencyModel_QueueingTime_ScalesWithInputLength(t *testing.T) {
	model := &RooflineLatencyModel{
		alphaCoeffs: []float64{100, 2, 50}, // alpha1 = 2
	}

	shortReq := &sim.Request{InputTokens: make([]int, 10)}
	longReq := &sim.Request{InputTokens: make([]int, 100)}

	shortTime := model.QueueingTime(shortReq)
	longTime := model.QueueingTime(longReq)

	if longTime <= shortTime {
		t.Errorf("QueueingTime(long input) = %d <= QueueingTime(short input) = %d, want strictly greater", longTime, shortTime)
	}
}

// TestRooflineLatencyModel_OutputTokenProcessingTime_ReturnsAlpha2 verifies:
// GIVEN a RooflineLatencyModel with alphaCoeffs
// WHEN OutputTokenProcessingTime is called
// THEN it MUST return alpha2.
func TestRooflineLatencyModel_OutputTokenProcessingTime_ReturnsAlpha2(t *testing.T) {
	model := &RooflineLatencyModel{
		alphaCoeffs: []float64{100, 2, 75},
	}

	result := model.OutputTokenProcessingTime()
	expected := int64(75) // alpha2

	if result != expected {
		t.Errorf("OutputTokenProcessingTime() = %d, want %d (alpha2)", result, expected)
	}
}

// TestRooflineLatencyModel_PostDecodeFixedOverhead_ReturnsZero verifies:
// GIVEN a RooflineLatencyModel
// WHEN PostDecodeFixedOverhead is called
// THEN it MUST return 0.
func TestRooflineLatencyModel_PostDecodeFixedOverhead_ReturnsZero(t *testing.T) {
	model := &RooflineLatencyModel{}

	result := model.PostDecodeFixedOverhead()

	if result != 0 {
		t.Errorf("PostDecodeFixedOverhead() = %d, want 0", result)
	}
}

// TestNewLatencyModel_Roofline_Success verifies:
// GIVEN valid roofline config (TP > 0, valid model/hw configs)
// WHEN NewLatencyModel is called with Backend="roofline" or ""
// THEN a RooflineLatencyModel MUST be returned with no error.
func TestNewLatencyModel_Roofline_Success(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: []float64{100, 2, 75},
		BetaCoeffs:  []float64{}, // roofline doesn't use BetaCoeffs
	}
	hw := sim.ModelHardwareConfig{
		Backend: "roofline",
		TP:      1,
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			NumKVHeads:      8,
			HiddenDim:       4096,
			IntermediateDim: 11008,
			BytesPerParam:   2.0,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.5,
			BwPeakTBs:  3.35,
			MfuPrefill: 0.5,
			MfuDecode:  0.3,
		},
	}

	model, err := NewLatencyModel(coeffs, hw)

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if model == nil {
		t.Fatal("expected non-nil model")
	}
	// Verify behavioral contract: roofline model should compute valid step times
	testBatch := []*sim.Request{
		{InputTokens: make([]int, 50), ProgressIndex: 0, NumNewTokens: 20},
	}
	stepTime := model.StepTime(testBatch)
	if stepTime < 1 {
		t.Errorf("StepTime returned %d, expected >= 1", stepTime)
	}
}

// TestNewLatencyModel_Roofline_ZeroTP verifies:
// GIVEN TP <= 0
// WHEN NewLatencyModel is called with Backend="roofline"
// THEN an error MUST be returned.
func TestNewLatencyModel_Roofline_ZeroTP(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: []float64{100, 2, 75},
	}
	hw := sim.ModelHardwareConfig{
		Backend: "roofline",
		TP:      0, // invalid
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			NumKVHeads:      8,
			HiddenDim:       4096,
			IntermediateDim: 11008,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.5,
			BwPeakTBs:  3.35,
		},
	}

	model, err := NewLatencyModel(coeffs, hw)

	if err == nil {
		t.Fatal("expected error for TP <= 0, got nil")
	}
	if model != nil {
		t.Errorf("expected nil model on error, got %T", model)
	}
	if !strings.Contains(err.Error(), "TP > 0") {
		t.Errorf("expected error message about TP, got: %v", err)
	}
}

// TestNewLatencyModel_InsufficientAlphaCoeffs verifies:
// GIVEN AlphaCoeffs with < 3 elements
// WHEN NewLatencyModel is called (any backend)
// THEN an error MUST be returned.
func TestNewLatencyModel_InsufficientAlphaCoeffs(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: []float64{100, 2}, // only 2 elements
		BetaCoeffs:  []float64{1000, 10, 5},
	}
	hw := sim.ModelHardwareConfig{
		Backend: "blackbox",
		TP:      1,
	}

	model, err := NewLatencyModel(coeffs, hw)

	if err == nil {
		t.Fatal("expected error for insufficient AlphaCoeffs, got nil")
	}
	if model != nil {
		t.Errorf("expected nil model on error, got %T", model)
	}
	if !strings.Contains(err.Error(), "AlphaCoeffs requires at least 3 elements") {
		t.Errorf("expected error message about AlphaCoeffs, got: %v", err)
	}
}

// TestNewLatencyModel_NaNInAlphaCoeffs verifies:
// GIVEN AlphaCoeffs containing NaN
// WHEN NewLatencyModel is called (any backend)
// THEN an error MUST be returned.
func TestNewLatencyModel_NaNInAlphaCoeffs(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: []float64{100, math.NaN(), 75},
		BetaCoeffs:  []float64{1000, 10, 5},
	}
	hw := sim.ModelHardwareConfig{
		Backend: "blackbox",
		TP:      1,
	}

	model, err := NewLatencyModel(coeffs, hw)

	if err == nil {
		t.Fatal("expected error for NaN in AlphaCoeffs, got nil")
	}
	if model != nil {
		t.Errorf("expected nil model on error, got %T", model)
	}
	if !strings.Contains(err.Error(), "NaN") && !strings.Contains(err.Error(), "AlphaCoeffs") {
		t.Errorf("expected error message about NaN in AlphaCoeffs, got: %v", err)
	}
}

// TestClampToInt64_NaN verifies:
// GIVEN a NaN float64
// WHEN clampToInt64 is called
// THEN it MUST return math.MaxInt64.
func TestClampToInt64_NaN(t *testing.T) {
	result := clampToInt64(math.NaN())
	if result != math.MaxInt64 {
		t.Errorf("clampToInt64(NaN) = %d, want %d", result, math.MaxInt64)
	}
}

// TestClampToInt64_Overflow verifies:
// GIVEN a float64 >= math.MaxInt64
// WHEN clampToInt64 is called
// THEN it MUST return math.MaxInt64.
func TestClampToInt64_Overflow(t *testing.T) {
	result := clampToInt64(float64(math.MaxInt64) + 1e9)
	if result != math.MaxInt64 {
		t.Errorf("clampToInt64(overflow) = %d, want %d", result, math.MaxInt64)
	}
}

// TestClampToInt64_Normal verifies:
// GIVEN a normal float64 within int64 range
// WHEN clampToInt64 is called
// THEN it MUST return the correct int64 value.
func TestClampToInt64_Normal(t *testing.T) {
	input := 12345.67
	expected := int64(12345)
	result := clampToInt64(input)
	if result != expected {
		t.Errorf("clampToInt64(%v) = %d, want %d", input, result, expected)
	}
}

// TestStepTime_AtLeastOne verifies invariant: all LatencyModel implementations
// MUST return StepTime >= 1 for any batch (including empty).
func TestStepTime_AtLeastOne(t *testing.T) {
	emptyBatch := []*sim.Request{}

	blackbox := &BlackboxLatencyModel{
		betaCoeffs:  []float64{0, 0, 0}, // zero coefficients — worst case
		alphaCoeffs: []float64{0, 0, 0},
	}
	assert.GreaterOrEqual(t, blackbox.StepTime(emptyBatch), int64(1),
		"blackbox with zero coefficients must still return >= 1")

	crossmodel := &CrossModelLatencyModel{
		betaCoeffs:  []float64{0, 0, 0, 0}, // zero coefficients — worst case
		alphaCoeffs: []float64{0, 0, 0},
		numLayers:   1,
		kvDimScaled: 0.0,
		isMoE:       0.0,
		isTP:        0.0,
	}
	assert.GreaterOrEqual(t, crossmodel.StepTime(emptyBatch), int64(1),
		"crossmodel with zero coefficients must still return >= 1")
}

// NOTE: Deprecation warning emission tests removed. sync.Once ensures warnings
// are emitted at most once per process, making test execution order-dependent.
// Warnings are verified via manual testing and visible in other test output.

func TestNewLatencyModel_Roofline_NoDeprecationWarning(t *testing.T) {
	// GIVEN a valid roofline latency model config
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: []float64{1.0, 2.0, 3.0},
		BetaCoeffs:  []float64{},
	}
	hw := sim.ModelHardwareConfig{
		Backend: "roofline",
		TP:      1,
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			HiddenDim:       4096,
			IntermediateDim: 11008,
			NumKVHeads:      32,
			BytesPerParam:   2.0,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.5,
			BwPeakTBs:  3.35,
			MfuPrefill: 0.5,
			MfuDecode:  0.3,
		},
	}

	// WHEN constructing the roofline latency model
	var logBuf bytes.Buffer
	oldOut := logrus.StandardLogger().Out
	logrus.SetOutput(&logBuf)
	defer logrus.SetOutput(oldOut)

	model, err := NewLatencyModel(coeffs, hw)

	// THEN no error is returned
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if model == nil {
		t.Fatal("expected non-nil model")
	}

	// AND no deprecation warning is logged
	logOutput := logBuf.String()
	if strings.Contains(logOutput, "deprecated") {
		t.Errorf("roofline backend should not emit deprecation warning, but got: %s", logOutput)
	}
}

func TestNewLatencyModel_TrainedPhysics_NoDeprecationWarning(t *testing.T) {
	// GIVEN a valid trained-physics latency model config
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: []float64{1.0, 2.0, 3.0},
		BetaCoeffs:  []float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0},
	}
	hw := sim.ModelHardwareConfig{
		Backend: "trained-physics",
		TP:      1,
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			HiddenDim:       4096,
			IntermediateDim: 11008,
			NumKVHeads:      32,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.5,
			BwPeakTBs:  3.35,
		},
	}

	// WHEN constructing the trained-physics latency model
	var logBuf bytes.Buffer
	oldOut := logrus.StandardLogger().Out
	logrus.SetOutput(&logBuf)
	defer logrus.SetOutput(oldOut)

	model, err := NewLatencyModel(coeffs, hw)

	// THEN no error is returned
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if model == nil {
		t.Fatal("expected non-nil model")
	}

	// AND no deprecation warning is logged
	logOutput := logBuf.String()
	if strings.Contains(logOutput, "deprecated") {
		t.Errorf("trained-physics backend should not emit deprecation warning, but got: %s", logOutput)
	}
}

// TestBlackboxLatencyModel_StepTime_Monotonic verifies:
// GIVEN a blackbox latency model
// WHEN StepTime is called with increasing prefill token counts
// THEN step time MUST increase (more prefill work → longer step time).
func TestBlackboxLatencyModel_StepTime_Monotonic(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{100, 10, 5},
		alphaCoeffs: []float64{50, 1, 25},
	}

	// Batch with 10 prefill tokens
	batch10 := []*sim.Request{
		{
			InputTokens:   make([]int, 20),
			ProgressIndex: 0,
			NumNewTokens:  10,
		},
	}

	// Batch with 50 prefill tokens
	batch50 := []*sim.Request{
		{
			InputTokens:   make([]int, 100),
			ProgressIndex: 0,
			NumNewTokens:  50,
		},
	}

	time10 := model.StepTime(batch10)
	time50 := model.StepTime(batch50)

	// THEN more prefill tokens must produce longer step time
	if time50 <= time10 {
		t.Errorf("StepTime(50 prefill tokens) = %d <= StepTime(10 prefill tokens) = %d, expected monotonicity", time50, time10)
	}
}

// TestRooflineLatencyModel_StepTime_PositiveAndMonotonic verifies:
// GIVEN a roofline latency model
// WHEN StepTime is called with varying batch sizes
// THEN step time MUST be positive and increase with more work.
func TestRooflineLatencyModel_StepTime_PositiveAndMonotonic(t *testing.T) {
	modelConfig := sim.ModelConfig{
		NumLayers:       32,
		NumHeads:        32,
		NumKVHeads:      8,
		HiddenDim:       4096,
		IntermediateDim: 11008,
		BytesPerParam:   2.0,
	}
	hwConfig := sim.HardwareCalib{
		TFlopsPeak: 989.5,
		BwPeakTBs:  3.35,
		MfuPrefill: 0.5,
		MfuDecode:  0.3,
	}
	model := &RooflineLatencyModel{
		modelConfig: modelConfig,
		hwConfig:    hwConfig,
		tp:          1,
		alphaCoeffs: []float64{100, 1, 50},
	}

	// Empty batch
	emptyTime := model.StepTime([]*sim.Request{})

	// Small batch
	smallBatch := []*sim.Request{
		{
			InputTokens:   make([]int, 50),
			ProgressIndex: 0,
			NumNewTokens:  20,
		},
	}
	smallTime := model.StepTime(smallBatch)

	// Large batch
	largeBatch := []*sim.Request{
		{
			InputTokens:   make([]int, 200),
			ProgressIndex: 0,
			NumNewTokens:  100,
		},
	}
	largeTime := model.StepTime(largeBatch)

	// THEN all times must be positive
	if emptyTime < 1 {
		t.Errorf("StepTime(empty) = %d, want >= 1", emptyTime)
	}
	if smallTime < 1 {
		t.Errorf("StepTime(small) = %d, want >= 1", smallTime)
	}
	if largeTime < 1 {
		t.Errorf("StepTime(large) = %d, want >= 1", largeTime)
	}

	// AND larger batches must take longer
	if smallTime <= emptyTime {
		t.Errorf("StepTime(small) = %d <= StepTime(empty) = %d, expected monotonicity", smallTime, emptyTime)
	}
	if largeTime <= smallTime {
		t.Errorf("StepTime(large) = %d <= StepTime(small) = %d, expected monotonicity", largeTime, smallTime)
	}
}

// TestNewLatencyModel_UnknownBackend_ReturnsError verifies:
// GIVEN an unknown backend string
// WHEN NewLatencyModel is called
// THEN an error MUST be returned indicating the backend is not supported.
func TestNewLatencyModel_UnknownBackend_ReturnsError(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: []float64{1.0, 2.0, 3.0},
		BetaCoeffs:  []float64{10.0, 20.0, 30.0},
	}
	hw := sim.ModelHardwareConfig{
		Backend: "unknown-backend",
		TP:      1,
	}

	model, err := NewLatencyModel(coeffs, hw)

	if err == nil {
		t.Fatal("expected error for unknown backend, got nil")
	}
	if model != nil {
		t.Errorf("expected nil model on error, got %T", model)
	}
	if !strings.Contains(err.Error(), "unknown-backend") {
		t.Errorf("expected error message to mention unknown backend, got: %v", err)
	}
}

// TestNewLatencyModel_NegativeCoefficients_ReturnsError verifies:
// GIVEN coefficients with negative values
// WHEN NewLatencyModel is called
// THEN an error MUST be returned rejecting negative coefficients.
func TestNewLatencyModel_NegativeCoefficients_ReturnsError(t *testing.T) {
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: []float64{1.0, -2.0, 3.0}, // alpha1 is negative
		BetaCoeffs:  []float64{10.0, 20.0, 30.0},
	}
	hw := sim.ModelHardwareConfig{
		Backend: "blackbox",
		TP:      1,
	}

	model, err := NewLatencyModel(coeffs, hw)

	if err == nil {
		t.Fatal("expected error for negative coefficients, got nil")
	}
	if model != nil {
		t.Errorf("expected nil model on error, got %T", model)
	}
	if !strings.Contains(err.Error(), "negative") && !strings.Contains(err.Error(), "AlphaCoeffs") {
		t.Errorf("expected error message about negative coefficients, got: %v", err)
	}
}
