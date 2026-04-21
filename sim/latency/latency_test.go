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
	// Type assertion to verify it's a BlackboxLatencyModel
	if _, ok := model.(*BlackboxLatencyModel); !ok {
		t.Errorf("expected *BlackboxLatencyModel, got %T", model)
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
	if _, ok := model.(*RooflineLatencyModel); !ok {
		t.Errorf("expected *RooflineLatencyModel, got %T", model)
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

func TestNewLatencyModel_Blackbox_EmitsDeprecationWarning(t *testing.T) {
	// GIVEN a valid blackbox latency model config
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: []float64{1.0, 2.0, 3.0},
		BetaCoeffs:  []float64{10.0, 20.0, 30.0},
	}
	hw := sim.ModelHardwareConfig{
		Backend: "blackbox",
		TP:      1,
		ModelConfig: sim.ModelConfig{
			NumLayers:       32,
			NumHeads:        32,
			HiddenDim:       4096,
			IntermediateDim: 11008,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.5,
			BwPeakTBs:  3.35,
		},
	}

	// WHEN constructing the blackbox latency model
	// Capture logrus output
	var logBuf bytes.Buffer
	oldOut := logrus.StandardLogger().Out
	logrus.SetOutput(&logBuf)
	defer logrus.SetOutput(oldOut)

	model, err := NewLatencyModel(coeffs, hw)

	// THEN no error is returned (backend is functional)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if model == nil {
		t.Fatal("expected non-nil model")
	}

	// AND deprecation warning is logged
	logOutput := logBuf.String()
	// Check for key parts of the warning (logrus escapes quotes in structured format)
	if !strings.Contains(logOutput, `blackbox`) ||
	   !strings.Contains(logOutput, `deprecated`) ||
	   !strings.Contains(logOutput, `trained-physics`) {
		t.Errorf("expected deprecation warning in log output, but got: %s", logOutput)
	}
}

func TestNewLatencyModel_Crossmodel_EmitsDeprecationWarning(t *testing.T) {
	// GIVEN a valid crossmodel latency model config
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: []float64{1.0, 2.0, 3.0},
		BetaCoeffs:  []float64{10.0, 20.0, 30.0, 40.0},
	}
	hw := sim.ModelHardwareConfig{
		Backend: "crossmodel",
		TP:      1,
		ModelConfig: sim.ModelConfig{
			NumLayers:  32,
			NumHeads:   32,
			HiddenDim:  4096,
			NumKVHeads: 8,
		},
		HWConfig: sim.HardwareCalib{
			TFlopsPeak: 989.5,
			BwPeakTBs:  3.35,
		},
	}

	// WHEN constructing the crossmodel latency model
	var logBuf bytes.Buffer
	oldOut := logrus.StandardLogger().Out
	logrus.SetOutput(&logBuf)
	defer logrus.SetOutput(oldOut)

	model, err := NewLatencyModel(coeffs, hw)

	// THEN no error is returned (backend is functional)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if model == nil {
		t.Fatal("expected non-nil model")
	}

	// AND deprecation warning is logged
	logOutput := logBuf.String()
	if !strings.Contains(logOutput, `crossmodel`) ||
	   !strings.Contains(logOutput, `deprecated`) ||
	   !strings.Contains(logOutput, `trained-physics`) {
		t.Errorf("expected deprecation warning in log output, but got: %s", logOutput)
	}
}

func TestNewLatencyModel_TrainedRoofline_EmitsDeprecationWarning(t *testing.T) {
	// GIVEN a valid trained-roofline latency model config
	coeffs := sim.LatencyCoeffs{
		AlphaCoeffs: []float64{1.0, 2.0, 3.0},
		BetaCoeffs:  []float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0},
	}
	hw := sim.ModelHardwareConfig{
		Backend: "trained-roofline",
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

	// WHEN constructing the trained-roofline latency model
	var logBuf bytes.Buffer
	oldOut := logrus.StandardLogger().Out
	logrus.SetOutput(&logBuf)
	defer logrus.SetOutput(oldOut)

	model, err := NewLatencyModel(coeffs, hw)

	// THEN no error is returned (backend is functional)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if model == nil {
		t.Fatal("expected non-nil model")
	}

	// AND deprecation warning is logged
	logOutput := logBuf.String()
	if !strings.Contains(logOutput, `trained-roofline`) ||
	   !strings.Contains(logOutput, `deprecated`) ||
	   !strings.Contains(logOutput, `trained-physics`) {
		t.Errorf("expected deprecation warning in log output, but got: %s", logOutput)
	}
}
