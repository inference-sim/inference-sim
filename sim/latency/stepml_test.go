package latency

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// validArtifactJSON is a minimal StepML artifact for testing.
const validArtifactJSON = `{
	"version": 1,
	"step_time": {
		"model_type": "linear",
		"intercept": 100.0,
		"feature_coefficients": {
			"prefill_tokens": 50.0,
			"decode_tokens": 5.0
		}
	},
	"queueing_time": {
		"model_type": "linear",
		"intercept": 10.0,
		"feature_coefficients": {
			"input_len": 0.5
		}
	},
	"output_token_processing_time_us": 2.0,
	"scheduling_processing_time_us": 5.0,
	"preemption_processing_time_us": 3.0
}`

func writeArtifact(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "model.json")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	return path
}

func TestLoadStepMLArtifact_Valid(t *testing.T) {
	path := writeArtifact(t, validArtifactJSON)
	art, err := LoadStepMLArtifact(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if art.Version != 1 {
		t.Errorf("version = %d, want 1", art.Version)
	}
	if art.StepTime.Intercept != 100.0 {
		t.Errorf("step_time intercept = %f, want 100.0", art.StepTime.Intercept)
	}
	if art.StepTime.FeatureCoefficients["prefill_tokens"] != 50.0 {
		t.Errorf("prefill_tokens coeff = %f, want 50.0", art.StepTime.FeatureCoefficients["prefill_tokens"])
	}
}

func TestLoadStepMLArtifact_MissingFile(t *testing.T) {
	_, err := LoadStepMLArtifact("/nonexistent/path.json")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestLoadStepMLArtifact_MissingStepTime(t *testing.T) {
	path := writeArtifact(t, `{"version": 1}`)
	_, err := LoadStepMLArtifact(path)
	if err == nil {
		t.Fatal("expected error for missing step_time")
	}
}

func TestLoadStepMLArtifact_InvalidVersion(t *testing.T) {
	path := writeArtifact(t, `{"version": 0, "step_time": {"model_type": "linear", "intercept": 0}}`)
	_, err := LoadStepMLArtifact(path)
	if err == nil {
		t.Fatal("expected error for version 0")
	}
}

func TestLoadStepMLArtifact_NaNCoefficient(t *testing.T) {
	// JSON doesn't support NaN natively; test via a model type error instead
	path := writeArtifact(t, `{"version": 1, "step_time": {"model_type": "neural_net", "intercept": 0}}`)
	_, err := LoadStepMLArtifact(path)
	if err == nil {
		t.Fatal("expected error for unsupported model_type")
	}
}

func TestStepMLLatencyModel_StepTime(t *testing.T) {
	path := writeArtifact(t, validArtifactJSON)
	art, err := LoadStepMLArtifact(path)
	if err != nil {
		t.Fatalf("load artifact: %v", err)
	}
	model := NewStepMLLatencyModel(art)

	// Batch: 1 prefill request (100 new tokens), 2 decode requests (1 new token each)
	batch := []*sim.Request{
		{InputTokens: make([]int, 200), ProgressIndex: 50, NumNewTokens: 100},  // prefill
		{InputTokens: make([]int, 100), OutputTokens: make([]int, 50), ProgressIndex: 150, NumNewTokens: 1}, // decode
		{InputTokens: make([]int, 80), OutputTokens: make([]int, 30), ProgressIndex: 110, NumNewTokens: 1},  // decode
	}

	// Expected: 100 (intercept) + 50*100 (prefill) + 5*2 (decode) = 100 + 5000 + 10 = 5110
	got := model.StepTime(batch)
	if got != 5110 {
		t.Errorf("StepTime = %d, want 5110", got)
	}
}

func TestStepMLLatencyModel_StepTime_Positive(t *testing.T) {
	// Test INV-M-1: predictions are always positive
	art := &StepMLArtifact{
		Version: 1,
		StepTime: &LinearModel{
			ModelType:           "linear",
			Intercept:           -1000,
			FeatureCoefficients: map[string]float64{},
		},
	}
	model := NewStepMLLatencyModel(art)
	got := model.StepTime([]*sim.Request{})
	if got < 1 {
		t.Errorf("StepTime = %d, want >= 1 (INV-M-1)", got)
	}
}

func TestStepMLLatencyModel_QueueingTime(t *testing.T) {
	path := writeArtifact(t, validArtifactJSON)
	art, err := LoadStepMLArtifact(path)
	if err != nil {
		t.Fatalf("load artifact: %v", err)
	}
	model := NewStepMLLatencyModel(art)

	req := &sim.Request{InputTokens: make([]int, 200)}
	// Expected: 10 (intercept) + 0.5 * 200 = 110
	got := model.QueueingTime(req)
	if got != 110 {
		t.Errorf("QueueingTime = %d, want 110", got)
	}
}

func TestStepMLLatencyModel_QueueingTime_NilModel(t *testing.T) {
	art := &StepMLArtifact{
		Version:  1,
		StepTime: &LinearModel{ModelType: "linear", Intercept: 10, FeatureCoefficients: map[string]float64{}},
	}
	model := NewStepMLLatencyModel(art)
	got := model.QueueingTime(&sim.Request{InputTokens: make([]int, 100)})
	if got != 0 {
		t.Errorf("QueueingTime with nil model = %d, want 0", got)
	}
}

func TestStepMLLatencyModel_ConstantMethods(t *testing.T) {
	path := writeArtifact(t, validArtifactJSON)
	art, err := LoadStepMLArtifact(path)
	if err != nil {
		t.Fatalf("load artifact: %v", err)
	}
	model := NewStepMLLatencyModel(art)

	if got := model.OutputTokenProcessingTime(); got != 2 {
		t.Errorf("OutputTokenProcessingTime = %d, want 2", got)
	}
	if got := model.SchedulingProcessingTime(); got != 5 {
		t.Errorf("SchedulingProcessingTime = %d, want 5", got)
	}
	if got := model.PreemptionProcessingTime(); got != 3 {
		t.Errorf("PreemptionProcessingTime = %d, want 3", got)
	}
}

func TestStepMLLatencyModel_KVFeatures(t *testing.T) {
	// Test that KV proxy features (ProgressIndex) are extracted and used
	art := &StepMLArtifact{
		Version: 1,
		StepTime: &LinearModel{
			ModelType: "linear",
			Intercept: 0,
			FeatureCoefficients: map[string]float64{
				"kv_sum":  0.1,
				"kv_mean": 10.0,
				"kv_max":  1.0,
			},
		},
	}
	model := NewStepMLLatencyModel(art)

	batch := []*sim.Request{
		{InputTokens: make([]int, 100), OutputTokens: make([]int, 50), ProgressIndex: 150, NumNewTokens: 1},
		{InputTokens: make([]int, 200), OutputTokens: make([]int, 100), ProgressIndex: 300, NumNewTokens: 1},
	}

	// kv_sum = 150 + 300 = 450 → 450 * 0.1 = 45
	// kv_mean = 225 → 225 * 10 = 2250
	// kv_max = 300 → 300 * 1 = 300
	// total = 45 + 2250 + 300 = 2595
	got := model.StepTime(batch)
	if got != 2595 {
		t.Errorf("StepTime with KV features = %d, want 2595", got)
	}
}

func TestExtractBatchFeatures(t *testing.T) {
	batch := []*sim.Request{
		{InputTokens: make([]int, 200), ProgressIndex: 50, NumNewTokens: 100},   // prefill: 50 < 200
		{InputTokens: make([]int, 100), OutputTokens: make([]int, 1), ProgressIndex: 150, NumNewTokens: 1}, // decode: 150 >= 100
	}

	f := extractBatchFeatures(batch)

	tests := []struct {
		name string
		want float64
	}{
		{"prefill_tokens", 100},
		{"decode_tokens", 1},
		{"num_prefill_reqs", 1},
		{"num_decode_reqs", 1},
		{"scheduled_tokens", 101},
		{"kv_sum", 200},    // 50 + 150
		{"kv_max", 150},
		{"kv_mean", 100},   // 200 / 2
	}
	for _, tt := range tests {
		if got := f[tt.name]; got != tt.want {
			t.Errorf("feature %q = %f, want %f", tt.name, got, tt.want)
		}
	}
}

func TestNewLatencyModel_StepMLDispatch(t *testing.T) {
	path := writeArtifact(t, validArtifactJSON)
	coeffs := sim.NewLatencyCoeffs(
		[]float64{0, 0, 0}, // beta (unused by stepml)
		[]float64{0, 0, 0}, // alpha (unused by stepml)
	)
	hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "", "", 0, false).
		WithStepMLModel(path)

	model, err := NewLatencyModel(coeffs, hw)
	if err != nil {
		t.Fatalf("NewLatencyModel with StepML: %v", err)
	}

	if _, ok := model.(*StepMLLatencyModel); !ok {
		t.Errorf("expected *StepMLLatencyModel, got %T", model)
	}
}

// regimeArtifactJSON has 2 regimes: decode-only and mixed-batch (fallback).
const regimeArtifactJSON = `{
	"version": 2,
	"step_time_regimes": [
		{
			"name": "decode_only",
			"condition": {"feature": "prefill_tokens", "op": "==", "value": 0},
			"model": {
				"model_type": "linear",
				"intercept": 50.0,
				"feature_coefficients": {"decode_tokens": 10.0, "kv_mean": 0.5}
			}
		},
		{
			"name": "mixed",
			"condition": null,
			"model": {
				"model_type": "linear",
				"intercept": 200.0,
				"feature_coefficients": {"prefill_tokens": 2.0, "decode_tokens": 8.0}
			}
		}
	],
	"output_token_processing_time_us": 1.0,
	"scheduling_processing_time_us": 2.0,
	"preemption_processing_time_us": 0.0
}`

func TestStepMLLatencyModel_RegimeDispatch(t *testing.T) {
	path := writeArtifact(t, regimeArtifactJSON)
	art, err := LoadStepMLArtifact(path)
	if err != nil {
		t.Fatalf("load regime artifact: %v", err)
	}
	model := NewStepMLLatencyModel(art)

	// Decode-only batch: 2 decode requests, 1 new token each
	decodeBatch := []*sim.Request{
		{InputTokens: make([]int, 100), OutputTokens: make([]int, 50), ProgressIndex: 150, NumNewTokens: 1},
		{InputTokens: make([]int, 80), OutputTokens: make([]int, 30), ProgressIndex: 110, NumNewTokens: 1},
	}
	// prefill_tokens=0 → decode_only regime
	// decode_tokens=2, kv_mean=(150+110)/2=130
	// Expected: 50 + 10*2 + 0.5*130 = 50 + 20 + 65 = 135
	got := model.StepTime(decodeBatch)
	if got != 135 {
		t.Errorf("StepTime(decode_only) = %d, want 135", got)
	}

	// Mixed batch: 1 prefill + 1 decode
	mixedBatch := []*sim.Request{
		{InputTokens: make([]int, 200), ProgressIndex: 50, NumNewTokens: 100},  // prefill
		{InputTokens: make([]int, 100), OutputTokens: make([]int, 50), ProgressIndex: 150, NumNewTokens: 1}, // decode
	}
	// prefill_tokens=100 > 0 → mixed regime
	// decode_tokens=1
	// Expected: 200 + 2*100 + 8*1 = 200 + 200 + 8 = 408
	got = model.StepTime(mixedBatch)
	if got != 408 {
		t.Errorf("StepTime(mixed) = %d, want 408", got)
	}
}

func TestStepMLLatencyModel_RegimeNoStepTime(t *testing.T) {
	// Artifact with only regimes (no step_time field) should work
	path := writeArtifact(t, regimeArtifactJSON)
	art, err := LoadStepMLArtifact(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if art.StepTime != nil {
		t.Error("expected nil step_time when using regimes only")
	}
	if len(art.StepTimeRegimes) != 2 {
		t.Errorf("got %d regimes, want 2", len(art.StepTimeRegimes))
	}
}

func TestExtractBatchFeatures_InteractionFeatures(t *testing.T) {
	batch := []*sim.Request{
		{InputTokens: make([]int, 200), ProgressIndex: 50, NumNewTokens: 100},   // prefill
		{InputTokens: make([]int, 100), OutputTokens: make([]int, 1), ProgressIndex: 150, NumNewTokens: 1}, // decode
	}

	f := extractBatchFeatures(batch)

	// prefill_x_decode = 100 * 1 = 100
	if got := f["prefill_x_decode"]; got != 100 {
		t.Errorf("prefill_x_decode = %f, want 100", got)
	}
	// decode_x_kv_mean = 1 * 100 = 100 (kv_mean = (50+150)/2 = 100)
	if got := f["decode_x_kv_mean"]; got != 100 {
		t.Errorf("decode_x_kv_mean = %f, want 100", got)
	}
	// prefill_sq = 100 * 100 = 10000
	if got := f["prefill_sq"]; got != 10000 {
		t.Errorf("prefill_sq = %f, want 10000", got)
	}
	// kv_std: pi values are 50 and 150, mean=100
	// variance = ((50-100)^2 + (150-100)^2) / 2 = (2500+2500)/2 = 2500
	// std = sqrt(2500) = 50
	if got := f["kv_std"]; got != 50 {
		t.Errorf("kv_std = %f, want 50", got)
	}
}

func TestStepMLLatencyModel_OverheadFloorAndCap(t *testing.T) {
	// Overhead floor: prediction < overhead → use overhead.
	// Overhead cap: prediction > 3*overhead → clamp to 3*overhead.
	art := &StepMLArtifact{
		Version: 1,
		StepTime: &LinearModel{
			ModelType:           "linear",
			Intercept:           100,
			FeatureCoefficients: map[string]float64{"decode_tokens": 200},
		},
		StepOverheadUs: 5000,
	}
	model := NewStepMLLatencyModel(art)

	// Small batch: 100 + 200*1 = 300 < 5000 → floored to 5000
	small := []*sim.Request{
		{InputTokens: make([]int, 10), OutputTokens: make([]int, 1), ProgressIndex: 10, NumNewTokens: 1},
	}
	got := model.StepTime(small)
	if got != 5000 {
		t.Errorf("StepTime(small) = %d, want 5000 (overhead floor)", got)
	}

	// Large batch: 100 + 200*100 = 20100, cap = 3*5000 = 15000 → capped
	var large []*sim.Request
	for i := 0; i < 100; i++ {
		large = append(large, &sim.Request{
			InputTokens: make([]int, 10), OutputTokens: make([]int, 1),
			ProgressIndex: 10, NumNewTokens: 1,
		})
	}
	got = model.StepTime(large)
	if got != 15000 {
		t.Errorf("StepTime(large) = %d, want 15000 (overhead cap)", got)
	}

	// Medium batch: 100 + 200*30 = 6100, between floor and cap → use as-is
	var medium []*sim.Request
	for i := 0; i < 30; i++ {
		medium = append(medium, &sim.Request{
			InputTokens: make([]int, 10), OutputTokens: make([]int, 1),
			ProgressIndex: 10, NumNewTokens: 1,
		})
	}
	got = model.StepTime(medium)
	if got != 6100 {
		t.Errorf("StepTime(medium) = %d, want 6100 (between floor and cap)", got)
	}
}

func TestNewLatencyModel_StepMLInvalidPath(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs(
		[]float64{0, 0, 0},
		[]float64{0, 0, 0},
	)
	hw := sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "", "", 0, false).
		WithStepMLModel("/nonexistent/model.json")

	_, err := NewLatencyModel(coeffs, hw)
	if err == nil {
		t.Fatal("expected error for nonexistent StepML model path")
	}
}

func TestNewLatencyModel_RooflineTakesPrecedence(t *testing.T) {
	// Verify that roofline dispatch happens before StepML (isolation guarantee)
	path := writeArtifact(t, validArtifactJSON)
	coeffs := sim.NewLatencyCoeffs(
		[]float64{0, 0, 0},
		[]float64{0, 0, 0},
	)
	hw := sim.NewModelHardwareConfig(
		sim.ModelConfig{NumLayers: 32, HiddenDim: 4096, NumHeads: 32, NumKVHeads: 32, VocabSize: 32000, BytesPerParam: 2, IntermediateDim: 11008},
		sim.HardwareCalib{TFlopsPeak: 989.5, BwPeakTBs: 3.35},
		"test-model", "H100", 1, true, // Roofline=true
	).WithStepMLModel(path)

	// This should fail on roofline validation (missing MFU database), NOT try StepML
	_, err := NewLatencyModel(coeffs, hw)
	if err == nil {
		t.Fatal("expected roofline validation error, not StepML dispatch")
	}
}
