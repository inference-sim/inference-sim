package sim

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGetHWConfig_MalformedJSON(t *testing.T) {
	// Create temp file with malformed JSON
	tmpDir := t.TempDir()
	badFile := filepath.Join(tmpDir, "bad_hw.json")
	if err := os.WriteFile(badFile, []byte(`{"H100": invalid`), 0644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	_, err := GetHWConfig(badFile, "H100")
	if err == nil {
		t.Error("expected error for malformed JSON, got nil")
	}
}

func TestGetHWConfig_UnknownGPU(t *testing.T) {
	// Create temp file with valid JSON but without the requested GPU
	tmpDir := t.TempDir()
	validFile := filepath.Join(tmpDir, "hw.json")
	content := `{"H100": {"TFlopsPeak": 1000, "BwPeakTBs": 3.35}}`
	if err := os.WriteFile(validFile, []byte(content), 0644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	_, err := GetHWConfig(validFile, "H200")
	if err == nil {
		t.Error("expected error for unknown GPU, got nil")
	}
	if err != nil && !strings.Contains(err.Error(), "H200") {
		t.Errorf("error should mention the unknown GPU name, got: %v", err)
	}
}

func TestGetHWConfig_ValidConfig(t *testing.T) {
	tmpDir := t.TempDir()
	validFile := filepath.Join(tmpDir, "hw.json")
	content := `{"H100": {"TFlopsPeak": 1000, "BwPeakTBs": 3.35}}`
	if err := os.WriteFile(validFile, []byte(content), 0644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	cfg, err := GetHWConfig(validFile, "H100")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if cfg.TFlopsPeak != 1000 {
		t.Errorf("expected TFlopsPeak=1000, got %v", cfg.TFlopsPeak)
	}
}

func TestGetModelConfig_MalformedJSON(t *testing.T) {
	tmpDir := t.TempDir()
	badFile := filepath.Join(tmpDir, "config.json")
	if err := os.WriteFile(badFile, []byte(`{"num_hidden_layers": invalid`), 0644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	_, err := GetModelConfig(badFile)
	if err == nil {
		t.Error("expected error for malformed JSON, got nil")
	}
}

func TestGetModelConfig_MissingTorchDtype(t *testing.T) {
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "config.json")
	// Valid JSON but missing torch_dtype
	content := `{"num_hidden_layers": 32, "hidden_size": 4096}`
	if err := os.WriteFile(configFile, []byte(content), 0644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	cfg, err := GetModelConfig(configFile)
	if err != nil {
		t.Errorf("should not error for missing torch_dtype (default to 0): %v", err)
	}
	if cfg == nil {
		t.Error("expected non-nil config")
	}
}

func TestGetModelConfig_ValidConfig(t *testing.T) {
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "config.json")
	content := `{
		"num_hidden_layers": 32,
		"hidden_size": 4096,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"vocab_size": 128256,
		"intermediate_size": 14336,
		"torch_dtype": "bfloat16"
	}`
	if err := os.WriteFile(configFile, []byte(content), 0644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	cfg, err := GetModelConfig(configFile)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if cfg.NumLayers != 32 {
		t.Errorf("expected NumLayers=32, got %v", cfg.NumLayers)
	}
	if cfg.BytesPerParam != 2 {
		t.Errorf("expected BytesPerParam=2 for bfloat16, got %v", cfg.BytesPerParam)
	}
}

func TestValidateRooflineConfig_ZeroModelFields_ReturnsError(t *testing.T) {
	hc := HardwareCalib{TFlopsPeak: 1000, BwPeakTBs: 3.35, BwEffConstant: 0.7, MfuPrefill: 0.5, MfuDecode: 0.3}

	tests := []struct {
		name  string
		mc    ModelConfig
		field string
	}{
		{"zero NumHeads", ModelConfig{NumHeads: 0, NumLayers: 32, HiddenDim: 4096, BytesPerParam: 2}, "NumHeads"},
		{"zero NumLayers", ModelConfig{NumHeads: 32, NumLayers: 0, HiddenDim: 4096, BytesPerParam: 2}, "NumLayers"},
		{"zero HiddenDim", ModelConfig{NumHeads: 32, NumLayers: 32, HiddenDim: 0, BytesPerParam: 2}, "HiddenDim"},
		{"zero BytesPerParam", ModelConfig{NumHeads: 32, NumLayers: 32, HiddenDim: 4096, BytesPerParam: 0}, "BytesPerParam"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// WHEN ValidateRooflineConfig is called
			err := ValidateRooflineConfig(tt.mc, hc)

			// THEN it returns an error mentioning the zero field
			if err == nil {
				t.Fatalf("expected error for %s, got nil", tt.field)
			}
			if !strings.Contains(err.Error(), tt.field) {
				t.Errorf("error should mention %s, got: %v", tt.field, err)
			}
		})
	}
}

func TestValidateRooflineConfig_ZeroHardwareFields_ReturnsAllErrors(t *testing.T) {
	// GIVEN a HardwareCalib with all critical fields zero (model config is valid)
	mc := ModelConfig{NumHeads: 32, NumLayers: 32, HiddenDim: 4096, BytesPerParam: 2}
	hc := HardwareCalib{} // all zero

	// WHEN ValidateRooflineConfig is called
	err := ValidateRooflineConfig(mc, hc)

	// THEN it returns an error mentioning every zero field
	if err == nil {
		t.Fatal("expected error for zero hardware fields, got nil")
	}
	errMsg := err.Error()
	for _, field := range []string{"TFlopsPeak", "BwPeakTBs", "BwEffConstant", "MfuPrefill", "MfuDecode"} {
		if !strings.Contains(errMsg, field) {
			t.Errorf("error should mention %s, got: %v", field, errMsg)
		}
	}
}

func TestValidateRooflineConfig_NaNInfFields_ReturnsErrors(t *testing.T) {
	// GIVEN a HardwareCalib with NaN and Inf fields (bypass <= 0 check)
	mc := ModelConfig{NumHeads: 32, NumLayers: 32, HiddenDim: 4096}
	hc := HardwareCalib{
		TFlopsPeak:    math.NaN(),
		BwPeakTBs:     math.Inf(1),
		BwEffConstant: 0.7,
		MfuPrefill:    0.5,
		MfuDecode:     math.NaN(),
	}

	// WHEN ValidateRooflineConfig is called
	err := ValidateRooflineConfig(mc, hc)

	// THEN it returns an error mentioning the invalid fields
	if err == nil {
		t.Fatal("expected error for NaN/Inf hardware fields, got nil")
	}
	errMsg := err.Error()
	for _, field := range []string{"TFlopsPeak", "BwPeakTBs", "MfuDecode"} {
		if !strings.Contains(errMsg, field) {
			t.Errorf("error should mention %s, got: %v", field, errMsg)
		}
	}
}

func TestValidateRooflineConfig_ValidConfig_ReturnsNil(t *testing.T) {
	// GIVEN valid ModelConfig and HardwareCalib
	mc := ModelConfig{NumHeads: 32, NumLayers: 32, HiddenDim: 4096, BytesPerParam: 2}
	hc := HardwareCalib{TFlopsPeak: 1000, BwPeakTBs: 3.35, BwEffConstant: 0.7, MfuPrefill: 0.5, MfuDecode: 0.3}

	// WHEN ValidateRooflineConfig is called
	err := ValidateRooflineConfig(mc, hc)

	// THEN it returns nil
	if err != nil {
		t.Errorf("expected nil error for valid config, got: %v", err)
	}
}

func TestNewSimulator_RooflineZeroNumHeads_ReturnsError(t *testing.T) {
	// GIVEN a SimConfig with Roofline=true and NumHeads=0
	cfg := SimConfig{
		Horizon:       100000,
		KVCacheConfig: NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		LatencyCoeffs: NewLatencyCoeffs(nil, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(
			ModelConfig{NumHeads: 0, NumLayers: 32, HiddenDim: 4096},
			HardwareCalib{TFlopsPeak: 1000, BwPeakTBs: 3.35, BwEffConstant: 0.7, MfuPrefill: 0.5, MfuDecode: 0.3},
			"", "", 1, true,
		),
	}

	// WHEN NewLatencyModel is called (roofline validation happens here)
	_, err := NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)

	// THEN it returns a non-nil error mentioning NumHeads
	if err == nil {
		t.Fatal("expected error for roofline with zero NumHeads, got nil")
	}
	if !strings.Contains(err.Error(), "NumHeads") {
		t.Errorf("error should mention NumHeads, got: %v", err)
	}
}

func TestNewSimulator_RooflineZeroTP_ReturnsError(t *testing.T) {
	// GIVEN a SimConfig with Roofline=true and TP=0
	cfg := SimConfig{
		Horizon:       100000,
		KVCacheConfig: NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		LatencyCoeffs: NewLatencyCoeffs(nil, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(
			ModelConfig{NumHeads: 32, NumLayers: 32, HiddenDim: 4096},
			HardwareCalib{TFlopsPeak: 1000, BwPeakTBs: 3.35, BwEffConstant: 0.7, MfuPrefill: 0.5, MfuDecode: 0.3},
			"", "", 0, true,
		),
	}

	// WHEN NewLatencyModel is called (roofline validation happens here)
	_, err := NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)

	// THEN it returns a non-nil error mentioning TP
	if err == nil {
		t.Fatal("expected error for roofline with zero TP, got nil")
	}
	if !strings.Contains(err.Error(), "TP") {
		t.Errorf("error should mention TP, got: %v", err)
	}
}

func TestNewSimulator_NonRooflineZeroNumHeads_Succeeds(t *testing.T) {
	// GIVEN a SimConfig with Roofline=false and NumHeads=0 (irrelevant)
	cfg := SimConfig{
		Horizon:             100000,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1, 2, 3}, []float64{1, 2, 3}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{NumHeads: 0}, HardwareCalib{}, "", "", 0, false),
	}

	// WHEN NewSimulator is called
	kvStore := NewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
	latencyModel, err := NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("NewLatencyModel: %v", err)
	}
	sim, err := NewSimulator(cfg, kvStore, latencyModel)

	// THEN it succeeds (roofline validation not applied)
	if err != nil {
		t.Fatalf("unexpected error for non-roofline mode: %v", err)
	}
	if sim == nil {
		t.Error("expected non-nil simulator")
	}
}
