package sim

import (
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
