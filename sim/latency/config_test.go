package latency_test

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
)

func TestGetHWConfig_MalformedJSON(t *testing.T) {
	// Create temp file with malformed JSON
	tmpDir := t.TempDir()
	badFile := filepath.Join(tmpDir, "bad_hw.json")
	if err := os.WriteFile(badFile, []byte(`{"H100": invalid`), 0644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	_, err := latency.GetHWConfig(badFile, "H100")
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

	_, err := latency.GetHWConfig(validFile, "H200")
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

	cfg, err := latency.GetHWConfig(validFile, "H100")
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

	_, err := latency.GetModelConfig(badFile)
	if err == nil {
		t.Error("expected error for malformed JSON, got nil")
	}
}

func TestGetModelConfig_MissingTorchDtype(t *testing.T) {
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "config.json")
	// Valid JSON but missing both torch_dtype and dtype
	content := `{"num_hidden_layers": 32, "hidden_size": 4096}`
	if err := os.WriteFile(configFile, []byte(content), 0644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	cfg, err := latency.GetModelConfig(configFile)
	if err != nil {
		t.Errorf("should not error for missing torch_dtype (default to 0): %v", err)
	}
	if cfg == nil {
		t.Fatal("expected non-nil config")
	}
	if cfg.BytesPerParam != 0 {
		t.Errorf("expected BytesPerParam=0 when both torch_dtype and dtype are missing, got %v", cfg.BytesPerParam)
	}
}

func TestGetModelConfig_DtypeFallback(t *testing.T) {
	// GIVEN a config.json with "dtype" instead of "torch_dtype" (e.g. GLM-5)
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "config.json")
	content := `{
		"num_hidden_layers": 40,
		"hidden_size": 4096,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"vocab_size": 151552,
		"intermediate_size": 13696,
		"dtype": "bfloat16"
	}`
	if err := os.WriteFile(configFile, []byte(content), 0644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	// WHEN GetModelConfig parses the config
	cfg, err := latency.GetModelConfig(configFile)

	// THEN BytesPerParam is resolved from the "dtype" field
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.BytesPerParam != 2 {
		t.Errorf("expected BytesPerParam=2 for bfloat16 via dtype fallback, got %v", cfg.BytesPerParam)
	}
	if cfg.NumLayers != 40 {
		t.Errorf("expected NumLayers=40, got %v", cfg.NumLayers)
	}
}

func TestGetModelConfig_TorchDtypeTakesPrecedenceOverDtype(t *testing.T) {
	// GIVEN a config.json with both "torch_dtype" and "dtype"
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "config.json")
	content := `{
		"num_hidden_layers": 32,
		"hidden_size": 4096,
		"num_attention_heads": 32,
		"torch_dtype": "float32",
		"dtype": "bfloat16"
	}`
	if err := os.WriteFile(configFile, []byte(content), 0644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	// WHEN GetModelConfig parses the config
	cfg, err := latency.GetModelConfig(configFile)

	// THEN torch_dtype wins (float32 = 4 bytes, not bfloat16 = 2 bytes)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.BytesPerParam != 4 {
		t.Errorf("expected BytesPerParam=4 (torch_dtype=float32 takes precedence), got %v", cfg.BytesPerParam)
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

	cfg, err := latency.GetModelConfig(configFile)
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

func TestGetModelConfig_MoEConfig_ParsesStandardFields(t *testing.T) {
	// GIVEN a Mixtral-style MoE config.json with standard field names
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "config.json")
	content := `{
		"num_hidden_layers": 32,
		"hidden_size": 4096,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"vocab_size": 32000,
		"intermediate_size": 14336,
		"torch_dtype": "bfloat16",
		"num_local_experts": 8,
		"num_experts_per_tok": 2
	}`
	if err := os.WriteFile(configFile, []byte(content), 0644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	// WHEN GetModelConfig parses the config
	cfg, err := latency.GetModelConfig(configFile)

	// THEN all fields are correctly extracted (MoE-specific fields are ignored but don't break parsing)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.NumLayers != 32 {
		t.Errorf("expected NumLayers=32, got %v", cfg.NumLayers)
	}
	if cfg.HiddenDim != 4096 {
		t.Errorf("expected HiddenDim=4096, got %v", cfg.HiddenDim)
	}
	if cfg.NumHeads != 32 {
		t.Errorf("expected NumHeads=32, got %v", cfg.NumHeads)
	}
	if cfg.NumKVHeads != 8 {
		t.Errorf("expected NumKVHeads=8 (GQA), got %v", cfg.NumKVHeads)
	}
	if cfg.IntermediateDim != 14336 {
		t.Errorf("expected IntermediateDim=14336, got %v", cfg.IntermediateDim)
	}
	if cfg.BytesPerParam != 2 {
		t.Errorf("expected BytesPerParam=2 for bfloat16, got %v", cfg.BytesPerParam)
	}
}

func TestGetModelConfig_FalconFieldNames(t *testing.T) {
	// GIVEN a Falcon-style config.json using non-standard field names:
	//   num_kv_heads (instead of num_key_value_heads)
	//   ffn_hidden_size (instead of intermediate_size)
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "config.json")
	content := `{
		"num_hidden_layers": 60,
		"hidden_size": 4096,
		"num_attention_heads": 32,
		"num_kv_heads": 8,
		"vocab_size": 65024,
		"ffn_hidden_size": 16384,
		"torch_dtype": "bfloat16"
	}`
	if err := os.WriteFile(configFile, []byte(content), 0644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	// WHEN GetModelConfig parses the config
	cfg, err := latency.GetModelConfig(configFile)

	// THEN fallback field names are used correctly
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.NumKVHeads != 8 {
		t.Errorf("expected NumKVHeads=8 via num_kv_heads fallback, got %v", cfg.NumKVHeads)
	}
	if cfg.IntermediateDim != 16384 {
		t.Errorf("expected IntermediateDim=16384 via ffn_hidden_size fallback, got %v", cfg.IntermediateDim)
	}
}

func TestGetModelConfig_GLMFieldNames(t *testing.T) {
	// GIVEN a GLM-style config.json using non-standard field names:
	//   multi_query_group_num (instead of num_key_value_heads)
	//   ffn_hidden_size (instead of intermediate_size)
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "config.json")
	content := `{
		"num_hidden_layers": 40,
		"hidden_size": 4096,
		"num_attention_heads": 32,
		"multi_query_group_num": 2,
		"vocab_size": 151552,
		"ffn_hidden_size": 13696,
		"dtype": "bfloat16"
	}`
	if err := os.WriteFile(configFile, []byte(content), 0644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	// WHEN GetModelConfig parses the config
	cfg, err := latency.GetModelConfig(configFile)

	// THEN fallback field names are used correctly
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.NumKVHeads != 2 {
		t.Errorf("expected NumKVHeads=2 via multi_query_group_num fallback, got %v", cfg.NumKVHeads)
	}
	if cfg.IntermediateDim != 13696 {
		t.Errorf("expected IntermediateDim=13696 via ffn_hidden_size fallback, got %v", cfg.IntermediateDim)
	}
	if cfg.BytesPerParam != 2 {
		t.Errorf("expected BytesPerParam=2 via dtype fallback, got %v", cfg.BytesPerParam)
	}
}

func TestGetModelConfig_StandardFieldsTakePrecedenceOverFallbacks(t *testing.T) {
	// GIVEN a config.json with both standard and fallback field names
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, "config.json")
	content := `{
		"num_hidden_layers": 32,
		"hidden_size": 4096,
		"num_attention_heads": 32,
		"num_key_value_heads": 8,
		"num_kv_heads": 4,
		"intermediate_size": 14336,
		"ffn_hidden_size": 11008,
		"torch_dtype": "bfloat16"
	}`
	if err := os.WriteFile(configFile, []byte(content), 0644); err != nil {
		t.Fatalf("failed to create test file: %v", err)
	}

	// WHEN GetModelConfig parses the config
	cfg, err := latency.GetModelConfig(configFile)

	// THEN standard field names take precedence
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.NumKVHeads != 8 {
		t.Errorf("expected NumKVHeads=8 (standard field takes precedence), got %v", cfg.NumKVHeads)
	}
	if cfg.IntermediateDim != 14336 {
		t.Errorf("expected IntermediateDim=14336 (standard field takes precedence), got %v", cfg.IntermediateDim)
	}
}

func TestValidateRooflineConfig_ZeroModelFields_ReturnsError(t *testing.T) {
	hc := sim.HardwareCalib{TFlopsPeak: 1000, BwPeakTBs: 3.35, BwEffConstant: 0.7, MfuPrefill: 0.5, MfuDecode: 0.3, MemoryGiB: 80.0}

	tests := []struct {
		name  string
		mc    sim.ModelConfig
		field string
	}{
		{"zero NumHeads", sim.ModelConfig{NumHeads: 0, NumLayers: 32, HiddenDim: 4096, BytesPerParam: 2}, "NumHeads"},
		{"zero NumLayers", sim.ModelConfig{NumHeads: 32, NumLayers: 0, HiddenDim: 4096, BytesPerParam: 2}, "NumLayers"},
		{"zero HiddenDim", sim.ModelConfig{NumHeads: 32, NumLayers: 32, HiddenDim: 0, BytesPerParam: 2}, "HiddenDim"},
		{"zero BytesPerParam", sim.ModelConfig{NumHeads: 32, NumLayers: 32, HiddenDim: 4096, BytesPerParam: 0}, "BytesPerParam"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// WHEN ValidateRooflineConfig is called
			err := latency.ValidateRooflineConfig(tt.mc, hc)

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
	mc := sim.ModelConfig{NumHeads: 32, NumLayers: 32, HiddenDim: 4096, BytesPerParam: 2}
	hc := sim.HardwareCalib{} // all zero

	// WHEN ValidateRooflineConfig is called
	err := latency.ValidateRooflineConfig(mc, hc)

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
	mc := sim.ModelConfig{NumHeads: 32, NumLayers: 32, HiddenDim: 4096}
	hc := sim.HardwareCalib{
		TFlopsPeak:    math.NaN(),
		BwPeakTBs:     math.Inf(1),
		BwEffConstant: 0.7,
		MfuPrefill:    0.5,
		MfuDecode:     math.NaN(),
		MemoryGiB:     math.Inf(-1),
	}

	// WHEN ValidateRooflineConfig is called
	err := latency.ValidateRooflineConfig(mc, hc)

	// THEN it returns an error mentioning the invalid fields
	if err == nil {
		t.Fatal("expected error for NaN/Inf hardware fields, got nil")
	}
	errMsg := err.Error()
	for _, field := range []string{"TFlopsPeak", "BwPeakTBs", "MfuDecode", "MemoryGiB"} {
		if !strings.Contains(errMsg, field) {
			t.Errorf("error should mention %s, got: %v", field, errMsg)
		}
	}
}

func TestValidateRooflineConfig_NaNMemoryGiB_ReturnsError(t *testing.T) {
	// NaN != 0 is true in IEEE 754, so NaN passes the outer guard and must
	// be caught by the inner math.IsNaN check. This test covers that path.
	mc := sim.ModelConfig{NumHeads: 32, NumLayers: 32, HiddenDim: 4096, BytesPerParam: 2}
	hc := sim.HardwareCalib{TFlopsPeak: 1000, BwPeakTBs: 3.35, BwEffConstant: 0.7, MfuPrefill: 0.5, MfuDecode: 0.3, MemoryGiB: math.NaN()}

	err := latency.ValidateRooflineConfig(mc, hc)

	if err == nil {
		t.Fatal("expected error for NaN MemoryGiB, got nil")
	}
	if !strings.Contains(err.Error(), "MemoryGiB") {
		t.Errorf("error should mention MemoryGiB, got: %v", err)
	}
}

func TestValidateRooflineConfig_NegativeMemoryGiB_ReturnsError(t *testing.T) {
	// A plain negative value (not -Inf) exercises the hc.MemoryGiB < 0 branch,
	// which is distinct from the math.IsInf path tested by NaNInfFields.
	mc := sim.ModelConfig{NumHeads: 32, NumLayers: 32, HiddenDim: 4096, BytesPerParam: 2}
	hc := sim.HardwareCalib{TFlopsPeak: 1000, BwPeakTBs: 3.35, BwEffConstant: 0.7, MfuPrefill: 0.5, MfuDecode: 0.3, MemoryGiB: -80.0}

	err := latency.ValidateRooflineConfig(mc, hc)

	if err == nil {
		t.Fatal("expected error for negative MemoryGiB, got nil")
	}
	if !strings.Contains(err.Error(), "MemoryGiB") {
		t.Errorf("error should mention MemoryGiB, got: %v", err)
	}
}

func TestValidateRooflineConfig_ValidConfig_ReturnsNil(t *testing.T) {
	// GIVEN valid ModelConfig and HardwareCalib
	mc := sim.ModelConfig{NumHeads: 32, NumLayers: 32, HiddenDim: 4096, BytesPerParam: 2}
	hc := sim.HardwareCalib{TFlopsPeak: 1000, BwPeakTBs: 3.35, BwEffConstant: 0.7, MfuPrefill: 0.5, MfuDecode: 0.3, MemoryGiB: 80.0}

	// WHEN ValidateRooflineConfig is called
	err := latency.ValidateRooflineConfig(mc, hc)

	// THEN it returns nil
	if err != nil {
		t.Errorf("expected nil error for valid config, got: %v", err)
	}
}

// TestNewLatencyModel_RooflineZeroNumHeads_ReturnsError verifies roofline rejects zero NumHeads.
func TestNewLatencyModel_RooflineZeroNumHeads_ReturnsError(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs(nil, []float64{100, 1, 100})
	hw := sim.NewModelHardwareConfig(
		sim.ModelConfig{NumHeads: 0, NumLayers: 32, HiddenDim: 4096},
		sim.HardwareCalib{TFlopsPeak: 1000, BwPeakTBs: 3.35, BwEffConstant: 0.7, MfuPrefill: 0.5, MfuDecode: 0.3, MemoryGiB: 80.0},
		"", "", 1, true,
	)

	// WHEN NewLatencyModel is called (roofline validation happens here)
	_, err := latency.NewLatencyModel(coeffs, hw)

	// THEN it returns a non-nil error mentioning NumHeads
	if err == nil {
		t.Fatal("expected error for roofline with zero NumHeads, got nil")
	}
	if !strings.Contains(err.Error(), "NumHeads") {
		t.Errorf("error should mention NumHeads, got: %v", err)
	}
}

// TestNewLatencyModel_RooflineZeroTP_ReturnsError verifies roofline rejects zero TP.
func TestNewLatencyModel_RooflineZeroTP_ReturnsError(t *testing.T) {
	coeffs := sim.NewLatencyCoeffs(nil, []float64{100, 1, 100})
	hw := sim.NewModelHardwareConfig(
		sim.ModelConfig{NumHeads: 32, NumLayers: 32, HiddenDim: 4096},
		sim.HardwareCalib{TFlopsPeak: 1000, BwPeakTBs: 3.35, BwEffConstant: 0.7, MfuPrefill: 0.5, MfuDecode: 0.3, MemoryGiB: 80.0},
		"", "", 0, true,
	)

	// WHEN NewLatencyModel is called (roofline validation happens here)
	_, err := latency.NewLatencyModel(coeffs, hw)

	// THEN it returns a non-nil error mentioning TP
	if err == nil {
		t.Fatal("expected error for roofline with zero TP, got nil")
	}
	if !strings.Contains(err.Error(), "TP") {
		t.Errorf("error should mention TP, got: %v", err)
	}
}

// R7 companion invariant: every GPU in hardware_config.json must have positive MemoryGiB.
// Survives refactoring — any GPU with valid memory passes regardless of exact value.
func TestGetHWConfig_AllGPUs_HavePositiveMemoryGiB(t *testing.T) {
	hwConfigPath := filepath.Join("..", "..", "hardware_config.json")
	for _, gpu := range []string{"H100", "A100-SXM", "A100-80"} {
		cfg, err := latency.GetHWConfig(hwConfigPath, gpu)
		if err != nil {
			t.Fatalf("GPU %q: %v", gpu, err)
		}
		if cfg.MemoryGiB <= 0 {
			t.Errorf("GPU %q: MemoryGiB must be > 0, got %v", gpu, cfg.MemoryGiB)
		}
	}
}

func TestGetHWConfig_MemoryGiB_ParsedFromRealConfig(t *testing.T) {
	// GIVEN the real hardware_config.json in the repo root
	hwConfigPath := filepath.Join("..", "..", "hardware_config.json")

	tests := []struct {
		name string
		gpu  string
	}{
		{"H100", "H100"},
		{"A100-SXM", "A100-SXM"},
		{"A100-80 alias (BC-14)", "A100-80"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// WHEN GetHWConfig is called for the GPU
			cfg, err := latency.GetHWConfig(hwConfigPath, tt.gpu)

			// THEN it succeeds and MemoryGiB is 80.0
			if err != nil {
				t.Fatalf("unexpected error for GPU %q: %v", tt.gpu, err)
			}
			if cfg.MemoryGiB != 80.0 {
				t.Errorf("expected MemoryGiB=80.0 for %q, got %v", tt.gpu, cfg.MemoryGiB)
			}
		})
	}
}
