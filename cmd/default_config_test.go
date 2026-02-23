package cmd

import (
	"os"
	"testing"
)

// TestGetCoefficients_StrictParsing_RejectsUnknownFields verifies BC-5 (R10):
// GIVEN a YAML file with a typo (unknown field "beta_coefs" instead of "beta_coeffs")
// WHEN GetCoefficients parses the file
// THEN it MUST panic with an error (strict parsing rejects unknown fields).
func TestGetCoefficients_StrictParsing_RejectsUnknownFields(t *testing.T) {
	tmpFile, err := os.CreateTemp(t.TempDir(), "defaults-*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	content := `
models:
  - id: "test-model"
    GPU: "H100"
    tensor_parallelism: 2
    vllm_version: "0.6.6"
    alpha_coeffs: [100, 1, 100]
    beta_coefs: [1000, 10, 5]
    total_kv_blocks: 100
`
	if _, err := tmpFile.WriteString(content); err != nil {
		t.Fatal(err)
	}
	_ = tmpFile.Close()

	defer func() {
		r := recover()
		if r == nil {
			t.Error("expected panic for unknown field 'beta_coefs', got none")
		}
	}()

	GetCoefficients("test-model", 2, "H100", "0.6.6", tmpFile.Name())
}

// TestGetCoefficients_ReturnsTotalKVBlocks_CallerMustCheckChanged verifies BC-5 (#285):
// GetCoefficients returns the model's default totalKVBlocks, but callers MUST
// use cmd.Flags().Changed() to avoid overwriting user-provided values (R18).
func TestGetCoefficients_ReturnsModelDefaultKVBlocks(t *testing.T) {
	// Skip if defaults.yaml not available
	path := "defaults.yaml"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		path = "../defaults.yaml"
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Skip("defaults.yaml not found, skipping integration test")
		}
	}

	// GIVEN a known model in defaults.yaml
	alpha, beta, kvBlocks := GetCoefficients(
		"meta-llama/llama-3.1-8b-instruct",
		1, "H100", "vllm/vllm-openai:v0.8.4",
		path,
	)

	// THEN coefficients should be non-nil
	if alpha == nil || beta == nil {
		t.Fatal("expected non-nil coefficients for known model")
	}

	// AND kvBlocks should be the model's default (non-zero)
	if kvBlocks == 0 {
		t.Error("expected non-zero totalKVBlocks from model defaults")
	}

	// The key invariant (R18): callers must check cmd.Flags().Changed("total-kv-blocks")
	// before using kvBlocks. The fix is at cmd/root.go:173.
	// This test documents the contract: GetCoefficients always returns the model
	// default. It's the caller's job to not overwrite user values.
	t.Logf("Model default totalKVBlocks: %d (callers must check Changed() before using)", kvBlocks)
}
