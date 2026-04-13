package cmd

import (
	"os"
	"testing"

	"github.com/sirupsen/logrus"
)

// TestGetCoefficients_StrictParsing_RejectsUnknownFields verifies BC-5 (R10):
// GIVEN a YAML file with a typo (unknown field "beta_coefs" instead of "beta_coeffs")
// WHEN GetCoefficients parses the file
// THEN it MUST call logrus.Fatalf (strict parsing rejects unknown fields).
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
`
	if _, err := tmpFile.WriteString(content); err != nil {
		t.Fatal(err)
	}
	_ = tmpFile.Close()

	// Intercept logrus.Fatalf to prevent os.Exit in tests
	fatalCalled := false
	logrus.StandardLogger().ExitFunc = func(int) { fatalCalled = true }
	defer func() { logrus.StandardLogger().ExitFunc = nil }()

	GetCoefficients("test-model", 2, "H100", "0.6.6", tmpFile.Name())

	if !fatalCalled {
		t.Error("expected logrus.Fatalf for unknown field 'beta_coefs', but it was not called")
	}
}

// TestGetCoefficients_ReturnsCoefficientsOnly verifies that GetCoefficients
// returns only alpha and beta coefficients, no longer returning KV blocks (#1035).
func TestGetCoefficients_ReturnsCoefficientsOnly(t *testing.T) {
	// Skip if defaults.yaml not available
	path := "defaults.yaml"
	if _, err := os.Stat(path); os.IsNotExist(err) {
		path = "../defaults.yaml"
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Skip("defaults.yaml not found, skipping integration test")
		}
	}

	// GIVEN a known model in defaults.yaml
	alpha, beta := GetCoefficients(
		"meta-llama/llama-3.1-8b-instruct",
		1, "H100", "vllm/vllm-openai:v0.11.0",
		path,
	)

	// THEN coefficients should be non-nil
	if alpha == nil || beta == nil {
		t.Fatal("expected non-nil coefficients for known model")
	}

	// GetCoefficients now returns only coefficients. KV blocks are determined by:
	// (1) CLI flag --total-kv-blocks, (2) auto-calculation, or (3) default value.
	t.Logf("Loaded alpha coefficients: %v", alpha)
	t.Logf("Loaded beta coefficients: %v", beta)
}
