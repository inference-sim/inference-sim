// Package testutil provides shared test infrastructure for the BLIS simulator.
// It consolidates golden dataset types and assertion helpers used across
// sim/ and sim/cluster/ test packages.
package testutil

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// GoldenDataset represents the structure of testdata/goldendataset.json.
type GoldenDataset struct {
	Tests []GoldenTestCase `json:"tests"`
}

// GoldenTestCase represents a single test case from the golden dataset.
type GoldenTestCase struct {
	Model                     string        `json:"model"`
	Workload                  string        `json:"workload"`
	Approach                  string        `json:"approach"`
	Rate                      float64       `json:"rate"`
	MaxPrompts                int           `json:"max-prompts"`
	PrefixTokens              int           `json:"prefix_tokens"`
	PromptTokens              int           `json:"prompt_tokens"`
	PromptTokensStdev         int           `json:"prompt_tokens_stdev"`
	PromptTokensMin           int           `json:"prompt_tokens_min"`
	PromptTokensMax           int           `json:"prompt_tokens_max"`
	OutputTokens              int           `json:"output_tokens"`
	OutputTokensStdev         int           `json:"output_tokens_stdev"`
	OutputTokensMin           int           `json:"output_tokens_min"`
	OutputTokensMax           int           `json:"output_tokens_max"`
	Hardware                  string        `json:"hardware"`
	TP                        int           `json:"tp"`
	Seed                      int64         `json:"seed"`
	MaxNumRunningReqs         int64         `json:"max-num-running-reqs"`
	MaxNumScheduledTokens     int64         `json:"max-num-scheduled-tokens"`
	MaxModelLen               int           `json:"max-model-len"`
	TotalKVBlocks             int64         `json:"total-kv-blocks"`
	BlockSizeInTokens         int64         `json:"block-size-in-tokens"`
	LongPrefillTokenThreshold int64         `json:"long-prefill-token-threshold"`
	AlphaCoeffs               []float64     `json:"alpha-coeffs"`
	BetaCoeffs                []float64     `json:"beta-coeffs"`
	Metrics                   GoldenMetrics `json:"metrics"`
}

// GoldenMetrics represents the expected metrics from a golden test case.
type GoldenMetrics struct {
	// Exact match metrics (integers)
	CompletedRequests int `json:"completed_requests"`
	TotalInputTokens  int `json:"total_input_tokens"`
	TotalOutputTokens int `json:"total_output_tokens"`

	// Deterministic floating-point metrics (derived from simulation clock)
	VllmEstimatedDurationS float64 `json:"vllm_estimated_duration_s"`
	ResponsesPerSec        float64 `json:"responses_per_sec"`
	TokensPerSec           float64 `json:"tokens_per_sec"`

	// E2E latency metrics
	E2EMeanMs float64 `json:"e2e_mean_ms"`
	E2EP90Ms  float64 `json:"e2e_p90_ms"`
	E2EP95Ms  float64 `json:"e2e_p95_ms"`
	E2EP99Ms  float64 `json:"e2e_p99_ms"`

	// TTFT latency metrics
	TTFTMeanMs float64 `json:"ttft_mean_ms"`
	TTFTP90Ms  float64 `json:"ttft_p90_ms"`
	TTFTP95Ms  float64 `json:"ttft_p95_ms"`
	TTFTP99Ms  float64 `json:"ttft_p99_ms"`

	// ITL latency metrics
	ITLMeanMs float64 `json:"itl_mean_ms"`
	ITLP90Ms  float64 `json:"itl_p90_ms"`
	ITLP95Ms  float64 `json:"itl_p95_ms"`
	ITLP99Ms  float64 `json:"itl_p99_ms"`

	// Scheduling delay
	SchedulingDelayP99Ms float64 `json:"scheduling_delay_p99_ms"`

	// Note: simulation_duration_s is wall clock time and NOT deterministic, so not tested
}

// LoadGoldenDataset loads the golden dataset from the testdata directory.
// The path is resolved relative to this source file: sim/internal/testutil/ → testdata/.
func LoadGoldenDataset(t *testing.T) *GoldenDataset {
	t.Helper()

	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("Failed to get current file path")
	}
	// Navigate from sim/internal/testutil/ to repo root testdata/
	path := filepath.Join(filepath.Dir(thisFile), "..", "..", "..", "testdata", "goldendataset.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("Failed to read golden dataset: %v", err)
	}

	var dataset GoldenDataset
	if err := json.Unmarshal(data, &dataset); err != nil {
		t.Fatalf("Failed to parse golden dataset: %v", err)
	}

	return &dataset
}

// AssertFloat64Equal compares two float64 values with relative tolerance.
func AssertFloat64Equal(t *testing.T, name string, want, got, relTol float64) {
	t.Helper()
	if want == 0 && got == 0 {
		return
	}
	diff := math.Abs(want - got)
	maxVal := math.Max(math.Abs(want), math.Abs(got))
	if diff/maxVal > relTol {
		t.Errorf("%s: got %v, want %v (diff=%v, relDiff=%v)", name, got, want, diff, diff/maxVal)
	}
}
