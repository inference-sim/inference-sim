package sim

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// GoldenDataset represents the structure of testdata/goldendataset.json
type GoldenDataset struct {
	Tests []GoldenTestCase `json:"tests"`
}

// GoldenTestCase represents a single test case from the golden dataset
type GoldenTestCase struct {
	Model                     string    `json:"model"`
	Workload                  string    `json:"workload"`
	Approach                  string    `json:"approach"`
	Rate                      float64   `json:"rate"`
	MaxPrompts                int       `json:"max-prompts"`
	PrefixTokens              int       `json:"prefix_tokens"`
	PromptTokens              int       `json:"prompt_tokens"`
	PromptTokensStdev         int       `json:"prompt_tokens_stdev"`
	PromptTokensMin           int       `json:"prompt_tokens_min"`
	PromptTokensMax           int       `json:"prompt_tokens_max"`
	OutputTokens              int       `json:"output_tokens"`
	OutputTokensStdev         int       `json:"output_tokens_stdev"`
	OutputTokensMin           int       `json:"output_tokens_min"`
	OutputTokensMax           int       `json:"output_tokens_max"`
	Hardware                  string    `json:"hardware"`
	TP                        int       `json:"tp"`
	Seed                      int64     `json:"seed"`
	MaxNumRunningReqs         int64     `json:"max-num-running-reqs"`
	MaxNumScheduledTokens     int64     `json:"max-num-scheduled-tokens"`
	MaxModelLen               int       `json:"max-model-len"`
	TotalKVBlocks             int64     `json:"total-kv-blocks"`
	BlockSizeInTokens         int64     `json:"block-size-in-tokens"`
	LongPrefillTokenThreshold int64     `json:"long-prefill-token-threshold"`
	AlphaCoeffs               []float64 `json:"alpha-coeffs"`
	BetaCoeffs                []float64 `json:"beta-coeffs"`
	Metrics                   GoldenMetrics `json:"metrics"`
}

// GoldenMetrics represents the expected metrics from a golden test case
type GoldenMetrics struct {
	CompletedRequests int `json:"completed_requests"`
	TotalInputTokens  int `json:"total_input_tokens"`
	TotalOutputTokens int `json:"total_output_tokens"`
}

// loadGoldenDataset loads the golden dataset from the testdata directory
func loadGoldenDataset(t *testing.T) *GoldenDataset {
	t.Helper()

	// Find testdata relative to this test file
	path := filepath.Join("..", "testdata", "goldendataset.json")
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

// TestSimulator_GoldenDataset verifies backward compatibility by running
// all test cases from the golden dataset and comparing results.
// This is the critical backward compatibility test for PR1.
func TestSimulator_GoldenDataset(t *testing.T) {
	dataset := loadGoldenDataset(t)

	if len(dataset.Tests) == 0 {
		t.Fatal("Golden dataset contains no test cases")
	}

	for _, tc := range dataset.Tests {
		t.Run(tc.Model, func(t *testing.T) {
			// Build GuideLLMConfig from test case
			guideLLMConfig := &GuideLLMConfig{
				Rate:               tc.Rate / 1e6, // Convert to internal rate format
				MaxPrompts:         tc.MaxPrompts,
				PrefixTokens:       tc.PrefixTokens,
				PromptTokens:       tc.PromptTokens,
				PromptTokensStdDev: tc.PromptTokensStdev,
				PromptTokensMin:    tc.PromptTokensMin,
				PromptTokensMax:    tc.PromptTokensMax,
				OutputTokens:       tc.OutputTokens,
				OutputTokensStdDev: tc.OutputTokensStdev,
				OutputTokensMin:    tc.OutputTokensMin,
				OutputTokensMax:    tc.OutputTokensMax,
			}

			// Create simulator with test case config
			sim := NewSimulator(
				math.MaxInt64,                // horizon
				tc.Seed,                      // seed (should be 42)
				tc.TotalKVBlocks,
				tc.BlockSizeInTokens,
				tc.MaxNumRunningReqs,
				tc.MaxNumScheduledTokens,
				tc.LongPrefillTokenThreshold,
				tc.BetaCoeffs,
				tc.AlphaCoeffs,
				guideLLMConfig,
				ModelConfig{},    // empty model config (not using roofline)
				HardwareCalib{},  // empty hardware config
				tc.Model,
				tc.Hardware,
				tc.TP,
				false, // roofline = false (using blackbox mode)
				"",    // no traces file
			)

			// Run simulation
			sim.Run()

			// Verify exact match on critical metrics
			// These metrics prove identical RNG sequences were generated
			if sim.Metrics.CompletedRequests != tc.Metrics.CompletedRequests {
				t.Errorf("completed_requests: got %d, want %d",
					sim.Metrics.CompletedRequests, tc.Metrics.CompletedRequests)
			}

			if sim.Metrics.TotalInputTokens != tc.Metrics.TotalInputTokens {
				t.Errorf("total_input_tokens: got %d, want %d",
					sim.Metrics.TotalInputTokens, tc.Metrics.TotalInputTokens)
			}

			if sim.Metrics.TotalOutputTokens != tc.Metrics.TotalOutputTokens {
				t.Errorf("total_output_tokens: got %d, want %d",
					sim.Metrics.TotalOutputTokens, tc.Metrics.TotalOutputTokens)
			}
		})
	}
}

// TestSimulator_WorkloadRNG_NotNil verifies the WorkloadRNG accessor never returns nil
func TestSimulator_WorkloadRNG_NotNil(t *testing.T) {
	sim := NewSimulator(
		1000000,  // horizon
		42,       // seed
		1000,     // totalKVBlocks
		16,       // blockSizeTokens
		256,      // maxRunningReqs
		2048,     // maxScheduledTokens
		0,        // longPrefillTokenThreshold
		[]float64{1000, 10, 5},  // betaCoeffs
		[]float64{100, 1, 100},  // alphaCoeffs
		&GuideLLMConfig{
			Rate:               10.0 / 1e6,
			MaxPrompts:         10,
			PrefixTokens:       0,
			PromptTokens:       100,
			PromptTokensStdDev: 10,
			PromptTokensMin:    10,
			PromptTokensMax:    200,
			OutputTokens:       50,
			OutputTokensStdDev: 10,
			OutputTokensMin:    10,
			OutputTokensMax:    100,
		},
		ModelConfig{},
		HardwareCalib{},
		"test-model",
		"H100",
		1,
		false,
		"",
	)

	rng := sim.WorkloadRNG()
	if rng == nil {
		t.Error("WorkloadRNG() returned nil")
	}

	// Should be able to draw values
	val := rng.Float64()
	if val < 0 || val >= 1 {
		t.Errorf("WorkloadRNG().Float64() = %v, want [0, 1)", val)
	}
}

// TestSimulator_DeterministicWorkload verifies same seed produces same workload
func TestSimulator_DeterministicWorkload(t *testing.T) {
	config := &GuideLLMConfig{
		Rate:               10.0 / 1e6,
		MaxPrompts:         50,
		PrefixTokens:       0,
		PromptTokens:       100,
		PromptTokensStdDev: 20,
		PromptTokensMin:    10,
		PromptTokensMax:    200,
		OutputTokens:       50,
		OutputTokensStdDev: 10,
		OutputTokensMin:    10,
		OutputTokensMax:    100,
	}

	// Create two simulators with same seed
	sim1 := NewSimulator(
		math.MaxInt64, 42, 10000, 16, 256, 2048, 0,
		[]float64{1000, 10, 5}, []float64{100, 1, 100},
		config, ModelConfig{}, HardwareCalib{},
		"test", "H100", 1, false, "",
	)

	sim2 := NewSimulator(
		math.MaxInt64, 42, 10000, 16, 256, 2048, 0,
		[]float64{1000, 10, 5}, []float64{100, 1, 100},
		config, ModelConfig{}, HardwareCalib{},
		"test", "H100", 1, false, "",
	)

	// Run both
	sim1.Run()
	sim2.Run()

	// Results must be identical
	if sim1.Metrics.CompletedRequests != sim2.Metrics.CompletedRequests {
		t.Errorf("Determinism broken: completed_requests %d vs %d",
			sim1.Metrics.CompletedRequests, sim2.Metrics.CompletedRequests)
	}

	if sim1.Metrics.TotalInputTokens != sim2.Metrics.TotalInputTokens {
		t.Errorf("Determinism broken: total_input_tokens %d vs %d",
			sim1.Metrics.TotalInputTokens, sim2.Metrics.TotalInputTokens)
	}

	if sim1.Metrics.TotalOutputTokens != sim2.Metrics.TotalOutputTokens {
		t.Errorf("Determinism broken: total_output_tokens %d vs %d",
			sim1.Metrics.TotalOutputTokens, sim2.Metrics.TotalOutputTokens)
	}
}
