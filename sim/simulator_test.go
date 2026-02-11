package sim

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
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

// loadGoldenDataset loads the golden dataset from the testdata directory
func loadGoldenDataset(t *testing.T) *GoldenDataset {
	t.Helper()

	// Find testdata relative to this test file using runtime.Caller
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("Failed to get current file path")
	}
	path := filepath.Join(filepath.Dir(thisFile), "..", "testdata", "goldendataset.json")
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
// This is the critical backward compatibility test ensuring RNG changes do not alter simulation outcomes.
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

			// === Exact match metrics (integers) ===
			// These prove identical RNG sequences were generated
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

			// === Compute derived metrics from simulation results ===
			vllmRuntime := float64(sim.Metrics.SimEndedTime) / float64(1e6)
			responsesPerSec := float64(sim.Metrics.CompletedRequests) / vllmRuntime
			tokensPerSec := float64(sim.Metrics.TotalOutputTokens) / vllmRuntime

			// TTFT statistics
			sortedTTFTs := make([]float64, 0, len(sim.Metrics.RequestTTFTs))
			for _, v := range sim.Metrics.RequestTTFTs {
				sortedTTFTs = append(sortedTTFTs, v)
			}
			sort.Float64s(sortedTTFTs)

			// E2E statistics
			sortedE2Es := make([]float64, 0, len(sim.Metrics.RequestE2Es))
			for _, v := range sim.Metrics.RequestE2Es {
				sortedE2Es = append(sortedE2Es, v)
			}
			sort.Float64s(sortedE2Es)

			// ITL statistics
			slices.Sort(sim.Metrics.AllITLs)

			// Scheduling delay statistics
			sortedSchedulingDelays := make([]float64, 0, len(sim.Metrics.RequestSchedulingDelays))
			for _, v := range sim.Metrics.RequestSchedulingDelays {
				sortedSchedulingDelays = append(sortedSchedulingDelays, float64(v))
			}
			sort.Float64s(sortedSchedulingDelays)

			// === Floating-point metrics with tolerance ===
			// Using relative tolerance of 1e-9 for deterministic floating-point comparisons
			const relTol = 1e-9

			// vLLM estimated duration (simulation clock based, deterministic)
			assertFloat64Equal(t, "vllm_estimated_duration_s", tc.Metrics.VllmEstimatedDurationS, vllmRuntime, relTol)

			// Throughput metrics
			assertFloat64Equal(t, "responses_per_sec", tc.Metrics.ResponsesPerSec, responsesPerSec, relTol)
			assertFloat64Equal(t, "tokens_per_sec", tc.Metrics.TokensPerSec, tokensPerSec, relTol)

			// E2E latency metrics
			assertFloat64Equal(t, "e2e_mean_ms", tc.Metrics.E2EMeanMs, CalculateMean(sortedE2Es), relTol)
			assertFloat64Equal(t, "e2e_p90_ms", tc.Metrics.E2EP90Ms, CalculatePercentile(sortedE2Es, 90), relTol)
			assertFloat64Equal(t, "e2e_p95_ms", tc.Metrics.E2EP95Ms, CalculatePercentile(sortedE2Es, 95), relTol)
			assertFloat64Equal(t, "e2e_p99_ms", tc.Metrics.E2EP99Ms, CalculatePercentile(sortedE2Es, 99), relTol)

			// TTFT latency metrics
			assertFloat64Equal(t, "ttft_mean_ms", tc.Metrics.TTFTMeanMs, CalculateMean(sortedTTFTs), relTol)
			assertFloat64Equal(t, "ttft_p90_ms", tc.Metrics.TTFTP90Ms, CalculatePercentile(sortedTTFTs, 90), relTol)
			assertFloat64Equal(t, "ttft_p95_ms", tc.Metrics.TTFTP95Ms, CalculatePercentile(sortedTTFTs, 95), relTol)
			assertFloat64Equal(t, "ttft_p99_ms", tc.Metrics.TTFTP99Ms, CalculatePercentile(sortedTTFTs, 99), relTol)

			// ITL latency metrics
			assertFloat64Equal(t, "itl_mean_ms", tc.Metrics.ITLMeanMs, CalculateMean(sim.Metrics.AllITLs), relTol)
			assertFloat64Equal(t, "itl_p90_ms", tc.Metrics.ITLP90Ms, CalculatePercentile(sim.Metrics.AllITLs, 90), relTol)
			assertFloat64Equal(t, "itl_p95_ms", tc.Metrics.ITLP95Ms, CalculatePercentile(sim.Metrics.AllITLs, 95), relTol)
			assertFloat64Equal(t, "itl_p99_ms", tc.Metrics.ITLP99Ms, CalculatePercentile(sim.Metrics.AllITLs, 99), relTol)

			// Scheduling delay
			assertFloat64Equal(t, "scheduling_delay_p99_ms", tc.Metrics.SchedulingDelayP99Ms, CalculatePercentile(sortedSchedulingDelays, 99), relTol)
		})
	}
}

// assertFloat64Equal compares two float64 values with relative tolerance
func assertFloat64Equal(t *testing.T, name string, want, got, relTol float64) {
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
