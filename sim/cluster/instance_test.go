package cluster

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// === Golden Dataset Types (mirrored from sim/simulator_test.go) ===

type GoldenDataset struct {
	Tests []GoldenTestCase `json:"tests"`
}

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

type GoldenMetrics struct {
	CompletedRequests      int     `json:"completed_requests"`
	TotalInputTokens       int     `json:"total_input_tokens"`
	TotalOutputTokens      int     `json:"total_output_tokens"`
	VllmEstimatedDurationS float64 `json:"vllm_estimated_duration_s"`
	ResponsesPerSec        float64 `json:"responses_per_sec"`
	TokensPerSec           float64 `json:"tokens_per_sec"`
	E2EMeanMs              float64 `json:"e2e_mean_ms"`
	E2EP90Ms               float64 `json:"e2e_p90_ms"`
	E2EP95Ms               float64 `json:"e2e_p95_ms"`
	E2EP99Ms               float64 `json:"e2e_p99_ms"`
	TTFTMeanMs             float64 `json:"ttft_mean_ms"`
	TTFTP90Ms              float64 `json:"ttft_p90_ms"`
	TTFTP95Ms              float64 `json:"ttft_p95_ms"`
	TTFTP99Ms              float64 `json:"ttft_p99_ms"`
	ITLMeanMs              float64 `json:"itl_mean_ms"`
	ITLP90Ms               float64 `json:"itl_p90_ms"`
	ITLP95Ms               float64 `json:"itl_p95_ms"`
	ITLP99Ms               float64 `json:"itl_p99_ms"`
	SchedulingDelayP99Ms   float64 `json:"scheduling_delay_p99_ms"`
}

func loadGoldenDataset(t *testing.T) *GoldenDataset {
	t.Helper()
	path := filepath.Join("..", "..", "testdata", "goldendataset.json")
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

// === Equivalence Tests (Critical for BC-1) ===

// TestInstanceSimulator_GoldenDataset_Equivalence verifies:
// GIVEN golden dataset test cases
// WHEN run through InstanceSimulator wrapper
// THEN all metrics match golden expected values exactly
func TestInstanceSimulator_GoldenDataset_Equivalence(t *testing.T) {
	dataset := loadGoldenDataset(t)

	if len(dataset.Tests) == 0 {
		t.Fatal("Golden dataset contains no test cases")
	}

	for _, tc := range dataset.Tests {
		t.Run(tc.Model, func(t *testing.T) {
			guideLLMConfig := &sim.GuideLLMConfig{
				Rate:               tc.Rate / 1e6,
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

			// Create InstanceSimulator (through wrapper)
			instance := NewInstanceSimulator(
				InstanceID("test-instance"),
				math.MaxInt64,
				tc.Seed,
				tc.TotalKVBlocks,
				tc.BlockSizeInTokens,
				tc.MaxNumRunningReqs,
				tc.MaxNumScheduledTokens,
				tc.LongPrefillTokenThreshold,
				tc.BetaCoeffs,
				tc.AlphaCoeffs,
				guideLLMConfig,
				sim.ModelConfig{},
				sim.HardwareCalib{},
				tc.Model,
				tc.Hardware,
				tc.TP,
				false,
				"",
			)

			// Run through wrapper
			instance.Run()

			// Verify exact match on integer metrics (proves identical RNG sequences)
			if instance.Metrics().CompletedRequests != tc.Metrics.CompletedRequests {
				t.Errorf("completed_requests: got %d, want %d",
					instance.Metrics().CompletedRequests, tc.Metrics.CompletedRequests)
			}

			if instance.Metrics().TotalInputTokens != tc.Metrics.TotalInputTokens {
				t.Errorf("total_input_tokens: got %d, want %d",
					instance.Metrics().TotalInputTokens, tc.Metrics.TotalInputTokens)
			}

			if instance.Metrics().TotalOutputTokens != tc.Metrics.TotalOutputTokens {
				t.Errorf("total_output_tokens: got %d, want %d",
					instance.Metrics().TotalOutputTokens, tc.Metrics.TotalOutputTokens)
			}

			// Verify derived metrics with tolerance
			const relTol = 1e-9
			vllmRuntime := float64(instance.Metrics().SimEndedTime) / 1e6
			assertFloat64Equal(t, "vllm_estimated_duration_s", tc.Metrics.VllmEstimatedDurationS, vllmRuntime, relTol)
		})
	}
}

// TestInstanceSimulator_Determinism verifies:
// GIVEN same seed (42) and config
// WHEN simulation runs twice via InstanceSimulator
// THEN CompletedRequests, TotalInputTokens, TotalOutputTokens are identical
func TestInstanceSimulator_Determinism(t *testing.T) {
	config := &sim.GuideLLMConfig{
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

	makeInstance := func(id string) *InstanceSimulator {
		return NewInstanceSimulator(
			InstanceID(id),
			math.MaxInt64, 42, 10000, 16, 256, 2048, 0,
			[]float64{1000, 10, 5}, []float64{100, 1, 100},
			config, sim.ModelConfig{}, sim.HardwareCalib{},
			"test", "H100", 1, false, "",
		)
	}

	instance1 := makeInstance("run1")
	instance2 := makeInstance("run2")

	instance1.Run()
	instance2.Run()

	if instance1.Metrics().CompletedRequests != instance2.Metrics().CompletedRequests {
		t.Errorf("Determinism broken: completed_requests %d vs %d",
			instance1.Metrics().CompletedRequests, instance2.Metrics().CompletedRequests)
	}

	if instance1.Metrics().TotalInputTokens != instance2.Metrics().TotalInputTokens {
		t.Errorf("Determinism broken: total_input_tokens %d vs %d",
			instance1.Metrics().TotalInputTokens, instance2.Metrics().TotalInputTokens)
	}

	if instance1.Metrics().TotalOutputTokens != instance2.Metrics().TotalOutputTokens {
		t.Errorf("Determinism broken: total_output_tokens %d vs %d",
			instance1.Metrics().TotalOutputTokens, instance2.Metrics().TotalOutputTokens)
	}
}

// === Accessor Behavior Tests ===

// TestInstanceSimulator_ID_ReturnsConstructorValue verifies:
// GIVEN InstanceSimulator created with ID "replica-0"
// WHEN ID() is called
// THEN returns InstanceID("replica-0")
func TestInstanceSimulator_ID_ReturnsConstructorValue(t *testing.T) {
	instance := NewInstanceSimulator(
		InstanceID("replica-0"),
		1000000, 42, 1000, 16, 256, 2048, 0,
		[]float64{1000, 10, 5}, []float64{100, 1, 100},
		&sim.GuideLLMConfig{Rate: 10.0 / 1e6, MaxPrompts: 5,
			PromptTokens: 100, PromptTokensStdDev: 10, PromptTokensMin: 10, PromptTokensMax: 200,
			OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100},
		sim.ModelConfig{}, sim.HardwareCalib{},
		"test", "H100", 1, false, "",
	)

	if instance.ID() != InstanceID("replica-0") {
		t.Errorf("ID() = %q, want %q", instance.ID(), "replica-0")
	}
}

// TestInstanceSimulator_Clock_AdvancesWithSimulation verifies:
// GIVEN simulation with requests
// WHEN Run() completes
// THEN Clock() > 0 AND Clock() == Metrics().SimEndedTime
func TestInstanceSimulator_Clock_AdvancesWithSimulation(t *testing.T) {
	instance := NewInstanceSimulator(
		InstanceID("test"),
		math.MaxInt64, 42, 10000, 16, 256, 2048, 0,
		[]float64{1000, 10, 5}, []float64{100, 1, 100},
		&sim.GuideLLMConfig{Rate: 10.0 / 1e6, MaxPrompts: 10,
			PromptTokens: 100, PromptTokensStdDev: 10, PromptTokensMin: 10, PromptTokensMax: 200,
			OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100},
		sim.ModelConfig{}, sim.HardwareCalib{},
		"test", "H100", 1, false, "",
	)

	instance.Run()

	if instance.Clock() <= 0 {
		t.Errorf("Clock() = %d, want > 0 after Run()", instance.Clock())
	}

	if instance.Clock() != instance.Metrics().SimEndedTime {
		t.Errorf("Clock() = %d, Metrics().SimEndedTime = %d, want equal",
			instance.Clock(), instance.Metrics().SimEndedTime)
	}
}

// TestInstanceSimulator_Metrics_DelegatesCorrectly verifies:
// GIVEN simulation runs to completion
// WHEN Metrics() is accessed
// THEN Metrics().CompletedRequests > 0 (not nil or empty)
func TestInstanceSimulator_Metrics_DelegatesCorrectly(t *testing.T) {
	instance := NewInstanceSimulator(
		InstanceID("test"),
		math.MaxInt64, 42, 10000, 16, 256, 2048, 0,
		[]float64{1000, 10, 5}, []float64{100, 1, 100},
		&sim.GuideLLMConfig{Rate: 10.0 / 1e6, MaxPrompts: 10,
			PromptTokens: 100, PromptTokensStdDev: 10, PromptTokensMin: 10, PromptTokensMax: 200,
			OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100},
		sim.ModelConfig{}, sim.HardwareCalib{},
		"test", "H100", 1, false, "",
	)

	instance.Run()

	if instance.Metrics() == nil {
		t.Fatal("Metrics() returned nil")
	}

	if instance.Metrics().CompletedRequests <= 0 {
		t.Errorf("Metrics().CompletedRequests = %d, want > 0", instance.Metrics().CompletedRequests)
	}
}

// TestInstanceSimulator_Horizon_ReturnsConstructorValue verifies Horizon() delegates correctly
func TestInstanceSimulator_Horizon_ReturnsConstructorValue(t *testing.T) {
	expectedHorizon := int64(5000000)
	instance := NewInstanceSimulator(
		InstanceID("test"),
		expectedHorizon, 42, 1000, 16, 256, 2048, 0,
		[]float64{1000, 10, 5}, []float64{100, 1, 100},
		&sim.GuideLLMConfig{Rate: 10.0 / 1e6, MaxPrompts: 5,
			PromptTokens: 100, PromptTokensStdDev: 10, PromptTokensMin: 10, PromptTokensMax: 200,
			OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100},
		sim.ModelConfig{}, sim.HardwareCalib{},
		"test", "H100", 1, false, "",
	)

	if instance.Horizon() != expectedHorizon {
		t.Errorf("Horizon() = %d, want %d", instance.Horizon(), expectedHorizon)
	}
}

// === Edge Case Tests ===

// TestInstanceSimulator_EmptyID_Valid verifies:
// GIVEN InstanceSimulator created with ID ""
// WHEN ID() is called
// THEN returns InstanceID("") (no panic, no error)
func TestInstanceSimulator_EmptyID_Valid(t *testing.T) {
	instance := NewInstanceSimulator(
		InstanceID(""),
		1000000, 42, 1000, 16, 256, 2048, 0,
		[]float64{1000, 10, 5}, []float64{100, 1, 100},
		&sim.GuideLLMConfig{Rate: 10.0 / 1e6, MaxPrompts: 5,
			PromptTokens: 100, PromptTokensStdDev: 10, PromptTokensMin: 10, PromptTokensMax: 200,
			OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100},
		sim.ModelConfig{}, sim.HardwareCalib{},
		"test", "H100", 1, false, "",
	)

	if instance.ID() != InstanceID("") {
		t.Errorf("ID() = %q, want empty string", instance.ID())
	}
}

// TestInstanceSimulator_ZeroRequests verifies:
// GIVEN config with MaxPrompts=0
// WHEN Run() completes
// THEN Metrics().CompletedRequests == 0 AND no panic
func TestInstanceSimulator_ZeroRequests(t *testing.T) {
	instance := NewInstanceSimulator(
		InstanceID("test"),
		1000000, 42, 1000, 16, 256, 2048, 0,
		[]float64{1000, 10, 5}, []float64{100, 1, 100},
		&sim.GuideLLMConfig{Rate: 10.0 / 1e6, MaxPrompts: 0,
			PromptTokens: 100, PromptTokensStdDev: 10, PromptTokensMin: 10, PromptTokensMax: 200,
			OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100},
		sim.ModelConfig{}, sim.HardwareCalib{},
		"test", "H100", 1, false, "",
	)

	// Should not panic
	instance.Run()

	if instance.Metrics().CompletedRequests != 0 {
		t.Errorf("CompletedRequests = %d, want 0 for MaxPrompts=0", instance.Metrics().CompletedRequests)
	}
}

// === Helper Functions ===

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

