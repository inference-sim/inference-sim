package sim

import (
	"fmt"
	"math"
	"slices"
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// TestSimulator_GoldenDataset verifies backward compatibility by running
// all test cases from the golden dataset and comparing results.
// This is the critical backward compatibility test ensuring RNG changes do not alter simulation outcomes.
func TestSimulator_GoldenDataset(t *testing.T) {
	dataset := testutil.LoadGoldenDataset(t)

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
			testutil.AssertFloat64Equal(t, "vllm_estimated_duration_s", tc.Metrics.VllmEstimatedDurationS, vllmRuntime, relTol)

			// Throughput metrics
			testutil.AssertFloat64Equal(t, "responses_per_sec", tc.Metrics.ResponsesPerSec, responsesPerSec, relTol)
			testutil.AssertFloat64Equal(t, "tokens_per_sec", tc.Metrics.TokensPerSec, tokensPerSec, relTol)

			// E2E latency metrics
			testutil.AssertFloat64Equal(t, "e2e_mean_ms", tc.Metrics.E2EMeanMs, CalculateMean(sortedE2Es), relTol)
			testutil.AssertFloat64Equal(t, "e2e_p90_ms", tc.Metrics.E2EP90Ms, CalculatePercentile(sortedE2Es, 90), relTol)
			testutil.AssertFloat64Equal(t, "e2e_p95_ms", tc.Metrics.E2EP95Ms, CalculatePercentile(sortedE2Es, 95), relTol)
			testutil.AssertFloat64Equal(t, "e2e_p99_ms", tc.Metrics.E2EP99Ms, CalculatePercentile(sortedE2Es, 99), relTol)

			// TTFT latency metrics
			testutil.AssertFloat64Equal(t, "ttft_mean_ms", tc.Metrics.TTFTMeanMs, CalculateMean(sortedTTFTs), relTol)
			testutil.AssertFloat64Equal(t, "ttft_p90_ms", tc.Metrics.TTFTP90Ms, CalculatePercentile(sortedTTFTs, 90), relTol)
			testutil.AssertFloat64Equal(t, "ttft_p95_ms", tc.Metrics.TTFTP95Ms, CalculatePercentile(sortedTTFTs, 95), relTol)
			testutil.AssertFloat64Equal(t, "ttft_p99_ms", tc.Metrics.TTFTP99Ms, CalculatePercentile(sortedTTFTs, 99), relTol)

			// ITL latency metrics
			testutil.AssertFloat64Equal(t, "itl_mean_ms", tc.Metrics.ITLMeanMs, CalculateMean(sim.Metrics.AllITLs), relTol)
			testutil.AssertFloat64Equal(t, "itl_p90_ms", tc.Metrics.ITLP90Ms, CalculatePercentile(sim.Metrics.AllITLs, 90), relTol)
			testutil.AssertFloat64Equal(t, "itl_p95_ms", tc.Metrics.ITLP95Ms, CalculatePercentile(sim.Metrics.AllITLs, 95), relTol)
			testutil.AssertFloat64Equal(t, "itl_p99_ms", tc.Metrics.ITLP99Ms, CalculatePercentile(sim.Metrics.AllITLs, 99), relTol)

			// Scheduling delay
			testutil.AssertFloat64Equal(t, "scheduling_delay_p99_ms", tc.Metrics.SchedulingDelayP99Ms, CalculatePercentile(sortedSchedulingDelays, 99), relTol)
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

// TestNewSimulatorWithoutWorkload_RunsEmpty verifies that a simulator created
// without any workload runs to completion without panic and produces zero results.
func TestNewSimulatorWithoutWorkload_RunsEmpty(t *testing.T) {
	sim := NewSimulatorWithoutWorkload(
		math.MaxInt64,              // horizon
		42,                         // seed
		10000,                      // totalKVBlocks
		16,                         // blockSizeTokens
		256,                        // maxRunningReqs
		2048,                       // maxScheduledTokens
		0,                          // longPrefillTokenThreshold
		[]float64{1000, 10, 5},     // betaCoeffs
		[]float64{100, 1, 100},     // alphaCoeffs
		ModelConfig{},              // modelConfig
		HardwareCalib{},            // hwConfig
		"test-model",               // model
		"H100",                     // GPU
		1,                          // tp
		false,                      // roofline
	)

	sim.Run()

	if sim.Metrics.CompletedRequests != 0 {
		t.Errorf("CompletedRequests: got %d, want 0", sim.Metrics.CompletedRequests)
	}
	if sim.Metrics.SimEndedTime != 0 {
		t.Errorf("SimEndedTime: got %d, want 0", sim.Metrics.SimEndedTime)
	}
}

// TestInjectArrival_RequestCompletes verifies that a single injected request
// is processed to completion by the simulator.
func TestInjectArrival_RequestCompletes(t *testing.T) {
	sim := NewSimulatorWithoutWorkload(
		math.MaxInt64,              // horizon
		42,                         // seed
		10000,                      // totalKVBlocks
		16,                         // blockSizeTokens
		256,                        // maxRunningReqs
		2048,                       // maxScheduledTokens
		0,                          // longPrefillTokenThreshold
		[]float64{1000, 10, 5},     // betaCoeffs
		[]float64{100, 1, 100},     // alphaCoeffs
		ModelConfig{},              // modelConfig
		HardwareCalib{},            // hwConfig
		"test-model",               // model
		"H100",                     // GPU
		1,                          // tp
		false,                      // roofline
	)

	req := &Request{
		ID:           "request_0",
		ArrivalTime:  0,
		InputTokens:  make([]int, 10),
		OutputTokens: make([]int, 5),
		State:        "queued",
	}

	sim.InjectArrival(req)
	sim.Run()

	if sim.Metrics.CompletedRequests != 1 {
		t.Errorf("CompletedRequests: got %d, want 1", sim.Metrics.CompletedRequests)
	}
	if len(sim.Metrics.Requests) < 1 {
		t.Errorf("len(Metrics.Requests): got %d, want >= 1", len(sim.Metrics.Requests))
	}
}

// TestInjectArrival_MultipleRequests verifies that multiple injected requests
// at staggered arrival times all complete successfully.
func TestInjectArrival_MultipleRequests(t *testing.T) {
	sim := NewSimulatorWithoutWorkload(
		math.MaxInt64,              // horizon
		42,                         // seed
		10000,                      // totalKVBlocks
		16,                         // blockSizeTokens
		256,                        // maxRunningReqs
		2048,                       // maxScheduledTokens
		0,                          // longPrefillTokenThreshold
		[]float64{1000, 10, 5},     // betaCoeffs
		[]float64{100, 1, 100},     // alphaCoeffs
		ModelConfig{},              // modelConfig
		HardwareCalib{},            // hwConfig
		"test-model",               // model
		"H100",                     // GPU
		1,                          // tp
		false,                      // roofline
	)

	for i := 0; i < 10; i++ {
		req := &Request{
			ID:           fmt.Sprintf("request_%d", i),
			ArrivalTime:  int64(i * 100000),
			InputTokens:  make([]int, 10),
			OutputTokens: make([]int, 5),
			State:        "queued",
		}
		sim.InjectArrival(req)
	}

	sim.Run()

	if sim.Metrics.CompletedRequests != 10 {
		t.Errorf("CompletedRequests: got %d, want 10", sim.Metrics.CompletedRequests)
	}
}
