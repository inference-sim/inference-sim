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
			sim := NewSimulator(SimConfig{
				Horizon:                   math.MaxInt64,
				Seed:                      tc.Seed,
				TotalKVBlocks:             tc.TotalKVBlocks,
				BlockSizeTokens:           tc.BlockSizeInTokens,
				MaxRunningReqs:            tc.MaxNumRunningReqs,
				MaxScheduledTokens:        tc.MaxNumScheduledTokens,
				LongPrefillTokenThreshold: tc.LongPrefillTokenThreshold,
				BetaCoeffs:                tc.BetaCoeffs,
				AlphaCoeffs:               tc.AlphaCoeffs,
				Model:                     tc.Model,
				GPU:                       tc.Hardware,
				TP:                        tc.TP,
				GuideLLMConfig: &GuideLLMConfig{
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
				},
			})

			// Run simulation
			sim.Run()

			// === Invariant: request conservation (issue #183) ===
			// With Horizon=MaxInt64, all injected requests must complete.
			// This catches silent-drop bugs independently of golden values.
			if sim.Metrics.CompletedRequests != tc.MaxPrompts {
				t.Errorf("request conservation violated: completed %d, injected %d",
					sim.Metrics.CompletedRequests, tc.MaxPrompts)
			}

			// === Invariant: KV block conservation (issue #200) ===
			// After all requests complete, all blocks should be released.
			if sim.KVCache.UsedBlocks() != 0 {
				t.Errorf("KV block leak: %d blocks still allocated after all requests completed",
					sim.KVCache.UsedBlocks())
			}

			// === Invariant: causality ===
			// For every completed request: E2E >= TTFT (both are positive durations).
			for id, ttft := range sim.Metrics.RequestTTFTs {
				e2e, ok := sim.Metrics.RequestE2Es[id]
				if ok && e2e < ttft {
					t.Errorf("causality violated for %s: E2E (%.2f) < TTFT (%.2f)", id, e2e, ttft)
				}
			}

			// === Exact match metrics (integers) ===
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

			sortedTTFTs := make([]float64, 0, len(sim.Metrics.RequestTTFTs))
			for _, v := range sim.Metrics.RequestTTFTs {
				sortedTTFTs = append(sortedTTFTs, v)
			}
			sort.Float64s(sortedTTFTs)

			sortedE2Es := make([]float64, 0, len(sim.Metrics.RequestE2Es))
			for _, v := range sim.Metrics.RequestE2Es {
				sortedE2Es = append(sortedE2Es, v)
			}
			sort.Float64s(sortedE2Es)

			slices.Sort(sim.Metrics.AllITLs)

			sortedSchedulingDelays := make([]float64, 0, len(sim.Metrics.RequestSchedulingDelays))
			for _, v := range sim.Metrics.RequestSchedulingDelays {
				sortedSchedulingDelays = append(sortedSchedulingDelays, float64(v))
			}
			sort.Float64s(sortedSchedulingDelays)

			const relTol = 1e-9

			testutil.AssertFloat64Equal(t, "vllm_estimated_duration_s", tc.Metrics.VllmEstimatedDurationS, vllmRuntime, relTol)
			testutil.AssertFloat64Equal(t, "responses_per_sec", tc.Metrics.ResponsesPerSec, responsesPerSec, relTol)
			testutil.AssertFloat64Equal(t, "tokens_per_sec", tc.Metrics.TokensPerSec, tokensPerSec, relTol)

			testutil.AssertFloat64Equal(t, "e2e_mean_ms", tc.Metrics.E2EMeanMs, CalculateMean(sortedE2Es), relTol)
			testutil.AssertFloat64Equal(t, "e2e_p90_ms", tc.Metrics.E2EP90Ms, CalculatePercentile(sortedE2Es, 90), relTol)
			testutil.AssertFloat64Equal(t, "e2e_p95_ms", tc.Metrics.E2EP95Ms, CalculatePercentile(sortedE2Es, 95), relTol)
			testutil.AssertFloat64Equal(t, "e2e_p99_ms", tc.Metrics.E2EP99Ms, CalculatePercentile(sortedE2Es, 99), relTol)

			testutil.AssertFloat64Equal(t, "ttft_mean_ms", tc.Metrics.TTFTMeanMs, CalculateMean(sortedTTFTs), relTol)
			testutil.AssertFloat64Equal(t, "ttft_p90_ms", tc.Metrics.TTFTP90Ms, CalculatePercentile(sortedTTFTs, 90), relTol)
			testutil.AssertFloat64Equal(t, "ttft_p95_ms", tc.Metrics.TTFTP95Ms, CalculatePercentile(sortedTTFTs, 95), relTol)
			testutil.AssertFloat64Equal(t, "ttft_p99_ms", tc.Metrics.TTFTP99Ms, CalculatePercentile(sortedTTFTs, 99), relTol)

			testutil.AssertFloat64Equal(t, "itl_mean_ms", tc.Metrics.ITLMeanMs, CalculateMean(sim.Metrics.AllITLs), relTol)
			testutil.AssertFloat64Equal(t, "itl_p90_ms", tc.Metrics.ITLP90Ms, CalculatePercentile(sim.Metrics.AllITLs, 90), relTol)
			testutil.AssertFloat64Equal(t, "itl_p95_ms", tc.Metrics.ITLP95Ms, CalculatePercentile(sim.Metrics.AllITLs, 95), relTol)
			testutil.AssertFloat64Equal(t, "itl_p99_ms", tc.Metrics.ITLP99Ms, CalculatePercentile(sim.Metrics.AllITLs, 99), relTol)

			testutil.AssertFloat64Equal(t, "scheduling_delay_p99_ms", tc.Metrics.SchedulingDelayP99Ms, CalculatePercentile(sortedSchedulingDelays, 99), relTol)
		})
	}
}

// TestSimulator_WorkloadRNG_NotNil verifies the WorkloadRNG accessor never returns nil
func TestSimulator_WorkloadRNG_NotNil(t *testing.T) {
	sim := NewSimulator(SimConfig{
		Horizon:       1000000,
		Seed:          42,
		TotalKVBlocks: 1000,
		BlockSizeTokens: 16,
		MaxRunningReqs: 256,
		MaxScheduledTokens: 2048,
		BetaCoeffs:    []float64{1000, 10, 5},
		AlphaCoeffs:   []float64{100, 1, 100},
		Model:         "test-model",
		GPU:           "H100",
		TP:            1,
		GuideLLMConfig: &GuideLLMConfig{
			Rate: 10.0 / 1e6, MaxPrompts: 10,
			PromptTokens: 100, PromptTokensStdDev: 10, PromptTokensMin: 10, PromptTokensMax: 200,
			OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100,
		},
	})

	rng := sim.WorkloadRNG()
	if rng == nil {
		t.Error("WorkloadRNG() returned nil")
	}

	val := rng.Float64()
	if val < 0 || val >= 1 {
		t.Errorf("WorkloadRNG().Float64() = %v, want [0, 1)", val)
	}
}

// TestSimulator_DeterministicWorkload verifies same seed produces same workload
func TestSimulator_DeterministicWorkload(t *testing.T) {
	cfg := SimConfig{
		Horizon:            math.MaxInt64,
		Seed:               42,
		TotalKVBlocks:      10000,
		BlockSizeTokens:    16,
		MaxRunningReqs:     256,
		MaxScheduledTokens: 2048,
		BetaCoeffs:         []float64{1000, 10, 5},
		AlphaCoeffs:        []float64{100, 1, 100},
		Model:              "test",
		GPU:                "H100",
		TP:                 1,
		GuideLLMConfig: &GuideLLMConfig{
			Rate: 10.0 / 1e6, MaxPrompts: 50,
			PromptTokens: 100, PromptTokensStdDev: 20, PromptTokensMin: 10, PromptTokensMax: 200,
			OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100,
		},
	}

	sim1 := NewSimulator(cfg)
	sim2 := NewSimulator(cfg)

	sim1.Run()
	sim2.Run()

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

// newTestSimConfig creates a SimConfig for tests that don't need workload generation.
func newTestSimConfig() SimConfig {
	return SimConfig{
		Horizon:            math.MaxInt64,
		Seed:               42,
		TotalKVBlocks:      10000,
		BlockSizeTokens:    16,
		MaxRunningReqs:     256,
		MaxScheduledTokens: 2048,
		BetaCoeffs:         []float64{1000, 10, 5},
		AlphaCoeffs:        []float64{100, 1, 100},
		Model:              "test-model",
		GPU:                "H100",
		TP:                 1,
	}
}

// TestNewSimulator_NoWorkload_EmptyQueue verifies that a SimConfig with no workload
// (both GuideLLMConfig nil and TracesWorkloadFilePath empty) creates a simulator
// with an empty EventQueue and runs to completion with zero results.
func TestNewSimulator_NoWorkload_EmptyQueue(t *testing.T) {
	sim := NewSimulator(newTestSimConfig())

	if len(sim.eventQueue) != 0 {
		t.Errorf("eventQueue length: got %d, want 0", len(sim.eventQueue))
	}

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
	sim := NewSimulator(newTestSimConfig())

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
	if sim.Metrics.KVAllocationFailures != 0 {
		t.Errorf("KVAllocationFailures: got %d, want 0 (no failures expected under normal conditions)", sim.Metrics.KVAllocationFailures)
	}
}

// TestInjectArrival_HandledByEmpty_StandaloneMode verifies #181 standalone boundary:
// GIVEN a standalone simulator (no cluster routing)
// WHEN a request is injected and completes
// THEN HandledBy in RequestMetrics is empty (no routing happened)
func TestInjectArrival_HandledByEmpty_StandaloneMode(t *testing.T) {
	sim := NewSimulator(newTestSimConfig())
	req := &Request{
		ID:           "request_0",
		ArrivalTime:  0,
		InputTokens:  make([]int, 10),
		OutputTokens: make([]int, 5),
		State:        "queued",
	}
	sim.InjectArrival(req)
	sim.Run()

	rm, ok := sim.Metrics.Requests["request_0"]
	if !ok {
		t.Fatal("request_0 not found in Metrics.Requests")
	}
	if rm.HandledBy != "" {
		t.Errorf("HandledBy: got %q, want empty (standalone mode)", rm.HandledBy)
	}
}

// TestInjectArrival_MultipleRequests verifies that multiple injected requests
// at staggered arrival times all complete successfully.
func TestInjectArrival_MultipleRequests(t *testing.T) {
	sim := NewSimulator(newTestSimConfig())

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

// failOnCompletionKVStore wraps a real KVStore but returns false from
// AllocateKVBlocks when the request has State == "completed". This works
// because simulator.go sets req.State = "completed" before calling
// AllocateKVBlocks for the final token, so the trigger precisely targets
// the completion-time allocation path described in issue #183.
type failOnCompletionKVStore struct {
	KVStore
	failCount int
}

func (f *failOnCompletionKVStore) AllocateKVBlocks(req *Request, startIndex, endIndex int64, cachedBlocks []int64) bool {
	if req.State == "completed" {
		f.failCount++
		return false
	}
	return f.KVStore.AllocateKVBlocks(req, startIndex, endIndex, cachedBlocks)
}

// TestStep_KVAllocFailAtCompletion_RequestNotSilentlyDropped verifies that
// when KV allocation fails at request completion time, the request is still
// counted as completed with full metrics recorded (not silently dropped).
// Regression test for issue #183.
func TestStep_KVAllocFailAtCompletion_RequestNotSilentlyDropped(t *testing.T) {
	// GIVEN a simulator with a KVStore that fails allocation at completion time
	cfg := newTestSimConfig()
	sim := NewSimulator(cfg)
	fakeKV := &failOnCompletionKVStore{KVStore: sim.KVCache}
	sim.KVCache = fakeKV

	// AND a request with output tokens that will reach the completion path
	req := &Request{
		ID:           "req-0",
		ArrivalTime:  0,
		InputTokens:  make([]int, 16), // 1 block worth of prefill
		OutputTokens: make([]int, 3),  // 3 decode tokens
		State:        "queued",
	}
	sim.InjectArrival(req)

	// WHEN the simulation runs to completion
	sim.Run()

	// THEN the fake KV store should have been triggered
	if fakeKV.failCount == 0 {
		t.Fatal("failOnCompletionKVStore was never triggered — test setup is invalid")
	}

	// AND the request should still be counted as completed (not silently dropped)
	if sim.Metrics.CompletedRequests != 1 {
		t.Errorf("CompletedRequests: got %d, want 1 (request was silently dropped — issue #183)", sim.Metrics.CompletedRequests)
	}

	// AND the KV allocation failure should be tracked in metrics
	if sim.Metrics.KVAllocationFailures != 1 {
		t.Errorf("KVAllocationFailures: got %d, want 1", sim.Metrics.KVAllocationFailures)
	}

	// AND request E2E/TTFT metrics should still be recorded
	if _, ok := sim.Metrics.RequestE2Es["req-0"]; !ok {
		t.Error("RequestE2Es missing for req-0 — metrics lost due to silent drop")
	}
	if _, ok := sim.Metrics.RequestTTFTs["req-0"]; !ok {
		t.Error("RequestTTFTs missing for req-0 — metrics lost due to silent drop")
	}
}
