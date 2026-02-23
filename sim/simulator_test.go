package sim

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"slices"
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// mustNewSimulator is a test helper that calls NewSimulator and fails the test on error.
func mustNewSimulator(t *testing.T, cfg SimConfig) *Simulator {
	t.Helper()
	s, err := NewSimulator(cfg)
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}
	return s
}

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
			sim := mustNewSimulator(t, SimConfig{
				Horizon:             math.MaxInt64,
				Seed:                tc.Seed,
				KVCacheConfig:       NewKVCacheConfig(tc.TotalKVBlocks, tc.BlockSizeInTokens, 0, 0, 0, 0),
				BatchConfig:         NewBatchConfig(tc.MaxNumRunningReqs, tc.MaxNumScheduledTokens, tc.LongPrefillTokenThreshold),
				LatencyCoeffs:       NewLatencyCoeffs(tc.BetaCoeffs, tc.AlphaCoeffs),
				ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, tc.Model, tc.Hardware, tc.TP, false),
				WorkloadConfig: NewWorkloadConfig(&GuideLLMConfig{
					Rate:               tc.Rate / 1e6,
					NumRequests:         tc.NumRequests,
					PrefixTokens:       tc.PrefixTokens,
					PromptTokens:       tc.PromptTokens,
					PromptTokensStdDev: tc.PromptTokensStdev,
					PromptTokensMin:    tc.PromptTokensMin,
					PromptTokensMax:    tc.PromptTokensMax,
					OutputTokens:       tc.OutputTokens,
					OutputTokensStdDev: tc.OutputTokensStdev,
					OutputTokensMin:    tc.OutputTokensMin,
					OutputTokensMax:    tc.OutputTokensMax,
				}, ""),
			})

			// Run simulation
			sim.Run()

			// === Invariant: request conservation (issue #183) ===
			// With Horizon=MaxInt64, all injected requests must complete.
			// This catches silent-drop bugs independently of golden values.
			if sim.Metrics.CompletedRequests != tc.NumRequests {
				t.Errorf("request conservation violated: completed %d, injected %d",
					sim.Metrics.CompletedRequests, tc.NumRequests)
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
	sim := mustNewSimulator(t, SimConfig{
		Horizon:             1000000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-model", "H100", 1, false),
		WorkloadConfig: NewWorkloadConfig(&GuideLLMConfig{
			Rate: 10.0 / 1e6, NumRequests: 10,
			PromptTokens: 100, PromptTokensStdDev: 10, PromptTokensMin: 10, PromptTokensMax: 200,
			OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100,
		}, ""),
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
		Horizon:             math.MaxInt64,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test", "H100", 1, false),
		WorkloadConfig: NewWorkloadConfig(&GuideLLMConfig{
			Rate: 10.0 / 1e6, NumRequests: 50,
			PromptTokens: 100, PromptTokensStdDev: 20, PromptTokensMin: 10, PromptTokensMax: 200,
			OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100,
		}, ""),
	}

	sim1 := mustNewSimulator(t, cfg)
	sim2 := mustNewSimulator(t, cfg)

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
		Horizon:             math.MaxInt64,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-model", "H100", 1, false),
	}
}

// TestNewSimulator_NoWorkload_EmptyQueue verifies that a SimConfig with no workload
// (both GuideLLMConfig nil and TracesWorkloadFilePath empty) creates a simulator
// with an empty EventQueue and runs to completion with zero results.
func TestNewSimulator_NoWorkload_EmptyQueue(t *testing.T) {
	sim := mustNewSimulator(t, newTestSimConfig())

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
	sim := mustNewSimulator(t, newTestSimConfig())

	req := &Request{
		ID:           "request_0",
		ArrivalTime:  0,
		InputTokens:  make([]int, 10),
		OutputTokens: make([]int, 5),
		State:        StateQueued,
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
	sim := mustNewSimulator(t, newTestSimConfig())
	req := &Request{
		ID:           "request_0",
		ArrivalTime:  0,
		InputTokens:  make([]int, 10),
		OutputTokens: make([]int, 5),
		State:        StateQueued,
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
	sim := mustNewSimulator(t, newTestSimConfig())

	for i := 0; i < 10; i++ {
		req := &Request{
			ID:           fmt.Sprintf("request_%d", i),
			ArrivalTime:  int64(i * 100000),
			InputTokens:  make([]int, 10),
			OutputTokens: make([]int, 5),
			State:        StateQueued,
		}
		sim.InjectArrival(req)
	}

	sim.Run()

	if sim.Metrics.CompletedRequests != 10 {
		t.Errorf("CompletedRequests: got %d, want 10", sim.Metrics.CompletedRequests)
	}
}

// failOnCompletionKVStore wraps a real KVStore but returns false from
// AllocateKVBlocks when the request has State == StateCompleted. This works
// because simulator.go sets req.State = StateCompleted before calling
// AllocateKVBlocks for the final token, so the trigger precisely targets
// the completion-time allocation path described in issue #183.
type failOnCompletionKVStore struct {
	KVStore
	failCount int
}

func (f *failOnCompletionKVStore) AllocateKVBlocks(req *Request, startIndex, endIndex int64, cachedBlocks []int64) bool {
	if req.State == StateCompleted {
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
	sim := mustNewSimulator(t, cfg)
	fakeKV := &failOnCompletionKVStore{KVStore: sim.KVCache}
	sim.KVCache = fakeKV

	// AND a request with output tokens that will reach the completion path
	req := &Request{
		ID:           "req-0",
		ArrivalTime:  0,
		InputTokens:  make([]int, 16), // 1 block worth of prefill
		OutputTokens: make([]int, 3),  // 3 decode tokens
		State:        StateQueued,
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

// =============================================================================
// Invariant Tests (Phase 4, issue #211)
//
// These tests verify simulation laws derived from the specification, not from
// running the code. They complement golden dataset tests (which verify output
// hasn't changed) with invariant tests (which verify output is correct).
// =============================================================================

// TestSimulator_RequestConservation_InfiniteHorizon_AllRequestsComplete verifies BC-1:
// GIVEN a simulator with Horizon=MaxInt64 and 50 injected requests
// WHEN the simulation runs to completion
// THEN CompletedRequests == 50 AND WaitQ is empty AND RunningBatch is empty.
func TestSimulator_RequestConservation_InfiniteHorizon_AllRequestsComplete(t *testing.T) {
	cfg := SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                99,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-conservation", "H100", 1, false),
		WorkloadConfig: NewWorkloadConfig(&GuideLLMConfig{
			Rate: 10.0 / 1e6, NumRequests: 50,
			PromptTokens: 100, PromptTokensStdDev: 20, PromptTokensMin: 10, PromptTokensMax: 200,
			OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100,
		}, ""),
	}

	sim := mustNewSimulator(t, cfg)
	sim.Run()

	// Three-term equation: injected == completed + queued + running
	injected := len(sim.Metrics.Requests)
	completed := sim.Metrics.CompletedRequests
	queued := sim.WaitQ.Len()
	running := 0
	if sim.RunningBatch != nil {
		running = len(sim.RunningBatch.Requests)
	}

	if completed+queued+running != injected {
		t.Errorf("request conservation violated: completed(%d) + queued(%d) + running(%d) = %d, injected = %d",
			completed, queued, running, completed+queued+running, injected)
	}

	// With infinite horizon, all should complete
	if completed != 50 {
		t.Errorf("infinite horizon: expected all 50 requests to complete, got %d", completed)
	}
	if queued != 0 {
		t.Errorf("infinite horizon: expected empty queue, got %d queued", queued)
	}
	if running != 0 {
		t.Errorf("infinite horizon: expected empty batch, got %d running", running)
	}
}

// TestSimulator_RequestConservation_FiniteHorizon_ThreeTermEquation verifies BC-2:
// GIVEN a simulator with a finite Horizon and requests injected at staggered times
//
//	(all arriving BEFORE the horizon, but late ones too large to complete)
//
// WHEN the simulation ends
// THEN completed + queued + running == injected (three-term conservation).
func TestSimulator_RequestConservation_FiniteHorizon_ThreeTermEquation(t *testing.T) {
	// CRITICAL: All injected requests MUST have ArrivalTime < Horizon.
	// Requests with ArrivalTime > Horizon have their ArrivalEvent never popped
	// from the event queue — they'd be in Metrics.Requests but NOT in
	// WaitQ/RunningBatch/completed, breaking the three-term equation.
	//
	// Strategy: inject early (small, fast) and late (large, slow) requests.
	// Early requests complete before the horizon. Late requests arrive before
	// the horizon but are too large to finish processing.
	cfg := SimConfig{
		Horizon:             500_000, // 0.5 seconds in ticks
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-conservation-finite", "H100", 1, false),
	}

	sim := mustNewSimulator(t, cfg)

	// Inject 10 early requests (small, will complete before horizon)
	for i := 0; i < 10; i++ {
		sim.InjectArrival(&Request{
			ID:           fmt.Sprintf("early_%d", i),
			ArrivalTime:  int64(i * 10000), // 0 to 90,000 ticks (well before 500k horizon)
			InputTokens:  make([]int, 20),
			OutputTokens: make([]int, 5),
			State:        StateQueued,
		})
	}

	// Inject 5 late requests (large, arrive before horizon but won't complete)
	for i := 0; i < 5; i++ {
		sim.InjectArrival(&Request{
			ID:           fmt.Sprintf("late_%d", i),
			ArrivalTime:  int64(300_000 + i*40_000), // 300,000 to 460,000 (all < 500k horizon)
			InputTokens:  make([]int, 200),           // large prefill
			OutputTokens: make([]int, 100),           // many decode tokens
			State:        StateQueued,
		})
	}

	sim.Run()

	injected := len(sim.Metrics.Requests)
	completed := sim.Metrics.CompletedRequests
	queued := sim.WaitQ.Len()
	running := 0
	if sim.RunningBatch != nil {
		running = len(sim.RunningBatch.Requests)
	}

	if completed+queued+running != injected {
		t.Errorf("request conservation violated: completed(%d) + queued(%d) + running(%d) = %d, injected = %d",
			completed, queued, running, completed+queued+running, injected)
	}

	// Verify we actually tested the non-trivial case: some but not all completed
	if completed == injected {
		t.Fatalf("all %d requests completed — horizon too long, three-term case untested", injected)
	}
	if completed == 0 {
		t.Fatalf("no requests completed — horizon too short, test setup invalid")
	}
}

// TestSimulator_Causality_FullChain_ArrivalToCompletion verifies BC-4:
// GIVEN a completed simulation with multiple requests
// WHEN examining per-request timing metrics
// THEN for every completed request:
//   - TTFT >= 0 (first token came after arrival — TTFT is a relative duration)
//   - E2E >= TTFT (completion came after first token)
//   - All ITL values >= 0 (no negative inter-token latencies)
func TestSimulator_Causality_FullChain_ArrivalToCompletion(t *testing.T) {
	cfg := SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                77,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-causality", "H100", 1, false),
		WorkloadConfig: NewWorkloadConfig(&GuideLLMConfig{
			Rate: 5.0 / 1e6, NumRequests: 30,
			PromptTokens: 100, PromptTokensStdDev: 20, PromptTokensMin: 10, PromptTokensMax: 200,
			OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100,
		}, ""),
	}

	sim := mustNewSimulator(t, cfg)
	sim.Run()

	if sim.Metrics.CompletedRequests == 0 {
		t.Fatal("no completed requests — test setup invalid")
	}

	for id := range sim.Metrics.Requests {
		ttft, hasTTFT := sim.Metrics.RequestTTFTs[id]
		e2e, hasE2E := sim.Metrics.RequestE2Es[id]

		if !hasTTFT || !hasE2E {
			continue // incomplete request (finite horizon) — skip
		}

		// TTFT and E2E are relative durations from arrival (in microseconds stored as float64).
		// Causality: TTFT >= 0 (first token came after arrival)
		if ttft < 0 {
			t.Errorf("causality violated for %s: TTFT (%.2f) < 0", id, ttft)
		}

		// Causality: E2E >= TTFT (completion came after first token)
		if e2e < ttft {
			t.Errorf("causality violated for %s: E2E (%.2f) < TTFT (%.2f)", id, e2e, ttft)
		}
	}

	// NC-1: All ITL values must be non-negative
	for i, itl := range sim.Metrics.AllITLs {
		if itl < 0 {
			t.Errorf("negative ITL at index %d: %d (time travel)", i, itl)
		}
	}
}

// TestSimulator_ClockMonotonicity_NeverDecreases verifies BC-6:
// GIVEN a simulator with injected requests
// WHEN processing events one at a time
// THEN the Clock value never decreases between consecutive events.
//
// This manual loop is functionally identical to Simulator.Run():
//
//	for HasPendingEvents { ProcessNextEvent; if Clock > Horizon { break } } + Finalize()
func TestSimulator_ClockMonotonicity_NeverDecreases(t *testing.T) {
	cfg := SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                55,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-monotonicity", "H100", 1, false),
		WorkloadConfig: NewWorkloadConfig(&GuideLLMConfig{
			Rate: 10.0 / 1e6, NumRequests: 20,
			PromptTokens: 50, PromptTokensStdDev: 10, PromptTokensMin: 10, PromptTokensMax: 100,
			OutputTokens: 20, OutputTokensStdDev: 5, OutputTokensMin: 5, OutputTokensMax: 40,
		}, ""),
	}

	sim := mustNewSimulator(t, cfg)

	prevClock := int64(0)
	eventCount := 0
	for sim.HasPendingEvents() {
		sim.ProcessNextEvent()
		eventCount++

		if sim.Clock < prevClock {
			t.Fatalf("clock monotonicity violated at event %d: clock went from %d to %d",
				eventCount, prevClock, sim.Clock)
		}
		prevClock = sim.Clock

		if sim.Clock > sim.Horizon {
			break
		}
	}
	sim.Finalize()

	if eventCount == 0 {
		t.Fatal("no events processed — test setup invalid")
	}
	t.Logf("clock monotonicity held across %d events (final clock: %d)", eventCount, sim.Clock)
}

// TestSimulator_Determinism_ByteIdenticalJSON verifies BC-8:
// GIVEN two simulator runs with identical config and seed
// WHEN both save results via SaveResults to temp files
// THEN the output files are identical after stripping wall-clock timestamps.
func TestSimulator_Determinism_ByteIdenticalJSON(t *testing.T) {
	cfg := SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-determinism", "H100", 1, false),
		WorkloadConfig: NewWorkloadConfig(&GuideLLMConfig{
			Rate: 5.0 / 1e6, NumRequests: 20,
			PromptTokens: 100, PromptTokensStdDev: 20, PromptTokensMin: 10, PromptTokensMax: 200,
			OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100,
		}, ""),
	}

	// Run 1
	sim1 := mustNewSimulator(t, cfg)
	sim1.Run()
	f1 := t.TempDir() + "/run1.json"
	sim1.Metrics.SaveResults("determinism-test", cfg.Horizon, cfg.TotalKVBlocks, f1)

	// Run 2
	sim2 := mustNewSimulator(t, cfg)
	sim2.Run()
	f2 := t.TempDir() + "/run2.json"
	sim2.Metrics.SaveResults("determinism-test", cfg.Horizon, cfg.TotalKVBlocks, f2)

	data1, err1 := os.ReadFile(f1)
	if err1 != nil {
		t.Fatalf("failed to read run1 output: %v", err1)
	}
	data2, err2 := os.ReadFile(f2)
	if err2 != nil {
		t.Fatalf("failed to read run2 output: %v", err2)
	}

	// All wall-clock fields removed from MetricsOutput — direct comparison is now valid
	if !bytes.Equal(data1, data2) {
		t.Error("determinism violation: normalized JSON differs between runs")
		lines1 := bytes.Split(data1, []byte("\n"))
		lines2 := bytes.Split(data2, []byte("\n"))
		maxLines := len(lines1)
		if len(lines2) > maxLines {
			maxLines = len(lines2)
		}
		for i := 0; i < maxLines; i++ {
			var l1, l2 []byte
			if i < len(lines1) {
				l1 = lines1[i]
			}
			if i < len(lines2) {
				l2 = lines2[i]
			}
			if !bytes.Equal(l1, l2) {
				t.Errorf("first difference at line %d:\n  run1: %s\n  run2: %s", i+1, l1, l2)
				break
			}
		}
	}
}

// TestSimulator_KVBlockConservation_PostSimulation_ZeroLeak verifies BC-10:
// GIVEN a completed simulation with all requests finished (infinite horizon)
// WHEN checking KV block accounting via KVStore interface
// THEN UsedBlocks() == 0 (no leaked blocks).
func TestSimulator_KVBlockConservation_PostSimulation_ZeroLeak(t *testing.T) {
	tests := []struct {
		name        string
		kvCPUBlocks int64
	}{
		{"single-tier", 0},
		{"tiered-gpu-cpu", 20},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := SimConfig{
				Horizon:             math.MaxInt64,
				Seed:                42,
				KVCacheConfig:       NewKVCacheConfig(10000, 16, tt.kvCPUBlocks, 0.8, 100.0, 0),
				BatchConfig:         NewBatchConfig(256, 2048, 0),
				LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
				ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-kv-conservation", "H100", 1, false),
				WorkloadConfig: NewWorkloadConfig(&GuideLLMConfig{
					Rate: 5.0 / 1e6, NumRequests: 20,
					PromptTokens: 100, PromptTokensStdDev: 20, PromptTokensMin: 10, PromptTokensMax: 200,
					OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100,
				}, ""),
			}

			sim := mustNewSimulator(t, cfg)
			sim.Run()

			if sim.KVCache.UsedBlocks() != 0 {
				t.Errorf("KV block leak: %d blocks still allocated after all requests completed (KVStore interface)",
					sim.KVCache.UsedBlocks())
			}

			if sim.KVCache.TotalCapacity() != cfg.TotalKVBlocks {
				t.Errorf("TotalCapacity changed: got %d, want %d", sim.KVCache.TotalCapacity(), cfg.TotalKVBlocks)
			}
		})
	}
}

func TestSimulator_ObservationMethods_MatchDirectAccess(t *testing.T) {
	// BC-1: Observation methods return same values as direct field access
	cfg := newTestSimConfig()
	s := mustNewSimulator(t, cfg)

	// Before any events: queue empty, batch empty
	if s.QueueDepth() != 0 {
		t.Errorf("QueueDepth: got %d, want 0", s.QueueDepth())
	}
	if s.BatchSize() != 0 {
		t.Errorf("BatchSize: got %d, want 0", s.BatchSize())
	}
	if s.CurrentClock() != 0 {
		t.Errorf("CurrentClock: got %d, want 0", s.CurrentClock())
	}
	if s.SimHorizon() != cfg.Horizon {
		t.Errorf("SimHorizon: got %d, want %d", s.SimHorizon(), cfg.Horizon)
	}

	// Inject a request and verify QueueDepth
	req := &Request{
		ID: "obs-test-1", ArrivalTime: 0,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
		State: StateQueued,
	}
	s.InjectArrival(req)
	s.ProcessNextEvent() // process arrival → queued
	if s.QueueDepth() != s.WaitQ.Len() {
		t.Errorf("QueueDepth mismatch: method=%d, direct=%d", s.QueueDepth(), s.WaitQ.Len())
	}
}

// TestWorkConserving_StepRestartsWhenWaitQNonEmpty verifies INV-8:
// GIVEN a simulator with MaxRunningReqs=1 and two requests arriving at t=0 and t=1
// WHEN the simulation runs to completion (infinite horizon, no further arrivals)
// THEN both requests complete (the second is not stranded when the first finishes)
// AND the second request's scheduling delay is bounded by the first's service time.
//
// Without the work-conserving fix (simulator.go Work-conserving comment block), the
// second request would be stranded forever: its QueuedEvent already fired (seeing
// stepEvent != nil), and when the first request completes, stepEvent is set to nil
// without checking WaitQ. No third arrival exists to trigger a new StepEvent via
// QueuedEvent.
func TestWorkConserving_StepRestartsWhenWaitQNonEmpty(t *testing.T) {
	cfg := SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(1, 2048, 0), // KEY: only one request can run at a time
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-work-conserving", "H100", 1, false),
	}

	s := mustNewSimulator(t, cfg)

	// Request A: arrives at t=0
	s.InjectArrival(&Request{
		ID: "req-A", ArrivalTime: 0,
		InputTokens:  make([]int, 10),
		OutputTokens: make([]int, 5),
		State:        StateQueued,
	})
	// Request B: arrives at t=1 (during A's service time, which is ~6000μs)
	s.InjectArrival(&Request{
		ID: "req-B", ArrivalTime: 1,
		InputTokens:  make([]int, 10),
		OutputTokens: make([]int, 5),
		State:        StateQueued,
	})

	s.Run()

	// BC-1: Both requests MUST complete.
	// Without the work-conserving fix, only req-A completes (req-B stranded forever).
	if s.Metrics.CompletedRequests != 2 {
		t.Fatalf("INV-8 violated: CompletedRequests = %d, want 2 "+
			"(second request stranded when running batch emptied without checking WaitQ)",
			s.Metrics.CompletedRequests)
	}

	// BC-2: req-B's scheduling delay must be bounded.
	// It should wait approximately one service time of req-A (~6000μs), not be stuck indefinitely.
	delayA := s.Metrics.RequestSchedulingDelays["req-A"]
	delayB := s.Metrics.RequestSchedulingDelays["req-B"]

	// req-A should be scheduled almost immediately (only alpha queueing delay)
	if delayA > 10000 {
		t.Errorf("req-A scheduling delay = %d μs, want < 10000 (should be near-immediate)", delayA)
	}

	// req-B should wait approximately one service time of req-A, not be unbounded.
	// With beta=[1000,10,5] and 10 input + 5 output tokens, service time is ~6000-7000μs.
	// We use a generous bound (40000μs ≈ 6× service time) to avoid brittleness.
	if delayB > 40000 {
		t.Errorf("req-B scheduling delay = %d μs, exceeds generous service time bound "+
			"(may indicate work-conserving violation)", delayB)
	}

	// req-B must have waited longer than req-A (it was queued behind A)
	if delayB <= delayA {
		t.Errorf("req-B scheduling delay (%d) <= req-A delay (%d), "+
			"but B should have waited for A to complete", delayB, delayA)
	}

	// INV-1: Request conservation still holds
	injected := len(s.Metrics.Requests)
	total := s.Metrics.CompletedRequests + s.WaitQ.Len()
	running := 0
	if s.RunningBatch != nil {
		running = len(s.RunningBatch.Requests)
	}
	total += running
	if total != injected {
		t.Errorf("INV-1 conservation: completed(%d) + queued(%d) + running(%d) = %d, injected = %d",
			s.Metrics.CompletedRequests, s.WaitQ.Len(), running, total, injected)
	}
}

// BC-1: Oversized request dropped at enqueue
func TestEnqueueRequest_OversizedInput_DroppedNotEnqueued(t *testing.T) {
	// GIVEN a simulator with 10 KV blocks of 16 tokens each (160 token capacity)
	cfg := SimConfig{
		Horizon:       1_000_000,
		Seed:          42,
		KVCacheConfig: NewKVCacheConfig(10, 16, 0, 0, 0, 0),
		BatchConfig:   NewBatchConfig(256, 2048, 0),
		LatencyCoeffs: NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
	}
	sim, err := NewSimulator(cfg)
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}

	// AND a request with 200 input tokens (needs ceil(200/16) = 13 blocks > 10 total)
	oversized := &Request{
		ID:          "oversized_req",
		InputTokens: make([]int, 200),
		State:       StateQueued,
	}
	// Register it in Metrics.Requests (simulating InjectArrival behavior)
	sim.Metrics.Requests[oversized.ID] = NewRequestMetrics(oversized, 0)

	// WHEN we try to enqueue it
	sim.EnqueueRequest(oversized)

	// THEN it must NOT be in the wait queue
	if sim.WaitQ.Len() != 0 {
		t.Errorf("WaitQ.Len() = %d, want 0 (oversized request should not be enqueued)", sim.WaitQ.Len())
	}

	// AND DroppedUnservable must be incremented
	if sim.Metrics.DroppedUnservable != 1 {
		t.Errorf("DroppedUnservable = %d, want 1", sim.Metrics.DroppedUnservable)
	}

	// AND the request must be removed from per-request tracking (BC-7)
	if _, exists := sim.Metrics.Requests[oversized.ID]; exists {
		t.Error("dropped request should be removed from Metrics.Requests")
	}

	// AND TotalInputTokens must NOT include the dropped request's tokens
	if sim.Metrics.TotalInputTokens != 0 {
		t.Errorf("TotalInputTokens = %d, want 0 (dropped request tokens not counted)", sim.Metrics.TotalInputTokens)
	}
}

// BC-2: Normal requests unaffected
func TestEnqueueRequest_NormalInput_Enqueued(t *testing.T) {
	// GIVEN a simulator with 100 KV blocks of 16 tokens each
	cfg := SimConfig{
		Horizon:       1_000_000,
		Seed:          42,
		KVCacheConfig: NewKVCacheConfig(100, 16, 0, 0, 0, 0),
		BatchConfig:   NewBatchConfig(256, 2048, 0),
		LatencyCoeffs: NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
	}
	sim, err := NewSimulator(cfg)
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}

	// AND a request that fits (100 tokens needs ceil(100/16) = 7 blocks <= 100 total)
	normal := &Request{
		ID:          "normal_req",
		InputTokens: make([]int, 100),
		State:       StateQueued,
	}

	// WHEN we enqueue it
	sim.EnqueueRequest(normal)

	// THEN it must be in the wait queue
	if sim.WaitQ.Len() != 1 {
		t.Errorf("WaitQ.Len() = %d, want 1", sim.WaitQ.Len())
	}

	// AND DroppedUnservable must remain 0
	if sim.Metrics.DroppedUnservable != 0 {
		t.Errorf("DroppedUnservable = %d, want 0", sim.Metrics.DroppedUnservable)
	}

	// AND TotalInputTokens must include the request's tokens
	if sim.Metrics.TotalInputTokens != 100 {
		t.Errorf("TotalInputTokens = %d, want 100", sim.Metrics.TotalInputTokens)
	}
}

// BC-3, BC-5, BC-6: Simulation terminates with mixed oversized and normal requests
func TestSimulator_OversizedRequests_TerminatesNoLivelock(t *testing.T) {
	// GIVEN a simulator with very small KV cache (50 blocks × 16 tokens = 800 tokens)
	// This is the exact reproduction case from issue #373
	cfg := SimConfig{
		Horizon:       10_000_000,
		Seed:          42,
		KVCacheConfig: NewKVCacheConfig(50, 16, 0, 0, 0, 0),
		BatchConfig:   NewBatchConfig(256, 2048, 0),
		LatencyCoeffs: NewLatencyCoeffs([]float64{6910, 17.67, 2.84}, []float64{0, 0, 0}),
	}
	sim, err := NewSimulator(cfg)
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}

	// AND a mix of requests: some fit, some don't
	// Request 0: 900 tokens → ceil(900/16) = 57 blocks > 50 → dropped
	oversized := &Request{
		ID:           "request_oversized",
		InputTokens:  make([]int, 900),
		OutputTokens: make([]int, 10),
		ArrivalTime:  100_000,
		State:        StateQueued,
	}
	// Request 1: 100 tokens → ceil(100/16) = 7 blocks <= 50 → fits
	normal := &Request{
		ID:           "request_normal",
		InputTokens:  make([]int, 100),
		OutputTokens: make([]int, 10),
		ArrivalTime:  200_000,
		State:        StateQueued,
	}

	sim.InjectArrival(oversized)
	sim.InjectArrival(normal)

	// WHEN we run the simulation
	sim.Run()

	// THEN it must terminate (reaching this line proves no livelock — BC-6)

	// AND the oversized request must be dropped
	if sim.Metrics.DroppedUnservable != 1 {
		t.Errorf("DroppedUnservable = %d, want 1", sim.Metrics.DroppedUnservable)
	}

	// AND the normal request must complete
	if sim.Metrics.CompletedRequests != 1 {
		t.Errorf("CompletedRequests = %d, want 1", sim.Metrics.CompletedRequests)
	}

	// AND conservation must hold (BC-5):
	// completed + still_queued + still_running + dropped = total injected into EnqueueRequest
	total := sim.Metrics.CompletedRequests + sim.Metrics.StillQueued + sim.Metrics.StillRunning + sim.Metrics.DroppedUnservable
	if total != 2 {
		t.Errorf("conservation: completed(%d) + queued(%d) + running(%d) + dropped(%d) = %d, want 2",
			sim.Metrics.CompletedRequests, sim.Metrics.StillQueued, sim.Metrics.StillRunning,
			sim.Metrics.DroppedUnservable, total)
	}
}

// BC-6: All oversized — simulation still terminates
func TestSimulator_AllOversized_TerminatesEmpty(t *testing.T) {
	// GIVEN a simulator with tiny KV cache
	cfg := SimConfig{
		Horizon:       10_000_000,
		Seed:          42,
		KVCacheConfig: NewKVCacheConfig(5, 16, 0, 0, 0, 0),
		BatchConfig:   NewBatchConfig(256, 2048, 0),
		LatencyCoeffs: NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
	}
	sim, err := NewSimulator(cfg)
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}

	// AND all requests are oversized (200 tokens → 13 blocks > 5 total)
	for i := 0; i < 5; i++ {
		req := &Request{
			ID:           fmt.Sprintf("request_%d", i),
			InputTokens:  make([]int, 200),
			OutputTokens: make([]int, 10),
			ArrivalTime:  int64(i) * 100_000,
			State:        StateQueued,
		}
		sim.InjectArrival(req)
	}

	// WHEN we run the simulation
	sim.Run()

	// THEN it must terminate (no livelock)
	// AND all requests must be dropped
	if sim.Metrics.DroppedUnservable != 5 {
		t.Errorf("DroppedUnservable = %d, want 5", sim.Metrics.DroppedUnservable)
	}

	// AND no requests completed
	if sim.Metrics.CompletedRequests != 0 {
		t.Errorf("CompletedRequests = %d, want 0", sim.Metrics.CompletedRequests)
	}
}

// TestRequestLifecycle_ValidTransitions verifies INV-2:
// GIVEN a request injected into the simulator
// WHEN the simulation runs to completion
// THEN the request transitions through queued → running → completed.
func TestRequestLifecycle_ValidTransitions(t *testing.T) {
	sim := mustNewSimulator(t, SimConfig{
		Horizon:       math.MaxInt64,
		Seed:          42,
		KVCacheConfig: NewKVCacheConfig(100, 16, 0, 0, 0, 0),
		BatchConfig:   NewBatchConfig(1, 2048, 0),
		LatencyCoeffs: NewLatencyCoeffs([]float64{100, 1, 1}, []float64{50, 0.1, 50}),
	})

	req := &Request{
		ID:           "lifecycle_test",
		InputTokens:  make([]int, 16),
		OutputTokens: make([]int, 4),
		ArrivalTime:  0,
		State:        StateQueued,
	}

	// GIVEN: request starts in queued state
	if req.State != StateQueued {
		t.Fatalf("initial state = %q, want %q", req.State, StateQueued)
	}

	sim.InjectArrival(req)

	// Process events one by one to observe state transitions
	sawRunning := false
	for sim.HasPendingEvents() {
		sim.ProcessNextEvent()
		if req.State == StateRunning {
			sawRunning = true
		}
		// THEN: completed must not occur before running
		if req.State == StateCompleted && !sawRunning {
			t.Fatal("request reached StateCompleted without transitioning through StateRunning")
		}
	}

	// THEN: request MUST have completed
	if req.State != StateCompleted {
		t.Errorf("final state = %q, want %q", req.State, StateCompleted)
	}
	// THEN: request MUST have been running at some point
	if !sawRunning {
		t.Error("request never entered StateRunning")
	}
}
