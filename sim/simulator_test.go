package sim

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// fixedOverheadModel is a test-only LatencyModel stub with configurable PostDecodeFixedOverhead.
type fixedOverheadModel struct {
	overhead int64
}

func (m *fixedOverheadModel) StepTime(batch []*Request) int64     { return 1 }
func (m *fixedOverheadModel) QueueingTime(req *Request) int64      { return 0 }
func (m *fixedOverheadModel) OutputTokenProcessingTime() int64     { return 0 }
func (m *fixedOverheadModel) PostDecodeFixedOverhead() int64       { return m.overhead }

// spyLatencyModel records the batch passed to StepTime for inspection.
// Used by BC-2/BC-3 tests to verify idle request filtering.
type spyLatencyModel struct {
	lastBatch []*Request
}

func (m *spyLatencyModel) StepTime(batch []*Request) int64 {
	m.lastBatch = batch
	return 1
}
func (m *spyLatencyModel) QueueingTime(req *Request) int64  { return 0 }
func (m *spyLatencyModel) OutputTokenProcessingTime() int64 { return 0 }
func (m *spyLatencyModel) PostDecodeFixedOverhead() int64   { return 0 }

// BC-2 (#963): StepTime receives only requests with NumNewTokens > 0.
// vLLM proof: scheduler.py:870 adds to scheduled_running_reqs only on
// successful allocation; scheduler.py:1236-1242 builds SchedulerOutput
// from scheduled_running_reqs, NOT self.running.
func TestExecuteBatchStep_FiltersIdleRequests_BC2(t *testing.T) {
	// GIVEN a running batch with 3 requests: 2 active (NumNewTokens > 0), 1 idle (NumNewTokens = 0)
	spy := &spyLatencyModel{}
	cfg := SimConfig{
		KVCacheConfig: NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:   NewBatchConfig(256, 2048, 0),
		Seed:          42,
	}
	kvStore := MustNewKVStoreFromConfig(cfg.KVCacheConfig)
	s, err := NewSimulator(cfg, kvStore, spy)
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}

	active1 := &Request{ID: "active1", InputTokens: make([]int, 32), OutputTokens: make([]int, 10),
		ProgressIndex: 32, NumNewTokens: 1, State: StateRunning}
	active2 := &Request{ID: "active2", InputTokens: make([]int, 16), OutputTokens: make([]int, 5),
		ProgressIndex: 16, NumNewTokens: 1, State: StateRunning}
	idle := &Request{ID: "idle", InputTokens: make([]int, 64), OutputTokens: make([]int, 20),
		ProgressIndex: 64, NumNewTokens: 0, State: StateRunning}

	s.RunningBatch = &Batch{Requests: []*Request{active1, active2, idle}}
	s.reqNumComputedTokens = map[string]int64{
		"active1": 32, "active2": 16, "idle": 64,
	}

	// WHEN executeBatchStep runs
	s.executeBatchStep(0)

	// THEN StepTime was called with only the 2 active requests (BC-2)
	if len(spy.lastBatch) != 2 {
		t.Fatalf("StepTime must receive only active requests, got %d", len(spy.lastBatch))
	}
	ids := map[string]bool{}
	for _, r := range spy.lastBatch {
		ids[r.ID] = true
	}
	if !ids["active1"] || !ids["active2"] {
		t.Errorf("StepTime batch missing active requests: got %v", ids)
	}
	if ids["idle"] {
		t.Errorf("StepTime batch should NOT contain idle request")
	}
}

// BC-3 (#963): Idle requests persist in RunningBatch after executeBatchStep.
func TestExecuteBatchStep_IdleRequestsPersist_BC3(t *testing.T) {
	// GIVEN a running batch with 1 active and 1 idle request
	spy := &spyLatencyModel{}
	cfg := SimConfig{
		KVCacheConfig: NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:   NewBatchConfig(256, 2048, 0),
		Seed:          42,
	}
	kvStore := MustNewKVStoreFromConfig(cfg.KVCacheConfig)
	s, err := NewSimulator(cfg, kvStore, spy)
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}

	active := &Request{ID: "active", InputTokens: make([]int, 32), OutputTokens: make([]int, 10),
		ProgressIndex: 32, NumNewTokens: 1, State: StateRunning}
	idle := &Request{ID: "idle", InputTokens: make([]int, 64), OutputTokens: make([]int, 20),
		ProgressIndex: 64, NumNewTokens: 0, State: StateRunning}

	s.RunningBatch = &Batch{Requests: []*Request{active, idle}}
	s.reqNumComputedTokens = map[string]int64{"active": 32, "idle": 64}

	// WHEN executeBatchStep runs
	s.executeBatchStep(0)

	// THEN both requests are still in RunningBatch (idle was not dropped) (BC-3)
	if len(s.RunningBatch.Requests) != 2 {
		t.Fatalf("idle requests must persist in RunningBatch, got %d requests", len(s.RunningBatch.Requests))
	}
	found := false
	for _, r := range s.RunningBatch.Requests {
		if r.ID == "idle" {
			found = true
		}
	}
	if !found {
		t.Error("idle request was dropped from RunningBatch (BC-3 violation)")
	}
}

// BC-1: Simulator.PostDecodeFixedOverhead() delegates to the underlying LatencyModel.
func TestSimulator_PostDecodeFixedOverhead_DelegatesToModel(t *testing.T) {
	tests := []struct {
		name     string
		overhead int64
	}{
		{"zero (blackbox/roofline)", 0},
		{"positive (trained-roofline)", 1850},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := newTestSimConfig()
			kvStore := MustNewKVStoreFromConfig(cfg.KVCacheConfig)
			model := &fixedOverheadModel{overhead: tc.overhead}
			s, err := NewSimulator(cfg, kvStore, model)
			if err != nil {
				t.Fatalf("NewSimulator: %v", err)
			}
			if got := s.PostDecodeFixedOverhead(); got != tc.overhead {
				t.Errorf("PostDecodeFixedOverhead() = %d, want %d", got, tc.overhead)
			}
		})
	}
}

// mustNewSimulator is a test helper that calls NewSimulator and fails the test on error.
// Honors KVCPUBlocks for tiered KV cache construction via MustNewKVStoreFromConfig.
func mustNewSimulator(t *testing.T, cfg SimConfig) *Simulator {
	t.Helper()
	kvStore := MustNewKVStoreFromConfig(cfg.KVCacheConfig)
	latencyModel, err := MustNewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("MustNewLatencyModel: %v", err)
	}
	s, err := NewSimulator(cfg, kvStore, latencyModel)
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
				ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, tc.Model, tc.Hardware, tc.TP, "blackbox", tc.MaxModelLen),
			})

			requests := testGenerateRequests(tc.Seed, math.MaxInt64, tc.Rate/1e6,
				tc.NumRequests, tc.PrefixTokens,
				tc.PromptTokens, tc.PromptTokensStdev, tc.PromptTokensMin, tc.PromptTokensMax,
				tc.OutputTokens, tc.OutputTokensStdev, tc.OutputTokensMin, tc.OutputTokensMax)
			injectRequests(sim, requests)

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
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-model", "H100", 1, "blackbox", 0),
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
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test", "H100", 1, "blackbox", 0),
	}

	requests := testGenerateRequests(42, math.MaxInt64, 10.0/1e6, 50,
		0, 100, 20, 10, 200, 50, 10, 10, 100)

	sim1 := mustNewSimulator(t, cfg)
	for _, req := range requests {
		sim1.InjectArrival(req)
	}

	// Re-generate identical requests for sim2 (same seed, same params)
	requests2 := testGenerateRequests(42, math.MaxInt64, 10.0/1e6, 50,
		0, 100, 20, 10, 200, 50, 10, 10, 100)

	sim2 := mustNewSimulator(t, cfg)
	for _, req := range requests2 {
		sim2.InjectArrival(req)
	}

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
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-model", "H100", 1, "blackbox", 0),
	}
}

func TestNewSimulator_NilKVStore_ReturnsError(t *testing.T) {
	cfg := newTestSimConfig()
	latencyModel, err := MustNewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("MustNewLatencyModel: %v", err)
	}
	_, err = NewSimulator(cfg, nil, latencyModel)
	if err == nil {
		t.Fatal("expected error for nil kvStore")
	}
	if err.Error() != "NewSimulator: kvStore must not be nil" {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestNewSimulator_BatchConfigValidation(t *testing.T) {
	tests := []struct {
		name           string
		maxRunning     int64
		maxTokens      int64
		prefillThresh  int64
		wantErrContain string
	}{
		{"zero_max_running", 0, 2048, 0, "MaxRunningReqs"},
		{"negative_max_running", -1, 2048, 0, "MaxRunningReqs"},
		{"zero_max_tokens", 256, 0, 0, "MaxScheduledTokens"},
		{"negative_max_tokens", 256, -1, 0, "MaxScheduledTokens"},
		{"negative_prefill_threshold", 256, 2048, -1, "LongPrefillTokenThreshold"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := newTestSimConfig()
			// Use struct literal to bypass NewBatchConfig validation — this tests
			// NewSimulator's defense-in-depth, not the constructor.
			cfg.BatchConfig = BatchConfig{
				MaxRunningReqs:            tc.maxRunning,
				MaxScheduledTokens:        tc.maxTokens,
				LongPrefillTokenThreshold: tc.prefillThresh,
			}
			kvStore := MustNewKVStoreFromConfig(cfg.KVCacheConfig)
			latencyModel, err := MustNewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
			if err != nil {
				t.Fatalf("MustNewLatencyModel: %v", err)
			}
			_, err = NewSimulator(cfg, kvStore, latencyModel)
			if err == nil {
				t.Fatalf("expected error for %s", tc.name)
			}
			if !strings.Contains(err.Error(), tc.wantErrContain) {
				t.Errorf("error %q should contain %q", err.Error(), tc.wantErrContain)
			}
		})
	}
}

func TestMustNewKVCacheState_NilFunc_Panics(t *testing.T) {
	// Save and restore the registered function
	saved := NewKVCacheStateFunc
	defer func() { NewKVCacheStateFunc = saved }()
	NewKVCacheStateFunc = nil

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for nil NewKVCacheStateFunc")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("expected string panic, got %T: %v", r, r)
		}
		expected := "NewKVCacheStateFunc not registered: import sim/kv to register it " +
			"(add: import _ \"github.com/inference-sim/inference-sim/sim/kv\")"
		if msg != expected {
			t.Errorf("panic message:\n  got:  %q\n  want: %q", msg, expected)
		}
	}()
	MustNewKVCacheState(100, 16)
}

func TestMustNewLatencyModel_NilFunc_Panics(t *testing.T) {
	// Save and restore the registered function
	saved := NewLatencyModelFunc
	defer func() { NewLatencyModelFunc = saved }()
	NewLatencyModelFunc = nil

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for nil NewLatencyModelFunc")
		}
		msg, ok := r.(string)
		if !ok {
			t.Fatalf("expected string panic, got %T: %v", r, r)
		}
		expected := "NewLatencyModelFunc not registered: import sim/latency to register it " +
			"(add: import _ \"github.com/inference-sim/inference-sim/sim/latency\")"
		if msg != expected {
			t.Errorf("panic message:\n  got:  %q\n  want: %q", msg, expected)
		}
	}()
	coeffs := NewLatencyCoeffs([]float64{1, 2, 3}, []float64{1, 2, 3})
	hw := NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, "blackbox", 0)
	_, _ = MustNewLatencyModel(coeffs, hw) //nolint:errcheck // expected to panic before returning
}

func TestNewSimulator_NilLatencyModel_ReturnsError(t *testing.T) {
	cfg := newTestSimConfig()
	kvStore := MustNewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
	_, err := NewSimulator(cfg, kvStore, nil)
	if err == nil {
		t.Fatal("expected error for nil latencyModel")
	}
	if err.Error() != "NewSimulator: latencyModel must not be nil" {
		t.Errorf("unexpected error message: %v", err)
	}
}

// BC-1: KV cache too small for MaxModelLen → error
func TestNewSimulator_MaxModelLen_KVTooSmall(t *testing.T) {
	cfg := newTestSimConfig()
	// 1024 tokens / 16 block size = 64 blocks needed, but only 50 available
	cfg.ModelHardwareConfig = NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test", "H100", 1, "blackbox", 1024)
	cfg.KVCacheConfig = NewKVCacheConfig(50, 16, 0, 0, 0, 0)

	kvStore := MustNewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
	latencyModel, err := MustNewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("MustNewLatencyModel: %v", err)
	}
	_, err = NewSimulator(cfg, kvStore, latencyModel)
	if err == nil {
		t.Fatal("expected error when KV cache too small for MaxModelLen")
	}
	if !strings.Contains(err.Error(), "KV cache too small for MaxModelLen") {
		t.Errorf("error should mention KV cache too small, got: %v", err)
	}
}

// BC-1: KV cache sufficient for MaxModelLen → no error
func TestNewSimulator_MaxModelLen_KVSufficient(t *testing.T) {
	cfg := newTestSimConfig()
	// 1024 tokens / 16 block size = 64 blocks needed, 100 available
	cfg.ModelHardwareConfig = NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test", "H100", 1, "blackbox", 1024)
	cfg.KVCacheConfig = NewKVCacheConfig(100, 16, 0, 0, 0, 0)
	_ = mustNewSimulator(t, cfg) // should not error
}

// BC-3: MaxModelLen=0 bypasses the KV check
func TestNewSimulator_MaxModelLen_Zero_NoValidation(t *testing.T) {
	cfg := newTestSimConfig()
	// MaxModelLen=0 (default) — no validation even with small KV cache
	cfg.ModelHardwareConfig = NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test", "H100", 1, "blackbox", 0)
	cfg.KVCacheConfig = NewKVCacheConfig(1, 16, 0, 0, 0, 0)
	_ = mustNewSimulator(t, cfg) // should not error
}

// TestNewSimulator_NoWorkload_EmptyQueue verifies that a SimConfig with no injected
// requests creates a simulator with an empty EventQueue and runs to completion
// with zero results.
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
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-conservation", "H100", 1, "blackbox", 0),
	}

	sim := mustNewSimulator(t, cfg)
	requests := testGenerateRequests(99, math.MaxInt64, 10.0/1e6, 50,
		0, 100, 20, 10, 200, 50, 10, 10, 100)
	injectRequests(sim, requests)
	sim.Run()

	// Five-term equation: injected == completed + queued + running + dropped + timedOut (INV-1)
	injected := 50 // independent count: we injected exactly 50 requests above
	completed := sim.Metrics.CompletedRequests
	queued := sim.WaitQ.Len()
	running := 0
	if sim.RunningBatch != nil {
		running = len(sim.RunningBatch.Requests)
	}
	dropped := sim.Metrics.DroppedUnservable
	timedOut := sim.Metrics.TimedOutRequests

	if completed+queued+running+dropped+timedOut != injected {
		t.Errorf("request conservation violated: completed(%d) + queued(%d) + running(%d) + dropped(%d) + timedOut(%d) = %d, injected = %d",
			completed, queued, running, dropped, timedOut, completed+queued+running+dropped+timedOut, injected)
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
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-conservation-finite", "H100", 1, "blackbox", 0),
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

	injected := 15 // independent count: 10 early + 5 late
	completed := sim.Metrics.CompletedRequests
	queued := sim.WaitQ.Len()
	running := 0
	if sim.RunningBatch != nil {
		running = len(sim.RunningBatch.Requests)
	}
	dropped := sim.Metrics.DroppedUnservable
	timedOut := sim.Metrics.TimedOutRequests

	sum := completed + queued + running + dropped + timedOut
	if sum != injected {
		t.Errorf("request conservation violated: completed(%d) + queued(%d) + running(%d) + dropped(%d) + timedOut(%d) = %d, injected = %d",
			completed, queued, running, dropped, timedOut, sum, injected)
	}

	// Verify we actually tested the non-trivial case: some but not all completed
	if completed == injected {
		t.Fatalf("all %d requests completed — horizon too long, conservation case untested", injected)
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
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-causality", "H100", 1, "blackbox", 0),
	}

	sim := mustNewSimulator(t, cfg)
	requests := testGenerateRequests(77, math.MaxInt64, 5.0/1e6, 30,
		0, 100, 20, 10, 200, 50, 10, 10, 100)
	injectRequests(sim, requests)
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
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-monotonicity", "H100", 1, "blackbox", 0),
	}

	sim := mustNewSimulator(t, cfg)
	requests := testGenerateRequests(55, math.MaxInt64, 10.0/1e6, 20,
		0, 50, 10, 10, 100, 20, 5, 5, 40)
	injectRequests(sim, requests)

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

// TestInjectArrival_BeyondHorizon_Warns verifies BC-8 (bugfix-audit):
// GIVEN a request with ArrivalTime > sim.Horizon
// WHEN InjectArrival is called
// THEN the request is still registered (backward compatible) and no panic occurs.
func TestInjectArrival_BeyondHorizon_Warns(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(100, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(10, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 1, 1}, []float64{50, 0.1, 50}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, "blackbox", 0),
	}
	sim := mustNewSimulator(t, cfg)
	req := &Request{
		ID: "beyond_horizon", InputTokens: make([]int, 16),
		OutputTokens: make([]int, 4), ArrivalTime: 2000, State: StateQueued,
	}
	sim.InjectArrival(req) // should not panic
	// Request is registered (backward compatible)
	if _, ok := sim.Metrics.Requests["beyond_horizon"]; !ok {
		t.Error("request should still be registered in Metrics.Requests")
	}
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
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-determinism", "H100", 1, "blackbox", 0),
	}

	// Run 1
	sim1 := mustNewSimulator(t, cfg)
	requests1 := testGenerateRequests(42, math.MaxInt64, 5.0/1e6, 20,
		0, 100, 20, 10, 200, 50, 10, 10, 100)
	injectRequests(sim1, requests1)
	sim1.Run()
	f1 := t.TempDir() + "/run1.json"
	if err := sim1.Metrics.SaveResults("determinism-test", cfg.Horizon, cfg.TotalKVBlocks, f1); err != nil {
		t.Fatalf("SaveResults run1: %v", err)
	}

	// Run 2
	sim2 := mustNewSimulator(t, cfg)
	requests2 := testGenerateRequests(42, math.MaxInt64, 5.0/1e6, 20,
		0, 100, 20, 10, 200, 50, 10, 10, 100)
	injectRequests(sim2, requests2)
	sim2.Run()
	f2 := t.TempDir() + "/run2.json"
	if err := sim2.Metrics.SaveResults("determinism-test", cfg.Horizon, cfg.TotalKVBlocks, f2); err != nil {
		t.Fatalf("SaveResults run2: %v", err)
	}

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
				ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-kv-conservation", "H100", 1, "blackbox", 0),
			}

			sim := mustNewSimulator(t, cfg)
			requests := testGenerateRequests(42, math.MaxInt64, 5.0/1e6, 20,
				0, 100, 20, 10, 200, 50, 10, 10, 100)
			injectRequests(sim, requests)
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
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-work-conserving", "H100", 1, "blackbox", 0),
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
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
		ModelHardwareConfig: ModelHardwareConfig{Backend: "blackbox"},
	}
	kvStore := MustNewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
	latencyModel, err := MustNewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("MustNewLatencyModel: %v", err)
	}
	sim, err := NewSimulator(cfg, kvStore, latencyModel)
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
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(100, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
		ModelHardwareConfig: ModelHardwareConfig{Backend: "blackbox"},
	}
	kvStore := MustNewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
	latencyModel, err := MustNewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("MustNewLatencyModel: %v", err)
	}
	sim, err := NewSimulator(cfg, kvStore, latencyModel)
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

// BC-2: Request with explicit budget exceeding MaxModelLen is dropped
func TestEnqueueRequest_MaxModelLen_Exceeded_Dropped(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, "blackbox", 512),
	}
	sim := mustNewSimulator(t, cfg)

	// Request: 300 input + 300 budget = 600 > MaxModelLen(512)
	req := &Request{
		ID:           "too_long",
		InputTokens:  make([]int, 300),
		OutputTokens: make([]int, 300),
		MaxOutputLen: 300, // client declares output budget
		State:        StateQueued,
	}
	sim.Metrics.Requests[req.ID] = NewRequestMetrics(req, 0)

	sim.EnqueueRequest(req)

	if sim.WaitQ.Len() != 0 {
		t.Errorf("WaitQ.Len() = %d, want 0", sim.WaitQ.Len())
	}
	if sim.Metrics.DroppedUnservable != 1 {
		t.Errorf("DroppedUnservable = %d, want 1", sim.Metrics.DroppedUnservable)
	}
	if _, exists := sim.Metrics.Requests[req.ID]; exists {
		t.Error("dropped request should be removed from Metrics.Requests")
	}
}

// BC-3: MaxModelLen=0 falls through to KV check only
func TestEnqueueRequest_MaxModelLen_Zero_FallsThroughToKV(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, "blackbox", 0),
	}
	sim := mustNewSimulator(t, cfg)

	// Large request that fits KV (100 tokens, 1000 blocks available) but would fail MaxModelLen if it were set
	req := &Request{
		ID:           "large_but_fits_kv",
		InputTokens:  make([]int, 100),
		OutputTokens: make([]int, 1000),
		State:        StateQueued,
	}

	sim.EnqueueRequest(req)

	if sim.WaitQ.Len() != 1 {
		t.Errorf("WaitQ.Len() = %d, want 1 (MaxModelLen=0 should not reject)", sim.WaitQ.Len())
	}
}

// BC-4: MaxOutputLen=0 → input-only check (oracle knowledge boundary: control plane
// never peeks at len(OutputTokens)). Runtime stop enforces output growth limit.
func TestEnqueueRequest_MaxOutputLen_OracleKnowledgeBoundary(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, "blackbox", 512),
	}
	sim := mustNewSimulator(t, cfg)

	// Case 1: MaxOutputLen=0 → auto-filled to 312 (512-200), input+budget=512 ≤ 512 → enqueued.
	// Auto-fill sets MaxOutputLen=312 (remaining context). Guard 1 budget check passes (200+312=512).
	// The control plane still doesn't peek at len(OutputTokens) (INV-9 preserved).
	reqFits := &Request{
		ID:           "input_fits_oracle",
		InputTokens:  make([]int, 200),
		OutputTokens: make([]int, 1000), // actual output exceeds context, but control plane can't see this
		MaxOutputLen: 0,                 // auto-filled to maxModelLen - input = 312
		State:        StateQueued,
	}
	sim.EnqueueRequest(reqFits)
	if sim.WaitQ.Len() != 1 {
		t.Errorf("WaitQ.Len() = %d, want 1 (MaxOutputLen=0 should only check input)", sim.WaitQ.Len())
	}

	// Case 2: MaxOutputLen=0, input exceeds MaxModelLen → dropped
	reqInputTooBig := &Request{
		ID:           "input_too_big",
		InputTokens:  make([]int, 600), // 600 > 512
		OutputTokens: make([]int, 10),
		MaxOutputLen: 0,
		State:        StateQueued,
	}
	sim.Metrics.Requests[reqInputTooBig.ID] = NewRequestMetrics(reqInputTooBig, 0)
	sim.EnqueueRequest(reqInputTooBig)
	if sim.WaitQ.Len() != 1 {
		t.Errorf("WaitQ.Len() = %d, want 1 (input exceeds MaxModelLen)", sim.WaitQ.Len())
	}
	if sim.Metrics.DroppedUnservable != 1 {
		t.Errorf("DroppedUnservable = %d, want 1", sim.Metrics.DroppedUnservable)
	}

	// Case 3: MaxOutputLen=400 (client budget) → total: 200 + 400 = 600 > 512 → dropped
	reqBudgetExceeds := &Request{
		ID:           "budget_exceeds",
		InputTokens:  make([]int, 200),
		OutputTokens: make([]int, 100), // actual output is 100, but client budget says 400
		MaxOutputLen: 400,
		State:        StateQueued,
	}
	sim.Metrics.Requests[reqBudgetExceeds.ID] = NewRequestMetrics(reqBudgetExceeds, 0)
	sim.EnqueueRequest(reqBudgetExceeds)
	if sim.WaitQ.Len() != 1 {
		t.Errorf("WaitQ.Len() = %d, want 1 (budget-exceeded request should be dropped)", sim.WaitQ.Len())
	}
	if sim.Metrics.DroppedUnservable != 2 {
		t.Errorf("DroppedUnservable = %d, want 2", sim.Metrics.DroppedUnservable)
	}
}

// Boundary test: input == maxModelLen with no budget → dropped (vLLM uses >= at serving.py:1542).
// A request whose input fills the entire context leaves no room for output.
func TestEnqueueRequest_InputEqualsMaxModelLen_Dropped(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, "blackbox", 512),
	}
	sim := mustNewSimulator(t, cfg)

	// input == MaxModelLen: fills the entire context, no room for output
	req := &Request{
		ID:           "exact_boundary",
		InputTokens:  make([]int, 512), // == MaxModelLen
		OutputTokens: make([]int, 10),
		State:        StateQueued,
	}
	sim.Metrics.Requests[req.ID] = NewRequestMetrics(req, 0)
	sim.EnqueueRequest(req)

	if sim.WaitQ.Len() != 0 {
		t.Errorf("WaitQ.Len() = %d, want 0 (input==MaxModelLen should be rejected)", sim.WaitQ.Len())
	}
	if sim.Metrics.DroppedUnservable != 1 {
		t.Errorf("DroppedUnservable = %d, want 1", sim.Metrics.DroppedUnservable)
	}
}

// Boundary test: input + budget == maxModelLen → accepted (exact fit, not exceeded).
func TestEnqueueRequest_ExactFit_Accepted(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, "blackbox", 512),
	}
	sim := mustNewSimulator(t, cfg)

	// input=200 + budget=312 = 512 == MaxModelLen → exact fit, should be accepted
	req := &Request{
		ID:           "exact_fit",
		InputTokens:  make([]int, 200),
		OutputTokens: make([]int, 312),
		MaxOutputLen: 312,
		State:        StateQueued,
	}
	sim.EnqueueRequest(req)
	if sim.WaitQ.Len() != 1 {
		t.Errorf("WaitQ.Len() = %d, want 1 (exact fit should be accepted)", sim.WaitQ.Len())
	}
}

// BC-7: Negative MaxOutputLen → warning + dropped
func TestEnqueueRequest_NegativeMaxOutputLen_Dropped(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{6910, 17.67, 2.84}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test", "H100", 1, "blackbox", 512),
	}
	sim := mustNewSimulator(t, cfg)

	req := &Request{
		ID:           "neg_budget",
		InputTokens:  make([]int, 100),
		OutputTokens: make([]int, 50),
		MaxOutputLen: -1,
		State:        StateQueued,
	}
	sim.Metrics.Requests[req.ID] = NewRequestMetrics(req, 0)
	sim.EnqueueRequest(req)

	if sim.WaitQ.Len() != 0 {
		t.Errorf("WaitQ.Len() = %d, want 0 (negative MaxOutputLen should be dropped)", sim.WaitQ.Len())
	}
	if sim.Metrics.DroppedUnservable != 1 {
		t.Errorf("DroppedUnservable = %d, want 1", sim.Metrics.DroppedUnservable)
	}
	if _, exists := sim.Metrics.Requests["neg_budget"]; exists {
		t.Error("Metrics.Requests should not contain dropped request")
	}
}

// BC-5: Runtime length cap force-completes request at MaxModelLen boundary.
// This is a defense-in-depth test: we directly place a request in the running batch
// with ProgressIndex already at MaxModelLen to simulate bypass of enqueue guard.
func TestProcessCompletions_RuntimeLengthCap(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, "blackbox", 100),
	}
	sim := mustNewSimulator(t, cfg)

	// Create a request with ProgressIndex at MaxModelLen boundary (bypassing enqueue guard)
	req := &Request{
		ID:           "length_capped",
		InputTokens:  make([]int, 50),
		OutputTokens: make([]int, 200), // would normally be longer than MaxModelLen
		State:        StateRunning,
		ProgressIndex: 100, // == MaxModelLen → should be force-completed
		ArrivalTime:  0,
	}
	sim.Metrics.Requests[req.ID] = NewRequestMetrics(req, 0)
	sim.RunningBatch = &Batch{Requests: []*Request{req}}

	// Simulate processCompletions
	remaining := sim.processCompletions(1000, 5000)

	// THEN the request is force-completed
	if len(remaining) != 0 {
		t.Errorf("remaining = %d, want 0 (length-capped request should be completed)", len(remaining))
	}
	if req.State != StateCompleted {
		t.Errorf("req.State = %s, want completed", req.State)
	}
	if sim.Metrics.CompletedRequests != 1 {
		t.Errorf("CompletedRequests = %d, want 1", sim.Metrics.CompletedRequests)
	}

	// BC-2: LengthCappedRequests counter incremented
	if sim.Metrics.LengthCappedRequests != 1 {
		t.Errorf("LengthCappedRequests = %d, want 1", sim.Metrics.LengthCappedRequests)
	}

	// LengthCapped flag set on request
	if !req.LengthCapped {
		t.Error("req.LengthCapped = false, want true")
	}

	// Conservation: no requests lost
	if sim.WaitQ.Len()+len(remaining)+sim.Metrics.CompletedRequests != 1 {
		t.Errorf("conservation violated: queued=%d + remaining=%d + completed=%d != 1",
			sim.WaitQ.Len(), len(remaining), sim.Metrics.CompletedRequests)
	}
}

// BC-5: End-to-end runtime length cap via sim.Run()
// Injects a request that passes the enqueue guard (MaxOutputLen=0 → input-only check)
// but has OutputTokens exceeding MaxModelLen. The runtime cap in processCompletions
// should force-complete the request at the MaxModelLen boundary.
func TestSimulator_RuntimeLengthCap_E2E(t *testing.T) {
	cfg := SimConfig{
		Horizon:             10_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{6910, 17.67, 2.84}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test", "H100", 1, "blackbox", 100),
	}
	sim := mustNewSimulator(t, cfg)

	// MaxOutputLen=0 → auto-filled to 50 (100-50). Guard 1 budget check: 50+50=100 ≤ 100 → enqueued.
	// But actual OutputTokens=200 means ProgressIndex will exceed MaxModelLen=100.
	req := &Request{
		ID:           "will_be_capped",
		InputTokens:  GenerateRandomTokenIDs(sim.WorkloadRNG(), 50),
		OutputTokens: GenerateRandomTokenIDs(sim.WorkloadRNG(), 200),
		ArrivalTime:  0,
		State:        StateQueued,
	}
	sim.InjectArrival(req)

	sim.Run()

	// Request completes (force-completed at MaxModelLen boundary)
	if sim.Metrics.CompletedRequests != 1 {
		t.Errorf("CompletedRequests = %d, want 1", sim.Metrics.CompletedRequests)
	}
	if sim.Metrics.LengthCappedRequests != 1 {
		t.Errorf("LengthCappedRequests = %d, want 1", sim.Metrics.LengthCappedRequests)
	}
	// INV-1: conservation holds
	total := sim.Metrics.CompletedRequests + sim.Metrics.StillQueued + sim.Metrics.StillRunning + sim.Metrics.DroppedUnservable
	if total != 1 {
		t.Errorf("INV-1: completed(%d)+queued(%d)+running(%d)+dropped(%d) = %d, want 1",
			sim.Metrics.CompletedRequests, sim.Metrics.StillQueued, sim.Metrics.StillRunning, sim.Metrics.DroppedUnservable, total)
	}
	// Output was truncated below the full 200
	if sim.Metrics.TotalOutputTokens >= 200 {
		t.Errorf("TotalOutputTokens = %d, want < 200 (force-completion should truncate)", sim.Metrics.TotalOutputTokens)
	}
	if sim.Metrics.TotalOutputTokens == 0 {
		t.Error("TotalOutputTokens = 0, want > 0 (some decode work should have happened)")
	}
	// Regression anchor: proactive cap stops at MaxModelLen(100)-1-input(50) = 49 decode tokens
	if sim.Metrics.TotalOutputTokens != 49 {
		t.Errorf("TotalOutputTokens = %d, want 49 (regression anchor)", sim.Metrics.TotalOutputTokens)
	}
	// KV blocks released
	if sim.KVCache.UsedBlocks() != 0 {
		t.Errorf("UsedBlocks = %d, want 0 (KV blocks should be released after force-completion)", sim.KVCache.UsedBlocks())
	}
}

// BC-7: INV-1 conservation holds when MaxModelLen drops requests.
// End-to-end test: inject mix of requests (some exceed budget, some fit),
// run to completion, verify injected == completed + dropped.
func TestSimulator_Conservation_WithMaxModelLen_Drops(t *testing.T) {
	cfg := SimConfig{
		Horizon:             10_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, "blackbox", 200),
	}
	sim := mustNewSimulator(t, cfg)
	rng := sim.WorkloadRNG()

	// Inject 20 requests: 10 that fit (input=50, output=50, budget=100 → 150 < 200)
	// and 10 that exceed (input=50, output=50, budget=200 → 250 > 200)
	numInjected := 0
	for i := 0; i < 10; i++ {
		req := &Request{
			ID:           fmt.Sprintf("fits_%d", i),
			InputTokens:  GenerateRandomTokenIDs(rng, 50),
			OutputTokens: GenerateRandomTokenIDs(rng, 50),
			MaxOutputLen: 100,
			ArrivalTime:  int64(i * 100000),
			State:        StateQueued,
		}
		sim.InjectArrival(req)
		numInjected++
	}
	for i := 0; i < 10; i++ {
		req := &Request{
			ID:           fmt.Sprintf("exceeds_%d", i),
			InputTokens:  GenerateRandomTokenIDs(rng, 50),
			OutputTokens: GenerateRandomTokenIDs(rng, 50),
			MaxOutputLen: 200, // 50 + 200 = 250 > MaxModelLen(200)
			ArrivalTime:  int64(i * 100000),
			State:        StateQueued,
		}
		sim.InjectArrival(req)
		numInjected++
	}

	sim.Run()

	// INV-1: injected == completed + still_queued + still_running + dropped
	total := sim.Metrics.CompletedRequests + sim.Metrics.StillQueued +
		sim.Metrics.StillRunning + sim.Metrics.DroppedUnservable
	if total != numInjected {
		t.Errorf("INV-1 violation: completed(%d) + queued(%d) + running(%d) + dropped(%d) = %d, want %d",
			sim.Metrics.CompletedRequests, sim.Metrics.StillQueued,
			sim.Metrics.StillRunning, sim.Metrics.DroppedUnservable,
			total, numInjected)
	}

	// Verify: 10 requests should be dropped, 10 should complete
	if sim.Metrics.DroppedUnservable != 10 {
		t.Errorf("DroppedUnservable = %d, want 10", sim.Metrics.DroppedUnservable)
	}
	if sim.Metrics.CompletedRequests != 10 {
		t.Errorf("CompletedRequests = %d, want 10", sim.Metrics.CompletedRequests)
	}
}

// BC-1: Negative MaxModelLen rejected by NewSimulator (defense-in-depth for struct literal bypass)
func TestNewSimulator_NegativeMaxModelLen_Error(t *testing.T) {
	cfg := newTestSimConfig()
	cfg.MaxModelLen = -5 // bypass canonical constructor's panic to test NewSimulator validation
	kvStore := MustNewKVStoreFromConfig(cfg.KVCacheConfig)
	latencyModel, err := MustNewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("MustNewLatencyModel: %v", err)
	}
	_, err = NewSimulator(cfg, kvStore, latencyModel)
	if err == nil {
		t.Fatal("expected error for negative MaxModelLen")
	}
	if !strings.Contains(err.Error(), "MaxModelLen") {
		t.Errorf("error %q should mention MaxModelLen", err.Error())
	}
}

// R3: NewModelHardwareConfig panics on negative MaxModelLen
func TestNewModelHardwareConfig_NegativeMaxModelLen_Panics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic for negative MaxModelLen")
		}
		msg := fmt.Sprintf("%v", r)
		if !strings.Contains(msg, "MaxModelLen") {
			t.Errorf("panic message %q should contain MaxModelLen", msg)
		}
	}()
	NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, "blackbox", -1)
}

// INV-9: Oracle Knowledge Boundary — control-plane functions must not reference OutputTokens.
// This is a structural enforcement test: it reads the source files for servability-decision
// functions and verifies zero references to OutputTokens.
func TestINV9_OracleKnowledgeBoundary_NoOutputTokensInControlPlane(t *testing.T) {
	// Control-plane files in sim/ that must not reference OutputTokens.
	// These files contain no metric-aggregate names like TotalOutputTokens,
	// so a whole-file scan is safe and maximally conservative.
	simControlPlaneFiles := []string{
		"admission.go",
		"routing.go",
		"routing_scorers.go",
		"routing_prefix_scorer.go",
		"scheduler.go",
		"priority.go",
	}

	for _, filename := range simControlPlaneFiles {
		data, err := os.ReadFile(filename)
		if err != nil {
			t.Fatalf("failed to read %s: %v", filename, err)
		}
		content := string(data)
		if strings.Contains(content, "OutputTokens") {
			t.Errorf("INV-9 violation: %s references OutputTokens — control-plane code must not access oracle output length", filename)
		}
	}

	// Cluster control-plane files that handle *Request in the routing pipeline.
	// These files may contain TotalOutputTokens (metric aggregation, not oracle access),
	// so we use line-level scanning with TotalOutputTokens exclusion.
	clusterControlPlaneFiles := []string{
		"cluster/cluster.go",
		"cluster/cluster_event.go",
		"cluster/snapshot.go",
		"cluster/counterfactual.go",
	}

	for _, filename := range clusterControlPlaneFiles {
		data, err := os.ReadFile(filename)
		if err != nil {
			t.Fatalf("failed to read %s: %v", filename, err)
		}
		for lineNum, line := range strings.Split(string(data), "\n") {
			// Remove known-safe metric aggregate names, then check for remaining OutputTokens
			cleaned := strings.ReplaceAll(line, "TotalOutputTokens", "")
			if strings.Contains(cleaned, "OutputTokens") {
				t.Errorf("INV-9 violation: %s line %d references OutputTokens — control-plane code must not access oracle output length",
					filename, lineNum+1)
			}
		}
	}

	// Additionally verify EnqueueRequest specifically (it's in simulator.go, which
	// also contains execution-engine code that legitimately uses OutputTokens).
	data, err := os.ReadFile("simulator.go")
	if err != nil {
		t.Fatalf("failed to read simulator.go: %v", err)
	}
	content := string(data)
	// Extract EnqueueRequest function body (between "func (sim *Simulator) EnqueueRequest" and the next "func ")
	startIdx := strings.Index(content, "func (sim *Simulator) EnqueueRequest")
	if startIdx < 0 {
		t.Fatal("EnqueueRequest function not found in simulator.go")
	}
	endIdx := strings.Index(content[startIdx+1:], "\nfunc ")
	if endIdx < 0 {
		t.Fatal("could not find end of EnqueueRequest function")
	}
	enqueueBody := content[startIdx : startIdx+1+endIdx]
	if strings.Contains(enqueueBody, "OutputTokens") {
		t.Error("INV-9 violation: EnqueueRequest references OutputTokens — enqueue guard must not access oracle output length")
	}
}

// BC-3, BC-5, BC-6: Simulation terminates with mixed oversized and normal requests
func TestSimulator_OversizedRequests_TerminatesNoLivelock(t *testing.T) {
	// GIVEN a simulator with very small KV cache (50 blocks × 16 tokens = 800 tokens)
	// This is the exact reproduction case from issue #373
	cfg := SimConfig{
		Horizon:             10_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(50, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{6910, 17.67, 2.84}, []float64{0, 0, 0}),
		ModelHardwareConfig: ModelHardwareConfig{Backend: "blackbox"},
	}
	kvStore := MustNewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
	latencyModel, err := MustNewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("MustNewLatencyModel: %v", err)
	}
	sim, err := NewSimulator(cfg, kvStore, latencyModel)
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
		Horizon:             10_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(5, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
		ModelHardwareConfig: ModelHardwareConfig{Backend: "blackbox"},
	}
	kvStore := MustNewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
	latencyModel, err := MustNewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("MustNewLatencyModel: %v", err)
	}
	sim, err := NewSimulator(cfg, kvStore, latencyModel)
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
		Horizon:             math.MaxInt64,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(100, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(1, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 1, 1}, []float64{50, 0.1, 50}),
		ModelHardwareConfig: ModelHardwareConfig{Backend: "blackbox"},
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

func TestStep_ZeroOutputTokens_TTFTBeforeE2E(t *testing.T) {
	// BC-5: TTFT must be recorded before E2E for zero-output-token requests.
	// This tests the two-pass invariant: executeBatchStep (Phase 2) records TTFT,
	// then processCompletions (Phase 3) records E2E.
	cfg := SimConfig{
		Horizon:             100_000_000,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(100, 10000, 100),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 2}, []float64{500, 1, 1000}),
		ModelHardwareConfig: ModelHardwareConfig{Backend: "blackbox"},
	}
	sim := mustNewSimulator(t, cfg)

	// Create a request with zero output tokens
	req := &Request{
		ID:           "zero-output",
		ArrivalTime:  0,
		InputTokens:  make([]int, 10),
		OutputTokens: []int{}, // zero output tokens
		State:        StateQueued,
	}
	sim.InjectArrival(req)

	// Run until completion
	sim.Run()

	// TTFT must be recorded
	ttft, hasTTFT := sim.Metrics.RequestTTFTs[req.ID]
	if !hasTTFT {
		t.Fatal("TTFT must be recorded for zero-output request")
	}
	if ttft <= 0 {
		t.Errorf("TTFT must be positive, got %f", ttft)
	}

	// E2E must be recorded
	e2e, hasE2E := sim.Metrics.RequestE2Es[req.ID]
	if !hasE2E {
		t.Fatal("E2E must be recorded for zero-output request")
	}
	if e2e <= 0 {
		t.Errorf("E2E must be positive, got %f", e2e)
	}

	// BC-5 ordering: TTFT must be <= E2E (TTFT recorded in Phase 2, E2E in Phase 3)
	if e2e < ttft {
		t.Errorf("BC-5 violated: E2E (%f) < TTFT (%f) — two-pass ordering broken", e2e, ttft)
	}

	// Request must have completed
	if sim.Metrics.CompletedRequests != 1 {
		t.Errorf("expected 1 completed request, got %d", sim.Metrics.CompletedRequests)
	}
}

// TestNewSimulator_NonRooflineZeroNumHeads_Succeeds verifies that non-roofline mode
// does not validate model config fields (NumHeads=0 is irrelevant for blackbox mode).
// Moved from model_hardware_config_test.go during PKG-2 extraction.
func TestNewSimulator_NonRooflineZeroNumHeads_Succeeds(t *testing.T) {
	// GIVEN a SimConfig with Roofline=false and NumHeads=0 (irrelevant)
	cfg := SimConfig{
		Horizon:             100000,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1, 2, 3}, []float64{1, 2, 3}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{NumHeads: 0}, HardwareCalib{}, "", "", 0, "blackbox", 0),
	}

	// WHEN NewSimulator is called
	kvStore := MustNewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
	latencyModel, err := MustNewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("MustNewLatencyModel: %v", err)
	}
	sim, err := NewSimulator(cfg, kvStore, latencyModel)

	// THEN it succeeds (roofline validation not applied)
	if err != nil {
		t.Fatalf("unexpected error for non-roofline mode: %v", err)
	}
	if sim == nil {
		t.Error("expected non-nil simulator")
	}
}

// BC-7: Chunked prefill + MaxModelLen — no spurious force-completion.
// GIVEN LongPrefillTokenThreshold=64, MaxModelLen=500, input=200, output=50
// WHEN the request undergoes multi-chunk prefill (4 chunks: 64,64,64,8)
// THEN the request completes normally (peak ProgressIndex=249 < 500),
//
//	LengthCappedRequests==0, TTFT recorded, TotalOutputTokens==49.
func TestSimulator_ChunkedPrefill_MaxModelLen_NoSpuriousCap(t *testing.T) {
	cfg := SimConfig{
		Horizon:             10_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 64), // LongPrefillTokenThreshold=64
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test", "H100", 1, "blackbox", 500),
	}
	sim := mustNewSimulator(t, cfg)

	req := &Request{
		ID:           "chunked_prefill",
		InputTokens:  GenerateRandomTokenIDs(sim.WorkloadRNG(), 200),
		OutputTokens: GenerateRandomTokenIDs(sim.WorkloadRNG(), 50),
		ArrivalTime:  0,
		State:        StateQueued,
	}
	sim.InjectArrival(req)
	sim.Run()

	// Request completes normally (no force-completion)
	if sim.Metrics.CompletedRequests != 1 {
		t.Errorf("CompletedRequests = %d, want 1", sim.Metrics.CompletedRequests)
	}
	if sim.Metrics.LengthCappedRequests != 0 {
		t.Errorf("LengthCappedRequests = %d, want 0 (no spurious force-completion during chunked prefill)", sim.Metrics.LengthCappedRequests)
	}

	// TTFT recorded — verifies TTFT fires on final prefill chunk (ProgressIndex=200), not intermediate chunks
	ttft, hasTTFT := sim.Metrics.RequestTTFTs["chunked_prefill"]
	if !hasTTFT || ttft <= 0 {
		t.Errorf("TTFT not recorded (hasTTFT=%v, ttft=%v); expected positive TTFT after chunked prefill", hasTTFT, ttft)
	}

	// TotalOutputTokens: decode runs PI 201→249 = 49 tokens.
	// Normal completion fires at PI == 200 + max(50,1) - 1 = 249.
	if sim.Metrics.TotalOutputTokens != 49 {
		t.Errorf("TotalOutputTokens = %d, want 49 (decode PI 201→249)", sim.Metrics.TotalOutputTokens)
	}

	// INV-1 conservation
	total := sim.Metrics.CompletedRequests + sim.Metrics.StillQueued + sim.Metrics.StillRunning + sim.Metrics.DroppedUnservable
	if total != 1 {
		t.Errorf("INV-1: completed(%d)+queued(%d)+running(%d)+dropped(%d) = %d, want 1",
			sim.Metrics.CompletedRequests, sim.Metrics.StillQueued, sim.Metrics.StillRunning, sim.Metrics.DroppedUnservable, total)
	}
}

// TestEnqueueRequest_AutoFill_MaxOutputLen verifies the engine-level auto-fill
// that sets MaxOutputLen = maxModelLen - len(InputTokens) when the client doesn't
// provide a budget (BC-1..BC-4, BC-8, BC-10).
func TestEnqueueRequest_AutoFill_MaxOutputLen(t *testing.T) {
	tests := []struct {
		name           string
		maxModelLen    int64
		inputLen       int
		initialMOL     int
		expectedMOL    int
		expectEnqueued bool
		desc           string
	}{
		{
			name:           "BC-1: auto-fill when client omits budget",
			maxModelLen:    1000,
			inputLen:       200,
			initialMOL:     0,
			expectedMOL:    800, // 1000 - 200
			expectEnqueued: true,
			desc:           "MaxOutputLen auto-filled to remaining context window",
		},
		{
			name:           "BC-2: no auto-fill when client sets budget",
			maxModelLen:    1000,
			inputLen:       200,
			initialMOL:     300,
			expectedMOL:    300, // unchanged
			expectEnqueued: true,
			desc:           "client-provided MaxOutputLen preserved",
		},
		{
			name:           "BC-3: no auto-fill in unlimited mode",
			maxModelLen:    0,
			inputLen:       200,
			initialMOL:     0,
			expectedMOL:    0, // unchanged
			expectEnqueued: true,
			desc:           "maxModelLen=0 means unlimited, no auto-fill",
		},
		{
			name:           "BC-4: no auto-fill when input exceeds context",
			maxModelLen:    100,
			inputLen:       150,
			initialMOL:     0,
			expectedMOL:    0, // unchanged (Guard 1 handles rejection)
			expectEnqueued: false,
			desc:           "input >= maxModelLen, Guard 1 drops it",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := SimConfig{
				KVCacheConfig:       NewKVCacheConfig(1000000, 16, 0, 0, 0, 0),
				BatchConfig:         NewBatchConfig(256, 4096, 0),
				LatencyCoeffs:       NewLatencyCoeffs([]float64{0, 0, 0}, []float64{0, 0, 0}),
				ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, "blackbox", tc.maxModelLen),
				Horizon:             1000000,
				Seed:                42,
			}
			s := mustNewSimulator(t, cfg)

			req := &Request{
				ID:           "test-req",
				InputTokens:  make([]int, tc.inputLen),
				OutputTokens: make([]int, 100),
				MaxOutputLen: tc.initialMOL,
				State:        StateQueued,
			}

			s.EnqueueRequest(req)

			if tc.expectEnqueued {
				if s.WaitQ.Len() != 1 {
					t.Fatalf("expected request enqueued, WaitQ.Len()=%d", s.WaitQ.Len())
				}
				if req.MaxOutputLen != tc.expectedMOL {
					t.Errorf("MaxOutputLen = %d, want %d (%s)",
						req.MaxOutputLen, tc.expectedMOL, tc.desc)
				}
			} else {
				if s.WaitQ.Len() != 0 {
					t.Fatalf("expected request dropped, WaitQ.Len()=%d", s.WaitQ.Len())
				}
			}
		})
	}
}

// TestSimulator_ProactiveCap_EliminatesOvershoot verifies BC-1 end-to-end:
// Output is MaxModelLen-1-input (49), not MaxModelLen-input (50).
func TestSimulator_ProactiveCap_EliminatesOvershoot(t *testing.T) {
	cfg := SimConfig{
		Horizon:             10_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{6910, 17.67, 2.84}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test", "H100", 1, "blackbox", 100),
	}
	sim := mustNewSimulator(t, cfg)

	req := &Request{
		ID:           "proactive_test",
		InputTokens:  GenerateRandomTokenIDs(sim.WorkloadRNG(), 50),
		OutputTokens: GenerateRandomTokenIDs(sim.WorkloadRNG(), 200),
		ArrivalTime:  0,
		State:        StateQueued,
	}
	sim.InjectArrival(req)
	sim.Run()

	if sim.Metrics.CompletedRequests != 1 {
		t.Fatalf("CompletedRequests = %d, want 1", sim.Metrics.CompletedRequests)
	}
	// Key: 49 tokens (MaxModelLen-1-input = 100-1-50), not 50 (old behavior)
	if sim.Metrics.TotalOutputTokens != 49 {
		t.Errorf("TotalOutputTokens = %d, want 49 (proactive cap: 100-1-50=49)", sim.Metrics.TotalOutputTokens)
	}
	// BC-5 fires at shifted boundary (PI >= maxModelLen-1)
	if sim.Metrics.LengthCappedRequests != 1 {
		t.Errorf("LengthCappedRequests = %d, want 1", sim.Metrics.LengthCappedRequests)
	}
	// INV-1 conservation
	total := sim.Metrics.CompletedRequests + sim.Metrics.StillQueued + sim.Metrics.StillRunning + sim.Metrics.DroppedUnservable
	if total != 1 {
		t.Errorf("INV-1 violated: %d+%d+%d+%d = %d, want 1",
			sim.Metrics.CompletedRequests, sim.Metrics.StillQueued, sim.Metrics.StillRunning, sim.Metrics.DroppedUnservable, total)
	}
}

// TestSimulator_ProactiveCap_MaxModelLen2_ZeroOutput verifies BC-11:
// With MaxModelLen=2 and input=1, the proactive cap allows 0 decode tokens.
func TestSimulator_ProactiveCap_MaxModelLen2_ZeroOutput(t *testing.T) {
	cfg := SimConfig{
		Horizon:             10_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(100, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test", "H100", 1, "blackbox", 2),
	}
	sim := mustNewSimulator(t, cfg)

	req := &Request{
		ID:           "boundary_2",
		InputTokens:  GenerateRandomTokenIDs(sim.WorkloadRNG(), 1),
		OutputTokens: GenerateRandomTokenIDs(sim.WorkloadRNG(), 10),
		ArrivalTime:  0,
		State:        StateQueued,
	}
	sim.InjectArrival(req)
	sim.Run()

	// MaxModelLen=2, input=1: proactive cap allows max(0, 2-1-1)=0 decode tokens
	if sim.Metrics.TotalOutputTokens != 0 {
		t.Errorf("TotalOutputTokens = %d, want 0 (MaxModelLen=2, input=1)", sim.Metrics.TotalOutputTokens)
	}
	if sim.Metrics.LengthCappedRequests != 1 {
		t.Errorf("LengthCappedRequests = %d, want 1", sim.Metrics.LengthCappedRequests)
	}
	// INV-1 conservation
	total := sim.Metrics.CompletedRequests + sim.Metrics.StillQueued + sim.Metrics.StillRunning + sim.Metrics.DroppedUnservable
	if total != 1 {
		t.Errorf("INV-1 violated: %d", total)
	}
}

// TestProcessCompletions_LengthCapped_MetricsRefreshed verifies BC-7:
// After force-completion, Metrics.Requests has LengthCapped=true.
func TestProcessCompletions_LengthCapped_MetricsRefreshed(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, "blackbox", 100),
	}
	sim := mustNewSimulator(t, cfg)

	req := &Request{
		ID:            "capped",
		InputTokens:   make([]int, 50),
		OutputTokens:  make([]int, 200),
		State:         StateRunning,
		ProgressIndex: 99, // >= maxModelLen-1 (100-1=99) → BC-5 fires
		ArrivalTime:   0,
		ITL:           []int64{5000, 5000, 5000},
	}
	sim.Metrics.Requests[req.ID] = NewRequestMetrics(req, 0)
	sim.RunningBatch = &Batch{Requests: []*Request{req}}

	sim.processCompletions(1000, 5000)

	if !req.LengthCapped {
		t.Error("req.LengthCapped = false, want true")
	}
	rm := sim.Metrics.Requests[req.ID]
	if !rm.LengthCapped {
		t.Error("Metrics.Requests[capped].LengthCapped = false, want true")
	}
}

// TestRecordRequestCompletion_LengthCapped_ITL verifies BC-8:
// Length-capped request uses len(req.ITL)-1 as ITL denominator.
func TestRecordRequestCompletion_LengthCapped_ITL(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, "blackbox", 100),
	}
	sim := mustNewSimulator(t, cfg)

	// 200 pre-determined output tokens, but only 3 actual decode steps (ITL entries)
	req := &Request{
		ID:             "capped_itl",
		InputTokens:    make([]int, 50),
		OutputTokens:   make([]int, 200),
		State:          StateCompleted,
		LengthCapped:   true,
		ArrivalTime:    0,
		FirstTokenTime: 10000,
		ITL:            []int64{5000, 5000, 5000}, // 3 intervals, sum=15000
	}
	sim.Metrics.Requests[req.ID] = NewRequestMetrics(req, 0)

	sim.recordRequestCompletion(req)

	// denominator = max(len(ITL)-1, 1) = max(3-1, 1) = 2 → avgITL = 15000/2 = 7500
	gotITL := sim.Metrics.RequestITLs[req.ID]
	if gotITL != 7500.0 {
		t.Errorf("RequestITLs = %f, want 7500 (denom=max(len(ITL)-1,1)=2, not len(OutputTokens)-1=199)", gotITL)
	}
}

// TestRecordRequestCompletion_NormalRequest_ITL verifies BC-9:
// Non-capped request ITL denominator unchanged.
func TestRecordRequestCompletion_NormalRequest_ITL(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "", "", 0, "blackbox", 0),
	}
	sim := mustNewSimulator(t, cfg)

	req := &Request{
		ID:             "normal",
		InputTokens:    make([]int, 50),
		OutputTokens:   make([]int, 10),
		State:          StateCompleted,
		ArrivalTime:    0,
		FirstTokenTime: 10000,
		ITL:            []int64{5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000}, // 9 intervals
	}
	sim.Metrics.Requests[req.ID] = NewRequestMetrics(req, 0)

	sim.recordRequestCompletion(req)

	// denominator = max(len(OutputTokens)-1, 1) = 9
	gotITL := sim.Metrics.RequestITLs[req.ID]
	if gotITL != 5000.0 {
		t.Errorf("RequestITLs = %f, want 5000", gotITL)
	}
}

// TestSimulator_Conservation_FiveTermWithTimeout verifies BC-4:
// GIVEN requests with short deadlines mixed with normal requests AND dropped requests
// WHEN simulation completes
// THEN completed + still_queued + still_running + dropped_unservable + timed_out == injected
func TestSimulator_Conservation_FiveTermWithTimeout(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(5, 16, 0, 0, 0, 0), // tiny KV: 5 blocks = 80 tokens capacity
		BatchConfig:         NewBatchConfig(1, 2048, 0),            // batch size 1 forces queuing
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test", "H100", 1, "blackbox", 0),
	}
	sim := mustNewSimulator(t, cfg)

	// Track injection count independently (Metrics.Requests deletes dropped entries)
	injectedCount := 0

	// 3 normal requests (small, will complete)
	for i := 0; i < 3; i++ {
		sim.InjectArrival(&Request{
			ID: fmt.Sprintf("normal_%d", i), ArrivalTime: int64(i * 5000),
			InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
			MaxOutputLen: 5, State: StateQueued,
		})
		injectedCount++
	}

	// 3 timeout requests (will timeout while queued due to batch size 1)
	for i := 0; i < 3; i++ {
		sim.InjectArrival(&Request{
			ID: fmt.Sprintf("timeout_%d", i), ArrivalTime: int64(i * 5000),
			InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
			MaxOutputLen: 5, State: StateQueued,
			Deadline: int64(i*5000 + 3000), // short deadline
		})
		injectedCount++
	}

	// 2 dropped requests (input too large for KV: 100 tokens > 80 token capacity)
	for i := 0; i < 2; i++ {
		sim.InjectArrival(&Request{
			ID: fmt.Sprintf("dropped_%d", i), ArrivalTime: int64(i * 5000),
			InputTokens: make([]int, 100), OutputTokens: make([]int, 5), // 100 tokens > 5 blocks * 16 = 80
			MaxOutputLen: 5, State: StateQueued,
		})
		injectedCount++
	}

	sim.Run()

	completed := sim.Metrics.CompletedRequests
	queued := sim.WaitQ.Len()
	running := 0
	if sim.RunningBatch != nil {
		running = len(sim.RunningBatch.Requests)
	}
	dropped := sim.Metrics.DroppedUnservable
	timedOut := sim.Metrics.TimedOutRequests

	fiveTermSum := completed + queued + running + dropped + timedOut
	if fiveTermSum != injectedCount {
		t.Errorf("BC-4 INV-1 5-term conservation violated: completed(%d) + queued(%d) + running(%d) + dropped(%d) + timedOut(%d) = %d, injected = %d",
			completed, queued, running, dropped, timedOut, fiveTermSum, injectedCount)
	}

	// All three terms must be exercised
	if timedOut == 0 {
		t.Error("expected at least 1 timed-out request")
	}
	if completed == 0 {
		t.Error("expected at least 1 completed request")
	}
	if dropped == 0 {
		t.Error("expected at least 1 dropped request (input > KV capacity)")
	}
}

// TestSimulator_Timeout_KVConservation verifies BC-13:
// GIVEN a running request with allocated KV blocks
// WHEN it times out
// THEN allocated + free = total (INV-4)
func TestSimulator_Timeout_KVConservation(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(100, 16, 0, 0, 0, 0), // small KV for observability
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{5000, 10, 5}, []float64{0, 0, 0}), // slower steps so timeout hits mid-execution
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test", "H100", 1, "blackbox", 0),
	}
	sim := mustNewSimulator(t, cfg)

	// Request with short deadline — will start running but timeout before completing
	sim.InjectArrival(&Request{
		ID: "timeout_kv", ArrivalTime: 0,
		InputTokens: make([]int, 50), OutputTokens: make([]int, 100),
		MaxOutputLen: 100, State: StateQueued,
		Deadline: 20000, // timeout at 20ms — step time is 5ms per step, not enough for 100 output tokens
	})

	sim.Run()

	// After simulation: all KV blocks must be freed (INV-4)
	usedBlocks := sim.KVCache.UsedBlocks()
	totalBlocks := sim.KVCache.TotalCapacity()
	freeBlocks := totalBlocks - usedBlocks
	if usedBlocks != 0 {
		t.Errorf("BC-13 INV-4: after timeout, used blocks = %d, want 0 (all freed)", usedBlocks)
	}
	if freeBlocks != totalBlocks {
		t.Errorf("BC-13 INV-4: free(%d) + used(%d) != total(%d)", freeBlocks, usedBlocks, totalBlocks)
	}
	if sim.Metrics.TimedOutRequests != 1 {
		t.Errorf("expected 1 timed-out request, got %d", sim.Metrics.TimedOutRequests)
	}
}

// TestEnqueueDecodeSubRequest_StepEventAtClusterTime verifies that when a decode
// sub-request is injected with a clusterTime ahead of the instance's internal clock,
// the StepEvent is scheduled at clusterTime rather than the stale instance time.
// GIVEN an idle simulator at internal clock = 0
// WHEN  EnqueueDecodeSubRequest is called with clusterTime = 50000 (ahead of internal clock)
// THEN  the earliest pending event is at clusterTime, preventing time-travel processing.
func TestEnqueueDecodeSubRequest_StepEventAtClusterTime(t *testing.T) {
	cfg := SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-clustertime", "H100", 1, "blackbox", 0),
	}
	s := mustNewSimulator(t, cfg)
	// Internal clock is 0 (idle simulator, no events processed).

	const clusterTime = int64(50000)
	req := &Request{
		ID:            "dec-1",
		InputTokens:   make([]int, 10),
		OutputTokens:  make([]int, 5),
		State:         StateQueued,
		ArrivalTime:   clusterTime,
		ProgressIndex: 10, // KV pre-allocated past input (as done by AllocateTransferredKV)
	}

	s.EnqueueDecodeSubRequest(req, clusterTime)

	// BC-1: A StepEvent must be scheduled (INV-8 work-conserving).
	if !s.HasPendingEvents() {
		t.Fatal("BC-1: no pending events after EnqueueDecodeSubRequest (INV-8 violated)")
	}

	// BC-2: The StepEvent must be at clusterTime, not at the stale internal clock (0).
	// Without the max(sim.Clock, clusterTime) fix, this would return 0, causing
	// the decode sub-request to be processed before its arrival time.
	got := s.PeekNextEventTime()
	if got != clusterTime {
		t.Errorf("BC-2: StepEvent time = %d, want %d (clusterTime); "+
			"decode sub-request would be processed at stale instance time %d",
			got, clusterTime, s.Clock)
	}
}

// TestEnqueueDecodeSubRequest_SimClockAhead_StepEventAtSimClock verifies the
// other branch of max(sim.Clock, clusterTime): when the instance has already
// advanced past the cluster time, the StepEvent is scheduled at sim.Clock.
// GIVEN a simulator that has processed events up to clock T_sim
// WHEN  EnqueueDecodeSubRequest is called with clusterTime < T_sim
// THEN  the StepEvent is at T_sim (not at clusterTime), preserving INV-3.
func TestEnqueueDecodeSubRequest_SimClockAhead_StepEventAtSimClock(t *testing.T) {
	cfg := SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test-simclockahead", "H100", 1, "blackbox", 0),
	}
	s := mustNewSimulator(t, cfg)

	// Advance the simulator's internal clock by processing a real request.
	s.InjectArrival(&Request{
		ID: "req-advance", ArrivalTime: 0,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
		State: StateQueued,
	})
	s.Run()
	advancedClock := s.Clock
	if advancedClock <= 0 {
		t.Fatalf("simulator clock did not advance after Run: clock = %d", advancedClock)
	}

	// Create a second simulator at the advanced clock state by injecting into a fresh sim
	// that has manually processed events to advance the clock.
	// Simpler approach: inject directly into the internal queue at an advanced time.
	cfg2 := cfg
	s2 := mustNewSimulator(t, cfg2)
	// Advance s2's clock by processing an arrival event.
	s2.InjectArrival(&Request{
		ID: "req-prime", ArrivalTime: 0,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
		State: StateQueued,
	})
	for s2.HasPendingEvents() {
		s2.ProcessNextEvent()
	}
	simClock := s2.Clock
	if simClock <= 0 {
		t.Fatalf("s2 clock did not advance: %d", simClock)
	}

	// clusterTime is behind the instance's current clock.
	clusterTime := simClock / 2
	if clusterTime <= 0 {
		t.Skip("clock did not advance enough to create clusterTime < simClock; skipping")
	}

	req := &Request{
		ID:            "dec-behind",
		InputTokens:   make([]int, 10),
		OutputTokens:  make([]int, 5),
		State:         StateQueued,
		ArrivalTime:   clusterTime,
		ProgressIndex: 10,
	}
	s2.EnqueueDecodeSubRequest(req, clusterTime)

	if !s2.HasPendingEvents() {
		t.Fatal("BC-1: no pending events after EnqueueDecodeSubRequest (INV-8 violated)")
	}

	// BC-2: The StepEvent must be at simClock (not clusterTime < simClock),
	// preserving clock monotonicity (INV-3).
	got := s2.PeekNextEventTime()
	if got != simClock {
		t.Errorf("BC-2: StepEvent time = %d, want %d (simClock); "+
			"using clusterTime (%d) would violate clock monotonicity INV-3",
			got, simClock, clusterTime)
	}
}
