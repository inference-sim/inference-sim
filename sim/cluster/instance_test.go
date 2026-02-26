package cluster

import (
	"fmt"
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// newTestSimConfig creates a SimConfig without workload for instance tests.
// All workload generation now happens externally — requests are passed via InjectRequest.
func newTestSimConfig() sim.SimConfig {
	return sim.SimConfig{
		Horizon:             math.MaxInt64,
		Seed:                42,
		KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test", "H100", 1, false),
	}
}

// newTestInstanceSimConfig creates a SimConfig without workload for instance tests.
func newTestInstanceSimConfig() sim.SimConfig {
	return newTestSimConfig()
}

// === Equivalence Tests (Critical for BC-1) ===

// TestInstanceSimulator_GoldenDataset_Equivalence verifies:
// GIVEN golden dataset test cases
// WHEN run through InstanceSimulator wrapper
// THEN all metrics match golden expected values exactly
func TestInstanceSimulator_GoldenDataset_Equivalence(t *testing.T) {
	dataset := testutil.LoadGoldenDataset(t)

	if len(dataset.Tests) == 0 {
		t.Fatal("Golden dataset contains no test cases")
	}

	for _, tc := range dataset.Tests {
		t.Run(tc.Model, func(t *testing.T) {
			instance := NewInstanceSimulator(
				InstanceID("test-instance"),
				sim.SimConfig{
					Horizon:             math.MaxInt64,
					Seed:                tc.Seed,
					KVCacheConfig:       sim.NewKVCacheConfig(tc.TotalKVBlocks, tc.BlockSizeInTokens, 0, 0, 0, 0),
					BatchConfig:         sim.NewBatchConfig(tc.MaxNumRunningReqs, tc.MaxNumScheduledTokens, tc.LongPrefillTokenThreshold),
					LatencyCoeffs:       sim.NewLatencyCoeffs(tc.BetaCoeffs, tc.AlphaCoeffs),
					ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, tc.Model, tc.Hardware, tc.TP, false),
				},
			)

			requests := testGenerateRequests(tc.Seed, math.MaxInt64, tc.Rate/1e6,
				tc.NumRequests, tc.PrefixTokens,
				tc.PromptTokens, tc.PromptTokensStdev, tc.PromptTokensMin, tc.PromptTokensMax,
				tc.OutputTokens, tc.OutputTokensStdev, tc.OutputTokensMin, tc.OutputTokensMax)
			for _, req := range requests {
				instance.InjectRequest(req)
			}

			instance.Run()

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

			const relTol = 1e-9
			vllmRuntime := float64(instance.Metrics().SimEndedTime) / 1e6
			testutil.AssertFloat64Equal(t, "vllm_estimated_duration_s", tc.Metrics.VllmEstimatedDurationS, vllmRuntime, relTol)
		})
	}
}

// TestInstanceSimulator_GoldenDataset_Invariants verifies system invariants alongside
// the golden dataset test (R7). Golden tests answer "did the output change?" but not
// "is the output correct?" These invariant tests verify laws from the specification.
func TestInstanceSimulator_GoldenDataset_Invariants(t *testing.T) {
	dataset := testutil.LoadGoldenDataset(t)

	if len(dataset.Tests) == 0 {
		t.Fatal("Golden dataset contains no test cases")
	}

	for _, tc := range dataset.Tests {
		t.Run(tc.Model, func(t *testing.T) {
			instance := NewInstanceSimulator(
				InstanceID("test-instance"),
				sim.SimConfig{
					Horizon:             math.MaxInt64,
					Seed:                tc.Seed,
					KVCacheConfig:       sim.NewKVCacheConfig(tc.TotalKVBlocks, tc.BlockSizeInTokens, 0, 0, 0, 0),
					BatchConfig:         sim.NewBatchConfig(tc.MaxNumRunningReqs, tc.MaxNumScheduledTokens, tc.LongPrefillTokenThreshold),
					LatencyCoeffs:       sim.NewLatencyCoeffs(tc.BetaCoeffs, tc.AlphaCoeffs),
					ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, tc.Model, tc.Hardware, tc.TP, false),
				},
			)

			requests := testGenerateRequests(tc.Seed, math.MaxInt64, tc.Rate/1e6,
				tc.NumRequests, tc.PrefixTokens,
				tc.PromptTokens, tc.PromptTokensStdev, tc.PromptTokensMin, tc.PromptTokensMax,
				tc.OutputTokens, tc.OutputTokensStdev, tc.OutputTokensMin, tc.OutputTokensMax)
			for _, req := range requests {
				instance.InjectRequest(req)
			}

			instance.Run()
			m := instance.Metrics()

			// INV-1: Request conservation
			// completed + still_queued + still_running + dropped == injected
			total := m.CompletedRequests + m.StillQueued + m.StillRunning + m.DroppedUnservable
			if total != tc.NumRequests {
				t.Errorf("INV-1 request conservation: completed(%d) + queued(%d) + running(%d) + dropped(%d) = %d, want %d (NumRequests)",
					m.CompletedRequests, m.StillQueued, m.StillRunning, m.DroppedUnservable, total, tc.NumRequests)
			}

			// INV-5: Causality — for every completed request, TTFT >= 0 and E2E >= TTFT
			for reqID, e2e := range m.RequestE2Es {
				ttft, hasTTFT := m.RequestTTFTs[reqID]
				if !hasTTFT {
					t.Errorf("INV-5 causality: request %q has E2E but no TTFT", reqID)
					continue
				}
				if ttft < 0 {
					t.Errorf("INV-5 causality: request %q TTFT = %f, want >= 0", reqID, ttft)
				}
				if e2e < ttft {
					t.Errorf("INV-5 causality: request %q E2E (%f) < TTFT (%f)", reqID, e2e, ttft)
				}
			}

			// INV-3: Clock monotonicity (SimEndedTime > 0 for non-empty workloads)
			if tc.NumRequests > 0 && m.SimEndedTime <= 0 {
				t.Errorf("INV-3 clock monotonicity: SimEndedTime = %d, want > 0 for %d requests",
					m.SimEndedTime, tc.NumRequests)
			}
		})
	}
}

// TestInstanceSimulator_Determinism verifies same seed produces identical results.
func TestInstanceSimulator_Determinism(t *testing.T) {
	cfg := newTestSimConfig()

	instance1 := NewInstanceSimulator(InstanceID("run1"), cfg)
	instance2 := NewInstanceSimulator(InstanceID("run2"), cfg)

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

func TestInstanceSimulator_ID_ReturnsConstructorValue(t *testing.T) {
	cfg := newTestSimConfig()
	cfg.Horizon = 1000000
	cfg.TotalKVBlocks = 1000
	instance := NewInstanceSimulator(InstanceID("replica-0"), cfg)

	if instance.ID() != InstanceID("replica-0") {
		t.Errorf("ID() = %q, want %q", instance.ID(), "replica-0")
	}
}

func TestInstanceSimulator_Clock_AdvancesWithSimulation(t *testing.T) {
	cfg := newTestSimConfig()
	cfg.Horizon = 10_000_000
	instance := NewInstanceSimulator(InstanceID("test"), cfg)

	// Inject requests before Run (workload is now external)
	for i, req := range newTestRequests(5) {
		req.ID = fmt.Sprintf("request_%d", i)
		instance.InjectRequest(req)
	}

	instance.Run()

	if instance.Clock() <= 0 {
		t.Errorf("Clock() = %d, want > 0 after Run()", instance.Clock())
	}
	if instance.Clock() != instance.Metrics().SimEndedTime {
		t.Errorf("Clock() = %d, Metrics().SimEndedTime = %d, want equal",
			instance.Clock(), instance.Metrics().SimEndedTime)
	}
}

func TestInstanceSimulator_Metrics_DelegatesCorrectly(t *testing.T) {
	cfg := newTestSimConfig()
	cfg.Horizon = 10_000_000
	instance := NewInstanceSimulator(InstanceID("test"), cfg)

	// Inject requests before Run (workload is now external)
	for i, req := range newTestRequests(5) {
		req.ID = fmt.Sprintf("request_%d", i)
		instance.InjectRequest(req)
	}

	instance.Run()

	if instance.Metrics() == nil {
		t.Fatal("Metrics() returned nil")
	}
	if instance.Metrics().CompletedRequests <= 0 {
		t.Errorf("Metrics().CompletedRequests = %d, want > 0", instance.Metrics().CompletedRequests)
	}
}

func TestInstanceSimulator_Horizon_ReturnsConstructorValue(t *testing.T) {
	cfg := newTestSimConfig()
	cfg.Horizon = 5000000
	cfg.TotalKVBlocks = 1000
	instance := NewInstanceSimulator(InstanceID("test"), cfg)

	if instance.Horizon() != 5000000 {
		t.Errorf("Horizon() = %d, want %d", instance.Horizon(), 5000000)
	}
}

// === Edge Case Tests ===

func TestInstanceSimulator_EmptyID_Valid(t *testing.T) {
	cfg := newTestSimConfig()
	cfg.Horizon = 1000000
	cfg.TotalKVBlocks = 1000
	instance := NewInstanceSimulator(InstanceID(""), cfg)

	if instance.ID() != InstanceID("") {
		t.Errorf("ID() = %q, want empty string", instance.ID())
	}
}

func TestInstanceSimulator_ZeroRequests(t *testing.T) {
	cfg := newTestSimConfig()
	cfg.Horizon = 1000000
	cfg.TotalKVBlocks = 1000
	instance := NewInstanceSimulator(InstanceID("test"), cfg)

	instance.Run()

	if instance.Metrics().CompletedRequests != 0 {
		t.Errorf("CompletedRequests = %d, want 0 for NumRequests=0", instance.Metrics().CompletedRequests)
	}
}

func TestInstanceSimulator_RunOnce_PanicsOnSecondCall(t *testing.T) {
	cfg := newTestSimConfig()
	cfg.Horizon = 1000000
	cfg.TotalKVBlocks = 1000
	instance := NewInstanceSimulator(InstanceID("test"), cfg)

	instance.Run()

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("Expected panic on second Run() call, but got none")
		}
		expected := "InstanceSimulator.Run() called more than once"
		if r != expected {
			t.Errorf("Panic message = %q, want %q", r, expected)
		}
	}()

	instance.Run()
}

// TestInstanceSimulator_ObservationMethods verifies QueueDepth, BatchSize, KVUtilization, FreeKVBlocks.
func TestInstanceSimulator_ObservationMethods(t *testing.T) {
	tests := []struct {
		name           string
		totalKVBlocks  int64
		wantQueueDepth int
		wantBatchSize  int
		wantFreeKV     int64
		wantKVUtilZero bool
	}{
		{
			name:           "fresh instance with no requests",
			totalKVBlocks:  100,
			wantQueueDepth: 0,
			wantBatchSize:  0,
			wantFreeKV:     100,
			wantKVUtilZero: true,
		},
		{
			name:           "different KV cache size",
			totalKVBlocks:  500,
			wantQueueDepth: 0,
			wantBatchSize:  0,
			wantFreeKV:     500,
			wantKVUtilZero: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := newTestInstanceSimConfig()
			cfg.Horizon = 1000000
			cfg.TotalKVBlocks = tc.totalKVBlocks
			inst := NewInstanceSimulator("obs-test", cfg)

			if got := inst.QueueDepth(); got != tc.wantQueueDepth {
				t.Errorf("QueueDepth() = %d, want %d", got, tc.wantQueueDepth)
			}
			if got := inst.BatchSize(); got != tc.wantBatchSize {
				t.Errorf("BatchSize() = %d, want %d", got, tc.wantBatchSize)
			}
			if got := inst.FreeKVBlocks(); got != tc.wantFreeKV {
				t.Errorf("FreeKVBlocks() = %d, want %d", got, tc.wantFreeKV)
			}
			kvUtil := inst.KVUtilization()
			if tc.wantKVUtilZero && kvUtil != 0 {
				t.Errorf("KVUtilization() = %f, want 0", kvUtil)
			}
		})
	}
}

func TestInstanceSimulator_BatchSize_NilRunningBatch(t *testing.T) {
	cfg := newTestInstanceSimConfig()
	cfg.Horizon = 1000000
	cfg.TotalKVBlocks = 100
	inst := NewInstanceSimulator("nil-batch", cfg)

	req := &sim.Request{
		ID:           "req_0",
		ArrivalTime:  0,
		InputTokens:  make([]int, 16),
		OutputTokens: make([]int, 1),
		State:        sim.StateQueued,
	}
	inst.InjectRequest(req)
	inst.hasRun = true
	inst.sim.Run()

	got := inst.BatchSize()
	if got != 0 {
		t.Errorf("BatchSize() after completed sim = %d, want 0", got)
	}
}

// TestInstanceSimulator_ProcessNextEvent_ReturnsEventsWithMonotonicTimestamps verifies:
// GIVEN a request injected via InjectRequestOnline
// WHEN ProcessNextEvent is called multiple times
// THEN events are returned with non-decreasing timestamps (clock monotonicity)
func TestInstanceSimulator_ProcessNextEvent_ReturnsEventsWithMonotonicTimestamps(t *testing.T) {
	cfg := newTestInstanceSimConfig()
	cfg.Horizon = 1000000
	cfg.TotalKVBlocks = 100
	inst := NewInstanceSimulator("return-test", cfg)
	inst.hasRun = true

	req := &sim.Request{
		ID:           "req_return",
		ArrivalTime:  100,
		InputTokens:  make([]int, 16),
		OutputTokens: make([]int, 1),
		State:        sim.StateQueued,
	}
	inst.InjectRequestOnline(req, 200)

	if !inst.HasPendingEvents() {
		t.Fatal("expected pending events after injection")
	}

	ev := inst.ProcessNextEvent()
	if ev == nil {
		t.Fatal("ProcessNextEvent returned nil")
	}

	// First event should have a non-negative timestamp
	if ev.Timestamp() < 0 {
		t.Errorf("first event timestamp = %d, want >= 0", ev.Timestamp())
	}

	// After first event executes, more events should be scheduled
	if !inst.HasPendingEvents() {
		t.Fatal("expected more events after first event, but no pending events")
	}

	ev2 := inst.ProcessNextEvent()
	if ev2 == nil {
		t.Fatal("second ProcessNextEvent returned nil")
	}

	// Second event should have timestamp >= first event (clock monotonicity)
	if ev2.Timestamp() < ev.Timestamp() {
		t.Errorf("second event timestamp (%d) < first (%d), violates monotonicity",
			ev2.Timestamp(), ev.Timestamp())
	}
}

func TestInstanceSimulator_InjectRequestOnline(t *testing.T) {
	cfg := newTestInstanceSimConfig()
	cfg.Horizon = 1000000
	cfg.TotalKVBlocks = 100
	inst := NewInstanceSimulator("online-test", cfg)

	inst.hasRun = true

	req := &sim.Request{
		ID:           "req_online",
		ArrivalTime:  100,
		InputTokens:  make([]int, 16),
		OutputTokens: make([]int, 1),
		State:        sim.StateQueued,
	}

	inst.InjectRequestOnline(req, 200)

	if !inst.HasPendingEvents() {
		t.Error("expected pending events after InjectRequestOnline, got none")
	}
}
