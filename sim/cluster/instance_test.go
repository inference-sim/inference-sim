package cluster

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// newTestSimConfigWithWorkload creates a SimConfig for instance tests with workload.
func newTestSimConfigWithWorkload(guideLLMConfig *sim.GuideLLMConfig) sim.SimConfig {
	return sim.SimConfig{
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
		GuideLLMConfig:     guideLLMConfig,
	}
}

// newTestInstanceSimConfig creates a SimConfig without workload for instance tests.
func newTestInstanceSimConfig() sim.SimConfig {
	return sim.SimConfig{
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
	}
}

// smallWorkload returns a small GuideLLMConfig for tests.
func smallWorkload(maxPrompts int) *sim.GuideLLMConfig {
	return &sim.GuideLLMConfig{
		Rate: 10.0 / 1e6, MaxPrompts: maxPrompts,
		PromptTokens: 100, PromptTokensStdDev: 10, PromptTokensMin: 10, PromptTokensMax: 200,
		OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100,
	}
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
					GuideLLMConfig: &sim.GuideLLMConfig{
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
				},
			)

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

// TestInstanceSimulator_Determinism verifies same seed produces identical results.
func TestInstanceSimulator_Determinism(t *testing.T) {
	cfg := newTestSimConfigWithWorkload(&sim.GuideLLMConfig{
		Rate: 10.0 / 1e6, MaxPrompts: 50,
		PromptTokens: 100, PromptTokensStdDev: 20, PromptTokensMin: 10, PromptTokensMax: 200,
		OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100,
	})

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
	cfg := newTestSimConfigWithWorkload(smallWorkload(5))
	cfg.Horizon = 1000000
	cfg.TotalKVBlocks = 1000
	instance := NewInstanceSimulator(InstanceID("replica-0"), cfg)

	if instance.ID() != InstanceID("replica-0") {
		t.Errorf("ID() = %q, want %q", instance.ID(), "replica-0")
	}
}

func TestInstanceSimulator_Clock_AdvancesWithSimulation(t *testing.T) {
	instance := NewInstanceSimulator(InstanceID("test"), newTestSimConfigWithWorkload(smallWorkload(10)))

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
	instance := NewInstanceSimulator(InstanceID("test"), newTestSimConfigWithWorkload(smallWorkload(10)))

	instance.Run()

	if instance.Metrics() == nil {
		t.Fatal("Metrics() returned nil")
	}
	if instance.Metrics().CompletedRequests <= 0 {
		t.Errorf("Metrics().CompletedRequests = %d, want > 0", instance.Metrics().CompletedRequests)
	}
}

func TestInstanceSimulator_Horizon_ReturnsConstructorValue(t *testing.T) {
	cfg := newTestSimConfigWithWorkload(smallWorkload(5))
	cfg.Horizon = 5000000
	cfg.TotalKVBlocks = 1000
	instance := NewInstanceSimulator(InstanceID("test"), cfg)

	if instance.Horizon() != 5000000 {
		t.Errorf("Horizon() = %d, want %d", instance.Horizon(), 5000000)
	}
}

// === Edge Case Tests ===

func TestInstanceSimulator_EmptyID_Valid(t *testing.T) {
	cfg := newTestSimConfigWithWorkload(smallWorkload(5))
	cfg.Horizon = 1000000
	cfg.TotalKVBlocks = 1000
	instance := NewInstanceSimulator(InstanceID(""), cfg)

	if instance.ID() != InstanceID("") {
		t.Errorf("ID() = %q, want empty string", instance.ID())
	}
}

func TestInstanceSimulator_ZeroRequests(t *testing.T) {
	cfg := newTestSimConfigWithWorkload(smallWorkload(0))
	cfg.Horizon = 1000000
	cfg.TotalKVBlocks = 1000
	instance := NewInstanceSimulator(InstanceID("test"), cfg)

	instance.Run()

	if instance.Metrics().CompletedRequests != 0 {
		t.Errorf("CompletedRequests = %d, want 0 for MaxPrompts=0", instance.Metrics().CompletedRequests)
	}
}

func TestInstanceSimulator_RunOnce_PanicsOnSecondCall(t *testing.T) {
	cfg := newTestSimConfigWithWorkload(smallWorkload(5))
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

// TestInstanceSimulator_ProcessNextEvent_ReturnsCorrectEventType verifies:
// GIVEN a request injected via InjectRequestOnline
// WHEN ProcessNextEvent is called
// THEN the returned event is an *sim.ArrivalEvent (the first event for an injected request)
func TestInstanceSimulator_ProcessNextEvent_ReturnsCorrectEventType(t *testing.T) {
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

	// First event for an injected request is an ArrivalEvent
	if _, ok := ev.(*sim.ArrivalEvent); !ok {
		t.Errorf("ProcessNextEvent returned %T, want *sim.ArrivalEvent", ev)
	}

	// After ArrivalEvent executes, a QueuedEvent should be scheduled
	if !inst.HasPendingEvents() {
		t.Fatal("expected QueuedEvent after ArrivalEvent, but no pending events")
	}

	ev2 := inst.ProcessNextEvent()
	if _, ok := ev2.(*sim.QueuedEvent); !ok {
		t.Errorf("second ProcessNextEvent returned %T, want *sim.QueuedEvent", ev2)
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
