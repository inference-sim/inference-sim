package cluster

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

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
			testutil.AssertFloat64Equal(t, "vllm_estimated_duration_s", tc.Metrics.VllmEstimatedDurationS, vllmRuntime, relTol)
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

// TestInstanceSimulator_RunOnce_PanicsOnSecondCall verifies run-once semantics:
// GIVEN an InstanceSimulator that has already Run()
// WHEN Run() is called again
// THEN it panics with "InstanceSimulator.Run() called more than once"
func TestInstanceSimulator_RunOnce_PanicsOnSecondCall(t *testing.T) {
	instance := NewInstanceSimulator(
		InstanceID("test"),
		1000000, 42, 1000, 16, 256, 2048, 0,
		[]float64{1000, 10, 5}, []float64{100, 1, 100},
		&sim.GuideLLMConfig{Rate: 10.0 / 1e6, MaxPrompts: 5,
			PromptTokens: 100, PromptTokensStdDev: 10, PromptTokensMin: 10, PromptTokensMax: 200,
			OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100},
		sim.ModelConfig{}, sim.HardwareCalib{},
		"test", "H100", 1, false, "",
	)

	// First run should succeed
	instance.Run()

	// Second run should panic
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

	instance.Run() // Should panic
}

// TestInstanceSimulator_ObservationMethods verifies BC-8:
// GIVEN an InstanceSimulator with known state
// WHEN QueueDepth, BatchSize, KVUtilization, FreeKVBlocks are called
// THEN they return correct values delegating to the wrapped Simulator
func TestInstanceSimulator_ObservationMethods(t *testing.T) {
	tests := []struct {
		name            string
		totalKVBlocks   int64
		wantQueueDepth  int
		wantBatchSize   int
		wantFreeKV      int64
		wantKVUtilZero  bool // true if KVUtilization should be 0
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
			inst := NewInstanceSimulatorWithoutWorkload(
				"obs-test", 1000000, 42, tc.totalKVBlocks, 16, 256, 2048, 0,
				[]float64{1000, 10, 5}, []float64{100, 1, 100},
				sim.ModelConfig{}, sim.HardwareCalib{}, "test", "H100", 1, false,
			)

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

// TestInstanceSimulator_BatchSize_NilRunningBatch verifies BatchSize returns 0
// when RunningBatch is nil (after simulation completes with all requests done).
func TestInstanceSimulator_BatchSize_NilRunningBatch(t *testing.T) {
	inst := NewInstanceSimulatorWithoutWorkload(
		"nil-batch", 1000000, 42, 100, 16, 256, 2048, 0,
		[]float64{1000, 10, 5}, []float64{100, 1, 100},
		sim.ModelConfig{}, sim.HardwareCalib{}, "test", "H100", 1, false,
	)

	// Inject a small request, run the simulation to completion
	req := &sim.Request{
		ID:           "req_0",
		ArrivalTime:  0,
		InputTokens:  make([]int, 16),
		OutputTokens: make([]int, 1),
		State:        "queued",
	}
	inst.InjectRequest(req)
	inst.hasRun = true
	inst.sim.Run()

	// After completion, RunningBatch may be nil
	got := inst.BatchSize()
	if got != 0 {
		t.Errorf("BatchSize() after completed sim = %d, want 0", got)
	}
}

// TestInstanceSimulator_InjectRequestOnline verifies InjectRequestOnline
// does NOT panic after hasRun is set (unlike InjectRequest).
func TestInstanceSimulator_InjectRequestOnline(t *testing.T) {
	inst := NewInstanceSimulatorWithoutWorkload(
		"online-test", 1000000, 42, 100, 16, 256, 2048, 0,
		[]float64{1000, 10, 5}, []float64{100, 1, 100},
		sim.ModelConfig{}, sim.HardwareCalib{}, "test", "H100", 1, false,
	)

	// Simulate that the event loop has started
	inst.hasRun = true

	req := &sim.Request{
		ID:           "req_online",
		ArrivalTime:  100,
		InputTokens:  make([]int, 16),
		OutputTokens: make([]int, 1),
		State:        "queued",
	}

	// Should NOT panic (unlike InjectRequest which would)
	inst.InjectRequestOnline(req, 200)

	// Verify the event was injected
	if !inst.HasPendingEvents() {
		t.Error("expected pending events after InjectRequestOnline, got none")
	}
}
