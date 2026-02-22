package cluster

import (
	"bytes"
	"encoding/json"
	"math"
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// newTestDeploymentConfig creates a DeploymentConfig suitable for testing.
func newTestDeploymentConfig(numInstances int) DeploymentConfig {
	return DeploymentConfig{
		NumInstances:              numInstances,
		Horizon:                   math.MaxInt64,
		Seed:                      42,
		TotalKVBlocks:             10000,
		BlockSizeTokens:           16,
		MaxRunningReqs:            256,
		MaxScheduledTokens:        2048,
		LongPrefillTokenThreshold: 0,
		BetaCoeffs:                []float64{1000, 10, 5},
		AlphaCoeffs:               []float64{100, 1, 100},
		ModelConfig:               sim.ModelConfig{},
		HWConfig:                  sim.HardwareCalib{},
		Model:                     "test-model",
		GPU:                       "H100",
		TP:                        1,
		Roofline:                  false,
	}
}

// newTestWorkload creates a GuideLLMConfig suitable for testing.
func newTestWorkload(numRequests int) *sim.GuideLLMConfig {
	return &sim.GuideLLMConfig{
		Rate:               10.0 / 1e6,
		NumRequests:         numRequests,
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
}

// mustRun is a test helper that calls Run and fails the test on error.
func mustRun(t *testing.T, cs *ClusterSimulator) {
	t.Helper()
	if err := cs.Run(); err != nil {
		t.Fatalf("ClusterSimulator.Run: %v", err)
	}
}

// TestPerInstanceMetrics_BeforeRun_Panics verifies run-once guard.
func TestPerInstanceMetrics_BeforeRun_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when calling PerInstanceMetrics before Run")
		}
	}()
	config := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(config, newTestWorkload(5), "")
	cs.PerInstanceMetrics() // should panic
}

// TestDeploymentConfig_ToSimConfig_FieldMapping verifies all fields are correctly mapped
// and workload fields are intentionally omitted.
func TestDeploymentConfig_ToSimConfig_FieldMapping(t *testing.T) {
	dc := DeploymentConfig{
		NumInstances:              3,
		Horizon:                   999,
		Seed:                      7,
		TotalKVBlocks:             500,
		BlockSizeTokens:           32,
		MaxRunningReqs:            128,
		MaxScheduledTokens:        4096,
		LongPrefillTokenThreshold: 512,
		BetaCoeffs:                []float64{1, 2, 3},
		AlphaCoeffs:               []float64{4, 5, 6},
		Model:                     "test-model",
		GPU:                       "H100",
		TP:                        2,
		Roofline:                  true,
		KVTransferBaseLatency:     42,
	}

	sc := dc.ToSimConfig()

	if sc.Horizon != 999 {
		t.Errorf("Horizon: got %d, want 999", sc.Horizon)
	}
	if sc.Seed != 7 {
		t.Errorf("Seed: got %d, want 7", sc.Seed)
	}
	if sc.TotalKVBlocks != 500 {
		t.Errorf("TotalKVBlocks: got %d, want 500", sc.TotalKVBlocks)
	}
	if sc.BlockSizeTokens != 32 {
		t.Errorf("BlockSizeTokens: got %d, want 32", sc.BlockSizeTokens)
	}
	if sc.MaxRunningReqs != 128 {
		t.Errorf("MaxRunningReqs: got %d, want 128", sc.MaxRunningReqs)
	}
	if sc.MaxScheduledTokens != 4096 {
		t.Errorf("MaxScheduledTokens: got %d, want 4096", sc.MaxScheduledTokens)
	}
	if sc.LongPrefillTokenThreshold != 512 {
		t.Errorf("LongPrefillTokenThreshold: got %d, want 512", sc.LongPrefillTokenThreshold)
	}
	if len(sc.BetaCoeffs) != 3 || sc.BetaCoeffs[0] != 1 {
		t.Errorf("BetaCoeffs: got %v, want [1 2 3]", sc.BetaCoeffs)
	}
	if len(sc.AlphaCoeffs) != 3 || sc.AlphaCoeffs[0] != 4 {
		t.Errorf("AlphaCoeffs: got %v, want [4 5 6]", sc.AlphaCoeffs)
	}
	if sc.Model != "test-model" {
		t.Errorf("Model: got %q, want %q", sc.Model, "test-model")
	}
	if sc.GPU != "H100" {
		t.Errorf("GPU: got %q, want %q", sc.GPU, "H100")
	}
	if sc.TP != 2 {
		t.Errorf("TP: got %d, want 2", sc.TP)
	}
	if !sc.Roofline {
		t.Error("Roofline: got false, want true")
	}
	if sc.KVTransferBaseLatency != 42 {
		t.Errorf("KVTransferBaseLatency: got %d, want 42", sc.KVTransferBaseLatency)
	}
	// Workload fields must be intentionally omitted
	if sc.GuideLLMConfig != nil {
		t.Error("GuideLLMConfig should be nil (workload generated centrally)")
	}
	if sc.TracesWorkloadFilePath != "" {
		t.Error("TracesWorkloadFilePath should be empty (workload generated centrally)")
	}
}

func TestDeploymentConfig_ToSimConfig_PrioritySchedulerFields(t *testing.T) {
	dc := DeploymentConfig{
		NumInstances:       1,
		Horizon:            999,
		Seed:               7,
		TotalKVBlocks:      500,
		BlockSizeTokens:    32,
		MaxRunningReqs:     128,
		MaxScheduledTokens: 4096,
		BetaCoeffs:         []float64{1, 2, 3},
		AlphaCoeffs:        []float64{4, 5, 6},
		PriorityPolicy:     "slo-based",
		Scheduler:          "priority-fcfs",
	}

	sc := dc.ToSimConfig()

	if sc.PriorityPolicy != "slo-based" {
		t.Errorf("PriorityPolicy: got %q, want %q", sc.PriorityPolicy, "slo-based")
	}
	if sc.Scheduler != "priority-fcfs" {
		t.Errorf("Scheduler: got %q, want %q", sc.Scheduler, "priority-fcfs")
	}
}

// TestClusterSimulator_SingleInstance_GoldenEquivalence verifies BC-7, BC-9:
// GIVEN each golden dataset test case configured as NumInstances=1 via ClusterSimulator
// WHEN Run() called
// THEN CompletedRequests, TotalInputTokens, TotalOutputTokens match golden values exactly.
func TestClusterSimulator_SingleInstance_GoldenEquivalence(t *testing.T) {
	dataset := testutil.LoadGoldenDataset(t)

	if len(dataset.Tests) == 0 {
		t.Fatal("Golden dataset contains no test cases")
	}

	for _, tc := range dataset.Tests {
		t.Run(tc.Model, func(t *testing.T) {
			config := DeploymentConfig{
				NumInstances:              1,
				Horizon:                   math.MaxInt64,
				Seed:                      tc.Seed,
				TotalKVBlocks:             tc.TotalKVBlocks,
				BlockSizeTokens:           tc.BlockSizeInTokens,
				MaxRunningReqs:            tc.MaxNumRunningReqs,
				MaxScheduledTokens:        tc.MaxNumScheduledTokens,
				LongPrefillTokenThreshold: tc.LongPrefillTokenThreshold,
				BetaCoeffs:                tc.BetaCoeffs,
				AlphaCoeffs:               tc.AlphaCoeffs,
				ModelConfig:               sim.ModelConfig{},
				HWConfig:                  sim.HardwareCalib{},
				Model:                     tc.Model,
				GPU:                       tc.Hardware,
				TP:                        tc.TP,
				Roofline:                  false,
			}

			workload := &sim.GuideLLMConfig{
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
			}

			cs := NewClusterSimulator(config, workload, "")
			mustRun(t, cs)

			m := cs.AggregatedMetrics()
			if m.CompletedRequests != tc.Metrics.CompletedRequests {
				t.Errorf("completed_requests: got %d, want %d",
					m.CompletedRequests, tc.Metrics.CompletedRequests)
			}
			if m.TotalInputTokens != tc.Metrics.TotalInputTokens {
				t.Errorf("total_input_tokens: got %d, want %d",
					m.TotalInputTokens, tc.Metrics.TotalInputTokens)
			}
			if m.TotalOutputTokens != tc.Metrics.TotalOutputTokens {
				t.Errorf("total_output_tokens: got %d, want %d",
					m.TotalOutputTokens, tc.Metrics.TotalOutputTokens)
			}
			// Verify timing: SimEndedTime must match golden vllm_estimated_duration_s
			vllmRuntime := float64(m.SimEndedTime) / 1e6
			testutil.AssertFloat64Equal(t,"vllm_estimated_duration_s",
				tc.Metrics.VllmEstimatedDurationS, vllmRuntime, 1e-9)
		})
	}
}

// TestClusterSimulator_MultiInstance_Determinism verifies BC-2:
// GIVEN N=4, seed=42, 100 requests
// WHEN run twice
// THEN per-instance and aggregated CompletedRequests are identical.
func TestClusterSimulator_MultiInstance_Determinism(t *testing.T) {
	config := newTestDeploymentConfig(4)
	workload := newTestWorkload(100)

	cs1 := NewClusterSimulator(config, workload, "")
	mustRun(t, cs1)

	cs2 := NewClusterSimulator(config, workload, "")
	mustRun(t, cs2)

	// Check aggregated
	if cs1.AggregatedMetrics().CompletedRequests != cs2.AggregatedMetrics().CompletedRequests {
		t.Errorf("aggregated CompletedRequests differ: %d vs %d",
			cs1.AggregatedMetrics().CompletedRequests, cs2.AggregatedMetrics().CompletedRequests)
	}

	// Check aggregated token counts and SimEndedTime
	a1, a2 := cs1.AggregatedMetrics(), cs2.AggregatedMetrics()
	if a1.TotalInputTokens != a2.TotalInputTokens {
		t.Errorf("aggregated TotalInputTokens differ: %d vs %d",
			a1.TotalInputTokens, a2.TotalInputTokens)
	}
	if a1.TotalOutputTokens != a2.TotalOutputTokens {
		t.Errorf("aggregated TotalOutputTokens differ: %d vs %d",
			a1.TotalOutputTokens, a2.TotalOutputTokens)
	}
	if a1.SimEndedTime != a2.SimEndedTime {
		t.Errorf("aggregated SimEndedTime differ: %d vs %d",
			a1.SimEndedTime, a2.SimEndedTime)
	}

	// Check per-instance (counts and timing)
	for i := 0; i < 4; i++ {
		m1, m2 := cs1.Instances()[i].Metrics(), cs2.Instances()[i].Metrics()
		if m1.CompletedRequests != m2.CompletedRequests {
			t.Errorf("instance %d CompletedRequests differ: %d vs %d", i, m1.CompletedRequests, m2.CompletedRequests)
		}
		if m1.TotalInputTokens != m2.TotalInputTokens {
			t.Errorf("instance %d TotalInputTokens differ: %d vs %d", i, m1.TotalInputTokens, m2.TotalInputTokens)
		}
		if m1.TotalOutputTokens != m2.TotalOutputTokens {
			t.Errorf("instance %d TotalOutputTokens differ: %d vs %d", i, m1.TotalOutputTokens, m2.TotalOutputTokens)
		}
		if m1.TTFTSum != m2.TTFTSum {
			t.Errorf("instance %d TTFTSum differ: %d vs %d", i, m1.TTFTSum, m2.TTFTSum)
		}
		if m1.SimEndedTime != m2.SimEndedTime {
			t.Errorf("instance %d SimEndedTime differ: %d vs %d", i, m1.SimEndedTime, m2.SimEndedTime)
		}
	}
}

// TestClusterSimulator_MultiInstance_AllComplete verifies BC-3, BC-5:
// GIVEN N=4, 100 requests
// WHEN run
// THEN aggregated CompletedRequests == 100 AND each instance completed > 0.
func TestClusterSimulator_MultiInstance_AllComplete(t *testing.T) {
	config := newTestDeploymentConfig(4)
	workload := newTestWorkload(100)

	cs := NewClusterSimulator(config, workload, "")
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	if m.CompletedRequests != 100 {
		t.Errorf("aggregated CompletedRequests: got %d, want 100", m.CompletedRequests)
	}

	for i, inst := range cs.Instances() {
		if inst.Metrics().CompletedRequests == 0 {
			t.Errorf("instance %d CompletedRequests == 0, want > 0", i)
		}
	}
}

// TestClusterSimulator_RoundRobin_EvenDistribution verifies BC-3:
// GIVEN N=3, 9 requests
// WHEN run
// THEN each instance has CompletedRequests == 3.
func TestClusterSimulator_RoundRobin_EvenDistribution(t *testing.T) {
	config := newTestDeploymentConfig(3)
	workload := newTestWorkload(9)

	cs := NewClusterSimulator(config, workload, "")
	mustRun(t, cs)

	for i, inst := range cs.Instances() {
		if inst.Metrics().CompletedRequests != 3 {
			t.Errorf("instance %d CompletedRequests: got %d, want 3",
				i, inst.Metrics().CompletedRequests)
		}
	}
}

// TestClusterSimulator_RoundRobin_UnevenDistribution verifies BC-3:
// GIVEN N=3, 10 requests
// WHEN run
// THEN instance 0 has 4, instances 1,2 have 3.
func TestClusterSimulator_RoundRobin_UnevenDistribution(t *testing.T) {
	config := newTestDeploymentConfig(3)
	workload := newTestWorkload(10)

	cs := NewClusterSimulator(config, workload, "")
	mustRun(t, cs)

	expected := []int{4, 3, 3}
	for i, inst := range cs.Instances() {
		if inst.Metrics().CompletedRequests != expected[i] {
			t.Errorf("instance %d CompletedRequests: got %d, want %d",
				i, inst.Metrics().CompletedRequests, expected[i])
		}
	}
}

// TestClusterSimulator_ZeroRequestInstances verifies C.4:
// GIVEN N=4, 2 requests
// WHEN run
// THEN instances 0,1 have CompletedRequests == 1, instances 2,3 have 0, no panic.
func TestClusterSimulator_ZeroRequestInstances(t *testing.T) {
	config := newTestDeploymentConfig(4)
	workload := newTestWorkload(2)

	cs := NewClusterSimulator(config, workload, "")
	mustRun(t, cs)

	expected := []int{1, 1, 0, 0}
	for i, inst := range cs.Instances() {
		if inst.Metrics().CompletedRequests != expected[i] {
			t.Errorf("instance %d CompletedRequests: got %d, want %d",
				i, inst.Metrics().CompletedRequests, expected[i])
		}
	}

	if cs.AggregatedMetrics().CompletedRequests != 2 {
		t.Errorf("aggregated CompletedRequests: got %d, want 2",
			cs.AggregatedMetrics().CompletedRequests)
	}
}

// TestClusterSimulator_AggregatedMetrics_Correctness verifies BC-7:
// GIVEN N=2
// WHEN run
// THEN aggregated == sum(per-instance) for counts, max for SimEndedTime.
func TestClusterSimulator_AggregatedMetrics_Correctness(t *testing.T) {
	config := newTestDeploymentConfig(2)
	workload := newTestWorkload(50)

	cs := NewClusterSimulator(config, workload, "")
	mustRun(t, cs)

	var sumCompleted, sumInput, sumOutput int
	var maxSimEnded, maxPeakKV int64
	var sumKVBlocksUsed float64
	for _, inst := range cs.Instances() {
		m := inst.Metrics()
		sumCompleted += m.CompletedRequests
		sumInput += m.TotalInputTokens
		sumOutput += m.TotalOutputTokens
		sumKVBlocksUsed += m.KVBlocksUsed
		if m.SimEndedTime > maxSimEnded {
			maxSimEnded = m.SimEndedTime
		}
		if m.PeakKVBlocksUsed > maxPeakKV {
			maxPeakKV = m.PeakKVBlocksUsed
		}
	}

	agg := cs.AggregatedMetrics()
	if agg.CompletedRequests != sumCompleted {
		t.Errorf("aggregated CompletedRequests: got %d, want %d (sum)", agg.CompletedRequests, sumCompleted)
	}
	if agg.TotalInputTokens != sumInput {
		t.Errorf("aggregated TotalInputTokens: got %d, want %d (sum)", agg.TotalInputTokens, sumInput)
	}
	if agg.TotalOutputTokens != sumOutput {
		t.Errorf("aggregated TotalOutputTokens: got %d, want %d (sum)", agg.TotalOutputTokens, sumOutput)
	}
	if agg.SimEndedTime != maxSimEnded {
		t.Errorf("aggregated SimEndedTime: got %d, want %d (max)", agg.SimEndedTime, maxSimEnded)
	}
	if agg.KVBlocksUsed != sumKVBlocksUsed {
		t.Errorf("aggregated KVBlocksUsed: got %v, want %v (sum)", agg.KVBlocksUsed, sumKVBlocksUsed)
	}
	if agg.PeakKVBlocksUsed != maxPeakKV {
		t.Errorf("aggregated PeakKVBlocksUsed: got %d, want %d (max)", agg.PeakKVBlocksUsed, maxPeakKV)
	}

	// Verify per-request map merging
	var sumRequests, sumTTFTs, sumE2Es, sumITLs, sumAllITLs int
	var sumTTFTSum, sumITLSum int64
	for _, inst := range cs.Instances() {
		m := inst.Metrics()
		sumRequests += len(m.Requests)
		sumTTFTs += len(m.RequestTTFTs)
		sumE2Es += len(m.RequestE2Es)
		sumITLs += len(m.RequestITLs)
		sumAllITLs += len(m.AllITLs)
		sumTTFTSum += m.TTFTSum
		sumITLSum += m.ITLSum
	}
	if len(agg.Requests) != sumRequests {
		t.Errorf("aggregated len(Requests): got %d, want %d (sum)", len(agg.Requests), sumRequests)
	}
	if len(agg.RequestTTFTs) != sumTTFTs {
		t.Errorf("aggregated len(RequestTTFTs): got %d, want %d (sum)", len(agg.RequestTTFTs), sumTTFTs)
	}
	if len(agg.RequestE2Es) != sumE2Es {
		t.Errorf("aggregated len(RequestE2Es): got %d, want %d (sum)", len(agg.RequestE2Es), sumE2Es)
	}
	if len(agg.AllITLs) != sumAllITLs {
		t.Errorf("aggregated len(AllITLs): got %d, want %d (sum)", len(agg.AllITLs), sumAllITLs)
	}
	if agg.TTFTSum != sumTTFTSum {
		t.Errorf("aggregated TTFTSum: got %d, want %d (sum)", agg.TTFTSum, sumTTFTSum)
	}
	if agg.ITLSum != sumITLSum {
		t.Errorf("aggregated ITLSum: got %d, want %d (sum)", agg.ITLSum, sumITLSum)
	}
}

// TestClusterSimulator_SharedClock_MonotonicGlobal verifies BC-6:
// GIVEN N=2
// WHEN run
// THEN cluster.Clock() >= every instance's Clock().
func TestClusterSimulator_SharedClock_MonotonicGlobal(t *testing.T) {
	config := newTestDeploymentConfig(2)
	workload := newTestWorkload(50)

	cs := NewClusterSimulator(config, workload, "")
	mustRun(t, cs)

	for i, inst := range cs.Instances() {
		if cs.Clock() < inst.Clock() {
			t.Errorf("cluster clock %d < instance %d clock %d",
				cs.Clock(), i, inst.Clock())
		}
	}
}

// TestClusterSimulator_RunOnce_Panics verifies C.3:
// GIVEN cluster has Run()
// WHEN Run() called again
// THEN panic.
func TestClusterSimulator_RunOnce_Panics(t *testing.T) {
	config := newTestDeploymentConfig(2)
	workload := newTestWorkload(10)

	cs := NewClusterSimulator(config, workload, "")
	mustRun(t, cs)

	defer func() {
		r := recover()
		if r == nil {
			t.Error("expected panic on second Run(), got none")
		}
		expected := "ClusterSimulator.Run() called more than once"
		if r != expected {
			t.Errorf("panic message = %q, want %q", r, expected)
		}
	}()
	mustRun(t, cs)
}

// TestNewClusterSimulator_ZeroInstances_Panics verifies C.4:
// GIVEN NumInstances=0
// WHEN NewClusterSimulator()
// THEN panic.
func TestNewClusterSimulator_ZeroInstances_Panics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Error("expected panic for NumInstances=0, got none")
		}
		expected := "ClusterSimulator: NumInstances must be >= 1"
		if r != expected {
			t.Errorf("panic message = %q, want %q", r, expected)
		}
	}()

	config := newTestDeploymentConfig(0)
	NewClusterSimulator(config, newTestWorkload(10), "")
}

// TestInstanceSimulator_InjectAfterRun_Panics verifies C.3:
// GIVEN instance has Run()
// WHEN InjectRequest() called
// THEN panic.
func TestInstanceSimulator_InjectAfterRun_Panics(t *testing.T) {
	inst := NewInstanceSimulator("test", newTestDeploymentConfig(1).ToSimConfig())
	inst.Run()

	defer func() {
		r := recover()
		if r == nil {
			t.Error("expected panic on InjectRequest after Run(), got none")
		}
		expected := "InstanceSimulator.InjectRequest() called after Run()"
		if r != expected {
			t.Errorf("panic message = %q, want %q", r, expected)
		}
	}()
	inst.InjectRequest(&sim.Request{
		ID: "req", ArrivalTime: 0, InputTokens: make([]int, 5),
		OutputTokens: make([]int, 3), State: sim.StateQueued,
	})
}

// TestClusterSimulator_GloballyUniqueRequestIDs verifies BC-4:
// GIVEN N=4, 20 requests
// WHEN run
// THEN len(AggregatedMetrics().Requests) == AggregatedMetrics().CompletedRequests
// AND all request IDs across instances are distinct.
func TestClusterSimulator_GloballyUniqueRequestIDs(t *testing.T) {
	config := newTestDeploymentConfig(4)
	workload := newTestWorkload(20)

	cs := NewClusterSimulator(config, workload, "")
	mustRun(t, cs)

	agg := cs.AggregatedMetrics()
	if len(agg.Requests) != agg.CompletedRequests {
		t.Errorf("len(Requests)=%d != CompletedRequests=%d — possible ID collision",
			len(agg.Requests), agg.CompletedRequests)
	}

	// Verify all IDs across instances are distinct
	seen := make(map[string]int) // request ID -> instance index
	for i, inst := range cs.Instances() {
		for id := range inst.Metrics().Requests {
			if prev, exists := seen[id]; exists {
				t.Errorf("duplicate request ID %q: instance %d and instance %d", id, prev, i)
			}
			seen[id] = i
		}
	}
}

// TestClusterSimulator_HorizonEnforcement verifies BC-8:
// GIVEN a finite horizon and enough requests to exceed it
// WHEN run
// THEN some requests are not completed AND cluster clock does not far exceed horizon.
func TestClusterSimulator_HorizonEnforcement(t *testing.T) {
	config := newTestDeploymentConfig(2)
	config.Horizon = 500000 // finite horizon
	workload := newTestWorkload(100)

	cs := NewClusterSimulator(config, workload, "")
	mustRun(t, cs)

	agg := cs.AggregatedMetrics()

	// With a tight horizon, not all requests should complete
	if agg.CompletedRequests >= 100 {
		t.Errorf("expected fewer than 100 completed requests with tight horizon, got %d",
			agg.CompletedRequests)
	}

	// SimEndedTime should be capped at horizon
	if agg.SimEndedTime > config.Horizon {
		t.Errorf("SimEndedTime %d exceeds horizon %d", agg.SimEndedTime, config.Horizon)
	}
}

// TestClusterSimulator_NilWorkload_Panics verifies the nil workload guard.
func TestClusterSimulator_NilWorkload_Panics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Error("expected panic for nil workload and empty traces path, got none")
		}
		expected := "ClusterSimulator: workload config is nil and no traces path provided"
		if r != expected {
			t.Errorf("panic message = %q, want %q", r, expected)
		}
	}()

	config := newTestDeploymentConfig(2)
	NewClusterSimulator(config, nil, "")
}

// TestClusterSimulator_AggregatedMetrics_BeforeRun_Panics verifies the hasRun guard.
func TestClusterSimulator_AggregatedMetrics_BeforeRun_Panics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Error("expected panic for AggregatedMetrics() before Run(), got none")
		}
		expected := "ClusterSimulator.AggregatedMetrics() called before Run()"
		if r != expected {
			t.Errorf("panic message = %q, want %q", r, expected)
		}
	}()

	config := newTestDeploymentConfig(2)
	cs := NewClusterSimulator(config, newTestWorkload(10), "")
	cs.AggregatedMetrics()
}

// TestClusterSimulator_HandledBy_PopulatedInMetrics verifies #181:
// GIVEN a 3-instance cluster with round-robin routing and 15 requests
// WHEN the simulation completes
// THEN every completed request's metrics has a non-empty HandledBy field
// AND each HandledBy value matches a valid instance ID
// AND per-instance metrics only contain requests handled by that instance
func TestClusterSimulator_HandledBy_PopulatedInMetrics(t *testing.T) {
	config := newTestDeploymentConfig(3)
	config.RoutingPolicy = "round-robin"
	workload := newTestWorkload(15)

	cs := NewClusterSimulator(config, workload, "")
	mustRun(t, cs)

	agg := cs.AggregatedMetrics()
	if agg.CompletedRequests == 0 {
		t.Fatal("expected completed requests, got 0")
	}

	// Build set of valid instance IDs
	validIDs := make(map[string]bool, len(cs.Instances()))
	for _, inst := range cs.Instances() {
		validIDs[string(inst.ID())] = true
	}

	// Verify every request in aggregated metrics has a valid HandledBy
	for reqID, rm := range agg.Requests {
		if rm.HandledBy == "" {
			t.Errorf("request %s: HandledBy is empty", reqID)
			continue
		}
		if !validIDs[rm.HandledBy] {
			t.Errorf("request %s: HandledBy=%q is not a valid instance ID", reqID, rm.HandledBy)
		}
	}

	// Verify per-instance consistency: each instance's metrics should only
	// contain requests with HandledBy matching that instance
	for _, inst := range cs.Instances() {
		instID := string(inst.ID())
		m := inst.Metrics()
		for reqID, rm := range m.Requests {
			if rm.HandledBy != instID {
				t.Errorf("instance %s contains request %s with HandledBy=%q (want %q)",
					instID, reqID, rm.HandledBy, instID)
			}
		}
	}
}

// TestClusterSimulator_HandledBy_SingleInstance verifies #181 boundary:
// GIVEN a 1-instance cluster
// WHEN the simulation completes
// THEN all requests have HandledBy == "instance_0"
func TestClusterSimulator_HandledBy_SingleInstance(t *testing.T) {
	config := newTestDeploymentConfig(1)
	workload := newTestWorkload(5)
	cs := NewClusterSimulator(config, workload, "")
	mustRun(t, cs)

	agg := cs.AggregatedMetrics()
	if agg.CompletedRequests == 0 {
		t.Fatal("expected completed requests, got 0")
	}
	for reqID, rm := range agg.Requests {
		if rm.HandledBy != "instance_0" {
			t.Errorf("request %s: HandledBy=%q, want %q", reqID, rm.HandledBy, "instance_0")
		}
	}
}

// === Routing Policy Tests ===

// TestClusterSimulator_RoutingPolicy_RoundRobinDefault verifies BC-6 (backward compatibility).
func TestClusterSimulator_RoutingPolicy_RoundRobinDefault(t *testing.T) {
	config := newTestDeploymentConfig(3)
	config.RoutingPolicy = "round-robin"
	workload := newTestWorkload(10)

	cs := NewClusterSimulator(config, workload, "")
	mustRun(t, cs)

	// Requests distributed evenly: 4, 3, 3 (or variant)
	counts := make(map[InstanceID]int)
	for _, inst := range cs.Instances() {
		counts[inst.ID()] = inst.Metrics().CompletedRequests
	}

	total := 0
	for _, count := range counts {
		total += count
		if count < 3 || count > 4 {
			t.Errorf("Expected 3-4 requests per instance, got %d", count)
		}
	}
	if total != 10 {
		t.Errorf("Expected 10 total completed requests, got %d", total)
	}
}

// TestClusterSimulator_RoutingPolicy_LeastLoaded verifies load-aware routing completes.
func TestClusterSimulator_RoutingPolicy_LeastLoaded(t *testing.T) {
	config := newTestDeploymentConfig(2)
	config.RoutingPolicy = "least-loaded"
	workload := newTestWorkload(5)

	cs := NewClusterSimulator(config, workload, "")
	mustRun(t, cs)

	if cs.AggregatedMetrics().CompletedRequests == 0 {
		t.Errorf("Expected non-zero completed requests, got 0")
	}
}

// TestClusterSimulator_AllRoutingPolicies_Smoke verifies all policies are exercisable.
func TestClusterSimulator_AllRoutingPolicies_Smoke(t *testing.T) {
	policies := []string{"round-robin", "least-loaded", "weighted", "prefix-affinity"}

	for _, policyName := range policies {
		t.Run(policyName, func(t *testing.T) {
			config := newTestDeploymentConfig(2)
			config.RoutingPolicy = policyName
			config.RoutingScorerConfigs = sim.DefaultScorerConfigs()
			workload := newTestWorkload(5)

			cs := NewClusterSimulator(config, workload, "")
			mustRun(t, cs)

			if cs.AggregatedMetrics().CompletedRequests == 0 {
				t.Errorf("Policy %q: expected non-zero completed requests", policyName)
			}
		})
	}
}

// === Benchmarks ===

func BenchmarkClusterSimulator_1K_1Instance(b *testing.B) {
	config := newTestDeploymentConfig(1)
	workload := newTestWorkload(1000)
	for i := 0; i < b.N; i++ {
		cs := NewClusterSimulator(config, workload, "")
		if err := cs.Run(); err != nil {
			b.Fatalf("cs.Run: %v", err)
		}
	}
}

func BenchmarkClusterSimulator_10K_4Instances(b *testing.B) {
	config := newTestDeploymentConfig(4)
	workload := newTestWorkload(10000)
	for i := 0; i < b.N; i++ {
		cs := NewClusterSimulator(config, workload, "")
		if err := cs.Run(); err != nil {
			b.Fatalf("cs.Run: %v", err)
		}
	}
}

func BenchmarkClusterSimulator_1K_10Instances(b *testing.B) {
	config := newTestDeploymentConfig(10)
	workload := newTestWorkload(1000)
	for i := 0; i < b.N; i++ {
		cs := NewClusterSimulator(config, workload, "")
		if err := cs.Run(); err != nil {
			b.Fatalf("cs.Run: %v", err)
		}
	}
}

// === Workload Parity Tests (F.4) ===

// TestClusterWorkloadGen_MatchesSimulator verifies D.6:
// GIVEN same seed and workload config
// WHEN sim.NewSimulator generates workload AND ClusterSimulator generates requests
// THEN count matches and for each request: ArrivalTime, len(InputTokens), len(OutputTokens) match.
func TestClusterWorkloadGen_MatchesSimulator(t *testing.T) {
	dataset := testutil.LoadGoldenDataset(t)
	if len(dataset.Tests) == 0 {
		t.Fatal("Golden dataset contains no test cases")
	}

	for _, tc := range dataset.Tests {
		t.Run(tc.Model, func(t *testing.T) {
			guideLLMConfig := &sim.GuideLLMConfig{
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
			}

			// Reference: sim.NewSimulator generates workload internally
			refSim, err := sim.NewSimulator(sim.SimConfig{
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
				GuideLLMConfig:            guideLLMConfig,
			})
			if err != nil {
				t.Fatalf("NewSimulator: %v", err)
			}

			// Cluster workload generation
			config := DeploymentConfig{
				NumInstances:              1,
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
			}
			cs := NewClusterSimulator(config, guideLLMConfig, "")
			requests := cs.generateRequestsFromDistribution()

			// Compare counts
			refCount := len(refSim.Metrics.Requests)
			if len(requests) != refCount {
				t.Fatalf("request count mismatch: cluster=%d, simulator=%d", len(requests), refCount)
			}

			// Compare each request
			for _, req := range requests {
				refMetric, ok := refSim.Metrics.Requests[req.ID]
				if !ok {
					t.Errorf("request %s not found in simulator Metrics.Requests", req.ID)
					continue
				}
				// ArrivalTime: compare via ArrivedAt (float64 seconds)
				gotArrived := float64(req.ArrivalTime) / 1e6
				if gotArrived != refMetric.ArrivedAt {
					t.Errorf("request %s ArrivalTime: cluster=%v, simulator=%v",
						req.ID, gotArrived, refMetric.ArrivedAt)
				}
				if len(req.InputTokens) != refMetric.NumPrefillTokens {
					t.Errorf("request %s InputTokens len: cluster=%d, simulator=%d",
						req.ID, len(req.InputTokens), refMetric.NumPrefillTokens)
				}
				if len(req.OutputTokens) != refMetric.NumDecodeTokens {
					t.Errorf("request %s OutputTokens len: cluster=%d, simulator=%d",
						req.ID, len(req.OutputTokens), refMetric.NumDecodeTokens)
				}
			}
		})
	}
}

// TestClusterWorkloadGen_Determinism verifies BC-2:
// GIVEN same seed
// WHEN called twice
// THEN request lists are identical.
func TestClusterWorkloadGen_Determinism(t *testing.T) {
	config := newTestDeploymentConfig(1)
	workload := newTestWorkload(50)

	cs1 := NewClusterSimulator(config, workload, "")
	reqs1 := cs1.generateRequestsFromDistribution()

	cs2 := NewClusterSimulator(config, workload, "")
	reqs2 := cs2.generateRequestsFromDistribution()

	if len(reqs1) != len(reqs2) {
		t.Fatalf("request count mismatch: %d vs %d", len(reqs1), len(reqs2))
	}

	for i := range reqs1 {
		if reqs1[i].ID != reqs2[i].ID {
			t.Errorf("request %d ID mismatch: %s vs %s", i, reqs1[i].ID, reqs2[i].ID)
		}
		if reqs1[i].ArrivalTime != reqs2[i].ArrivalTime {
			t.Errorf("request %d ArrivalTime mismatch: %d vs %d",
				i, reqs1[i].ArrivalTime, reqs2[i].ArrivalTime)
		}
		if len(reqs1[i].InputTokens) != len(reqs2[i].InputTokens) {
			t.Errorf("request %d InputTokens len mismatch: %d vs %d",
				i, len(reqs1[i].InputTokens), len(reqs2[i].InputTokens))
		}
		if len(reqs1[i].OutputTokens) != len(reqs2[i].OutputTokens) {
			t.Errorf("request %d OutputTokens len mismatch: %d vs %d",
				i, len(reqs1[i].OutputTokens), len(reqs2[i].OutputTokens))
		}
	}
}

func TestAggregateMetrics_IncludesKVCacheFields(t *testing.T) {
	// GIVEN a cluster simulation with 2 instances
	cfg := newTestDeploymentConfig(2)
	cs := NewClusterSimulator(cfg, newTestWorkload(10), "")
	mustRun(t, cs)

	agg := cs.AggregatedMetrics()
	perInst := cs.PerInstanceMetrics()

	// THEN PreemptionCount MUST be the sum of per-instance counts
	expectedPreemption := int64(0)
	for _, m := range perInst {
		expectedPreemption += m.PreemptionCount
	}
	if agg.PreemptionCount != expectedPreemption {
		t.Errorf("PreemptionCount: got %d, want %d (sum of per-instance)", agg.PreemptionCount, expectedPreemption)
	}

	// THEN KVAllocationFailures MUST be the sum of per-instance counts
	expectedKVFailures := int64(0)
	for _, m := range perInst {
		expectedKVFailures += m.KVAllocationFailures
	}
	if agg.KVAllocationFailures != expectedKVFailures {
		t.Errorf("KVAllocationFailures: got %d, want %d (sum of per-instance)", agg.KVAllocationFailures, expectedKVFailures)
	}

	// THEN CacheHitRate MUST be the average of per-instance rates
	expectedCacheHit := 0.0
	for _, m := range perInst {
		expectedCacheHit += m.CacheHitRate
	}
	expectedCacheHit /= float64(len(perInst))
	if math.Abs(agg.CacheHitRate-expectedCacheHit) > 1e-9 {
		t.Errorf("CacheHitRate: got %f, want %f (average of per-instance)", agg.CacheHitRate, expectedCacheHit)
	}

	// THEN KVThrashingRate MUST be the average of per-instance rates
	expectedThrashing := 0.0
	for _, m := range perInst {
		expectedThrashing += m.KVThrashingRate
	}
	expectedThrashing /= float64(len(perInst))
	if math.Abs(agg.KVThrashingRate-expectedThrashing) > 1e-9 {
		t.Errorf("KVThrashingRate: got %f, want %f (average of per-instance)", agg.KVThrashingRate, expectedThrashing)
	}
}

func TestAggregateMetrics_SingleInstance_AverageEqualsSelf(t *testing.T) {
	// GIVEN a cluster with exactly 1 instance (edge case: average = self)
	cfg := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(cfg, newTestWorkload(5), "")
	mustRun(t, cs)

	agg := cs.AggregatedMetrics()
	perInst := cs.PerInstanceMetrics()

	// THEN for a single instance, aggregated values MUST equal the instance values
	if agg.PreemptionCount != perInst[0].PreemptionCount {
		t.Errorf("PreemptionCount: got %d, want %d (single instance)", agg.PreemptionCount, perInst[0].PreemptionCount)
	}
	if math.Abs(agg.CacheHitRate-perInst[0].CacheHitRate) > 1e-9 {
		t.Errorf("CacheHitRate: got %f, want %f (single instance)", agg.CacheHitRate, perInst[0].CacheHitRate)
	}
	if math.Abs(agg.KVThrashingRate-perInst[0].KVThrashingRate) > 1e-9 {
		t.Errorf("KVThrashingRate: got %f, want %f (single instance)", agg.KVThrashingRate, perInst[0].KVThrashingRate)
	}
}

// =============================================================================
// Cluster-Level Invariant Tests (Phase 4, issue #211)
// =============================================================================

// TestClusterSimulator_RequestConservation_SumAcrossInstances verifies BC-3:
// GIVEN N=4 instances and 100 requests
// WHEN the cluster simulation completes (infinite horizon)
// THEN sum of per-instance CompletedRequests == 100 == aggregated CompletedRequests.
func TestClusterSimulator_RequestConservation_SumAcrossInstances(t *testing.T) {
	config := newTestDeploymentConfig(4)
	workload := newTestWorkload(100)

	cs := NewClusterSimulator(config, workload, "")
	mustRun(t, cs)

	sumCompleted := 0
	for _, inst := range cs.Instances() {
		sumCompleted += inst.Metrics().CompletedRequests
	}

	agg := cs.AggregatedMetrics()

	// Conservation: sum of parts == whole
	if sumCompleted != agg.CompletedRequests {
		t.Errorf("conservation: sum of instance completions (%d) != aggregated (%d)",
			sumCompleted, agg.CompletedRequests)
	}

	// Conservation: injected == completed
	if agg.CompletedRequests != 100 {
		t.Errorf("conservation: aggregated completions (%d) != injected (100)",
			agg.CompletedRequests)
	}
}

// TestClusterSimulator_Causality_PerInstance verifies BC-5:
// GIVEN a cluster simulation with multiple instances
// WHEN examining per-instance metrics
// THEN for every completed request: TTFT >= 0, E2E >= TTFT, and all ITL >= 0.
func TestClusterSimulator_Causality_PerInstance(t *testing.T) {
	config := newTestDeploymentConfig(3)
	workload := newTestWorkload(50)

	cs := NewClusterSimulator(config, workload, "")
	mustRun(t, cs)

	totalChecked := 0
	for idx, inst := range cs.Instances() {
		m := inst.Metrics()
		for id, ttft := range m.RequestTTFTs {
			e2e, ok := m.RequestE2Es[id]
			if !ok {
				continue
			}
			// TTFT is a relative duration from arrival — must be non-negative
			if ttft < 0 {
				t.Errorf("causality violated: instance %d, request %s: TTFT (%.2f) < 0", idx, id, ttft)
			}
			// E2E must be >= TTFT
			if e2e < ttft {
				t.Errorf("causality violated: instance %d, request %s: E2E (%.2f) < TTFT (%.2f)",
					idx, id, e2e, ttft)
			}
			totalChecked++
		}

		for i, itl := range m.AllITLs {
			if itl < 0 {
				t.Errorf("negative ITL: instance %d, index %d: %d", idx, i, itl)
			}
		}
	}

	if totalChecked == 0 {
		t.Fatal("no completed requests checked — test setup invalid")
	}
}

// TestClusterSimulator_ClockMonotonicity_ClusterDominatesInstances verifies BC-7:
// GIVEN a cluster simulation with non-trivial workload
// WHEN the simulation completes
// THEN cluster.Clock() >= every instance's Clock().
func TestClusterSimulator_ClockMonotonicity_ClusterDominatesInstances(t *testing.T) {
	config := newTestDeploymentConfig(4)
	workload := newTestWorkload(100)

	cs := NewClusterSimulator(config, workload, "")
	mustRun(t, cs)

	for i, inst := range cs.Instances() {
		if cs.Clock() < inst.Clock() {
			t.Errorf("clock monotonicity violated: cluster clock (%d) < instance %d clock (%d)",
				cs.Clock(), i, inst.Clock())
		}
	}
}

// TestClusterSimulator_Determinism_ByteIdenticalAggregation verifies BC-9:
// GIVEN two cluster runs with identical config and seed
// WHEN both aggregate metrics
// THEN all integer metrics match exactly AND per-request metrics (sorted, JSON) are byte-identical.
func TestClusterSimulator_Determinism_ByteIdenticalAggregation(t *testing.T) {
	run := func() *sim.Metrics {
		config := newTestDeploymentConfig(3)
		workload := newTestWorkload(50)
		cs := NewClusterSimulator(config, workload, "")
		mustRun(t, cs)
		return cs.AggregatedMetrics()
	}

	m1 := run()
	m2 := run()

	// Compare integer fields
	if m1.CompletedRequests != m2.CompletedRequests {
		t.Errorf("determinism: CompletedRequests %d vs %d", m1.CompletedRequests, m2.CompletedRequests)
	}
	if m1.TotalInputTokens != m2.TotalInputTokens {
		t.Errorf("determinism: TotalInputTokens %d vs %d", m1.TotalInputTokens, m2.TotalInputTokens)
	}
	if m1.TotalOutputTokens != m2.TotalOutputTokens {
		t.Errorf("determinism: TotalOutputTokens %d vs %d", m1.TotalOutputTokens, m2.TotalOutputTokens)
	}
	if m1.SimEndedTime != m2.SimEndedTime {
		t.Errorf("determinism: SimEndedTime %d vs %d", m1.SimEndedTime, m2.SimEndedTime)
	}
	if m1.TTFTSum != m2.TTFTSum {
		t.Errorf("determinism: TTFTSum %d vs %d", m1.TTFTSum, m2.TTFTSum)
	}
	if m1.ITLSum != m2.ITLSum {
		t.Errorf("determinism: ITLSum %d vs %d", m1.ITLSum, m2.ITLSum)
	}

	// Compare per-request maps via JSON serialization (catches map ordering issues)
	j1, _ := json.Marshal(sortedRequestMetrics(m1.Requests))
	j2, _ := json.Marshal(sortedRequestMetrics(m2.Requests))
	if !bytes.Equal(j1, j2) {
		t.Error("determinism: per-request metrics JSON differs between runs")
	}
}

// sortedRequestMetrics returns RequestMetrics in sorted order for deterministic comparison.
func sortedRequestMetrics(m map[string]sim.RequestMetrics) []sim.RequestMetrics {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	result := make([]sim.RequestMetrics, len(keys))
	for i, k := range keys {
		result[i] = m[k]
	}
	return result
}

// TestClusterSimulator_Conservation_PolicyMatrix verifies INV-1 at cluster level
// across 10 policy combinations (promoted from H12 hypothesis experiment):
// GIVEN each policy combination with infinite horizon and ample resources
// WHEN the cluster simulation completes
// THEN completed + still_queued + still_running == injected (three-term conservation)
// AND all requests complete (infinite horizon, no resource pressure).
func TestClusterSimulator_Conservation_PolicyMatrix(t *testing.T) {
	matrix := []struct {
		name            string
		numInstances    int
		routingPolicy   string
		scorerConfigs   []sim.ScorerConfig
		scheduler       string
		priorityPolicy  string
		admissionPolicy string
	}{
		{"round-robin/fcfs/2inst", 2, "round-robin", nil, "fcfs", "constant", "always-admit"},
		{"least-loaded/fcfs/3inst", 3, "least-loaded", nil, "fcfs", "constant", "always-admit"},
		{"weighted/fcfs/2inst", 2, "weighted", sim.DefaultScorerConfigs(), "fcfs", "constant", "always-admit"},
		{"prefix-affinity/fcfs/2inst", 2, "prefix-affinity", nil, "fcfs", "constant", "always-admit"},
		{"round-robin/sjf/3inst", 3, "round-robin", nil, "sjf", "constant", "always-admit"},
		{"round-robin/priority-fcfs/slo/2inst", 2, "round-robin", nil, "priority-fcfs", "slo-based", "always-admit"},
		{"least-loaded/priority-fcfs/slo/3inst", 3, "least-loaded", nil, "priority-fcfs", "slo-based", "always-admit"},
		{"weighted/sjf/4inst", 4, "weighted", sim.DefaultScorerConfigs(), "sjf", "constant", "always-admit"},
		{"round-robin/fcfs/token-bucket/2inst", 2, "round-robin", nil, "fcfs", "constant", "token-bucket"},
		{"least-loaded/fcfs/4inst", 4, "least-loaded", nil, "fcfs", "constant", "always-admit"},
	}

	const numRequests = 50

	for _, tc := range matrix {
		t.Run(tc.name, func(t *testing.T) {
			config := newTestDeploymentConfig(tc.numInstances)
			config.RoutingPolicy = tc.routingPolicy
			config.RoutingScorerConfigs = tc.scorerConfigs
			config.Scheduler = tc.scheduler
			config.PriorityPolicy = tc.priorityPolicy
			config.AdmissionPolicy = tc.admissionPolicy
			// Token bucket with generous capacity so all requests are admitted
			if tc.admissionPolicy == "token-bucket" {
				config.TokenBucketCapacity = 1e6
				config.TokenBucketRefillRate = 1e6
			}

			workload := newTestWorkload(numRequests)
			cs := NewClusterSimulator(config, workload, "")
			mustRun(t, cs)

			agg := cs.AggregatedMetrics()
			injected := len(agg.Requests)

			// BC-3: Three-term conservation equation (INV-1)
			conservation := agg.CompletedRequests + agg.StillQueued + agg.StillRunning
			if conservation != injected {
				t.Errorf("INV-1 conservation: completed(%d) + queued(%d) + running(%d) = %d, injected = %d",
					agg.CompletedRequests, agg.StillQueued, agg.StillRunning, conservation, injected)
			}

			// BC-4: All complete under infinite horizon with ample resources
			if agg.CompletedRequests != numRequests {
				t.Errorf("infinite horizon: CompletedRequests = %d, want %d",
					agg.CompletedRequests, numRequests)
			}

			// Cross-check: sum of per-instance completions == aggregated
			sumCompleted := 0
			for _, inst := range cs.Instances() {
				sumCompleted += inst.Metrics().CompletedRequests
			}
			if sumCompleted != agg.CompletedRequests {
				t.Errorf("aggregation: sum(per-instance) = %d, aggregated = %d",
					sumCompleted, agg.CompletedRequests)
			}
		})
	}
}

// TestClusterSimulator_Determinism_PrefixAffinity_ByteIdentical verifies INV-6
// for routing policies that use stateful scorers with internal maps (promoted from H13):
// GIVEN identical config with prefix-affinity or weighted routing (includes prefix scorer)
// WHEN run twice with same seed
// THEN per-request metrics JSON is byte-identical.
//
// This specifically targets the PrefixCacheIndex LRU which uses map iteration internally.
// Non-deterministic map iteration in scoring or eviction would cause divergence here.
func TestClusterSimulator_Determinism_PrefixAffinity_ByteIdentical(t *testing.T) {
	policies := []struct {
		name          string
		routingPolicy string
		scorerConfigs []sim.ScorerConfig
	}{
		{"prefix-affinity", "prefix-affinity", sim.DefaultScorerConfigs()},
		{"weighted-default", "weighted", sim.DefaultScorerConfigs()},
	}

	for _, pol := range policies {
		t.Run(pol.name, func(t *testing.T) {
			mkSim := func() *ClusterSimulator {
				config := newTestDeploymentConfig(3)
				config.RoutingPolicy = pol.routingPolicy
				config.RoutingScorerConfigs = pol.scorerConfigs
				// Use prefix tokens to exercise the prefix cache index
				workload := &sim.GuideLLMConfig{
					Rate:               10.0 / 1e6,
					NumRequests:        30,
					PrefixTokens:       32,
					PromptTokens:       100,
					PromptTokensStdDev: 20,
					PromptTokensMin:    10,
					PromptTokensMax:    200,
					OutputTokens:       50,
					OutputTokensStdDev: 10,
					OutputTokensMin:    10,
					OutputTokensMax:    100,
				}
				cs := NewClusterSimulator(config, workload, "")
				mustRun(t, cs)
				return cs
			}

			cs1 := mkSim()
			cs2 := mkSim()

			m1 := cs1.AggregatedMetrics()
			m2 := cs2.AggregatedMetrics()

			// Integer fields must match exactly
			if m1.CompletedRequests != m2.CompletedRequests {
				t.Errorf("CompletedRequests: %d vs %d", m1.CompletedRequests, m2.CompletedRequests)
			}
			if m1.TotalInputTokens != m2.TotalInputTokens {
				t.Errorf("TotalInputTokens: %d vs %d", m1.TotalInputTokens, m2.TotalInputTokens)
			}
			if m1.TotalOutputTokens != m2.TotalOutputTokens {
				t.Errorf("TotalOutputTokens: %d vs %d", m1.TotalOutputTokens, m2.TotalOutputTokens)
			}
			if m1.SimEndedTime != m2.SimEndedTime {
				t.Errorf("SimEndedTime: %d vs %d", m1.SimEndedTime, m2.SimEndedTime)
			}

			// Per-request metrics must be byte-identical (sorted JSON)
			j1, _ := json.Marshal(sortedRequestMetrics(m1.Requests))
			j2, _ := json.Marshal(sortedRequestMetrics(m2.Requests))
			if !bytes.Equal(j1, j2) {
				t.Error("INV-6 violated: per-request metrics JSON differs between runs " +
					"(likely non-deterministic map iteration in prefix cache or scorer)")
			}
		})
	}
}

// TestClusterSimulator_OverloadConservation verifies INV-1 under 10x overload
// (promoted from H-Overload hypothesis experiment, PR #335):
// GIVEN a 4-instance cluster at extreme overload rate
// WHEN the simulation runs to a finite horizon
// THEN conservation holds:
//   - always-admit: completed + still_queued + still_running == injected
//   - token-bucket: completed + still_queued + still_running + rejected == total_generated
//
// AND no panics occur (BC-5).
func TestClusterSimulator_OverloadConservation(t *testing.T) {
	// Use a high rate relative to capacity to create genuine overload.
	// With beta=[1000,10,5], 4 instances, max-running=256: capacity is very high
	// due to batching. A rate of 500 req/s with only 200 requests and a short
	// horizon creates a burst that overloads the system.
	cases := []struct {
		name            string
		admissionPolicy string
		// Token bucket params (only used when admission is "token-bucket")
		tbCapacity   float64
		tbRefillRate float64
	}{
		{"always-admit", "always-admit", 0, 0},
		{"token-bucket", "token-bucket", 5000, 10000},
	}

	const (
		numRequests  = 500
		numInstances = 4
		rateReqPerS  = 50_000.0
		maxRunning   = 2 // Tightly constrain batch size to create genuine overload
		// All 500 requests arrive in ~10ms (500/50000). With max-running=2
		// per instance (8 total slots), service time far exceeds horizon.
		horizon = 100_000 // 0.1 seconds in microsecond ticks
	)

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			config := newTestDeploymentConfig(numInstances)
			config.Horizon = horizon
			config.MaxRunningReqs = maxRunning
			config.AdmissionPolicy = tc.admissionPolicy
			config.RoutingPolicy = "least-loaded"
			config.Scheduler = "fcfs"
			config.PriorityPolicy = "constant"
			if tc.admissionPolicy == "token-bucket" {
				config.TokenBucketCapacity = tc.tbCapacity
				config.TokenBucketRefillRate = tc.tbRefillRate
			}

			workload := &sim.GuideLLMConfig{
				Rate:               rateReqPerS / 1e6,
				NumRequests:        numRequests,
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

			cs := NewClusterSimulator(config, workload, "")
			mustRun(t, cs)

			agg := cs.AggregatedMetrics()
			injected := len(agg.Requests)
			rejected := cs.RejectedRequests()

			// BC-1/BC-2: Conservation equation
			conservation := agg.CompletedRequests + agg.StillQueued + agg.StillRunning
			if tc.admissionPolicy == "always-admit" {
				// Three-term conservation: no rejections
				if conservation != injected {
					t.Errorf("INV-1 conservation (always-admit): completed(%d) + queued(%d) + running(%d) = %d, want %d (injected)",
						agg.CompletedRequests, agg.StillQueued, agg.StillRunning, conservation, injected)
				}
				if rejected != 0 {
					t.Errorf("always-admit should have 0 rejections, got %d", rejected)
				}
			} else {
				// Four-term conservation: injected + rejected == total generated
				totalGenerated := injected + rejected
				if conservation != injected {
					t.Errorf("INV-1 conservation (token-bucket): completed(%d) + queued(%d) + running(%d) = %d, want %d (injected)",
						agg.CompletedRequests, agg.StillQueued, agg.StillRunning, conservation, injected)
				}
				if totalGenerated != numRequests {
					t.Errorf("four-term conservation: injected(%d) + rejected(%d) = %d, want %d (total generated)",
						injected, rejected, totalGenerated, numRequests)
				}
			}

			// Verify overload: under finite horizon, not all requests should complete
			// (this confirms the test is actually exercising overload, not a trivial case)
			if agg.CompletedRequests == numRequests && tc.admissionPolicy == "always-admit" {
				t.Logf("warning: all %d requests completed — overload may not be genuine (increase rate or decrease horizon)", numRequests)
			}
		})
	}
}

// TestClusterSimulator_SchedulerLiveness verifies scheduler liveness (INV-2)
// across all scheduler types (promoted from H-Liveness hypothesis experiment, PR #335):
// GIVEN each scheduler (fcfs, sjf, priority-fcfs) with a mixed workload and
//
//	batch-constrained config (max-running=8) that forces queueing
//
// WHEN the simulation runs to completion (infinite horizon, ample resources)
// THEN all requests complete: still_queued == 0, still_running == 0
// AND completed == injected (conservation + liveness combined).
func TestClusterSimulator_SchedulerLiveness(t *testing.T) {
	schedulers := []struct {
		name           string
		scheduler      string
		priorityPolicy string
	}{
		{"fcfs", "fcfs", "constant"},
		{"sjf", "sjf", "constant"},
		{"priority-fcfs", "priority-fcfs", "slo-based"},
	}

	const (
		numRequests  = 100
		numInstances = 4
		rateReqPerS  = 200.0
		maxRunning   = 8 // Constrains batch size to force queueing
	)

	for _, tc := range schedulers {
		t.Run(tc.name, func(t *testing.T) {
			config := newTestDeploymentConfig(numInstances)
			config.Horizon = math.MaxInt64 // Infinite horizon — all requests must complete
			config.MaxRunningReqs = maxRunning
			config.RoutingPolicy = "least-loaded"
			config.AdmissionPolicy = "always-admit"
			config.Scheduler = tc.scheduler
			config.PriorityPolicy = tc.priorityPolicy

			// Mixed workload: varying prompt and output sizes to exercise scheduler ordering
			workload := &sim.GuideLLMConfig{
				Rate:               rateReqPerS / 1e6,
				NumRequests:        numRequests,
				PrefixTokens:       0,
				PromptTokens:       200,
				PromptTokensStdDev: 100,
				PromptTokensMin:    32,
				PromptTokensMax:    512,
				OutputTokens:       128,
				OutputTokensStdDev: 64,
				OutputTokensMin:    16,
				OutputTokensMax:    256,
			}

			cs := NewClusterSimulator(config, workload, "")
			mustRun(t, cs)

			agg := cs.AggregatedMetrics()
			injected := len(agg.Requests)

			// BC-3: Liveness — no requests stranded
			if agg.StillQueued != 0 {
				t.Errorf("liveness: still_queued = %d, want 0 (scheduler %s)", agg.StillQueued, tc.scheduler)
			}
			if agg.StillRunning != 0 {
				t.Errorf("liveness: still_running = %d, want 0 (scheduler %s)", agg.StillRunning, tc.scheduler)
			}

			// BC-4: Conservation + liveness → all complete
			if agg.CompletedRequests != injected {
				t.Errorf("conservation+liveness: completed = %d, injected = %d (scheduler %s)",
					agg.CompletedRequests, injected, tc.scheduler)
			}
		})
	}
}
