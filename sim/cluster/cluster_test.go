package cluster

import (
	"bytes"
	"encoding/json"
	"math"
	"reflect"
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// newTestDeploymentConfig creates a DeploymentConfig suitable for testing.
func newTestDeploymentConfig(numInstances int) DeploymentConfig {
	return DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 1, false),
		},
		NumInstances: numInstances,
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
	cs := NewClusterSimulator(config, newTestRequests(5))
	cs.PerInstanceMetrics() // should panic
}

// TestDeploymentConfig_ToSimConfig_ReturnsEmbeddedSimConfig verifies that
// ToSimConfig() returns exactly the embedded SimConfig (BC-1).
func TestDeploymentConfig_ToSimConfig_ReturnsEmbeddedSimConfig(t *testing.T) {
	dc := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             999,
			Seed:                7,
			KVCacheConfig:       sim.NewKVCacheConfig(500, 32, 0, 0, 0, 42),
			BatchConfig:         sim.NewBatchConfig(128, 4096, 512),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1, 2, 3}, []float64{4, 5, 6}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 2, true),
			PolicyConfig:        sim.NewPolicyConfig("slo-based", "priority-fcfs"),
		},
		NumInstances:    3,
		AdmissionPolicy: "token-bucket",
		TraceLevel:      "decisions",
	}

	sc := dc.ToSimConfig()

	// BC-1: ToSimConfig returns exactly the embedded SimConfig
	// Note: SimConfig contains slices (BetaCoeffs, AlphaCoeffs) so direct
	// == comparison won't compile. Use reflect.DeepEqual instead.
	if !reflect.DeepEqual(sc, dc.SimConfig) {
		t.Errorf("ToSimConfig() differs from embedded SimConfig:\n  got:  %+v\n  want: %+v", sc, dc.SimConfig)
	}

	// BC-4: WorkloadConfig is an empty struct (cluster generates workload centrally)
	if sc.WorkloadConfig != (sim.WorkloadConfig{}) {
		t.Error("WorkloadConfig should be zero-valued (workload generated centrally)")
	}
}

// TestDeploymentConfig_NoFieldShadowing verifies that no directly-declared
// DeploymentConfig field shares a name with any SimConfig field (BC-6).
// After SimConfig decomposition, this recursively collects promoted field names
// from embedded sub-configs (KVCacheConfig, BatchConfig, etc.).
func TestDeploymentConfig_NoFieldShadowing(t *testing.T) {
	dcType := reflect.TypeOf(DeploymentConfig{})
	scType := reflect.TypeOf(sim.SimConfig{})

	// Recursively collect all field names from SimConfig (including promoted from embedded structs)
	simFields := make(map[string]bool)
	var collectFields func(t reflect.Type)
	collectFields = func(t reflect.Type) {
		for i := 0; i < t.NumField(); i++ {
			field := t.Field(i)
			if field.Anonymous {
				collectFields(field.Type)
			} else {
				simFields[field.Name] = true
			}
		}
	}
	collectFields(scType)

	// Check each directly-declared DeploymentConfig field (skip embedded SimConfig)
	for i := 0; i < dcType.NumField(); i++ {
		field := dcType.Field(i)
		if field.Anonymous {
			continue // skip the embedded SimConfig itself
		}
		if simFields[field.Name] {
			t.Errorf("DeploymentConfig field %q shadows SimConfig field — use promoted access instead", field.Name)
		}
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
				SimConfig: sim.SimConfig{
					Horizon:             math.MaxInt64,
					Seed:                tc.Seed,
					KVCacheConfig:       sim.NewKVCacheConfig(tc.TotalKVBlocks, tc.BlockSizeInTokens, 0, 0, 0, 0),
					BatchConfig:         sim.NewBatchConfig(tc.MaxNumRunningReqs, tc.MaxNumScheduledTokens, tc.LongPrefillTokenThreshold),
					LatencyCoeffs:       sim.NewLatencyCoeffs(tc.BetaCoeffs, tc.AlphaCoeffs),
					ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, tc.Model, tc.Hardware, tc.TP, false),
				},
				NumInstances: 1,
			}

			requests := testGenerateRequests(tc.Seed, math.MaxInt64, tc.Rate/1e6,
				tc.NumRequests, tc.PrefixTokens,
				tc.PromptTokens, tc.PromptTokensStdev, tc.PromptTokensMin, tc.PromptTokensMax,
				tc.OutputTokens, tc.OutputTokensStdev, tc.OutputTokensMin, tc.OutputTokensMax)

			cs := NewClusterSimulator(config, requests)
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

// TestClusterSimulator_SingleInstance_GoldenInvariants verifies R7 companion:
// GIVEN each golden dataset test case configured as NumInstances=1
// WHEN Run() completes
// THEN INV-1 (conservation), INV-5 (causality) hold for every test case.
func TestClusterSimulator_SingleInstance_GoldenInvariants(t *testing.T) {
	dataset := testutil.LoadGoldenDataset(t)

	for _, tc := range dataset.Tests {
		t.Run(tc.Model+"_invariants", func(t *testing.T) {
			config := DeploymentConfig{
				SimConfig: sim.SimConfig{
					Horizon:             math.MaxInt64,
					Seed:                tc.Seed,
					KVCacheConfig:       sim.NewKVCacheConfig(tc.TotalKVBlocks, tc.BlockSizeInTokens, 0, 0, 0, 0),
					BatchConfig:         sim.NewBatchConfig(tc.MaxNumRunningReqs, tc.MaxNumScheduledTokens, tc.LongPrefillTokenThreshold),
					LatencyCoeffs:       sim.NewLatencyCoeffs(tc.BetaCoeffs, tc.AlphaCoeffs),
					ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, tc.Model, tc.Hardware, tc.TP, false),
				},
				NumInstances: 1,
			}

			requests := testGenerateRequests(tc.Seed, math.MaxInt64, tc.Rate/1e6,
				tc.NumRequests, tc.PrefixTokens,
				tc.PromptTokens, tc.PromptTokensStdev, tc.PromptTokensMin, tc.PromptTokensMax,
				tc.OutputTokens, tc.OutputTokensStdev, tc.OutputTokensMin, tc.OutputTokensMax)

			cs := NewClusterSimulator(config, requests)
			mustRun(t, cs)
			m := cs.AggregatedMetrics()

			// INV-1: Request conservation — compare against tc.NumRequests (independent source).
			conservation := m.CompletedRequests + m.StillQueued + m.StillRunning + m.DroppedUnservable
			if conservation != tc.NumRequests {
				t.Errorf("INV-1 conservation: completed(%d) + queued(%d) + running(%d) + dropped(%d) = %d, want numRequests(%d)",
					m.CompletedRequests, m.StillQueued, m.StillRunning, m.DroppedUnservable,
					conservation, tc.NumRequests)
			}

			// INV-5: Causality — TTFT >= 0 and E2E >= TTFT for all completed requests
			for reqID, ttft := range m.RequestTTFTs {
				if ttft < 0 {
					t.Errorf("INV-5 causality: request %s TTFT = %f < 0", reqID, ttft)
				}
				if e2e, ok := m.RequestE2Es[reqID]; ok {
					if e2e < ttft {
						t.Errorf("INV-5 causality: request %s E2E(%f) < TTFT(%f)", reqID, e2e, ttft)
					}
				}
			}
		})
	}
}

// TestClusterSimulator_MultiInstance_Determinism verifies BC-2:
// GIVEN N=4, seed=42, 100 requests
// WHEN run twice
// THEN per-instance and aggregated CompletedRequests are identical.
func TestClusterSimulator_MultiInstance_Determinism(t *testing.T) {
	config := newTestDeploymentConfig(4)

	cs1 := NewClusterSimulator(config, newTestRequests(100))
	mustRun(t, cs1)

	cs2 := NewClusterSimulator(config, newTestRequests(100))
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
	requests := newTestRequests(100)

	cs := NewClusterSimulator(config, requests)
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
	requests := newTestRequests(9)

	cs := NewClusterSimulator(config, requests)
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
	requests := newTestRequests(10)

	cs := NewClusterSimulator(config, requests)
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
	requests := newTestRequests(2)

	cs := NewClusterSimulator(config, requests)
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
	requests := newTestRequests(50)

	cs := NewClusterSimulator(config, requests)
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
	requests := newTestRequests(50)

	cs := NewClusterSimulator(config, requests)
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
	requests := newTestRequests(10)

	cs := NewClusterSimulator(config, requests)
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
	NewClusterSimulator(config, newTestRequests(10))
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
	requests := newTestRequests(20)

	cs := NewClusterSimulator(config, requests)
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
	requests := newTestRequests(100)

	cs := NewClusterSimulator(config, requests)
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

// TestClusterSimulator_NilRequests_NoPanic verifies that nil/empty requests are accepted
// (the new constructor no longer panics on nil requests — it simply produces zero arrivals).
func TestClusterSimulator_NilRequests_NoPanic(t *testing.T) {
	config := newTestDeploymentConfig(2)

	// nil requests: should not panic
	cs := NewClusterSimulator(config, nil)
	mustRun(t, cs)

	if cs.AggregatedMetrics().CompletedRequests != 0 {
		t.Errorf("expected 0 completed requests with nil requests, got %d",
			cs.AggregatedMetrics().CompletedRequests)
	}

	// empty requests: should not panic
	cs2 := NewClusterSimulator(config, []*sim.Request{})
	mustRun(t, cs2)

	if cs2.AggregatedMetrics().CompletedRequests != 0 {
		t.Errorf("expected 0 completed requests with empty requests, got %d",
			cs2.AggregatedMetrics().CompletedRequests)
	}
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
	cs := NewClusterSimulator(config, newTestRequests(10))
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
	requests := newTestRequests(15)

	cs := NewClusterSimulator(config, requests)
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
	requests := newTestRequests(5)
	cs := NewClusterSimulator(config, requests)
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
	requests := newTestRequests(10)

	cs := NewClusterSimulator(config, requests)
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
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	if cs.AggregatedMetrics().CompletedRequests == 0 {
		t.Errorf("Expected non-zero completed requests, got 0")
	}
}

// TestClusterSimulator_AllRoutingPolicies_Smoke verifies all policies are exercisable.
func TestClusterSimulator_AllRoutingPolicies_Smoke(t *testing.T) {
	policies := []string{"round-robin", "least-loaded", "weighted"}

	for _, policyName := range policies {
		t.Run(policyName, func(t *testing.T) {
			config := newTestDeploymentConfig(2)
			config.RoutingPolicy = policyName
			config.RoutingScorerConfigs = sim.DefaultScorerConfigs()
			requests := newTestRequests(5)

			cs := NewClusterSimulator(config, requests)
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
	for i := 0; i < b.N; i++ {
		requests := newTestRequests(1000)
		cs := NewClusterSimulator(config, requests)
		if err := cs.Run(); err != nil {
			b.Fatalf("cs.Run: %v", err)
		}
	}
}

func BenchmarkClusterSimulator_10K_4Instances(b *testing.B) {
	config := newTestDeploymentConfig(4)
	for i := 0; i < b.N; i++ {
		requests := newTestRequests(10000)
		cs := NewClusterSimulator(config, requests)
		if err := cs.Run(); err != nil {
			b.Fatalf("cs.Run: %v", err)
		}
	}
}

func BenchmarkClusterSimulator_1K_10Instances(b *testing.B) {
	config := newTestDeploymentConfig(10)
	for i := 0; i < b.N; i++ {
		requests := newTestRequests(1000)
		cs := NewClusterSimulator(config, requests)
		if err := cs.Run(); err != nil {
			b.Fatalf("cs.Run: %v", err)
		}
	}
}

func TestAggregateMetrics_IncludesKVCacheFields(t *testing.T) {
	// GIVEN a cluster simulation with 2 instances
	cfg := newTestDeploymentConfig(2)
	cs := NewClusterSimulator(cfg, newTestRequests(10))
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
	cs := NewClusterSimulator(cfg, newTestRequests(5))
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
	requests := newTestRequests(100)

	cs := NewClusterSimulator(config, requests)
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
	requests := newTestRequests(50)

	cs := NewClusterSimulator(config, requests)
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
	requests := newTestRequests(100)

	cs := NewClusterSimulator(config, requests)
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
		requests := newTestRequests(50)
		cs := NewClusterSimulator(config, requests)
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

// meanMapValues computes the arithmetic mean of all values in a map.
// Panics on empty map (test infrastructure — should never be empty).
func meanMapValues(m map[string]float64) float64 {
	if len(m) == 0 {
		panic("meanMapValues: empty map")
	}
	sum := 0.0
	for _, v := range m {
		sum += v
	}
	return sum / float64(len(m))
}

// TestClusterSimulator_Conservation_PolicyMatrix verifies INV-1 at cluster level
// across 10 policy combinations (promoted from H12 hypothesis experiment):
// GIVEN each policy combination with infinite horizon and ample resources
// WHEN the cluster simulation completes
// THEN completed + still_queued + still_running == len(Requests) (map-based conservation)
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

			requests := newTestRequests(numRequests)
			cs := NewClusterSimulator(config, requests)
			mustRun(t, cs)

			agg := cs.AggregatedMetrics()
			injected := len(agg.Requests)

			// INV-1 conservation (map-based): len(Requests) == completed + queued + running.
			// Three-term because dropped requests are deleted from the Requests map.
			// The four-term formula (including dropped) is verified via InjectedRequests
			// in TestSaveResults_DroppedUnservable_InJSON.
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

// TestClusterSimulator_Determinism_WeightedPrefixScorer_ByteIdentical verifies INV-6
// for weighted routing with stateful scorers that use internal maps (promoted from H13):
// GIVEN identical config with weighted routing (includes prefix-affinity scorer)
// WHEN run twice with same seed
// THEN per-request metrics JSON is byte-identical.
//
// This specifically targets the PrefixCacheIndex LRU which uses map iteration internally.
// Non-deterministic map iteration in scoring or eviction would cause divergence here.
func TestClusterSimulator_Determinism_WeightedPrefixScorer_ByteIdentical(t *testing.T) {
	policies := []struct {
		name          string
		routingPolicy string
		scorerConfigs []sim.ScorerConfig
	}{
		{"weighted-default", "weighted", sim.DefaultScorerConfigs()},
	}

	for _, pol := range policies {
		t.Run(pol.name, func(t *testing.T) {
			mkSim := func() *ClusterSimulator {
				config := newTestDeploymentConfig(3)
				config.RoutingPolicy = pol.routingPolicy
				config.RoutingScorerConfigs = pol.scorerConfigs
				// Use prefix tokens to exercise the prefix cache index
				requests := testGenerateRequests(42, math.MaxInt64, 10.0/1e6, 30,
					32, 100, 20, 10, 200, 50, 10, 10, 100)
				cs := NewClusterSimulator(config, requests)
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

			requests := testGenerateRequests(42, math.MaxInt64, rateReqPerS/1e6, numRequests,
				0, 100, 20, 10, 200, 50, 10, 10, 100)

			cs := NewClusterSimulator(config, requests)
			mustRun(t, cs)

			agg := cs.AggregatedMetrics()
			injected := len(agg.Requests)
			rejected := cs.RejectedRequests()

			// INV-1 conservation (map-based): len(Requests) == completed + queued + running.
			// Three-term because dropped requests are deleted from the Requests map.
			conservation := agg.CompletedRequests + agg.StillQueued + agg.StillRunning
			if tc.admissionPolicy == "always-admit" {
				// No rejections expected
				if conservation != injected {
					t.Errorf("INV-1 conservation (always-admit): completed(%d) + queued(%d) + running(%d) = %d, want %d (injected)",
						agg.CompletedRequests, agg.StillQueued, agg.StillRunning, conservation, injected)
				}
				if rejected != 0 {
					t.Errorf("always-admit should have 0 rejections, got %d", rejected)
				}
			} else {
				// Pipeline conservation: injected + rejected == total generated
				totalGenerated := injected + rejected
				if conservation != injected {
					t.Errorf("INV-1 conservation (token-bucket): completed(%d) + queued(%d) + running(%d) = %d, want %d (injected)",
						agg.CompletedRequests, agg.StillQueued, agg.StillRunning, conservation, injected)
				}
				if totalGenerated != numRequests {
					t.Errorf("pipeline conservation: injected(%d) + rejected(%d) = %d, want %d (total generated)",
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
			requests := testGenerateRequests(42, math.MaxInt64, rateReqPerS/1e6, numRequests,
				0, 200, 100, 32, 512, 128, 64, 16, 256)

			cs := NewClusterSimulator(config, requests)
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

// TestClusterSimulator_AdmissionLatency_ExactOffset verifies that admission
// latency creates an exact additive offset in TTFT and E2E
// (promoted from H26 experiment, PR #372, issue #378):
// GIVEN constant token lengths, low rate (no queuing), and deterministic seed
// WHEN the cluster runs with AdmissionLatency=0, 10000 (10ms), and 50000 (50ms)
// THEN TTFT and E2E deltas MUST match the admission latency exactly (within 0.1ms)
// AND the linearity ratio (50ms/10ms) MUST equal 5.0 (within 0.01).
func TestClusterSimulator_AdmissionLatency_ExactOffset(t *testing.T) {
	const (
		numRequests  = 50
		numInstances = 4
		rateReqPerS  = 10.0
		inputTokens  = 128
		outputTokens = 32
	)

	// Constant tokens (zero stddev) eliminates variance.
	mkRequests := func() []*sim.Request {
		return testGenerateRequests(42, math.MaxInt64, rateReqPerS/1e6, numRequests,
			0, inputTokens, 0, inputTokens, inputTokens, outputTokens, 0, outputTokens, outputTokens)
	}

	runWithLatency := func(latencyUS int64) *sim.Metrics {
		config := newTestDeploymentConfig(numInstances)
		config.RoutingPolicy = "least-loaded"
		config.AdmissionLatency = latencyUS
		cs := NewClusterSimulator(config, mkRequests())
		mustRun(t, cs)
		return cs.AggregatedMetrics()
	}

	mA := runWithLatency(0)      // baseline
	mB := runWithLatency(10000)  // 10ms
	mC := runWithLatency(50000)  // 50ms

	// Compute mean TTFT and E2E (in ticks/microseconds), convert to ms
	ttftA := meanMapValues(mA.RequestTTFTs) / 1000.0
	ttftB := meanMapValues(mB.RequestTTFTs) / 1000.0
	ttftC := meanMapValues(mC.RequestTTFTs) / 1000.0

	e2eA := meanMapValues(mA.RequestE2Es) / 1000.0
	e2eB := meanMapValues(mB.RequestE2Es) / 1000.0
	e2eC := meanMapValues(mC.RequestE2Es) / 1000.0

	// BC-1: TTFT and E2E deltas must match admission latency (within 0.1ms)
	const tol = 0.1 // ms

	ttftDeltaB := ttftB - ttftA
	e2eDeltaB := e2eB - e2eA
	if math.Abs(ttftDeltaB-10.0) > tol {
		t.Errorf("BC-1 TTFT delta (10ms latency): got %.4f ms, want 10.0 ± %.1f ms", ttftDeltaB, tol)
	}
	if math.Abs(e2eDeltaB-10.0) > tol {
		t.Errorf("BC-1 E2E delta (10ms latency): got %.4f ms, want 10.0 ± %.1f ms", e2eDeltaB, tol)
	}

	ttftDeltaC := ttftC - ttftA
	e2eDeltaC := e2eC - e2eA
	if math.Abs(ttftDeltaC-50.0) > tol {
		t.Errorf("BC-1 TTFT delta (50ms latency): got %.4f ms, want 50.0 ± %.1f ms", ttftDeltaC, tol)
	}
	if math.Abs(e2eDeltaC-50.0) > tol {
		t.Errorf("BC-1 E2E delta (50ms latency): got %.4f ms, want 50.0 ± %.1f ms", e2eDeltaC, tol)
	}

	// BC-2: Linearity check — 50ms/10ms ratio must be 5.0
	if e2eDeltaB > 0 {
		ratio := e2eDeltaC / e2eDeltaB
		if math.Abs(ratio-5.0) > 0.01 {
			t.Errorf("BC-2 linearity: E2E delta ratio (50ms/10ms) = %.4f, want 5.0 ± 0.01", ratio)
		}
	} else {
		t.Error("BC-2: E2E delta for 10ms config is <= 0, cannot check linearity")
	}

	// Sanity: all requests completed in all configs
	if mA.CompletedRequests != numRequests {
		t.Errorf("baseline: completed %d, want %d", mA.CompletedRequests, numRequests)
	}
	if mB.CompletedRequests != numRequests {
		t.Errorf("10ms config: completed %d, want %d", mB.CompletedRequests, numRequests)
	}
	if mC.CompletedRequests != numRequests {
		t.Errorf("50ms config: completed %d, want %d", mC.CompletedRequests, numRequests)
	}
}

// TestClusterSimulator_FullStackConservation verifies INV-1 conservation
// across the full policy stack: weighted routing + admission control +
// priority scheduling (promoted from H25 experiment, PR #372, issue #379):
// GIVEN weighted routing (prefix-affinity:3,queue-depth:2,kv-utilization:2),
//
//	priority-FCFS scheduling, and multiple admission/KV configurations
//
// WHEN the simulation completes
// THEN conservation holds: completed + still_queued + still_running == len(Requests)
// AND preemptions are triggered in the constrained-KV config (stress path exercised)
// AND pipeline conservation holds for token-bucket: len(Requests) + rejected == total.
func TestClusterSimulator_FullStackConservation(t *testing.T) {
	const (
		numRequests  = 50
		numInstances = 4
		rateReqPerS  = 200.0
	)

	mkRequests := func() []*sim.Request {
		return testGenerateRequests(42, math.MaxInt64, rateReqPerS/1e6, numRequests,
			32, 128, 32, 32, 256, 64, 16, 16, 128)
	}

	mkFullStackConfig := func() DeploymentConfig {
		config := newTestDeploymentConfig(numInstances)
		config.RoutingPolicy = "weighted"
		config.RoutingScorerConfigs = sim.DefaultScorerConfigs()
		config.Scheduler = "priority-fcfs"
		config.PriorityPolicy = "slo-based"
		config.AdmissionPolicy = "always-admit"
		return config
	}

	t.Run("always-admit/ample-kv", func(t *testing.T) {
		// BC-3: Happy path — all modules active, ample resources
		config := mkFullStackConfig()
		cs := NewClusterSimulator(config, mkRequests())
		mustRun(t, cs)

		agg := cs.AggregatedMetrics()
		injected := len(agg.Requests)

		// INV-1 conservation (map-based three-term)
		conservation := agg.CompletedRequests + agg.StillQueued + agg.StillRunning
		if conservation != injected {
			t.Errorf("INV-1: completed(%d) + queued(%d) + running(%d) = %d, want %d (injected)",
				agg.CompletedRequests, agg.StillQueued, agg.StillRunning, conservation, injected)
		}

		// All requests complete under infinite horizon with ample resources
		if agg.CompletedRequests != numRequests {
			t.Errorf("expected all %d requests to complete, got %d", numRequests, agg.CompletedRequests)
		}

		// No requests dropped as unservable (ample KV)
		if agg.DroppedUnservable != 0 {
			t.Errorf("expected 0 DroppedUnservable with ample KV, got %d", agg.DroppedUnservable)
		}
	})

	t.Run("always-admit/constrained-kv", func(t *testing.T) {
		// BC-4: Stress path — constrained KV blocks force preemptions.
		// Uses high rate (2000/s) with finite horizon to keep many requests in-flight.
		// TotalKVBlocks=50 per instance with 10 blocks/request means only ~5 concurrent
		// requests can hold KV. With high arrival rate, batch formation tries to schedule
		// more, triggering preemptions. MaxRunningReqs=256 (default) allows large batches.
		// 50 >= 10 (max single request input blocks: ceil((32+128)/16)) so no DroppedUnservable.
		config := mkFullStackConfig()
		config.TotalKVBlocks = 50
		config.BlockSizeTokens = 16
		config.Horizon = 500000 // 0.5 seconds — many requests still in-flight at end
		constRequests := testGenerateRequests(42, math.MaxInt64, 2000.0/1e6, numRequests,
			32, 128, 0, 128, 128, 64, 0, 64, 64)
		cs := NewClusterSimulator(config, constRequests)
		mustRun(t, cs)

		agg := cs.AggregatedMetrics()
		injected := len(agg.Requests)

		// INV-1 conservation (map-based three-term)
		conservation := agg.CompletedRequests + agg.StillQueued + agg.StillRunning
		if conservation != injected {
			t.Errorf("INV-1: completed(%d) + queued(%d) + running(%d) = %d, want %d (injected)",
				agg.CompletedRequests, agg.StillQueued, agg.StillRunning, conservation, injected)
		}

		// Verify stress path is actually exercised: preemptions must occur
		if agg.PreemptionCount == 0 {
			t.Error("expected preemptions with constrained batch+KV (50 blocks per instance) at rate=2000, got 0 — test is not exercising the stress path")
		}

		// Verify no requests dropped as unservable (max input = ceil((32+128)/16) = 10 blocks ≤ 50)
		if agg.DroppedUnservable != 0 {
			t.Errorf("expected 0 DroppedUnservable with 50 blocks per instance (max request needs 10 blocks), got %d", agg.DroppedUnservable)
		}
	})

	t.Run("token-bucket", func(t *testing.T) {
		// BC-5: Pipeline conservation with admission rejections
		config := mkFullStackConfig()
		config.AdmissionPolicy = "token-bucket"
		config.TokenBucketCapacity = 500
		config.TokenBucketRefillRate = 300
		cs := NewClusterSimulator(config, mkRequests())
		mustRun(t, cs)

		agg := cs.AggregatedMetrics()
		injected := len(agg.Requests)
		rejected := cs.RejectedRequests()

		// INV-1 conservation (map-based three-term)
		conservation := agg.CompletedRequests + agg.StillQueued + agg.StillRunning
		if conservation != injected {
			t.Errorf("INV-1: completed(%d) + queued(%d) + running(%d) = %d, want %d (injected)",
				agg.CompletedRequests, agg.StillQueued, agg.StillRunning, conservation, injected)
		}

		// Pipeline conservation: injected + rejected == total generated
		if injected+rejected != numRequests {
			t.Errorf("pipeline conservation: injected(%d) + rejected(%d) = %d, want %d",
				injected, rejected, injected+rejected, numRequests)
		}

		// Sanity: token-bucket should reject some requests (not all admitted)
		if rejected == 0 {
			t.Error("expected some rejections with token-bucket(cap=500,refill=300) at rate=200, got 0")
		}
	})
}
