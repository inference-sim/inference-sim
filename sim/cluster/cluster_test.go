package cluster

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// newTestDeploymentConfig creates a DeploymentConfig suitable for testing.
func newTestDeploymentConfig(numInstances int) DeploymentConfig {
	return DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyModel:        testFakeLatency(),
			ModelHardwareConfig: sim.NewModelHardwareConfig(testModelConfig(), "test-model", "H100", 1, 1, false, 0),
		},
		NumInstances: numInstances,
	}
}

// mustRun is a test helper that calls Run and fails the test on error.
// mustRun takes the slice of testing.TB that *rapid.T also provides, so properties use it too.
func mustRun(t interface {
	Helper()
	Fatalf(string, ...any)
}, cs *ClusterSimulator) {
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
	cs := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(5)), nil)
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
			LatencyModel:        testFakeLatency(),
			ModelHardwareConfig: sim.NewModelHardwareConfig(testModelConfig(), "test-model", "H100", 2, 1, false, 0),
			PolicyConfig:        sim.NewPolicyConfig("priority-fcfs", ""),
		},
		NumInstances: 3,
		TraceLevel:   "decisions",
	}

	sc := dc.ToSimConfig()

	// BC-1: ToSimConfig returns exactly the embedded SimConfig
	// Note: SimConfig contains slices and maps, so direct == comparison won't
	// compile. Use reflect.DeepEqual instead.
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
					BatchConfig:         sim.NewBatchConfig(tc.MaxNumSeqs, tc.MaxNumBatchedTokens, tc.LongPrefillTokenThreshold),
					LatencyModel:        testFakeLatency(),
					ModelHardwareConfig: sim.NewModelHardwareConfig(testModelConfig(), tc.Model, tc.Hardware, tc.TP, 1, false, 0),
				},
				NumInstances: 1,
			}

			requests := testGenerateRequests(tc.Seed, math.MaxInt64, tc.Rate/1e6,
				tc.NumRequests, tc.PrefixTokens,
				tc.PromptTokens, tc.PromptTokensStdev, tc.PromptTokensMin, tc.PromptTokensMax,
				tc.OutputTokens, tc.OutputTokensStdev, tc.OutputTokensMin, tc.OutputTokensMax)

			cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
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
			testutil.AssertFloat64Equal(t, "vllm_estimated_duration_s",
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
					BatchConfig:         sim.NewBatchConfig(tc.MaxNumSeqs, tc.MaxNumBatchedTokens, tc.LongPrefillTokenThreshold),
					LatencyModel:        testFakeLatency(),
					ModelHardwareConfig: sim.NewModelHardwareConfig(testModelConfig(), tc.Model, tc.Hardware, tc.TP, 1, false, 0),
				},
				NumInstances: 1,
			}

			requests := testGenerateRequests(tc.Seed, math.MaxInt64, tc.Rate/1e6,
				tc.NumRequests, tc.PrefixTokens,
				tc.PromptTokens, tc.PromptTokensStdev, tc.PromptTokensMin, tc.PromptTokensMax,
				tc.OutputTokens, tc.OutputTokensStdev, tc.OutputTokensMin, tc.OutputTokensMax)

			cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
			mustRun(t, cs)
			m := cs.AggregatedMetrics()

			// INV-1: Request conservation — compare against tc.NumRequests (independent source).
			assertClusterINV1Conservation(t, cs, tc.NumRequests, noRejections, tc.Model+"/"+tc.Workload)

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

	cs1 := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(100)), nil)
	mustRun(t, cs1)

	cs2 := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(100)), nil)
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

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
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

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
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

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
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

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
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

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
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

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
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

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
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
	NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(10)), nil)
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
		ID: "req", ArrivalTime: 0, InputTokens: make([]sim.TokenID, 5),
		OutputTokens: make([]sim.TokenID, 3), State: sim.StateQueued,
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

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
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

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
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
	cs := NewClusterSimulator(config, NewSliceRequestSource(nil), nil)
	mustRun(t, cs)

	if cs.AggregatedMetrics().CompletedRequests != 0 {
		t.Errorf("expected 0 completed requests with nil requests, got %d",
			cs.AggregatedMetrics().CompletedRequests)
	}

	// empty requests: should not panic
	cs2 := NewClusterSimulator(config, NewSliceRequestSource([]*sim.Request{}), nil)
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
	cs := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(10)), nil)
	cs.AggregatedMetrics()
}

// TestClusterSimulator_HandledBy_PopulatedInMetrics verifies #181:
// GIVEN a 3-instance cluster and 15 requests
// WHEN the simulation completes
// THEN every completed request's metrics has a non-empty HandledBy field
// AND each HandledBy value matches a valid instance ID
// AND per-instance metrics only contain requests handled by that instance
func TestClusterSimulator_HandledBy_PopulatedInMetrics(t *testing.T) {
	config := newTestDeploymentConfig(3)
	requests := newTestRequests(15)

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
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
	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
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

// === Benchmarks ===

func BenchmarkClusterSimulator_1K_1Instance(b *testing.B) {
	config := newTestDeploymentConfig(1)
	for i := 0; i < b.N; i++ {
		requests := newTestRequests(1000)
		cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
		if err := cs.Run(); err != nil {
			b.Fatalf("cs.Run: %v", err)
		}
	}
}

func BenchmarkClusterSimulator_10K_4Instances(b *testing.B) {
	config := newTestDeploymentConfig(4)
	for i := 0; i < b.N; i++ {
		requests := newTestRequests(10000)
		cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
		if err := cs.Run(); err != nil {
			b.Fatalf("cs.Run: %v", err)
		}
	}
}

func BenchmarkClusterSimulator_1K_10Instances(b *testing.B) {
	config := newTestDeploymentConfig(10)
	for i := 0; i < b.N; i++ {
		requests := newTestRequests(1000)
		cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
		if err := cs.Run(); err != nil {
			b.Fatalf("cs.Run: %v", err)
		}
	}
}

func TestAggregateMetrics_IncludesKVCacheFields(t *testing.T) {
	// GIVEN a cluster simulation with 2 instances
	cfg := newTestDeploymentConfig(2)
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(newTestRequests(10)), nil)
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
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(newTestRequests(5)), nil)
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

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
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

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
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

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
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
		cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
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
// across scheduler and cluster-size combinations (promoted from H12 hypothesis experiment):
// GIVEN each combination with infinite horizon and ample resources
// WHEN the cluster simulation completes
// THEN the eleven-term conservation equation holds against an independent count, with
// no rejections permitted,
// AND all requests complete (infinite horizon, no resource pressure).
func TestClusterSimulator_Conservation_PolicyMatrix(t *testing.T) {
	matrix := []struct {
		name         string
		numInstances int
		scheduler    string
	}{
		{"fcfs/2inst", 2, "fcfs"},
		{"fcfs/3inst", 3, "fcfs"},
		{"fcfs/4inst", 4, "fcfs"},
		{"sjf/3inst", 3, "sjf"},
		{"sjf/4inst", 4, "sjf"},
		{"priority-fcfs/2inst", 2, "priority-fcfs"},
		{"priority-fcfs/3inst", 3, "priority-fcfs"},
	}

	const numRequests = 50

	for _, tc := range matrix {
		t.Run(tc.name, func(t *testing.T) {
			config := newTestDeploymentConfig(tc.numInstances)
			config.Scheduler = tc.scheduler

			requests := newTestRequests(numRequests)
			cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
			mustRun(t, cs)

			agg := cs.AggregatedMetrics()
			assertClusterINV1Conservation(t, cs, numRequests, noRejections, "infinite horizon")

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

// TestClusterSimulator_OverloadConservation verifies INV-1 under 10x overload
// (promoted from H-Overload hypothesis experiment, PR #335):
// GIVEN a 4-instance cluster at extreme overload rate
// WHEN the simulation runs to a finite horizon
// THEN conservation holds:
//   - admit-all: completed + still_queued + still_running == injected
//   - reject-half: completed + still_queued + still_running + rejected == total_generated
//
// AND no panics occur (BC-5).
func TestClusterSimulator_OverloadConservation(t *testing.T) {
	// Use a high rate relative to capacity to create genuine overload.
	// With 4 instances and max-running=256, capacity is very high due to batching. A rate of 500 req/s with only 200 requests and a short
	// horizon creates a burst that overloads the system.
	cases := []struct {
		name    string
		gateway func() *fakeGateway
	}{
		{"always-admit", func() *fakeGateway { return &fakeGateway{} }},
		{"reject-half", rejectEveryOtherGateway},
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
			config.MaxNumSeqs = maxRunning
			config.Scheduler = "fcfs"

			requests := testGenerateRequests(42, math.MaxInt64, rateReqPerS/1e6, numRequests,
				0, 100, 20, 10, 200, 50, 10, 10, 100)

			cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
			cs.SetGateway(tc.gateway())
			mustRun(t, cs)

			agg := cs.AggregatedMetrics()

			assertClusterINV1Conservation(t, cs, numRequests, cs.RejectedRequests(), tc.name)

			// The full-pipeline clause, against an observation independent of the
			// counters: the Metrics.Requests map. The helper cannot do this — it
			// folds the pipeline clause into the same equality, and the map is not a
			// general baseline: the drop guards, drain redirect and PD parent collapse
			// delete from it (undercount), while the timeout paths do not delete at all
			// (overcount). Sound here only because this fixture does neither, which the
			// guards below pin.
			if agg.DroppedUnservable != 0 {
				t.Fatalf("fixture now drops %d requests, so len(Requests) undercounts and is no longer a valid injected count", agg.DroppedUnservable)
			}
			if agg.TimedOutRequests != 0 {
				t.Fatalf("fixture now times out %d requests, which stay in Requests, so len(Requests) overcounts and is no longer a valid injected count", agg.TimedOutRequests)
			}
			if injected := len(agg.Requests); injected+cs.RejectedRequests() != numRequests {
				t.Errorf("INV-1 pipeline: len(Requests)=%d + rejected=%d = %d, want %d generated",
					injected, cs.RejectedRequests(), injected+cs.RejectedRequests(), numRequests)
			}

			if tc.name == "always-admit" {
				if rejected := cs.RejectedRequests(); rejected != 0 {
					t.Errorf("always-admit should have 0 rejections, got %d", rejected)
				}
			} else if cs.RejectedRequests() == 0 {
				t.Error("reject-half gateway rejected nothing, so the rejected term is untested")
			}

			// Verify overload: under finite horizon, not all requests should complete
			// (this confirms the test is actually exercising overload, not a trivial case)
			if agg.CompletedRequests == numRequests && tc.name == "always-admit" {
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
			config.MaxNumSeqs = maxRunning
			config.Scheduler = tc.scheduler

			// Mixed workload: varying prompt and output sizes to exercise scheduler ordering
			requests := testGenerateRequests(42, math.MaxInt64, rateReqPerS/1e6, numRequests,
				0, 200, 100, 32, 512, 128, 64, 16, 256)

			cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
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

// TestClusterSimulator_GatewayHold_ExactOffset verifies that time a request spends in the
// gateway before dispatch creates an exact additive offset in TTFT and E2E, which are measured
// from arrival (promoted from H26 experiment, PR #372, issue #378):
// GIVEN constant token lengths, low rate (no queuing), and deterministic seed
// WHEN the gateway holds each request 0, 10000 (10ms), and 50000 (50ms) before dispatch
// THEN TTFT and E2E deltas MUST match the hold exactly (within 0.1ms)
// AND the linearity ratio (50ms/10ms) MUST equal 5.0 (within 0.01).
func TestClusterSimulator_GatewayHold_ExactOffset(t *testing.T) {
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
		cs := NewClusterSimulator(config, NewSliceRequestSource(mkRequests()), nil)
		cs.SetGateway(&fakeGateway{holdUs: latencyUS})
		mustRun(t, cs)
		return cs.AggregatedMetrics()
	}

	mA := runWithLatency(0)     // baseline
	mB := runWithLatency(10000) // 10ms
	mC := runWithLatency(50000) // 50ms

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
// across the full stack: gateway rejections + priority scheduling + KV pressure
// (promoted from H25 experiment, PR #372, issue #379):
// GIVEN priority-FCFS scheduling and multiple gateway/KV configurations
// WHEN the simulation completes
// THEN conservation holds: completed + still_queued + still_running == len(Requests)
// AND preemptions are triggered in the constrained-KV config (stress path exercised)
// AND pipeline conservation holds with gateway rejections: len(Requests) + rejected == total.
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
		config.Scheduler = "priority-fcfs"
		return config
	}

	t.Run("always-admit/ample-kv", func(t *testing.T) {
		// BC-3: Happy path — all modules active, ample resources
		config := mkFullStackConfig()
		cs := NewClusterSimulator(config, NewSliceRequestSource(mkRequests()), nil)
		mustRun(t, cs)

		agg := cs.AggregatedMetrics()

		assertClusterINV1Conservation(t, cs, numRequests, noRejections, "always-admit/ample-kv")

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
		// more, triggering preemptions. MaxNumSeqs=256 (default) allows large batches.
		// 50 >= 10 (max single request input blocks: ceil((32+128)/16)) so no DroppedUnservable.
		config := mkFullStackConfig()
		config.TotalKVBlocks = 50
		config.BlockSizeTokens = 16
		config.Horizon = 500000 // 0.5 seconds — many requests still in-flight at end
		constRequests := testGenerateRequests(42, math.MaxInt64, 2000.0/1e6, numRequests,
			32, 128, 0, 128, 128, 64, 0, 64, 64)
		cs := NewClusterSimulator(config, NewSliceRequestSource(constRequests), nil)
		mustRun(t, cs)

		agg := cs.AggregatedMetrics()

		assertClusterINV1Conservation(t, cs, len(constRequests), noRejections, "always-admit/constrained-kv")

		// Verify stress path is actually exercised: preemptions must occur
		if agg.PreemptionCount == 0 {
			t.Error("expected preemptions with constrained batch+KV (50 blocks per instance) at rate=2000, got 0 — test is not exercising the stress path")
		}

		// Verify no requests dropped as unservable (max input = ceil((32+128)/16) = 10 blocks ≤ 50)
		if agg.DroppedUnservable != 0 {
			t.Errorf("expected 0 DroppedUnservable with 50 blocks per instance (max request needs 10 blocks), got %d", agg.DroppedUnservable)
		}
	})

	t.Run("gateway-rejections", func(t *testing.T) {
		// BC-5: Pipeline conservation with admission rejections
		config := mkFullStackConfig()
		cs := NewClusterSimulator(config, NewSliceRequestSource(mkRequests()), nil)
		cs.SetGateway(rejectEveryOtherGateway())
		mustRun(t, cs)

		agg := cs.AggregatedMetrics()
		rejected := cs.RejectedRequests()

		assertClusterINV1Conservation(t, cs, numRequests, rejected, "gateway-rejections")

		// The full-pipeline clause against the Metrics.Requests map, independent of
		// the counters the helper reads. Sound here only because nothing is dropped and
		// nothing times out; see the longer note at the equivalent check in
		// TestClusterSimulator_OverloadConservation.
		if agg.DroppedUnservable != 0 {
			t.Fatalf("fixture now drops %d requests, so len(Requests) undercounts and is no longer a valid injected count", agg.DroppedUnservable)
		}
		if agg.TimedOutRequests != 0 {
			t.Fatalf("fixture now times out %d requests, which stay in Requests, so len(Requests) overcounts and is no longer a valid injected count", agg.TimedOutRequests)
		}
		if injected := len(agg.Requests); injected+rejected != numRequests {
			t.Errorf("INV-1 pipeline: len(Requests)=%d + rejected=%d = %d, want %d generated",
				injected, rejected, injected+rejected, numRequests)
		}

		// Sanity: the gateway rejects every other request
		if rejected != numRequests/2 {
			t.Errorf("expected %d rejections from the reject-every-other gateway, got %d", numRequests/2, rejected)
		}
	})
}

// BC-6: Cluster-mode MaxModelLen drops surface in aggregated metrics with INV-1 conservation.
// Exercises Guard 1a (input >= MaxModelLen) and Guard 1b (input + budget > MaxModelLen).
// Differs from TestClusterSimulator_InFlightRequests_DroppedUnservable_Decrements which tests
// the KV-capacity Guard 2 path.
func TestClusterSimulator_MaxModelLen_DroppedUnservable(t *testing.T) {
	const maxModelLen = 200

	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             10_000_000,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyModel:        testFakeZeroQueueing(),
			ModelHardwareConfig: sim.NewModelHardwareConfig(testModelConfig(), "test", "H100", 1, 1, false, maxModelLen),
		},
		NumInstances: 2,
	}

	rng := sim.NewPartitionedRNG(sim.NewSimulationKey(42))
	workloadRNG := rng.ForSubsystem(sim.SubsystemWorkload)

	var requests []*sim.Request
	numFit, numGuard1a, numGuard1b := 0, 0, 0

	// 5 requests that fit (input=50, output=50, no budget → auto-filled to 150, budget check passes)
	for i := 0; i < 5; i++ {
		req := &sim.Request{
			ID:           fmt.Sprintf("fit_%d", i),
			InputTokens:  sim.GenerateRandomTokenIDs(workloadRNG, 50),
			OutputTokens: sim.GenerateRandomTokenIDs(workloadRNG, 50),
			ArrivalTime:  int64(i * 100_000),
			State:        sim.StateQueued,
		}
		requests = append(requests, req)
		numFit++
	}

	// 3 requests dropped by Guard 1a (input >= MaxModelLen)
	for i := 0; i < 3; i++ {
		req := &sim.Request{
			ID:           fmt.Sprintf("guard1a_%d", i),
			InputTokens:  sim.GenerateRandomTokenIDs(workloadRNG, 200), // input == MaxModelLen → dropped
			OutputTokens: sim.GenerateRandomTokenIDs(workloadRNG, 10),
			ArrivalTime:  int64((5 + i) * 100_000),
			State:        sim.StateQueued,
		}
		requests = append(requests, req)
		numGuard1a++
	}

	// 2 requests dropped by Guard 1b (input + budget > MaxModelLen)
	for i := 0; i < 2; i++ {
		req := &sim.Request{
			ID:           fmt.Sprintf("guard1b_%d", i),
			InputTokens:  sim.GenerateRandomTokenIDs(workloadRNG, 100),
			OutputTokens: sim.GenerateRandomTokenIDs(workloadRNG, 50),
			MaxOutputLen: 150, // 100 + 150 = 250 > MaxModelLen(200)
			ArrivalTime:  int64((8 + i) * 100_000),
			State:        sim.StateQueued,
		}
		requests = append(requests, req)
		numGuard1b++
	}

	totalInjected := numFit + numGuard1a + numGuard1b
	expectedDropped := numGuard1a + numGuard1b

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
	mustRun(t, cs)

	agg := cs.AggregatedMetrics()

	// DroppedUnservable matches expected oversized count
	if agg.DroppedUnservable != expectedDropped {
		t.Errorf("DroppedUnservable = %d, want %d (Guard1a=%d + Guard1b=%d)",
			agg.DroppedUnservable, expectedDropped, numGuard1a, numGuard1b)
	}

	assertClusterINV1Conservation(t, cs, totalInjected, noRejections, "oversized-request guards")

	// Post-simulation drain: all requests completed or dropped, nothing in-flight
	if agg.StillQueued != 0 || agg.StillRunning != 0 {
		t.Errorf("StillQueued=%d, StillRunning=%d; want both 0 (all should complete within horizon)",
			agg.StillQueued, agg.StillRunning)
	}

	// inFlightRequests drains to 0 for each instance (direct access, same package)
	for _, inst := range cs.instances {
		instID := string(inst.ID())
		inflight := cs.inFlightRequests[instID]
		if inflight != 0 {
			t.Errorf("inFlightRequests[%s] = %d, want 0 (should drain after completion+drops)", instID, inflight)
		}
	}

	// Metrics.Requests map excludes dropped request IDs
	for _, req := range requests {
		_, exists := agg.Requests[req.ID]
		isDropped := strings.HasPrefix(req.ID, "guard1")
		if exists && isDropped {
			t.Errorf("Metrics.Requests should NOT contain dropped request %s", req.ID)
		}
		if !exists && !isDropped {
			t.Errorf("Metrics.Requests should contain fit request %s", req.ID)
		}
	}
}

// TestClusterSimulator_GatewayAccessors_ZeroWithoutGatewayQueue verifies the gateway-queue
// accessors read zero when the gateway holds nothing.
func TestClusterSimulator_GatewayAccessors_ZeroWithoutGatewayQueue(t *testing.T) {
	config := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(3)), nil)
	mustRun(t, cs)

	if cs.GatewayQueueDepth() != 0 {
		t.Errorf("GatewayQueueDepth should be 0 when the gateway holds nothing, got %d", cs.GatewayQueueDepth())
	}
	if cs.GatewayQueueRejected() != 0 {
		t.Errorf("GatewayQueueRejected should be 0 when the gateway holds nothing, got %d", cs.GatewayQueueRejected())
	}
}

// TestNewClusterSimulator_UsesPoolGPUType verifies the observable hardware commitment:
// when NodePools are configured, GPU() on each placed instance reflects the pool's
// gpu_type (authoritative), not the CLI --gpu flag.
//
// GPU() is a public behavioral API that exposes the hardware commitment made at construction.
// The refactor-survival test passes: any reimplementation preserving SC-004 behavior
// ("pool gpu_type overrides CLI --gpu when NodePools are present") would produce the same
// GPU() result. The counter-assertion (no NodePools → CLI flag preserved) confirms the
// override is specific to the NodePools construction path, not a general override of the CLI flag.
func TestNewClusterSimulator_UsesPoolGPUType(t *testing.T) {
	// GIVEN: CLI --gpu flag = "H100", pool gpu_type = "A100" — they differ intentionally.
	sharedConfig := sim.SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		ModelHardwareConfig: sim.NewModelHardwareConfig(testModelConfig(), "test-model", "H100", 1, 1, false, 0),
		KVCacheConfig:       sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(4, 2048, 0),
		LatencyModel:        testFakeLatency(),
	}

	// WHEN: NodePools path — pool is authoritative (SC-004).
	withPools := DeploymentConfig{
		SimConfig:    sharedConfig,
		NumInstances: 2,
		NodePools: []NodePoolConfig{
			{Name: "a100-pool", GPUType: "A100", GPUsPerNode: 8, InitialNodes: 2, MaxNodes: 2, GPUMemoryGiB: 80},
		},
		InstanceLifecycle: InstanceLifecycleConfig{},
	}
	cs := NewClusterSimulator(withPools, NewSliceRequestSource(nil), nil)
	if len(cs.instances) != 2 {
		t.Fatalf("expected 2 placed instances, got %d", len(cs.instances))
	}
	// THEN: GPU() = "A100" (pool gpu_type overrides CLI --gpu flag "H100").
	for _, inst := range cs.instances {
		if got := inst.GPU(); got != "A100" {
			t.Errorf("pool path: instance %s: GPU() = %q, want %q (pool gpu_type must override CLI --gpu)", inst.ID(), got, "A100")
		}
	}

	// Counter-assertion: without NodePools, CLI --gpu flag "H100" is preserved unchanged.
	// This confirms the pool override is specific to the NodePools construction path.
	withoutPools := DeploymentConfig{
		SimConfig:         sharedConfig,
		NumInstances:      2,
		InstanceLifecycle: InstanceLifecycleConfig{},
	}
	cs2 := NewClusterSimulator(withoutPools, NewSliceRequestSource(nil), nil)
	if len(cs2.instances) != 2 {
		t.Fatalf("counter-assertion: expected 2 instances, got %d", len(cs2.instances))
	}
	for _, inst := range cs2.instances {
		if got := inst.GPU(); got != "H100" {
			t.Errorf("no-pools path: instance %s: GPU() = %q, want %q (CLI --gpu must be preserved when no NodePools)", inst.ID(), got, "H100")
		}
	}
}

// TestNewClusterSimulator_NoNodePools_DeterminismPreserved verifies INV-6:
// when NodePools is empty, two runs with the same seed produce byte-identical scalar metrics.
func TestNewClusterSimulator_NoNodePools_DeterminismPreserved(t *testing.T) {
	// NodePools intentionally empty (not set in newTestDeploymentConfig)
	config := newTestDeploymentConfig(2)

	cs1 := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(100)), nil)
	mustRun(t, cs1)

	cs2 := NewClusterSimulator(config, NewSliceRequestSource(newTestRequests(100)), nil)
	mustRun(t, cs2)

	m1 := cs1.AggregatedMetrics()
	m2 := cs2.AggregatedMetrics()

	// Compare all scalar fields (INV-6: same seed must produce identical results).
	if m1.CompletedRequests != m2.CompletedRequests {
		t.Errorf("CompletedRequests: run1=%d, run2=%d (INV-6 violation)", m1.CompletedRequests, m2.CompletedRequests)
	}
	if m1.TotalInputTokens != m2.TotalInputTokens {
		t.Errorf("TotalInputTokens: run1=%d, run2=%d (INV-6 violation)", m1.TotalInputTokens, m2.TotalInputTokens)
	}
	if m1.TotalOutputTokens != m2.TotalOutputTokens {
		t.Errorf("TotalOutputTokens: run1=%d, run2=%d (INV-6 violation)", m1.TotalOutputTokens, m2.TotalOutputTokens)
	}
	if m1.SimEndedTime != m2.SimEndedTime {
		t.Errorf("SimEndedTime: run1=%d, run2=%d (INV-6 violation)", m1.SimEndedTime, m2.SimEndedTime)
	}
	if m1.PeakKVBlocksUsed != m2.PeakKVBlocksUsed {
		t.Errorf("PeakKVBlocksUsed: run1=%d, run2=%d (INV-6 violation)", m1.PeakKVBlocksUsed, m2.PeakKVBlocksUsed)
	}
	if m1.PreemptionCount != m2.PreemptionCount {
		t.Errorf("PreemptionCount: run1=%d, run2=%d (INV-6 violation)", m1.PreemptionCount, m2.PreemptionCount)
	}
	if m1.KVAllocationFailures != m2.KVAllocationFailures {
		t.Errorf("KVAllocationFailures: run1=%d, run2=%d (INV-6 violation)", m1.KVAllocationFailures, m2.KVAllocationFailures)
	}
	if m1.StillQueued != m2.StillQueued {
		t.Errorf("StillQueued: run1=%d, run2=%d (INV-6 violation)", m1.StillQueued, m2.StillQueued)
	}
	if m1.StillRunning != m2.StillRunning {
		t.Errorf("StillRunning: run1=%d, run2=%d (INV-6 violation)", m1.StillRunning, m2.StillRunning)
	}
	if m1.DroppedUnservable != m2.DroppedUnservable {
		t.Errorf("DroppedUnservable: run1=%d, run2=%d (INV-6 violation)", m1.DroppedUnservable, m2.DroppedUnservable)
	}
	if m1.LengthCappedRequests != m2.LengthCappedRequests {
		t.Errorf("LengthCappedRequests: run1=%d, run2=%d (INV-6 violation)", m1.LengthCappedRequests, m2.LengthCappedRequests)
	}
	if m1.TimedOutRequests != m2.TimedOutRequests {
		t.Errorf("TimedOutRequests: run1=%d, run2=%d (INV-6 violation)", m1.TimedOutRequests, m2.TimedOutRequests)
	}
	if m1.TTFTSum != m2.TTFTSum {
		t.Errorf("TTFTSum: run1=%d, run2=%d (INV-6 violation)", m1.TTFTSum, m2.TTFTSum)
	}
	if m1.ITLSum != m2.ITLSum {
		t.Errorf("ITLSum: run1=%d, run2=%d (INV-6 violation)", m1.ITLSum, m2.ITLSum)
	}
}

// TestNodeReadyEvent_DeferredConstruction_UsesPoolGPUType verifies US2 deferred construction:
// GIVEN a cluster with NodePools but InitialNodes=0 (no initial capacity), all instances start pending.
// WHEN a NodeReadyEvent fires (node provisioned and marked ready).
// THEN the newly constructed instance uses the pool's GPU type (SC-003: pool-authoritative).
// THEN the instance is registered with the snapshot provider and is routable (SC-003).
func TestNodeReadyEvent_DeferredConstruction_UsesPoolGPUType(t *testing.T) {
	// GIVEN: CLI --gpu flag = "H100", pool gpu_type = "A100" — they differ intentionally.
	// InitialNodes=0 means no nodes at startup → all instances deferred (pending).
	cfg := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             1_000_000,
			Seed:                42,
			ModelHardwareConfig: sim.NewModelHardwareConfig(testModelConfig(), "test-model", "H100", 1, 1, false, 0),
			KVCacheConfig:       sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(4, 2048, 0),
			LatencyModel:        testFakeLatency(),
		},
		NumInstances: 2,
		NodePools: []NodePoolConfig{
			{
				Name:         "a100-pool",
				GPUType:      "A100",
				GPUsPerNode:  8,
				InitialNodes: 0, // no initial capacity — all instances start pending
				MaxNodes:     2,
				GPUMemoryGiB: 80,
			},
		},
	}

	cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)

	// Precondition: all instances are pending (no capacity at startup).
	if len(cs.instances) != 0 {
		t.Fatalf("precondition failed: expected 0 placed instances at startup (InitialNodes=0), got %d", len(cs.instances))
	}
	if cs.placement == nil {
		t.Fatal("precondition failed: placement must be non-nil when NodePools are configured")
	}

	// Provision a node in the pool (sets it to Provisioning state).
	node, _ := cs.placement.ProvisionNode("a100-pool", 0)

	// Fire NodeReadyEvent directly (simulate provisioning delay elapsing).
	event := &NodeReadyEvent{timestamp: 0, nodeID: node.ID}
	event.Execute(cs)

	// THEN: instances grew — deferred construction fired.
	if len(cs.instances) == 0 {
		t.Fatal("NodeReadyEvent.Execute did not construct any deferred instances")
	}

	// THEN: pool GPU type ("A100") is used, not CLI --gpu ("H100").
	for _, inst := range cs.instances {
		if got := inst.GPU(); got != "A100" {
			t.Errorf("deferred instance %s: GPU() = %q, want %q (pool gpu_type must override CLI --gpu in deferred path)",
				inst.ID(), got, "A100")
		}
		// Verify scheduleInstanceLoadedEvent fired and instance reached Active state
		// (WarmUpRequestCount=0 and loading delay=0, so Loading → Active immediately).
		if got := inst.State; got != sim.InstanceStateActive {
			t.Errorf("instance %s: State = %v, want sim.InstanceStateActive (scheduleInstanceLoadedEvent must fire)", inst.ID(), got)
		}
	}

	// THEN: the deferred instance is visible to the gateway and routable (SC-003).
	for _, inst := range cs.instances {
		if !containsInstance(cs.Instances(), inst.ID()) || !inst.IsRoutable() {
			t.Errorf("deferred instance %s is not visible through Instances() as routable", inst.ID())
		}
	}
}

// TestClusterSimulator_SessionTerminalStateCompleteness verifies INV-11 (BC-3):
// every session reaches exactly one terminal state after ClusterSimulator.Run().
// With the default fake latency model and a 500s horizon, all sessions
// complete normally (sessionCompleted path). This exercises the full DES
// pipeline: seed injected → rounds executed → OnComplete returns nil exactly once.
// Uses a set (not a counter) to prevent two-bugs-cancel false pass scenarios.
func TestClusterSimulator_SessionTerminalStateCompleteness(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	inputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 20},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler (input): %v", err)
	}
	outputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 10},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler (output): %v", err)
	}

	const numSessions = 4
	const maxRounds = 3
	const horizon = int64(500_000_000) // 500 seconds

	blueprints := make([]workload.SessionBlueprint, numSessions)
	seeds := make([]*sim.Request, numSessions)
	for i := 0; i < numSessions; i++ {
		sessID := fmt.Sprintf("t21_sess_%d", i)
		blueprints[i] = workload.SessionBlueprint{
			SessionID:     sessID,
			ClientID:      "test-client",
			MaxRounds:     maxRounds,
			ThinkTimeUs:   1_000_000, // 1 second
			Horizon:       horizon,
			InputSampler:  inputSampler,
			OutputSampler: outputSampler,
			RNG:           rand.New(rand.NewSource(rng.Int63())),
			TenantID:      "test-tenant",
			SLOClass:      "standard",
			Model:         "test-model",
		}
		seeds[i] = &sim.Request{
			ID:           fmt.Sprintf("t21_sess_%d_r0", i),
			ArrivalTime:  int64(i * 1_000_000),
			InputTokens:  make([]sim.TokenID, 20),
			OutputTokens: make([]sim.TokenID, 10),
			MaxOutputLen: 10,
			State:        sim.StateQueued,
			SessionID:    sessID,
			RoundIndex:   0,
		}
	}

	sm := workload.NewSessionManager(blueprints)
	// Use a set (not a counter) so that one session called twice AND another
	// silently abandoned cannot cancel out to give a false pass.
	terminatedSessions := make(map[string]bool)
	onDone := func(req *sim.Request, tick int64) []*sim.Request {
		followUps := sm.OnComplete(req, tick)
		// A nil return means this session just reached a terminal state.
		// Exactly one nil return per session guaranteed by the sessionActive guard.
		if followUps == nil && req.SessionID != "" {
			terminatedSessions[req.SessionID] = true
		}
		return followUps
	}

	config := newTestDeploymentConfig(1)
	config.Horizon = horizon
	cs := NewClusterSimulator(config, NewSliceRequestSource(seeds), onDone)
	mustRun(t, cs)

	// INV-11: every session reached exactly one terminal state
	if len(terminatedSessions) != numSessions {
		t.Errorf("BC-3 (INV-11): terminal sessions = %d, want %d (no session silently abandoned)",
			len(terminatedSessions), numSessions)
	}
}

// TestClusterSimulator_SessionFollowUpCausality verifies INV-10 (BC-4):
// round[N+1].ArrivalTime >= round[N].completionTick + ThinkTimeUs, where
// completionTick comes from actual DES execution (not a hardcoded value).
// Also verifies the DES clock advanced past seed arrival (catches wrong-tick bugs).
func TestClusterSimulator_SessionFollowUpCausality(t *testing.T) {
	inputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 20},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler (input): %v", err)
	}
	outputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 10},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler (output): %v", err)
	}

	const thinkTimeUs = int64(500_000) // 500ms
	const numSessions = 3
	const maxRounds = 3

	rng := rand.New(rand.NewSource(7))
	blueprints := make([]workload.SessionBlueprint, numSessions)
	seeds := make([]*sim.Request, numSessions)
	for i := 0; i < numSessions; i++ {
		sessID := fmt.Sprintf("t22_sess_%d", i)
		blueprints[i] = workload.SessionBlueprint{
			SessionID:     sessID,
			ClientID:      "test-client",
			MaxRounds:     maxRounds,
			ThinkTimeUs:   thinkTimeUs,
			Horizon:       math.MaxInt64,
			InputSampler:  inputSampler,
			OutputSampler: outputSampler,
			RNG:           rand.New(rand.NewSource(rng.Int63())),
			TenantID:      "test-tenant",
			SLOClass:      "standard",
			Model:         "test-model",
		}
		seeds[i] = &sim.Request{
			ID:           fmt.Sprintf("t22_sess_%d_r0", i),
			ArrivalTime:  int64(i * 2_000_000),
			InputTokens:  make([]sim.TokenID, 20),
			OutputTokens: make([]sim.TokenID, 10),
			MaxOutputLen: 10,
			State:        sim.StateQueued,
			SessionID:    sessID,
			RoundIndex:   0,
		}
	}

	type roundKey struct {
		sessID string
		round  int
	}
	completionTick := make(map[roundKey]int64)
	arrivalTime := make(map[roundKey]int64)
	for i, s := range seeds {
		arrivalTime[roundKey{fmt.Sprintf("t22_sess_%d", i), 0}] = s.ArrivalTime
	}

	sm := workload.NewSessionManager(blueprints)
	onDone := func(req *sim.Request, tick int64) []*sim.Request {
		if req.SessionID != "" {
			completionTick[roundKey{req.SessionID, req.RoundIndex}] = tick
		}
		followUps := sm.OnComplete(req, tick)
		for _, fu := range followUps {
			arrivalTime[roundKey{fu.SessionID, fu.RoundIndex}] = fu.ArrivalTime
		}
		return followUps
	}

	config := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(config, NewSliceRequestSource(seeds), onDone)
	mustRun(t, cs)

	// Verify INV-10 for every consecutive round pair.
	// Also verify DES clock advanced past seed arrival (catches wrong-tick bugs).
	for i := 0; i < numSessions; i++ {
		sessID := fmt.Sprintf("t22_sess_%d", i)
		seedArrival := seeds[i].ArrivalTime
		for n := 0; n < maxRounds-1; n++ {
			ck, okC := completionTick[roundKey{sessID, n}]
			arr, okA := arrivalTime[roundKey{sessID, n + 1}]
			if !okC || !okA {
				continue // round may not have run (horizon or budget) — skip
			}
			if ck <= seedArrival {
				t.Errorf("BC-4: session %s round[%d] completionTick=%d <= seedArrival=%d (DES clock did not advance)",
					sessID, n, ck, seedArrival)
			}
			if arr < ck+thinkTimeUs {
				t.Errorf("BC-4 (INV-10): session %s round[%d].arrival=%d < round[%d].completion(%d)+thinkTime(%d)=%d",
					sessID, n+1, arr, n, ck, thinkTimeUs, ck+thinkTimeUs)
			}
		}
	}
	// Guard against vacuous pass: verify all rounds completed (MaxInt64 horizon,
	// no budget cap, so every round of every session must have a completion tick).
	expectedEntries := numSessions * maxRounds
	if len(completionTick) != expectedEntries {
		t.Errorf("BC-4: completionTick entries = %d, want %d (not all rounds completed — causality check may be incomplete)",
			len(completionTick), expectedEntries)
	}
}

// TestClusterSimulator_MultiTurnSession_EndToEnd verifies BC-5, BC-6, BC-7:
// INV-1 conservation holds with dynamic follow-up injection, follow-ups have
// RoundIndex > 0, and every session generates at least one follow-up.
// This is the first test of the standard (non-disaggregated) cluster path
// with real multi-turn session management.
// NOTE: totalInjected counts seeds plus dynamically injected follow-ups, so it is
// the one baseline the helper's "eager slice" precondition does not cover — the
// test must compute it, which it does in onDone below.
func TestClusterSimulator_MultiTurnSession_EndToEnd(t *testing.T) {
	inputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 20},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler (input): %v", err)
	}
	outputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 10},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler (output): %v", err)
	}

	const numSessions = 3
	const maxRounds = 3

	rng := rand.New(rand.NewSource(13))
	blueprints := make([]workload.SessionBlueprint, numSessions)
	seeds := make([]*sim.Request, numSessions)
	for i := 0; i < numSessions; i++ {
		sessID := fmt.Sprintf("t23_sess_%d", i)
		blueprints[i] = workload.SessionBlueprint{
			SessionID:     sessID,
			ClientID:      "test-client",
			MaxRounds:     maxRounds,
			ThinkTimeUs:   100_000, // 100ms
			Horizon:       math.MaxInt64,
			InputSampler:  inputSampler,
			OutputSampler: outputSampler,
			RNG:           rand.New(rand.NewSource(rng.Int63())),
			TenantID:      "test-tenant",
			SLOClass:      "standard",
			Model:         "test-model",
		}
		seeds[i] = &sim.Request{
			ID:           fmt.Sprintf("t23_sess_%d_r0", i),
			ArrivalTime:  int64(i * 1_000_000),
			InputTokens:  make([]sim.TokenID, 20),
			OutputTokens: make([]sim.TokenID, 10),
			MaxOutputLen: 10,
			State:        sim.StateQueued,
			SessionID:    sessID,
			RoundIndex:   0,
		}
	}

	sm := workload.NewSessionManager(blueprints)
	totalInjected := numSessions // seeds
	followUpCount := 0
	followUpsBySession := make(map[string][]int)

	onDone := func(req *sim.Request, tick int64) []*sim.Request {
		followUps := sm.OnComplete(req, tick)
		for _, fu := range followUps {
			totalInjected++
			followUpCount++
			if fu.SessionID != "" {
				followUpsBySession[fu.SessionID] = append(followUpsBySession[fu.SessionID], fu.RoundIndex)
			}
		}
		return followUps
	}

	config := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(config, NewSliceRequestSource(seeds), onDone)
	mustRun(t, cs)

	metrics := cs.AggregatedMetrics()

	// BC-5 (INV-1): conservation with dynamic follow-up injection
	assertClusterINV1Conservation(t, cs, totalInjected, noRejections, "multi-turn end-to-end")

	if metrics.CompletedRequests == 0 {
		t.Error("BC-5: CompletedRequests = 0, expected > 0 (work must be done)")
	}

	// BC-6: no follow-up has RoundIndex == 0 (only seeds are round 0)
	for sessID, rounds := range followUpsBySession {
		for _, ri := range rounds {
			if ri == 0 {
				t.Errorf("BC-6: session %s generated a follow-up with RoundIndex=0", sessID)
			}
		}
	}

	// BC-7: every session generated at least one follow-up
	for i := 0; i < numSessions; i++ {
		sessID := fmt.Sprintf("t23_sess_%d", i)
		if len(followUpsBySession[sessID]) == 0 {
			t.Errorf("BC-7: session %s generated 0 follow-ups, want >= 1 (MaxRounds=%d)", sessID, maxRounds)
		}
	}

	// Sanity: exact follow-up count with MaxInt64 horizon (no interruptions)
	expectedFollowUps := numSessions * (maxRounds - 1)
	if followUpCount != expectedFollowUps {
		t.Errorf("follow-up count = %d, want %d (%d sessions × %d follow-ups each)",
			followUpCount, expectedFollowUps, numSessions, maxRounds-1)
	}
}

// ---------------------------------------------------------------------------
// Autoscaler wiring tests
// ---------------------------------------------------------------------------

func TestNewClusterSimulator_AutoscalerWiredWhenEnabled(t *testing.T) {
	cfg := newTestDeploymentConfig(2)
	cfg.ModelAutoscalerIntervalUs = 30_000_000
	// AutoscalerAnalyzerConfig zero values → defaults applied inside constructor.
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
	if cs.autoscaler == nil {
		t.Fatal("autoscaler must not be nil when ModelAutoscalerIntervalUs > 0")
	}
	if cs.autoscaler.collector == nil {
		t.Error("autoscaler.collector must not be nil")
	}
	if cs.autoscaler.analyzer == nil {
		t.Error("autoscaler.analyzer must not be nil")
	}
	if cs.autoscaler.engine == nil {
		t.Error("autoscaler.engine must not be nil")
	}
	if cs.autoscaler.actuator == nil {
		t.Error("autoscaler.actuator must not be nil")
	}
}

func TestNewClusterSimulator_AutoscalerNilWhenDisabled(t *testing.T) {
	cfg := newTestDeploymentConfig(2)
	// ModelAutoscalerIntervalUs == 0 (default) → autoscaler stays nil.
	cs := NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil)
	if cs.autoscaler != nil {
		t.Error("autoscaler must be nil when ModelAutoscalerIntervalUs == 0")
	}
}

// TestAutoscaler_RequestBoundedRun_Terminates verifies that a request-bounded run
// (Horizon == math.MaxInt64) with the autoscaler enabled terminates correctly.
// Regression test for Bug 2: scheduleNextTick had no termination guard, causing
// an infinite event loop on request-bounded runs.
func TestAutoscaler_RequestBoundedRun_Terminates(t *testing.T) {
	cfg := newTestDeploymentConfig(1)
	cfg.ModelAutoscalerIntervalUs = 100_000 // 100 ms ticks — fires many times during test
	reqs := newTestRequests(10)

	cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), nil)

	done := make(chan error, 1)
	go func() { done <- cs.Run() }()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Run() returned error: %v", err)
		}
		// INV-1: all requests must be accounted for — premature termination would
		// silently lose requests while still passing the timeout check.
		agg := cs.AggregatedMetrics()
		if agg.CompletedRequests != len(reqs) {
			t.Errorf("INV-1: expected %d completed, got %d (queued=%d, running=%d)",
				len(reqs), agg.CompletedRequests, agg.StillQueued, agg.StillRunning)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Run() did not terminate within 2s — autoscaler tick loop is likely infinite (Bug 2 regression)")
	}
}

// TestPushArrival_CoInvariant_SessionFollowUpsProcessed verifies the
// pendingArrivals co-invariant end-to-end: session follow-up requests
// injected via pushArrival (inside the OnRequestDone callback) are fully
// processed by the cluster simulation.
//
// If pushArrival failed to increment pendingArrivals for follow-ups, the
// autoscaler's scheduleNextTick termination guard (pendingArrivals <= 0)
// would fire prematurely, causing Run() to return before follow-ups complete
// and breaking INV-1 conservation.
func TestPushArrival_CoInvariant_SessionFollowUpsProcessed(t *testing.T) {
	cfg := newTestDeploymentConfig(1)
	cfg.ModelAutoscalerIntervalUs = 100_000 // enable autoscaler to exercise termination guard

	const initial = 3
	reqs := newTestRequests(initial)

	// Each initial request generates exactly one follow-up; follow-ups generate none.
	followUpsIssued := 0
	onDone := func(req *sim.Request, tick int64) []*sim.Request {
		if followUpsIssued >= initial {
			return nil
		}
		followUpsIssued++
		return []*sim.Request{{
			ID:           fmt.Sprintf("followup-%d", followUpsIssued),
			ArrivalTime:  tick,
			InputTokens:  make([]sim.TokenID, 50),
			OutputTokens: make([]sim.TokenID, 20),
			MaxOutputLen: 20,
			State:        sim.StateQueued,
		}}
	}

	cs := NewClusterSimulator(cfg, NewSliceRequestSource(reqs), onDone)

	done := make(chan error, 1)
	go func() { done <- cs.Run() }()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Run() returned error: %v", err)
		}
		agg := cs.AggregatedMetrics()
		want := initial * 2 // initial + one follow-up each
		if agg.CompletedRequests != want {
			t.Errorf("completed = %d, want %d — follow-up requests not tracked by pendingArrivals co-invariant",
				agg.CompletedRequests, want)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Run() timed out — autoscaler terminated early, pendingArrivals co-invariant likely broken for follow-up requests")
	}
}

// newBatchTestRequests creates n requests with the given SLOClass,
// arriving every 10µs starting at t=0, with 50 input tokens and 20 output tokens.
func newBatchTestRequests(n int, sloClass string) []*sim.Request {
	reqs := make([]*sim.Request, n)
	for i := range reqs {
		reqs[i] = &sim.Request{
			ID:           fmt.Sprintf("req_%s_%d", sloClass, i),
			ArrivalTime:  int64(i) * 10,
			SLOClass:     sloClass,
			InputTokens:  make([]sim.TokenID, 50),
			OutputTokens: make([]sim.TokenID, 20),
			State:        sim.StateQueued,
		}
	}
	return reqs
}

// TestBatchRequestsNotSerialized verifies that batch and background requests are
// NOT serialized — they flow through admission like standard requests.
// This is a regression guard for issue #965.
func TestBatchRequestsNotSerialized(t *testing.T) {
	for _, sloClass := range []string{"batch", "background"} {
		t.Run(sloClass, func(t *testing.T) {
			requests := newBatchTestRequests(10, sloClass)
			cfg := newTestDeploymentConfig(1)
			cs := NewClusterSimulator(cfg, NewSliceRequestSource(requests), nil)
			mustRun(t, cs)

			// Always-admit must not reject batch/background requests.
			if cs.RejectedRequests() != 0 {
				t.Errorf("expected 0 rejections under always-admit, got %d", cs.RejectedRequests())
			}

			m := cs.AggregatedMetrics()
			if m.CompletedRequests != 10 {
				t.Fatalf("completed %d requests, want 10", m.CompletedRequests)
			}

			// Batch/background requests must flow through admission concurrently (not
			// serialized): their mean TTFT must match the same burst sent as "standard".
			// Serializing 10 requests would inflate it several-fold; 1.5x is the margin.
			baseCS := NewClusterSimulator(newTestDeploymentConfig(1), NewSliceRequestSource(newBatchTestRequests(10, "standard")), nil)
			mustRun(t, baseCS)
			base := baseCS.AggregatedMetrics()
			if base.CompletedRequests != 10 {
				t.Fatalf("baseline completed %d requests, want 10", base.CompletedRequests)
			}
			ttftMean := float64(m.TTFTSum) / float64(m.CompletedRequests)
			baseMean := float64(base.TTFTSum) / float64(base.CompletedRequests)
			if ttftMean > 1.5*baseMean {
				t.Errorf("mean TTFT %.0fµs > 1.5x the standard-class baseline %.0fµs: %s requests are being serialized (regression: #965)",
					ttftMean, baseMean, sloClass)
			}
		})
	}
}

// TestClusterSimulator_OrphanedTimeout_DoesNotInflateSimEndedTime verifies that
// the cluster path's prevClusterClock save/restore (cluster.go) prevents orphaned
// TimeoutEvents from inflating AggregatedMetrics().SimEndedTime. This is the
// cluster-mode companion to TestTimeout_OrphanedTimeout_DoesNotInflateSimEndedTime
// in sim/timeout_test.go.
//
// Scenario: 5 requests complete quickly but each bears a 300s Deadline
// (mimicking DefaultTimeoutUs). Without the c.clock restore in the cluster loop,
// c.clock would advance to arrival+300s for the last request, inflating SimEndedTime.
func TestClusterSimulator_OrphanedTimeout_DoesNotInflateSimEndedTime(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             500_000_000, // 500s — beyond workload.DefaultTimeoutUs (300s)
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyModel:        testFakeZeroQueueing(),
			ModelHardwareConfig: sim.NewModelHardwareConfig(testModelConfig(), "test", "H100", 1, 1, false, 0),
		},
		NumInstances: 2,
	}

	rng := sim.NewPartitionedRNG(sim.NewSimulationKey(42))
	workloadRNG := rng.ForSubsystem(sim.SubsystemWorkload)

	requests := make([]*sim.Request, 5)
	for i := 0; i < 5; i++ {
		arrivalTime := int64(i * 100_000) // stagger arrivals by 100ms
		requests[i] = &sim.Request{
			ID:           fmt.Sprintf("r%d", i),
			ArrivalTime:  arrivalTime,
			InputTokens:  sim.GenerateRandomTokenIDs(workloadRNG, 10),
			OutputTokens: sim.GenerateRandomTokenIDs(workloadRNG, 5),
			State:        sim.StateQueued,
			MaxOutputLen: 5,
			Deadline:     arrivalTime + workload.DefaultTimeoutUs,
		}
	}

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()

	if m.CompletedRequests != 5 {
		t.Fatalf("CompletedRequests: got %d, want 5", m.CompletedRequests)
	}
	if m.TimedOutRequests != 0 {
		t.Errorf("TimedOutRequests: got %d, want 0 (orphaned timeouts must not fire)", m.TimedOutRequests)
	}

	// SimEndedTime must reflect actual completion, not the last orphaned timeout.
	// Last arrival at 400_000 µs; per-request completion takes a few tens of ms.
	// Lower bound (> 400_000): last arrival advanced the clock past 400ms.
	// Upper bound (< 1_000_000): 300× below workload.DefaultTimeoutUs; catches inflation.
	if m.SimEndedTime <= 400_000 {
		t.Errorf("SimEndedTime too low: got %d µs, want > 400_000 µs", m.SimEndedTime)
	}
	if m.SimEndedTime > 1_000_000 {
		t.Errorf("SimEndedTime inflated by orphaned timeout in cluster path: got %d µs (%.1fs), want < 1_000_000 µs",
			m.SimEndedTime, float64(m.SimEndedTime)/1e6)
	}
}

// TestClusterSimulator_MixedOrphanedAndGenuineTimeout_CorrectMetrics verifies that
// the lazy-cancellation guard skips only completed-request (orphaned) TimeoutEvents
// and does not suppress genuine timeouts for still-queued requests.
//
// Scenario (1 instance, batch-size 1):
//   - r0: arrives at 0, 300s deadline → completes normally (~6ms), orphaned timeout skipped
//   - r1: arrives at 0, 300s deadline → queued behind r0, completes normally (~12ms), orphaned timeout skipped
//   - r2: arrives at 0, 5000µs deadline → queued behind r0; r0 takes ~6ms so r2 times out while waiting
//
// Expected: CompletedRequests=2, TimedOutRequests=1, SimEndedTime reflects r1's
// completion time (~12ms), not r0/r1's orphaned 300s timeouts.
func TestClusterSimulator_MixedOrphanedAndGenuineTimeout_CorrectMetrics(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             500_000_000,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(1, 2048, 0), // max 1 running — forces queuing
			LatencyModel:        testFakeZeroQueueing(),
			ModelHardwareConfig: sim.NewModelHardwareConfig(testModelConfig(), "test", "H100", 1, 1, false, 0),
		},
		NumInstances: 1,
	}

	// r0 and r1 complete before their 300s deadline (orphaned TimeoutEvents).
	// r2 has a 5000µs deadline — shorter than the time r0 takes to complete (~6ms),
	// so r2 times out while queued. This verifies the guard uses State==StateCompleted
	// (not a broader condition) and does not suppress genuine timeouts.
	requests := []*sim.Request{
		{
			ID: "r0", ArrivalTime: 0,
			InputTokens: make([]sim.TokenID, 10), OutputTokens: make([]sim.TokenID, 5),
			State: sim.StateQueued, MaxOutputLen: 5,
			Deadline: workload.DefaultTimeoutUs,
		},
		{
			ID: "r1", ArrivalTime: 0,
			InputTokens: make([]sim.TokenID, 10), OutputTokens: make([]sim.TokenID, 5),
			State: sim.StateQueued, MaxOutputLen: 5,
			Deadline: workload.DefaultTimeoutUs,
		},
		{
			ID: "r2", ArrivalTime: 0,
			InputTokens: make([]sim.TokenID, 10), OutputTokens: make([]sim.TokenID, 5),
			State: sim.StateQueued, MaxOutputLen: 5,
			Deadline: 5_000, // 5ms — r0 takes ~6ms, so r2 times out while queued
		},
	}

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()

	if m.CompletedRequests != 2 {
		t.Errorf("CompletedRequests: got %d, want 2", m.CompletedRequests)
	}
	if m.TimedOutRequests != 1 {
		t.Errorf("TimedOutRequests: got %d, want 1 (r2 must genuinely time out)", m.TimedOutRequests)
	}
	// INV-1: 3 injected = 2 completed + 1 timed-out
	if got := m.CompletedRequests + m.TimedOutRequests; got != 3 {
		t.Errorf("conservation: CompletedRequests(%d) + TimedOutRequests(%d) = %d, want 3",
			m.CompletedRequests, m.TimedOutRequests, got)
	}
	// r1 completes after r0 (~12ms total); SimEndedTime should reflect r1's
	// completion, not r0/r1's orphaned 300s timeouts.
	// Lower bound > 10_000: r0 takes ~6ms and r1 runs after, so SimEndedTime
	// must exceed 10ms — verifying both requests actually ran, not just r2's timeout.
	if m.SimEndedTime <= 10_000 {
		t.Errorf("SimEndedTime too low: got %d µs, want > 10_000 µs (r0+r1 must both complete)", m.SimEndedTime)
	}
	if m.SimEndedTime > 100_000 {
		t.Errorf("SimEndedTime inflated by orphaned timeout: got %d µs (%.1fs), want < 100_000 µs",
			m.SimEndedTime, float64(m.SimEndedTime)/1e6)
	}
}

// TestClusterSimulator_PD_OrphanedTimeout_TimingNotInflated verifies that with
// PD disaggregation, orphaned TimeoutEvents on the prefill instance do not
// inflate per-request phase timestamps set by detectPrefillCompletions via
// c.clock. If c.clock were not restored after a skipped orphaned timeout,
// KVTransferStartedEvent.time and parent.PrefillCompleteTime could be set to
// the orphaned timestamp (~300s) rather than the actual prefill completion time.
//
// Scenario: 4 requests through a 2P+2D cluster, each with a workload.DefaultTimeoutUs
// deadline. All complete well before the deadline. Assert that PrefillCompleteTime,
// TransferStartTime, and the causality chain are bounded by actual work time.
func TestClusterSimulator_PD_OrphanedTimeout_TimingNotInflated(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.Horizon = 500_000_000 // 500s — beyond workload.DefaultTimeoutUs

	rng := sim.NewPartitionedRNG(sim.NewSimulationKey(42))
	workloadRNG := rng.ForSubsystem(sim.SubsystemWorkload)

	const n = 4
	requests := make([]*sim.Request, n)
	for i := 0; i < n; i++ {
		arrivalTime := int64(i * 10_000)
		requests[i] = &sim.Request{
			ID:           fmt.Sprintf("pd_r%d", i),
			ArrivalTime:  arrivalTime,
			InputTokens:  sim.GenerateRandomTokenIDs(workloadRNG, 10),
			OutputTokens: sim.GenerateRandomTokenIDs(workloadRNG, 5),
			State:        sim.StateQueued,
			MaxOutputLen: 5,
			Deadline:     arrivalTime + workload.DefaultTimeoutUs,
		}
	}

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	if m.CompletedRequests != n {
		t.Fatalf("CompletedRequests: got %d, want %d", m.CompletedRequests, n)
	}
	if m.TimedOutRequests != 0 {
		t.Errorf("TimedOutRequests: got %d, want 0", m.TimedOutRequests)
	}

	// Phase timestamps set by detectPrefillCompletions/detectDecodeCompletions
	// use c.clock. If c.clock were not restored after an orphaned timeout skip,
	// these would be inflated to ~300s. Assert all are bounded well below 1s.
	const phaseThreshold = int64(1_000_000) // 1s
	for _, parent := range cs.parentRequests {
		if parent.PrefillCompleteTime > phaseThreshold {
			t.Errorf("parent %s: PrefillCompleteTime inflated: %d µs (%.1fs), want < %d µs",
				parent.ID, parent.PrefillCompleteTime, float64(parent.PrefillCompleteTime)/1e6, phaseThreshold)
		}
		if parent.TransferStartTime > phaseThreshold {
			t.Errorf("parent %s: TransferStartTime inflated: %d µs (%.1fs), want < %d µs",
				parent.ID, parent.TransferStartTime, float64(parent.TransferStartTime)/1e6, phaseThreshold)
		}
		// Causality: each phase must not precede the prior one.
		chain := []struct {
			name  string
			value int64
		}{
			{"ArrivalTime", parent.ArrivalTime},
			{"PrefillCompleteTime", parent.PrefillCompleteTime},
			{"TransferStartTime", parent.TransferStartTime},
			{"TransferCompleteTime", parent.TransferCompleteTime},
			{"DecodeEnqueueTime", parent.DecodeEnqueueTime},
		}
		for i := 1; i < len(chain); i++ {
			if chain[i].value < chain[i-1].value {
				t.Errorf("parent %s causality: %s (%d) < %s (%d)",
					parent.ID, chain[i].name, chain[i].value, chain[i-1].name, chain[i-1].value)
			}
		}
	}
}
