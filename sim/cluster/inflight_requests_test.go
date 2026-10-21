package cluster

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestClusterSimulator_InFlightRequests_CounterfactualIncludesInFlight verifies:
// GIVEN tracing with counterfactual analysis, a gateway that reports scores, and
//
//	simultaneous arrivals (each dispatch sees the previous ones in flight)
//
// WHEN CandidateScore is recorded during routing decisions
// THEN at least one candidate has InFlightRequests > 0 (proving the field is populated)
func TestClusterSimulator_InFlightRequests_CounterfactualIncludesInFlight(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             10000000,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(10, 2048, 0),
			LatencyModel:        testFakeLatency(),
			ModelHardwareConfig: sim.NewModelHardwareConfig(testModelConfig(), "", "", 1, 1, false, 0),
		},
		NumInstances:    1,
		TraceLevel:      "decisions",
		CounterfactualK: 1,
	}

	// Pre-generate requests at t=0 so later dispatches see earlier ones in flight
	reqs := make([]*sim.Request, 3)
	for i := range reqs {
		reqs[i] = &sim.Request{
			ID:           "cf_req_" + string(rune('a'+i)),
			ArrivalTime:  0,
			InputTokens:  make([]sim.TokenID, 16),
			OutputTokens: make([]sim.TokenID, 8),
			State:        sim.StateQueued,
		}
	}

	cs := NewClusterSimulator(config, NewSliceRequestSource(reqs), nil)
	cs.SetGateway(&fakeGateway{scores: map[string]float64{"instance_0": 1}})

	mustRun(t, cs)

	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}

	// Verify at least one candidate has InFlightRequests > 0 (proving the field
	// is populated from actual cluster state, not just defaulting to zero)
	foundInFlight := false
	totalCandidates := 0
	for _, r := range tr.Routings {
		for _, c := range r.Candidates {
			totalCandidates++
			if c.InFlightRequests > 0 {
				foundInFlight = true
			}
		}
	}
	if totalCandidates == 0 {
		t.Error("expected candidates in routing records with counterfactual-k=1")
	}
	if !foundInFlight {
		t.Error("expected at least one candidate with InFlightRequests > 0, " +
			"but all were 0 — field may not be populated from cluster state")
	}
}

// TestClusterSimulator_InFlightRequests_DrainsToZeroAfterCompletion verifies BC-4:
// GIVEN a cluster with requests that all complete before horizon
// WHEN simulation ends
// THEN all inFlightRequests values are 0
func TestClusterSimulator_InFlightRequests_DrainsToZeroAfterCompletion(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             10000000,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(10, 2048, 0),
			LatencyModel:        testFakeLatency(),
			ModelHardwareConfig: sim.NewModelHardwareConfig(testModelConfig(), "", "", 1, 1, false, 0),
		},
		NumInstances: 2,
	}
	requests := testGenerateRequests(42, 10000000, 2.0/1e6, 6,
		0, 16, 0, 16, 16, 8, 0, 8, 8)
	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)

	mustRun(t, cs)

	for instID, inflight := range cs.inFlightRequests {
		if inflight != 0 {
			t.Errorf("instance %s: inFlightRequests = %d after completion, want 0", instID, inflight)
		}
	}

	m := cs.AggregatedMetrics()
	if m.CompletedRequests == 0 {
		t.Error("no requests completed — test setup issue")
	}
}

// TestClusterSimulator_InFlightRequests_DroppedUnservable_Decrements verifies BC-7:
// GIVEN a cluster where TotalKVBlocks is too small for some requests
// WHEN requests are routed and some are dropped as unservable
// THEN inFlightRequests drains correctly (accounting for drops)
func TestClusterSimulator_InFlightRequests_DroppedUnservable_Decrements(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             10000000,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(5, 16, 0, 0, 0, 0), // Very small — will force drops
			BatchConfig:         sim.NewBatchConfig(10, 2048, 0),
			LatencyModel:        testFakeLatency(),
			ModelHardwareConfig: sim.NewModelHardwareConfig(testModelConfig(), "", "", 1, 1, false, 0),
		},
		NumInstances: 1,
	}
	// Requests with large input that exceeds KV capacity
	reqs := make([]*sim.Request, 3)
	for i := range reqs {
		reqs[i] = &sim.Request{
			ID:           fmt.Sprintf("drop_req_%d", i),
			ArrivalTime:  int64(i * 1000),
			InputTokens:  make([]sim.TokenID, 200), // 200 tokens / 16 block_size = 13 blocks > 5 total
			OutputTokens: make([]sim.TokenID, 8),
			State:        sim.StateQueued,
		}
	}
	cs := NewClusterSimulator(config, NewSliceRequestSource(reqs), nil)
	mustRun(t, cs)

	// All requests should be dropped as unservable
	m := cs.AggregatedMetrics()
	if m.DroppedUnservable == 0 {
		t.Fatal("expected DroppedUnservable > 0 — test setup issue (KV blocks too large)")
	}

	// InFlightRequests must still drain to match expected
	for _, inst := range cs.Instances() {
		instID := string(inst.ID())
		inflight := cs.inFlightRequests[instID]
		im := inst.Metrics()
		expected := im.StillQueued + im.StillRunning
		if inflight != expected {
			t.Errorf("instance %s: inFlightRequests=%d, expected %d (StillQueued=%d + StillRunning=%d)",
				instID, inflight, expected, im.StillQueued, im.StillRunning)
		}
	}
}

// TestClusterSimulator_InFlightRequests_CompletionBasedDecrement verifies BC-3:
// GIVEN simultaneous requests dispatched to one instance
// WHEN a request enters the queue (QueuedEvent fires)
// THEN InFlightRequests does NOT decrement (stays elevated)
// AND InFlightRequests decrements only when the request completes
func TestClusterSimulator_InFlightRequests_CompletionBasedDecrement(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             10000000,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(10, 2048, 0),
			LatencyModel:        testFakeLatency(),
			ModelHardwareConfig: sim.NewModelHardwareConfig(testModelConfig(), "", "", 1, 1, false, 0),
		},
		NumInstances:    1,
		TraceLevel:      "decisions",
		CounterfactualK: 1,
	}
	reqs := make([]*sim.Request, 4)
	for i := range reqs {
		reqs[i] = &sim.Request{
			ID:           fmt.Sprintf("causal_req_%d", i),
			ArrivalTime:  0,
			InputTokens:  make([]sim.TokenID, 16),
			OutputTokens: make([]sim.TokenID, 8),
			State:        sim.StateQueued,
		}
	}
	cs := NewClusterSimulator(config, NewSliceRequestSource(reqs), nil)
	cs.SetGateway(&fakeGateway{scores: map[string]float64{"instance_0": 1}})
	mustRun(t, cs)

	// Key assertion: at least one routing decision sees InFlightRequests > QueueDepth + BatchSize
	// This proves the in-flight window extends beyond queue absorption
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}
	foundElevated := false
	for _, r := range tr.Routings {
		for _, c := range r.Candidates {
			if c.InFlightRequests > c.QueueDepth+c.BatchSize {
				foundElevated = true
				break
			}
		}
	}
	if !foundElevated {
		t.Error("expected at least one routing decision where InFlightRequests > QueueDepth + BatchSize, " +
			"proving the in-flight window extends beyond queue absorption")
	}

	// Post-simulation: must drain per BC-4
	for instID, inflight := range cs.inFlightRequests {
		if inflight != 0 {
			t.Errorf("instance %s: inFlightRequests = %d, want 0", instID, inflight)
		}
	}
}

// TestCounterfactual_IncludesInFlightRequests verifies:
// GIVEN instances with different in-flight counts and explicit router scores
// WHEN counterfactual builds the candidate list
// THEN candidates are ranked by score, each carries its instance's InFlightRequests, and the
// regret is the best score minus the chosen instance's score
func TestCounterfactual_IncludesInFlightRequests(t *testing.T) {
	cs := NewClusterSimulator(newTestDeploymentConfig(2), NewSliceRequestSource(nil), nil)
	cs.inFlightRequests["instance_0"] = 3
	cs.inFlightRequests["instance_1"] = 0
	scores := map[string]float64{"instance_0": 0.3, "instance_1": 0.8}

	candidates, regret := cs.counterfactual("instance_0", scores, 2)

	if len(candidates) != 2 {
		t.Fatalf("expected 2 candidates, got %d", len(candidates))
	}
	if candidates[0].InstanceID != "instance_1" || candidates[1].InstanceID != "instance_0" {
		t.Errorf("candidates not ranked by score: %s, %s", candidates[0].InstanceID, candidates[1].InstanceID)
	}
	want := map[string]int{"instance_0": 3, "instance_1": 0}
	for _, c := range candidates {
		if c.InFlightRequests != want[c.InstanceID] {
			t.Errorf("%s InFlightRequests = %d, want %d", c.InstanceID, c.InFlightRequests, want[c.InstanceID])
		}
	}
	if diff := regret - 0.5; diff > 1e-9 || diff < -1e-9 {
		t.Errorf("regret = %v, want 0.5", regret)
	}

	top1, _ := cs.counterfactual("instance_0", scores, 1)
	if len(top1) != 1 || top1[0].InstanceID != "instance_1" {
		t.Errorf("k=1 candidates = %+v, want only instance_1", top1)
	}
}
