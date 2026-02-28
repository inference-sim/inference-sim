package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestClusterSimulator_InFlightRequests_DrainsToZeroAfterProcessing verifies:
// GIVEN a 2-instance cluster with 4+ requests using weighted routing
// WHEN run to completion
// THEN all inFlightRequests values are 0 (every routed request was absorbed)
func TestClusterSimulator_InFlightRequests_DrainsToZeroAfterProcessing(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:       10000000,
			Seed:          42,
			KVCacheConfig: sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:   sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs: sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
		},
		NumInstances:         2,
		RoutingPolicy:        "weighted",
		RoutingScorerConfigs: sim.DefaultScorerConfigs(),
	}
	requests := testGenerateRequests(42, 10000000, 2.0/1e6, 6,
		0, 16, 0, 16, 16, 8, 0, 8, 8)
	cs := NewClusterSimulator(config, requests)

	mustRun(t, cs)

	for instID, pending := range cs.inFlightRequests {
		if pending != 0 {
			t.Errorf("instance %s: inFlightRequests = %d after completion, want 0", instID, pending)
		}
	}

	// Sanity check: requests were actually processed
	m := cs.AggregatedMetrics()
	if m.CompletedRequests == 0 {
		t.Error("no requests completed — test setup issue")
	}
}

// TestClusterSimulator_InFlightRequests_VisibleInRoutingState verifies:
// GIVEN a 1-instance cluster with pre-generated requests at identical timestamps
//
//	and non-zero routing latency (so routing decisions overlap with pending state)
//
// WHEN routing decisions are traced
// THEN at least one routing decision observes InFlightRequests > 0
//
// Design: With RoutingLatency=100, request N's routing decision occurs at T_N + 100.
// If request N+1 arrives at T_N+1 < T_N + 100 + queueing_delay, the QueuedEvent from
// request N hasn't fired yet, so InFlightRequests > 0 is visible to request N+1's
// routing decision.
func TestClusterSimulator_InFlightRequests_VisibleInRoutingState(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:       10000000,
			Seed:          42,
			KVCacheConfig: sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:   sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs: sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
		},
		NumInstances:    1,
		RoutingLatency:  100, // Creates window where pending is visible
		TraceLevel:      "decisions",
		CounterfactualK: 1,
	}

	// Pre-generate requests at the same arrival time so their routing decisions
	// happen sequentially within the same tick window
	reqs := make([]*sim.Request, 3)
	for i := range reqs {
		reqs[i] = &sim.Request{
			ID:           "vis_req_" + string(rune('a'+i)),
			ArrivalTime:  0, // All arrive at t=0
			InputTokens:  make([]int, 16),
			OutputTokens: make([]int, 8),
			State:        sim.StateQueued,
		}
	}

	cs := NewClusterSimulator(config, reqs)

	mustRun(t, cs)

	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}

	// Check if any candidate score in any routing record has InFlightRequests > 0
	foundPending := false
	for _, r := range tr.Routings {
		for _, c := range r.Candidates {
			if c.InFlightRequests > 0 {
				foundPending = true
				break
			}
		}
		if foundPending {
			break
		}
	}

	if !foundPending {
		t.Error("expected at least one routing decision to observe InFlightRequests > 0, " +
			"but all candidates had InFlightRequests = 0")
	}

	// All pending must drain to zero after completion
	for instID, pending := range cs.inFlightRequests {
		if pending != 0 {
			t.Errorf("instance %s: inFlightRequests = %d after completion, want 0", instID, pending)
		}
	}
}

// TestClusterSimulator_InFlightRequests_CounterfactualIncludesInFlight verifies:
// GIVEN tracing with counterfactual analysis and routing latency (to create pending state)
// WHEN CandidateScore is recorded during routing decisions
// THEN at least one candidate has InFlightRequests > 0 (proving the field is populated)
func TestClusterSimulator_InFlightRequests_CounterfactualIncludesInFlight(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:       10000000,
			Seed:          42,
			KVCacheConfig: sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:   sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs: sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
		},
		NumInstances:    1,
		RoutingLatency:  100, // Creates pending state visible to routing
		TraceLevel:      "decisions",
		CounterfactualK: 1,
	}

	// Pre-generate requests at t=0 to create routing overlap
	reqs := make([]*sim.Request, 3)
	for i := range reqs {
		reqs[i] = &sim.Request{
			ID:           "cf_req_" + string(rune('a'+i)),
			ArrivalTime:  0,
			InputTokens:  make([]int, 16),
			OutputTokens: make([]int, 8),
			State:        sim.StateQueued,
		}
	}

	cs := NewClusterSimulator(config, reqs)

	mustRun(t, cs)

	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}

	// Verify at least one candidate has InFlightRequests > 0 (proving the field
	// is populated from actual cluster state, not just defaulting to zero)
	foundPending := false
	totalCandidates := 0
	for _, r := range tr.Routings {
		for _, c := range r.Candidates {
			totalCandidates++
			if c.InFlightRequests > 0 {
				foundPending = true
			}
		}
	}
	if totalCandidates == 0 {
		t.Error("expected candidates in routing records with counterfactual-k=1")
	}
	if !foundPending {
		t.Error("expected at least one candidate with InFlightRequests > 0, " +
			"but all were 0 — field may not be populated from cluster state")
	}
}

// TestComputeCounterfactual_IncludesInFlightRequests verifies:
// GIVEN snapshots with different InFlightRequests values and explicit scores
// WHEN computeCounterfactual builds the candidate list
// THEN each candidate's InFlightRequests matches its source snapshot
func TestComputeCounterfactual_IncludesInFlightRequests(t *testing.T) {
	snapshots := []sim.RoutingSnapshot{
		{ID: "inst_0", QueueDepth: 2, BatchSize: 1, InFlightRequests: 3},
		{ID: "inst_1", QueueDepth: 0, BatchSize: 0, InFlightRequests: 0},
	}
	scores := map[string]float64{"inst_0": 0.3, "inst_1": 0.8}

	candidates, _ := computeCounterfactual("inst_0", scores, snapshots, 2)

	if len(candidates) != 2 {
		t.Fatalf("expected 2 candidates, got %d", len(candidates))
	}

	// Find inst_0's candidate and verify InFlightRequests is preserved
	for _, c := range candidates {
		if c.InstanceID == "inst_0" {
			if c.InFlightRequests != 3 {
				t.Errorf("inst_0 InFlightRequests = %d, want 3", c.InFlightRequests)
			}
		}
		if c.InstanceID == "inst_1" {
			if c.InFlightRequests != 0 {
				t.Errorf("inst_1 InFlightRequests = %d, want 0", c.InFlightRequests)
			}
		}
	}
}
