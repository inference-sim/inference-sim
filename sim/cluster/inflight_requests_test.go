package cluster

import (
	"fmt"
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

// TestClusterSimulator_InFlightRequests_DrainsToZeroAfterCompletion verifies BC-4:
// GIVEN a cluster with requests that all complete before horizon
// WHEN simulation ends
// THEN all inFlightRequests values are 0
func TestClusterSimulator_InFlightRequests_DrainsToZeroAfterCompletion(t *testing.T) {
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
			Horizon:       10000000,
			Seed:          42,
			KVCacheConfig: sim.NewKVCacheConfig(5, 16, 0, 0, 0, 0), // Very small — will force drops
			BatchConfig:   sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs: sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
		},
		NumInstances: 1,
	}
	// Requests with large input that exceeds KV capacity
	reqs := make([]*sim.Request, 3)
	for i := range reqs {
		reqs[i] = &sim.Request{
			ID:           fmt.Sprintf("drop_req_%d", i),
			ArrivalTime:  int64(i * 1000),
			InputTokens:  make([]int, 200), // 200 tokens / 16 block_size = 13 blocks > 5 total
			OutputTokens: make([]int, 8),
			State:        sim.StateQueued,
		}
	}
	cs := NewClusterSimulator(config, reqs)
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
// GIVEN requests routed with routing latency
// WHEN a request enters the queue (QueuedEvent fires)
// THEN InFlightRequests does NOT decrement (stays elevated)
// AND InFlightRequests decrements only when the request completes
func TestClusterSimulator_InFlightRequests_CompletionBasedDecrement(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:       10000000,
			Seed:          42,
			KVCacheConfig: sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:   sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs: sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
		},
		NumInstances:    1,
		RoutingLatency:  100,
		TraceLevel:      "decisions",
		CounterfactualK: 1,
	}
	reqs := make([]*sim.Request, 4)
	for i := range reqs {
		reqs[i] = &sim.Request{
			ID:           fmt.Sprintf("causal_req_%d", i),
			ArrivalTime:  0,
			InputTokens:  make([]int, 16),
			OutputTokens: make([]int, 8),
			State:        sim.StateQueued,
		}
	}
	cs := NewClusterSimulator(config, reqs)
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
