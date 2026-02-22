package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestClusterSimulator_PendingRequests_DrainsToZeroAfterProcessing verifies:
// GIVEN a 2-instance cluster with 4+ requests using weighted routing
// WHEN run to completion
// THEN all pendingRequests values are 0 (every routed request was absorbed)
func TestClusterSimulator_PendingRequests_DrainsToZeroAfterProcessing(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon: 10000000, Seed: 42,
			TotalKVBlocks: 100, BlockSizeTokens: 16,
			MaxRunningReqs: 10, MaxScheduledTokens: 2048,
			BetaCoeffs: []float64{1000, 10, 5}, AlphaCoeffs: []float64{100, 50, 25},
		},
		NumInstances:         2,
		RoutingPolicy:        "weighted",
		RoutingScorerConfigs: sim.DefaultScorerConfigs(),
	}
	workload := &sim.GuideLLMConfig{
		Rate: 2.0 / 1e6, NumRequests: 6,
		PromptTokens: 16, OutputTokens: 8,
		PromptTokensStdDev: 0, OutputTokensStdDev: 0,
		PromptTokensMin: 16, PromptTokensMax: 16,
		OutputTokensMin: 8, OutputTokensMax: 8,
	}
	cs := NewClusterSimulator(config, workload, "")

	mustRun(t, cs)

	for instID, pending := range cs.pendingRequests {
		if pending != 0 {
			t.Errorf("instance %s: pendingRequests = %d after completion, want 0", instID, pending)
		}
	}

	// Sanity check: requests were actually processed
	m := cs.AggregatedMetrics()
	if m.CompletedRequests == 0 {
		t.Error("no requests completed — test setup issue")
	}
}

// TestClusterSimulator_PendingRequests_VisibleInRoutingState verifies:
// GIVEN a 1-instance cluster with pre-generated requests at identical timestamps
//
//	and non-zero routing latency (so routing decisions overlap with pending state)
//
// WHEN routing decisions are traced
// THEN at least one routing decision observes PendingRequests > 0
//
// Design: With RoutingLatency=100, request N's routing decision occurs at T_N + 100.
// If request N+1 arrives at T_N+1 < T_N + 100 + queueing_delay, the QueuedEvent from
// request N hasn't fired yet, so PendingRequests > 0 is visible to request N+1's
// routing decision.
func TestClusterSimulator_PendingRequests_VisibleInRoutingState(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon: 10000000, Seed: 42,
			TotalKVBlocks: 100, BlockSizeTokens: 16,
			MaxRunningReqs: 10, MaxScheduledTokens: 2048,
			BetaCoeffs: []float64{1000, 10, 5}, AlphaCoeffs: []float64{100, 50, 25},
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

	cs := NewClusterSimulator(config, &sim.GuideLLMConfig{
		Rate: 1.0 / 1e6, NumRequests: 0,
		PromptTokens: 16, OutputTokens: 8,
		PromptTokensStdDev: 0, OutputTokensStdDev: 0,
		PromptTokensMin: 16, PromptTokensMax: 16,
		OutputTokensMin: 8, OutputTokensMax: 8,
	}, "")
	cs.SetPreGeneratedRequests(reqs)

	mustRun(t, cs)

	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}

	// Check if any candidate score in any routing record has PendingRequests > 0
	foundPending := false
	for _, r := range tr.Routings {
		for _, c := range r.Candidates {
			if c.PendingRequests > 0 {
				foundPending = true
				break
			}
		}
		if foundPending {
			break
		}
	}

	if !foundPending {
		t.Error("expected at least one routing decision to observe PendingRequests > 0, " +
			"but all candidates had PendingRequests = 0")
	}

	// All pending must drain to zero after completion
	for instID, pending := range cs.pendingRequests {
		if pending != 0 {
			t.Errorf("instance %s: pendingRequests = %d after completion, want 0", instID, pending)
		}
	}
}

// TestClusterSimulator_PendingRequests_CausalDecrement verifies:
// GIVEN a 1-instance cluster with routing latency and pre-generated requests at t=0
// WHEN requests are processed
// THEN pending drains to zero (each QueuedEvent triggered exactly one decrement)
// AND at least one routing decision observes PendingRequests > 0 (proving decrement
//
//	happens on QueuedEvent, not prematurely on ArrivalEvent)
func TestClusterSimulator_PendingRequests_CausalDecrement(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon: 10000000, Seed: 42,
			TotalKVBlocks: 100, BlockSizeTokens: 16,
			MaxRunningReqs: 10, MaxScheduledTokens: 2048,
			BetaCoeffs: []float64{1000, 10, 5}, AlphaCoeffs: []float64{100, 50, 25},
		},
		NumInstances:    1,
		RoutingLatency:  100, // Creates overlap window where pending is visible
		TraceLevel:      "decisions",
		CounterfactualK: 1,
	}

	// Pre-generate requests all at t=0 so they route in quick succession.
	// With RoutingLatency=100, routing decisions occur at t=100, and the QueuedEvent
	// from the first request won't have fired yet when the second routes.
	reqs := make([]*sim.Request, 4)
	for i := range reqs {
		reqs[i] = &sim.Request{
			ID:           "pending_req_" + string(rune('0'+i)),
			ArrivalTime:  0,
			InputTokens:  make([]int, 16),
			OutputTokens: make([]int, 8),
			State:        sim.StateQueued,
		}
	}

	cs := NewClusterSimulator(config, &sim.GuideLLMConfig{
		Rate: 1.0 / 1e6, NumRequests: 0,
		PromptTokens: 16, OutputTokens: 8,
		PromptTokensStdDev: 0, OutputTokensStdDev: 0,
		PromptTokensMin: 16, PromptTokensMax: 16,
		OutputTokensMin: 8, OutputTokensMax: 8,
	}, "")
	cs.SetPreGeneratedRequests(reqs)

	mustRun(t, cs)

	// After simulation, all pending must be zero
	for instID, pending := range cs.pendingRequests {
		if pending != 0 {
			t.Errorf("instance %s: pendingRequests = %d, want 0", instID, pending)
		}
	}

	// Verify routing decisions observed PendingRequests > 0 (proving the decrement
	// doesn't happen prematurely on ArrivalEvent — it waits for QueuedEvent)
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}

	foundNonZero := false
	for _, r := range tr.Routings {
		for _, c := range r.Candidates {
			if c.PendingRequests > 0 {
				foundNonZero = true
			}
		}
	}

	if !foundNonZero {
		t.Error("expected at least one routing decision to observe PendingRequests > 0, " +
			"but all candidates had PendingRequests = 0")
	}
}

// TestClusterSimulator_PendingRequests_CounterfactualIncludesPending verifies:
// GIVEN tracing with counterfactual analysis and routing latency (to create pending state)
// WHEN CandidateScore is recorded during routing decisions
// THEN at least one candidate has PendingRequests > 0 (proving the field is populated)
func TestClusterSimulator_PendingRequests_CounterfactualIncludesPending(t *testing.T) {
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon: 10000000, Seed: 42,
			TotalKVBlocks: 100, BlockSizeTokens: 16,
			MaxRunningReqs: 10, MaxScheduledTokens: 2048,
			BetaCoeffs: []float64{1000, 10, 5}, AlphaCoeffs: []float64{100, 50, 25},
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

	cs := NewClusterSimulator(config, &sim.GuideLLMConfig{
		Rate: 1.0 / 1e6, NumRequests: 0,
		PromptTokens: 16, OutputTokens: 8,
		PromptTokensStdDev: 0, OutputTokensStdDev: 0,
		PromptTokensMin: 16, PromptTokensMax: 16,
		OutputTokensMin: 8, OutputTokensMax: 8,
	}, "")
	cs.SetPreGeneratedRequests(reqs)

	mustRun(t, cs)

	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}

	// Verify at least one candidate has PendingRequests > 0 (proving the field
	// is populated from actual cluster state, not just defaulting to zero)
	foundPending := false
	totalCandidates := 0
	for _, r := range tr.Routings {
		for _, c := range r.Candidates {
			totalCandidates++
			if c.PendingRequests > 0 {
				foundPending = true
			}
		}
	}
	if totalCandidates == 0 {
		t.Error("expected candidates in routing records with counterfactual-k=1")
	}
	if !foundPending {
		t.Error("expected at least one candidate with PendingRequests > 0, " +
			"but all were 0 — field may not be populated from cluster state")
	}
}

// TestComputeCounterfactual_IncludesPendingRequests verifies:
// GIVEN snapshots with different PendingRequests values and explicit scores
// WHEN computeCounterfactual builds the candidate list
// THEN each candidate's PendingRequests matches its source snapshot
func TestComputeCounterfactual_IncludesPendingRequests(t *testing.T) {
	snapshots := []sim.RoutingSnapshot{
		{ID: "inst_0", QueueDepth: 2, BatchSize: 1, PendingRequests: 3},
		{ID: "inst_1", QueueDepth: 0, BatchSize: 0, PendingRequests: 0},
	}
	scores := map[string]float64{"inst_0": 0.3, "inst_1": 0.8}

	candidates, _ := computeCounterfactual("inst_0", scores, snapshots, 2)

	if len(candidates) != 2 {
		t.Fatalf("expected 2 candidates, got %d", len(candidates))
	}

	// Find inst_0's candidate and verify PendingRequests is preserved
	for _, c := range candidates {
		if c.InstanceID == "inst_0" {
			if c.PendingRequests != 3 {
				t.Errorf("inst_0 PendingRequests = %d, want 3", c.PendingRequests)
			}
		}
		if c.InstanceID == "inst_1" {
			if c.PendingRequests != 0 {
				t.Errorf("inst_1 PendingRequests = %d, want 0", c.PendingRequests)
			}
		}
	}
}
