package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestComputeCounterfactual_WithScores_TopKSortedByScore(t *testing.T) {
	// GIVEN 3 instances with explicit scores from WeightedScoring
	snapshots := []sim.RoutingSnapshot{
		{ID: "i_0", QueueDepth: 5, BatchSize: 2, KVUtilization: 0.3},
		{ID: "i_1", QueueDepth: 1, BatchSize: 0, KVUtilization: 0.1},
		{ID: "i_2", QueueDepth: 3, BatchSize: 1, KVUtilization: 0.5},
	}
	scores := map[string]float64{"i_0": 0.4, "i_1": 0.9, "i_2": 0.6}

	// WHEN computing top-2 counterfactual, chosen is i_2 (score 0.6)
	candidates, regret := computeCounterfactual("i_2", scores, snapshots, 2)

	// THEN top-2 sorted by score desc: i_1 (0.9), i_2 (0.6)
	if len(candidates) != 2 {
		t.Fatalf("expected 2 candidates, got %d", len(candidates))
	}
	if candidates[0].InstanceID != "i_1" || candidates[0].Score != 0.9 {
		t.Errorf("first candidate: got %s/%.1f, want i_1/0.9", candidates[0].InstanceID, candidates[0].Score)
	}
	if candidates[1].InstanceID != "i_2" || candidates[1].Score != 0.6 {
		t.Errorf("second candidate: got %s/%.1f, want i_2/0.6", candidates[1].InstanceID, candidates[1].Score)
	}

	// THEN regret = best(0.9) - chosen(0.6) = 0.3
	if regret < 0.299 || regret > 0.301 {
		t.Errorf("expected regret ~0.3, got %.6f", regret)
	}
}

func TestComputeCounterfactual_ChosenIsBest_ZeroRegret(t *testing.T) {
	// GIVEN chosen instance has the highest score
	snapshots := []sim.RoutingSnapshot{
		{ID: "i_0", QueueDepth: 1},
		{ID: "i_1", QueueDepth: 5},
	}
	scores := map[string]float64{"i_0": 0.8, "i_1": 0.2}

	// WHEN chosen is i_0 (best score)
	_, regret := computeCounterfactual("i_0", scores, snapshots, 2)

	// THEN regret = 0
	if regret != 0 {
		t.Errorf("expected 0 regret, got %.6f", regret)
	}
}

func TestComputeCounterfactual_NilScores_UsesLoadFallback(t *testing.T) {
	// GIVEN nil scores (RoundRobin/LeastLoaded) and different loads
	snapshots := []sim.RoutingSnapshot{
		{ID: "i_0", QueueDepth: 10, BatchSize: 5}, // load=15, score=-15
		{ID: "i_1", QueueDepth: 1, BatchSize: 0},  // load=1,  score=-1
		{ID: "i_2", QueueDepth: 3, BatchSize: 2},  // load=5,  score=-5
	}

	// WHEN computing with nil scores, chosen is i_0 (worst load)
	candidates, regret := computeCounterfactual("i_0", nil, snapshots, 3)

	// THEN sorted by score desc (least loaded first): i_1(-1), i_2(-5), i_0(-15)
	if len(candidates) != 3 {
		t.Fatalf("expected 3 candidates, got %d", len(candidates))
	}
	if candidates[0].InstanceID != "i_1" {
		t.Errorf("first candidate: got %s, want i_1 (least loaded)", candidates[0].InstanceID)
	}
	if candidates[2].InstanceID != "i_0" {
		t.Errorf("last candidate: got %s, want i_0 (most loaded)", candidates[2].InstanceID)
	}

	// THEN regret = best(-1) - chosen(-15) = 14
	if regret < 13.99 || regret > 14.01 {
		t.Errorf("expected regret ~14, got %.6f", regret)
	}
}

func TestComputeCounterfactual_NilScores_IncludesPendingRequests(t *testing.T) {
	// GIVEN nil scores and instances where PendingRequests breaks the tie (#175)
	snapshots := []sim.RoutingSnapshot{
		{ID: "i_0", QueueDepth: 5, BatchSize: 3, PendingRequests: 4}, // load=12, score=-12
		{ID: "i_1", QueueDepth: 5, BatchSize: 3, PendingRequests: 0}, // load=8,  score=-8
	}

	// WHEN computing counterfactual with nil scores, chosen is i_0
	candidates, regret := computeCounterfactual("i_0", nil, snapshots, 2)

	// THEN i_1 ranks first (lower load including PendingRequests)
	if len(candidates) != 2 {
		t.Fatalf("expected 2 candidates, got %d", len(candidates))
	}
	if candidates[0].InstanceID != "i_1" {
		t.Errorf("first candidate: got %s, want i_1 (lower total load)", candidates[0].InstanceID)
	}
	// THEN PendingRequests is captured in the CandidateScore record
	if candidates[0].PendingRequests != 0 {
		t.Errorf("candidate i_1 PendingRequests: got %d, want 0", candidates[0].PendingRequests)
	}
	if candidates[1].PendingRequests != 4 {
		t.Errorf("candidate i_0 PendingRequests: got %d, want 4", candidates[1].PendingRequests)
	}

	// THEN regret = best(-8) - chosen(-12) = 4
	if regret < 3.99 || regret > 4.01 {
		t.Errorf("expected regret ~4, got %.6f", regret)
	}
}

func TestComputeCounterfactual_KZero_ReturnsNilAndZero(t *testing.T) {
	// GIVEN k=0
	snapshots := []sim.RoutingSnapshot{{ID: "i_0"}}

	// WHEN computing with k=0
	candidates, regret := computeCounterfactual("i_0", nil, snapshots, 0)

	// THEN nil candidates and zero regret
	if candidates != nil {
		t.Errorf("expected nil candidates, got %v", candidates)
	}
	if regret != 0 {
		t.Errorf("expected 0 regret, got %.6f", regret)
	}
}

func TestComputeCounterfactual_KExceedsInstances_ClampsToLen(t *testing.T) {
	// GIVEN 2 instances but k=10
	snapshots := []sim.RoutingSnapshot{
		{ID: "i_0", QueueDepth: 1},
		{ID: "i_1", QueueDepth: 2},
	}
	scores := map[string]float64{"i_0": 0.8, "i_1": 0.3}

	// WHEN k > len(snapshots)
	candidates, _ := computeCounterfactual("i_1", scores, snapshots, 10)

	// THEN returns all instances (clamped)
	if len(candidates) != 2 {
		t.Fatalf("expected 2 candidates (clamped), got %d", len(candidates))
	}
}

func TestComputeCounterfactual_TiedScores_BreaksByInstanceID(t *testing.T) {
	// GIVEN equal scores, different IDs
	snapshots := []sim.RoutingSnapshot{
		{ID: "i_b", QueueDepth: 1},
		{ID: "i_a", QueueDepth: 1},
	}
	scores := map[string]float64{"i_a": 0.5, "i_b": 0.5}

	// WHEN computing candidates
	candidates, _ := computeCounterfactual("i_b", scores, snapshots, 2)

	// THEN tie-broken by ID ascending: i_a before i_b
	if candidates[0].InstanceID != "i_a" {
		t.Errorf("expected i_a first (ID tie-break), got %s", candidates[0].InstanceID)
	}
}

func TestCopyScores_NilInput_ReturnsNil(t *testing.T) {
	if got := copyScores(nil); got != nil {
		t.Errorf("expected nil, got %v", got)
	}
}

func TestCopyScores_ReturnsIndependentCopy(t *testing.T) {
	original := map[string]float64{"a": 1.0, "b": 2.0}
	copied := copyScores(original)

	// Mutate original
	original["a"] = 99.0

	// Copy should be unaffected
	if copied["a"] != 1.0 {
		t.Errorf("copy was mutated: got %.1f, want 1.0", copied["a"])
	}
}
