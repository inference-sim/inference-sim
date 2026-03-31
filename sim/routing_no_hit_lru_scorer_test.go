package sim_test

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestNoHitLRU_WarmRequest_NeutralScores verifies warm requests get 0.5 for all instances.
func TestNoHitLRU_WarmRequest_NeutralScores(t *testing.T) {
	// Instance A has cached blocks → warm request
	cqf := makeCacheQueryFn(map[string]int{"A": 5, "B": 0})
	policy := sim.NewRoutingPolicy("weighted", []sim.ScorerConfig{
		{Name: "no-hit-lru", Weight: 1.0},
	}, 16, nil, cqf)

	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2}}
	state := &sim.RouterState{
		Snapshots: []sim.RoutingSnapshot{{ID: "A"}, {ID: "B"}},
	}
	decision := policy.Route(req, state)

	for _, id := range []string{"A", "B"} {
		if s := decision.Scores[id]; s < 0.49 || s > 0.51 {
			t.Errorf("expected %s score ≈ 0.5 (warm), got %f", id, s)
		}
	}
}

// TestNoHitLRU_ColdRequest_NeverUsedEndpointsPreferred verifies never-used endpoints
// get the highest score on cold requests.
func TestNoHitLRU_ColdRequest_NeverUsedEndpointsPreferred(t *testing.T) {
	cqf := makeCacheQueryFn(map[string]int{"A": 0, "B": 0, "C": 0})
	policy := sim.NewRoutingPolicy("weighted", []sim.ScorerConfig{
		{Name: "no-hit-lru", Weight: 1.0},
	}, 16, nil, cqf)

	state := &sim.RouterState{
		Snapshots: []sim.RoutingSnapshot{{ID: "A"}, {ID: "B"}, {ID: "C"}},
	}

	// First cold request: all never-used, all should score 1.0
	req1 := &sim.Request{ID: "r1", InputTokens: []int{1}}
	d1 := policy.Route(req1, state)

	for _, id := range []string{"A", "B", "C"} {
		if s := d1.Scores[id]; s < 0.99 {
			t.Errorf("first request: expected %s score ≈ 1.0 (all never-used), got %f", id, s)
		}
	}

	// After routing to one instance, that instance is now "used".
	// Next cold request: the used instance should score lower.
	req2 := &sim.Request{ID: "r2", InputTokens: []int{2}}
	d2 := policy.Route(req2, state)

	usedID := d1.TargetInstance
	for _, snap := range state.Snapshots {
		if snap.ID != usedID {
			// Never-used instances should score higher than the used one
			if d2.Scores[snap.ID] < d2.Scores[usedID] {
				t.Errorf("never-used %s (%.3f) should score >= used %s (%.3f)",
					snap.ID, d2.Scores[snap.ID], usedID, d2.Scores[usedID])
			}
		}
	}
}

// TestNoHitLRU_SingleEndpoint_ScoresOne verifies single endpoint always scores 1.0.
func TestNoHitLRU_SingleEndpoint_ScoresOne(t *testing.T) {
	cqf := makeCacheQueryFn(map[string]int{"A": 0})
	policy := sim.NewRoutingPolicy("weighted", []sim.ScorerConfig{
		{Name: "no-hit-lru", Weight: 1.0},
	}, 16, nil, cqf)

	req := &sim.Request{ID: "r1", InputTokens: []int{1}}
	state := &sim.RouterState{
		Snapshots: []sim.RoutingSnapshot{{ID: "A"}},
	}
	decision := policy.Route(req, state)

	if s := decision.Scores["A"]; s < 0.99 {
		t.Errorf("expected single endpoint score ≈ 1.0, got %f", s)
	}
}

// TestNoHitLRU_NilCacheQueryFn_Panics verifies factory panics without cacheQueryFn.
func TestNoHitLRU_NilCacheQueryFn_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for nil cacheQueryFn")
		}
	}()
	sim.NewRoutingPolicy("weighted", []sim.ScorerConfig{
		{Name: "no-hit-lru", Weight: 1.0},
	}, 16, nil)
}

// TestNoHitLRU_RankFiltersNonSnapshotInstances verifies that LRU ranking only considers
// instances present in the current routing snapshots. This prevents rank inflation from
// non-routable or filtered-out instances that may exist in the LRU.
func TestNoHitLRU_RankFiltersNonSnapshotInstances(t *testing.T) {
	// cacheQueryFn includes all 4 instances, but snapshots only include A, B, C
	cqf := makeCacheQueryFn(map[string]int{"A": 0, "B": 0, "C": 0, "D": 0})
	policy := sim.NewRoutingPolicy("weighted", []sim.ScorerConfig{
		{Name: "no-hit-lru", Weight: 1.0},
	}, 16, nil, cqf)

	// Route to D first (using full snapshot set) to put D in the LRU
	fullState := &sim.RouterState{
		Snapshots: []sim.RoutingSnapshot{{ID: "A"}, {ID: "B"}, {ID: "C"}, {ID: "D"}},
	}
	req1 := &sim.Request{ID: "r1", InputTokens: []int{1}}
	policy.Route(req1, fullState)

	// Route with D as target (route a second request to make sure D is in LRU)
	req2 := &sim.Request{ID: "r2", InputTokens: []int{2}}
	policy.Route(req2, fullState)

	// Now route with D filtered out of snapshots (e.g., non-routable).
	// Scores for A, B, C should still span the full [0, 1] range.
	filteredState := &sim.RouterState{
		Snapshots: []sim.RoutingSnapshot{{ID: "A"}, {ID: "B"}, {ID: "C"}},
	}
	req3 := &sim.Request{ID: "r3", InputTokens: []int{3}}
	d3 := policy.Route(req3, filteredState)

	// At least one instance should score 1.0 (never-used instances among the candidates)
	maxScore := 0.0
	for _, snap := range filteredState.Snapshots {
		if d3.Scores[snap.ID] > maxScore {
			maxScore = d3.Scores[snap.ID]
		}
	}
	if maxScore < 0.99 {
		t.Errorf("expected at least one candidate to score ≈ 1.0, max was %f (rank filtering may be broken)", maxScore)
	}

	// No score should be negative (would indicate non-snapshot entries inflating ranks)
	for _, snap := range filteredState.Snapshots {
		if d3.Scores[snap.ID] < 0 {
			t.Errorf("score for %s is negative (%f) — non-snapshot LRU entries are inflating ranks",
				snap.ID, d3.Scores[snap.ID])
		}
	}
}

// TestNoHitLRU_CombinedWithPrecisePrefix verifies the two scorers compose correctly.
func TestNoHitLRU_CombinedWithPrecisePrefix(t *testing.T) {
	// A has cached blocks, B and C do not
	cqf := makeCacheQueryFn(map[string]int{"A": 10, "B": 0, "C": 0})
	policy := sim.NewRoutingPolicy("weighted", []sim.ScorerConfig{
		{Name: "precise-prefix-cache", Weight: 2.0},
		{Name: "no-hit-lru", Weight: 1.0},
	}, 16, nil, cqf)

	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2}}
	state := &sim.RouterState{
		Snapshots: []sim.RoutingSnapshot{{ID: "A"}, {ID: "B"}, {ID: "C"}},
	}
	decision := policy.Route(req, state)

	// A has highest prefix score (1.0) + warm neutral (0.5) → should win
	if decision.TargetInstance != "A" {
		t.Errorf("expected target A (warm hit), got %s", decision.TargetInstance)
	}
}
