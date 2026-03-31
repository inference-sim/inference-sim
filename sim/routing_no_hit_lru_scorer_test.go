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
