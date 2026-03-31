package sim_test

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// makeCacheQueryFn creates a CacheQueryFn from a static map of instance ID → cached block count.
func makeCacheQueryFn(counts map[string]int) sim.CacheQueryFn {
	cqf := make(sim.CacheQueryFn, len(counts))
	for id, count := range counts {
		count := count // capture
		cqf[id] = func(tokens []int) int { return count }
	}
	return cqf
}

// TestPrecisePrefixCache_MinMaxNormalization verifies that the precise-prefix-cache scorer
// uses min-max normalization: highest raw → 1.0, lowest raw → 0.0.
func TestPrecisePrefixCache_MinMaxNormalization(t *testing.T) {
	cqf := makeCacheQueryFn(map[string]int{"A": 10, "B": 5, "C": 0})
	policy := sim.NewRoutingPolicy("weighted", []sim.ScorerConfig{
		{Name: "precise-prefix-cache", Weight: 1.0},
	}, 16, nil, cqf)

	req := &sim.Request{ID: "r1", InputTokens: []int{1, 2, 3}}
	state := &sim.RouterState{
		Snapshots: []sim.RoutingSnapshot{
			{ID: "A"}, {ID: "B"}, {ID: "C"},
		},
	}
	decision := policy.Route(req, state)

	// Instance A has most cached blocks → should be selected
	if decision.TargetInstance != "A" {
		t.Errorf("expected target A (highest cache), got %s", decision.TargetInstance)
	}

	// Verify score values: A=1.0, B=0.5, C=0.0
	if s := decision.Scores["A"]; s < 0.99 || s > 1.01 {
		t.Errorf("expected A score ≈ 1.0, got %f", s)
	}
	if s := decision.Scores["B"]; s < 0.49 || s > 0.51 {
		t.Errorf("expected B score ≈ 0.5, got %f", s)
	}
	if s := decision.Scores["C"]; s < -0.01 || s > 0.01 {
		t.Errorf("expected C score ≈ 0.0, got %f", s)
	}
}

// TestPrecisePrefixCache_AllEqual_AllScoreOne verifies all-equal raw scores → all 1.0.
func TestPrecisePrefixCache_AllEqual_AllScoreOne(t *testing.T) {
	cqf := makeCacheQueryFn(map[string]int{"A": 3, "B": 3})
	policy := sim.NewRoutingPolicy("weighted", []sim.ScorerConfig{
		{Name: "precise-prefix-cache", Weight: 1.0},
	}, 16, nil, cqf)

	req := &sim.Request{ID: "r1", InputTokens: []int{1}}
	state := &sim.RouterState{
		Snapshots: []sim.RoutingSnapshot{{ID: "A"}, {ID: "B"}},
	}
	decision := policy.Route(req, state)

	for _, id := range []string{"A", "B"} {
		if s := decision.Scores[id]; s < 0.99 || s > 1.01 {
			t.Errorf("expected %s score ≈ 1.0 (all-equal), got %f", id, s)
		}
	}
}

// TestPrecisePrefixCache_ZeroCachedBlocks verifies that when no instance has cached blocks,
// all scores are 1.0 (all-equal case).
func TestPrecisePrefixCache_ZeroCachedBlocks(t *testing.T) {
	cqf := makeCacheQueryFn(map[string]int{"A": 0, "B": 0})
	policy := sim.NewRoutingPolicy("weighted", []sim.ScorerConfig{
		{Name: "precise-prefix-cache", Weight: 1.0},
	}, 16, nil, cqf)

	req := &sim.Request{ID: "r1", InputTokens: []int{1}}
	state := &sim.RouterState{
		Snapshots: []sim.RoutingSnapshot{{ID: "A"}, {ID: "B"}},
	}
	decision := policy.Route(req, state)

	for _, id := range []string{"A", "B"} {
		if s := decision.Scores[id]; s < 0.99 {
			t.Errorf("expected %s score ≈ 1.0 (all zero), got %f", id, s)
		}
	}
}

// TestPrecisePrefixCache_NilCacheQueryFn_Panics verifies factory panics without cacheQueryFn.
func TestPrecisePrefixCache_NilCacheQueryFn_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for nil cacheQueryFn")
		}
	}()
	sim.NewRoutingPolicy("weighted", []sim.ScorerConfig{
		{Name: "precise-prefix-cache", Weight: 1.0},
	}, 16, nil)
}
