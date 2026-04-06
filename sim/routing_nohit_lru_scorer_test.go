package sim

import "testing"

// TestNoHitLRU_ColdRequestDistribution verifies BC-3: cold requests distribute
// to never-used instances first, then by LRU position.
func TestNoHitLRU_ColdRequestDistribution(t *testing.T) {
	// All instances return 0 cached blocks (cold)
	cacheQueryFn := cacheQueryFn{
		"a": func(tokens []int) int { return 0 },
		"b": func(tokens []int) int { return 0 },
		"c": func(tokens []int) int { return 0 },
	}
	scorer, obs := newNoHitLRUScorer(cacheQueryFn)
	snapshots := []RoutingSnapshot{{ID: "a"}, {ID: "b"}, {ID: "c"}}

	// First cold request: all never-used → a=1.0, b=0.5, c=0.0 (positional)
	req1 := &Request{ID: "r1", InputTokens: []int{1}}
	scores1 := scorer(req1, snapshots)
	if scores1["a"] != 1.0 {
		t.Errorf("r1: a got %.3f, want 1.0 (first never-used)", scores1["a"])
	}
	if scores1["b"] != 0.5 {
		t.Errorf("r1: b got %.3f, want 0.5 (second never-used)", scores1["b"])
	}
	if scores1["c"] != 0.0 {
		t.Errorf("r1: c got %.3f, want 0.0 (third never-used)", scores1["c"])
	}

	// Simulate routing to "a" — observer updates LRU
	obs(req1, "a")

	// Second cold request: "a" is now most-recently-used
	// b, c are never-used (rank 0, 1), a is used (rank 2)
	req2 := &Request{ID: "r2", InputTokens: []int{1}}
	scores2 := scorer(req2, snapshots)
	if scores2["b"] != 1.0 {
		t.Errorf("r2: b got %.3f, want 1.0 (first never-used)", scores2["b"])
	}
	if scores2["c"] != 0.5 {
		t.Errorf("r2: c got %.3f, want 0.5 (second never-used)", scores2["c"])
	}
	if scores2["a"] != 0.0 {
		t.Errorf("r2: a got %.3f, want 0.0 (most-recently-used)", scores2["a"])
	}

	// Route to "b", then third request: a is older than b
	obs(req2, "b")
	req3 := &Request{ID: "r3", InputTokens: []int{1}}
	scores3 := scorer(req3, snapshots)
	// c is never-used (rank 0 → 1.0), a is oldest-used (rank 1 → 0.5), b is newest-used (rank 2 → 0.0)
	if scores3["c"] != 1.0 {
		t.Errorf("r3: c got %.3f, want 1.0 (never-used)", scores3["c"])
	}
	if scores3["a"] != 0.5 {
		t.Errorf("r3: a got %.3f, want 0.5 (oldest-used)", scores3["a"])
	}
	if scores3["b"] != 0.0 {
		t.Errorf("r3: b got %.3f, want 0.0 (newest-used)", scores3["b"])
	}
}

// TestNoHitLRU_WarmRequestNeutral verifies BC-4: warm requests → all 0.5.
func TestNoHitLRU_WarmRequestNeutral(t *testing.T) {
	cacheQueryFn := cacheQueryFn{
		"a": func(tokens []int) int { return 3 }, // has cached blocks
		"b": func(tokens []int) int { return 0 },
	}
	scorer, _ := newNoHitLRUScorer(cacheQueryFn)
	scores := scorer(
		&Request{ID: "r1", InputTokens: []int{1}},
		[]RoutingSnapshot{{ID: "a"}, {ID: "b"}},
	)
	for id, score := range scores {
		if score != 0.5 {
			t.Errorf("warm request: instance %s got %.3f, want 0.5", id, score)
		}
	}
}

// TestNoHitLRU_ObserverColdOnly verifies BC-5: observer doesn't update LRU on warm requests.
func TestNoHitLRU_ObserverColdOnly(t *testing.T) {
	// Use a mutable flag so we can switch from warm to cold mid-test.
	warmA := true
	cacheQueryFn := cacheQueryFn{
		"a": func(tokens []int) int {
			if warmA {
				return 3
			}
			return 0
		},
		"b": func(tokens []int) int { return 0 },
	}
	scorer, obs := newNoHitLRUScorer(cacheQueryFn)
	snapshots := []RoutingSnapshot{{ID: "a"}, {ID: "b"}}

	// Route a warm request to "a" — observer should NOT update LRU
	req := &Request{ID: "r1", InputTokens: []int{1}}
	scorer(req, snapshots)
	obs(req, "a") // should NOT update LRU (warm)

	// Switch to cold and verify "a" is still never-used (LRU not updated by warm routing)
	warmA = false
	req2 := &Request{ID: "r2", InputTokens: []int{1}}
	scores := scorer(req2, snapshots)
	if scores["a"] != 1.0 {
		t.Errorf("a got %.3f, want 1.0 (never-used — warm routing should not update LRU)", scores["a"])
	}
}

// TestNoHitLRU_SingleInstanceCold verifies single instance cold → score 1.0.
func TestNoHitLRU_SingleInstanceCold(t *testing.T) {
	cacheQueryFn := cacheQueryFn{
		"only": func(tokens []int) int { return 0 },
	}
	scorer, _ := newNoHitLRUScorer(cacheQueryFn)
	scores := scorer(
		&Request{ID: "r1", InputTokens: []int{1}},
		[]RoutingSnapshot{{ID: "only"}},
	)
	if scores["only"] != 1.0 {
		t.Errorf("single instance cold: got %.3f, want 1.0", scores["only"])
	}
}

// TestNoHitLRU_NilCacheQueryFn verifies nil cacheQueryFn → all 0.5 (neutral).
func TestNoHitLRU_NilCacheQueryFn(t *testing.T) {
	scorer, _ := newNoHitLRUScorer(nil)
	scores := scorer(
		&Request{ID: "r1", InputTokens: []int{1}},
		[]RoutingSnapshot{{ID: "a"}, {ID: "b"}},
	)
	for id, score := range scores {
		if score != 0.5 {
			t.Errorf("nil cacheQueryFn: instance %s got %.3f, want 0.5", id, score)
		}
	}
}

// TestNoHitLRU_Determinism verifies INV-6: identical routing sequences produce
// identical LRU state and scores.
func TestNoHitLRU_Determinism(t *testing.T) {
	makeCacheQueryFn := func() cacheQueryFn {
		return cacheQueryFn{
			"a": func(tokens []int) int { return 0 },
			"b": func(tokens []int) int { return 0 },
			"c": func(tokens []int) int { return 0 },
		}
	}
	snapshots := []RoutingSnapshot{{ID: "a"}, {ID: "b"}, {ID: "c"}}
	requests := []*Request{
		{ID: "r1", InputTokens: []int{1}},
		{ID: "r2", InputTokens: []int{2}},
		{ID: "r3", InputTokens: []int{3}},
	}
	routeTargets := []string{"a", "b", "a"}

	// Run sequence twice with fresh state
	var scores1, scores2 []map[string]float64
	for _, run := range []int{1, 2} {
		scorer, obs := newNoHitLRUScorer(makeCacheQueryFn())
		var runScores []map[string]float64
		for i, req := range requests {
			s := scorer(req, snapshots)
			obs(req, routeTargets[i])
			// Copy scores
			cp := make(map[string]float64, len(s))
			for k, v := range s {
				cp[k] = v
			}
			runScores = append(runScores, cp)
		}
		if run == 1 {
			scores1 = runScores
		} else {
			scores2 = runScores
		}
	}

	// Verify identical
	for i := range scores1 {
		for id, v1 := range scores1[i] {
			if v2, ok := scores2[i][id]; !ok || v1 != v2 {
				t.Errorf("request %d, instance %s: run1=%.3f, run2=%.3f (determinism violation)", i, id, v1, v2)
			}
		}
	}
}

// TestNoHitLRU_LRURepromotion verifies that routing the same instance twice
// re-promotes it to most-recently-used, producing correct LRU rank scores.
func TestNoHitLRU_LRURepromotion(t *testing.T) {
	cqf := cacheQueryFn{
		"a": func(tokens []int) int { return 0 },
		"b": func(tokens []int) int { return 0 },
		"c": func(tokens []int) int { return 0 },
	}
	scorer, obs := newNoHitLRUScorer(cqf)
	snapshots := []RoutingSnapshot{{ID: "a"}, {ID: "b"}, {ID: "c"}}

	// Route: a, b, a (re-promote a to most-recently-used)
	for _, target := range []string{"a", "b", "a"} {
		req := &Request{ID: "r-" + target, InputTokens: []int{1}}
		scorer(req, snapshots)
		obs(req, target)
	}

	// After [a, b, a]: lruOrder = [a, b]. c is never-used.
	// c=1.0 (never-used, rank 0), b=0.5 (oldest-used, rank 1), a=0.0 (newest-used, rank 2)
	req := &Request{ID: "r-check", InputTokens: []int{1}}
	scores := scorer(req, snapshots)
	if scores["c"] != 1.0 {
		t.Errorf("c: got %.3f, want 1.0 (never-used)", scores["c"])
	}
	if scores["b"] != 0.5 {
		t.Errorf("b: got %.3f, want 0.5 (oldest-used)", scores["b"])
	}
	if scores["a"] != 0.0 {
		t.Errorf("a: got %.3f, want 0.0 (newest-used, re-promoted)", scores["a"])
	}
}

// TestNoHitLRU_ObserverMismatchedReqID verifies that the observer ignores calls
// with a request ID different from the last scored request (safety net against
// stale LRU updates).
func TestNoHitLRU_ObserverMismatchedReqID(t *testing.T) {
	cqf := cacheQueryFn{
		"a": func(tokens []int) int { return 0 },
		"b": func(tokens []int) int { return 0 },
	}
	scorer, obs := newNoHitLRUScorer(cqf)
	snapshots := []RoutingSnapshot{{ID: "a"}, {ID: "b"}}

	// Score req1 (cold) — sets lastReqID = "r1"
	req1 := &Request{ID: "r1", InputTokens: []int{1}}
	scorer(req1, snapshots)

	// Call observer with DIFFERENT request — should NOT update LRU
	req2 := &Request{ID: "r2", InputTokens: []int{1}}
	obs(req2, "a")

	// Score req3 (cold) — "a" should still be never-used (observer was no-op)
	req3 := &Request{ID: "r3", InputTokens: []int{1}}
	scores := scorer(req3, snapshots)
	if scores["a"] != 1.0 {
		t.Errorf("a: got %.3f, want 1.0 (never-used — mismatched observer should be no-op)", scores["a"])
	}
}
