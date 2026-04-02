package sim

import "testing"

// TestPrecisePrefixCache_MinMaxNormalization verifies BC-1: min-max normalization
// produces correct scores for varying cached block counts.
func TestPrecisePrefixCache_MinMaxNormalization(t *testing.T) {
	cacheQueryFn := cacheQueryFn{
		"inst-0": func(tokens []int) int { return 5 },
		"inst-1": func(tokens []int) int { return 3 },
		"inst-2": func(tokens []int) int { return 0 },
	}
	scorer, _ := newPrecisePrefixCacheScorer(cacheQueryFn)
	req := &Request{ID: "r1", InputTokens: []int{1, 2, 3, 4, 5}}
	snapshots := []RoutingSnapshot{
		{ID: "inst-0"},
		{ID: "inst-1"},
		{ID: "inst-2"},
	}
	scores := scorer(req, snapshots)

	tests := []struct {
		id    string
		want  float64
		descr string
	}{
		{"inst-0", 1.0, "highest cached blocks → 1.0"},
		{"inst-1", 0.6, "intermediate → (3-0)/(5-0) = 0.6"},
		{"inst-2", 0.0, "lowest cached blocks → 0.0"},
	}
	for _, tt := range tests {
		got := scores[tt.id]
		if got < tt.want-0.001 || got > tt.want+0.001 {
			t.Errorf("%s: got %.3f, want %.3f (%s)", tt.id, got, tt.want, tt.descr)
		}
	}
}

// TestPrecisePrefixCache_AllEqual verifies BC-2: all-equal cached blocks produce uniform scores.
// All-zero → 0.5 (neutral, no cache data to differentiate).
// All-equal nonzero → 1.0 (all instances equally good).
func TestPrecisePrefixCache_AllEqual(t *testing.T) {
	tests := []struct {
		name   string
		counts map[string]int
		want   float64
	}{
		{"all zero", map[string]int{"a": 0, "b": 0, "c": 0}, 0.5},
		{"all equal nonzero", map[string]int{"a": 4, "b": 4, "c": 4}, 1.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cqf := make(cacheQueryFn, len(tt.counts))
			ids := []string{"a", "b", "c"}
			for _, id := range ids {
				count := tt.counts[id]
				cqf[id] = func(tokens []int) int { return count }
			}
			scorer, _ := newPrecisePrefixCacheScorer(cqf)
			req := &Request{ID: "r1", InputTokens: []int{1}}
			var snapshots []RoutingSnapshot
			for _, id := range ids {
				snapshots = append(snapshots, RoutingSnapshot{ID: id})
			}
			scores := scorer(req, snapshots)
			for id, score := range scores {
				if score != tt.want {
					t.Errorf("instance %s: got %.3f, want %.3f (all-equal case)", id, score, tt.want)
				}
			}
		})
	}
}

// TestPrecisePrefixCache_ObserverIsNil verifies BC-8: no observer for stateless scorer.
func TestPrecisePrefixCache_ObserverIsNil(t *testing.T) {
	_, obs := newPrecisePrefixCacheScorer(nil)
	if obs != nil {
		t.Error("expected nil observer for precise-prefix-cache scorer")
	}
}

// TestPrecisePrefixCache_SingleInstance verifies single instance scores 1.0.
func TestPrecisePrefixCache_SingleInstance(t *testing.T) {
	cacheQueryFn := cacheQueryFn{
		"only": func(tokens []int) int { return 3 },
	}
	scorer, _ := newPrecisePrefixCacheScorer(cacheQueryFn)
	scores := scorer(&Request{ID: "r1", InputTokens: []int{1}}, []RoutingSnapshot{{ID: "only"}})
	if scores["only"] != 1.0 {
		t.Errorf("single instance: got %.3f, want 1.0", scores["only"])
	}
}

// TestPrecisePrefixCache_NilCacheQueryFn verifies nil cacheQueryFn → all 1.0.
func TestPrecisePrefixCache_NilCacheQueryFn(t *testing.T) {
	scorer, _ := newPrecisePrefixCacheScorer(nil)
	scores := scorer(&Request{ID: "r1", InputTokens: []int{1}}, []RoutingSnapshot{
		{ID: "a"}, {ID: "b"},
	})
	for id, score := range scores {
		if score != 1.0 {
			t.Errorf("nil cacheQueryFn: instance %s got %.3f, want 1.0", id, score)
		}
	}
}

// TestPrecisePrefixCache_MissingInstanceInCacheQueryFn verifies that an instance
// present in snapshots but absent from cacheQueryFn is treated as having zero
// cached blocks (not a panic or undefined behavior).
func TestPrecisePrefixCache_MissingInstanceInCacheQueryFn(t *testing.T) {
	// cacheQueryFn has "a" and "b" but NOT "c"
	cqf := cacheQueryFn{
		"a": func(tokens []int) int { return 5 },
		"b": func(tokens []int) int { return 0 },
	}
	scorer, _ := newPrecisePrefixCacheScorer(cqf)
	scores := scorer(
		&Request{ID: "r1", InputTokens: []int{1}},
		[]RoutingSnapshot{{ID: "a"}, {ID: "b"}, {ID: "c"}},
	)
	// "c" missing from cacheQueryFn → treated as 0 cached blocks (same as "b")
	if scores["a"] != 1.0 {
		t.Errorf("a: got %.3f, want 1.0", scores["a"])
	}
	if scores["c"] != 0.0 {
		t.Errorf("c (missing): got %.3f, want 0.0 (treated as zero cache)", scores["c"])
	}
}
