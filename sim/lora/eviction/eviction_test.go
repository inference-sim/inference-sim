package eviction

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// The eviction package holds the pluggable adapter-eviction policies extracted
// from the resident set's hardcoded LRU (Backend Swap). These contract tests
// assert the observable behavior of the baseline lru policy and the registry,
// independent of internal representation.

// TestLRU_SelectsFirstCandidate verifies lru evicts the least-recently-used
// unpinned adapter — candidates[0], since candidates are LRU→MRU ordered (BC-1).
func TestLRU_SelectsFirstCandidate(t *testing.T) {
	victim, ok := lru{}.SelectVictim(sim.EvictionContext{
		Candidates: []string{"a", "b", "c"},
		RankOf:     func(string) (int, bool) { return 0, false },
	})
	if !ok || victim != "a" {
		t.Fatalf("SelectVictim([a,b,c]) = (%q, %v), want (\"a\", true)", victim, ok)
	}
}

// TestLRU_NoCandidates verifies lru reports no victim when nothing is evictable
// (every slot pinned or the set empty) — the unit half of BC-4.
func TestLRU_NoCandidates(t *testing.T) {
	victim, ok := lru{}.SelectVictim(sim.EvictionContext{
		Candidates: nil,
		RankOf:     func(string) (int, bool) { return 0, false },
	})
	if ok || victim != "" {
		t.Fatalf("SelectVictim(nil) = (%q, %v), want (\"\", false)", victim, ok)
	}
}

// TestLRU_IgnoresRank verifies lru is recency-only: it never consults RankOf, so
// even when a later candidate is the cheaper-to-reload adapter the LRU one is
// still evicted (BC-2). Guards against a rank-aware regression sneaking into lru.
func TestLRU_IgnoresRank(t *testing.T) {
	victim, ok := lru{}.SelectVictim(sim.EvictionContext{
		Candidates: []string{"a", "b"},
		// b is the "cheaper" (lower-rank) adapter — an accessor a rank-aware policy
		// would prefer to evict. lru must ignore this and still evict a.
		RankOf: func(id string) (int, bool) {
			if id == "b" {
				return 1, true
			}
			return 64, true
		},
	})
	if !ok || victim != "a" {
		t.Fatalf("SelectVictim ignoring rank = (%q, %v), want (\"a\", true)", victim, ok)
	}
}

// TestNew_LRURegistered verifies the baseline lru policy is registered and
// constructible by name (the factory returns lru by default in B-3).
func TestNew_LRURegistered(t *testing.T) {
	pol, err := New("lru")
	if err != nil {
		t.Fatalf("New(lru): unexpected error %v", err)
	}
	if pol == nil {
		t.Fatalf("New(lru): got nil policy")
	}
}

// TestNew_UnknownName verifies an unregistered name is a validation error, not a
// silent nil — the CLI surfaces valid names (Principle V).
func TestNew_UnknownName(t *testing.T) {
	pol, err := New("bogus")
	if err == nil {
		t.Fatalf("New(bogus): want error, got nil")
	}
	if pol != nil {
		t.Fatalf("New(bogus): want nil policy, got %v", pol)
	}
}

// TestNew_UnknownName_DeterministicMessage verifies the unknown-policy error text
// is reproducible run-to-run (R2 / INV-6): the valid-names list must be sorted, so
// two calls yield byte-identical messages regardless of map iteration order.
func TestNew_UnknownName_DeterministicMessage(t *testing.T) {
	_, err1 := New("bogus")
	_, err2 := New("bogus")
	if err1 == nil || err2 == nil {
		t.Fatalf("New(bogus): want error both times, got (%v, %v)", err1, err2)
	}
	if err1.Error() != err2.Error() {
		t.Fatalf("unknown-policy error not deterministic:\n first: %q\nsecond: %q", err1, err2)
	}
}
