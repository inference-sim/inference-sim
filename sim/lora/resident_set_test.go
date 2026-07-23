package lora

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/lora/eviction"
)

// The resident-adapter set models the finite per-instance GPU adapter slots
// (data-model.md "Resident-Adapter Set"): a capacity-bounded LRU over adapter
// ids with pinning for in-flight requests. These contract tests assert the
// observable laws — capacity bound, LRU eviction order, and pin protection —
// independent of the internal linked-list representation.

// TestResidentSet_StoreMakesResident verifies a stored id becomes resident at
// MRU and that touch/store on an absent id behave as documented.
func TestResidentSet_StoreMakesResident(t *testing.T) {
	s := newResidentSet(2)
	if s.Len() != 0 {
		t.Fatalf("new set: len = %d, want 0", s.Len())
	}

	if evicted, ok := s.Store("a"); !ok || evicted != "" {
		t.Fatalf("store(a) into empty set: got (%q, %v), want (\"\", true)", evicted, ok)
	}
	if !s.IsResident("a") {
		t.Fatalf("store(a): a not resident")
	}
	if s.Len() != 1 {
		t.Fatalf("after store(a): len = %d, want 1", s.Len())
	}

	// touch on an absent id is a no-op (does not make it resident).
	s.Touch("ghost")
	if s.IsResident("ghost") {
		t.Fatalf("touch(ghost) made an absent id resident")
	}
	if s.Len() != 1 {
		t.Fatalf("touch(absent) changed len to %d, want 1", s.Len())
	}

	// re-storing a resident id is idempotent on membership (equivalent to touch).
	if evicted, ok := s.Store("a"); !ok || evicted != "" {
		t.Fatalf("store(a) again: got (%q, %v), want (\"\", true)", evicted, ok)
	}
	if s.Len() != 1 {
		t.Fatalf("re-store(a): len = %d, want 1", s.Len())
	}
}

// TestResidentSet_CapacityInvariant verifies len never exceeds capacity as more
// distinct adapters than slots are stored (new invariant: resident ≤ capacity).
func TestResidentSet_CapacityInvariant(t *testing.T) {
	const capacity = 3
	s := newResidentSet(capacity)
	ids := []string{"a", "b", "c", "d", "e", "f"}
	for _, id := range ids {
		s.Store(id)
		if s.Len() > capacity {
			t.Fatalf("after store(%s): len = %d exceeds capacity %d", id, s.Len(), capacity)
		}
	}
	if s.Len() != capacity {
		t.Fatalf("final len = %d, want %d (full)", s.Len(), capacity)
	}
}

// TestResidentSet_EvictsLeastRecentlyUsed pins down the LRU order deterministically:
// touch promotes to MRU so the next store evicts the true least-recently-used id.
// No randomness is involved — the eviction victim is a pure function of access order.
func TestResidentSet_EvictsLeastRecentlyUsed(t *testing.T) {
	s := newResidentSet(3)
	s.Store("a")
	s.Store("b")
	s.Store("c") // LRU order (old→new): a, b, c

	s.Touch("a") // promote a to MRU: b, c, a

	// Storing d is at capacity → evicts the LRU, which is now b.
	evicted, ok := s.Store("d")
	if !ok {
		t.Fatalf("store(d): ok = false, want true (b is evictable)")
	}
	if evicted != "b" {
		t.Fatalf("store(d) evicted %q, want \"b\" (least recently used)", evicted)
	}
	if s.IsResident("b") {
		t.Fatalf("b should have been evicted")
	}
	for _, id := range []string{"a", "c", "d"} {
		if !s.IsResident(id) {
			t.Fatalf("%s should still be resident after evicting b", id)
		}
	}

	// Next store evicts c (now the LRU: c, a, d → c).
	if evicted, ok := s.Store("e"); !ok || evicted != "c" {
		t.Fatalf("store(e) evicted (%q, %v), want (\"c\", true)", evicted, ok)
	}
}

// TestResidentSet_PinnedNeverEvicted verifies a pinned (in-flight) adapter is
// never chosen as an eviction victim even when it is the least recently used
// (US2 scenario 3). Eviction falls through to the next non-pinned candidate.
func TestResidentSet_PinnedNeverEvicted(t *testing.T) {
	s := newResidentSet(3)
	s.Store("a")
	s.Store("b")
	s.Store("c") // LRU order: a, b, c

	s.Pin("a") // a is in use — protected despite being LRU

	evicted, ok := s.Store("d")
	if !ok {
		t.Fatalf("store(d): ok = false, want true (b/c are evictable)")
	}
	if evicted == "a" {
		t.Fatalf("evicted pinned adapter a")
	}
	if evicted != "b" {
		t.Fatalf("store(d) evicted %q, want \"b\" (LRU non-pinned)", evicted)
	}
	if !s.IsResident("a") {
		t.Fatalf("pinned a must remain resident")
	}
}

// TestResidentSet_AllPinnedAtCapacityRejects verifies that when the set is full
// and every resident adapter is pinned, a new store is refused (ok=false) rather
// than violating the capacity bound or evicting an in-use adapter. The cold-load
// gate (PR3) makes such a request wait.
func TestResidentSet_AllPinnedAtCapacityRejects(t *testing.T) {
	s := newResidentSet(2)
	s.Store("a")
	s.Store("b")
	s.Pin("a")
	s.Pin("b")

	evicted, ok := s.Store("c")
	if ok {
		t.Fatalf("store(c) with all pinned: ok = true, want false")
	}
	if evicted != "" {
		t.Fatalf("store(c) with all pinned: evicted %q, want \"\"", evicted)
	}
	if s.IsResident("c") {
		t.Fatalf("c admitted despite full+all-pinned set")
	}
	if s.Len() != 2 {
		t.Fatalf("len = %d after refused store, want 2", s.Len())
	}
}

// TestResidentSet_PinRefcount verifies pinning is reference-counted: an adapter
// used by multiple concurrent in-flight requests stays protected until the last
// user unpins it. This guards against unpinning a still-in-use adapter.
func TestResidentSet_PinRefcount(t *testing.T) {
	s := newResidentSet(2)
	s.Store("a")
	s.Store("b") // LRU: a, b

	s.Pin("a")
	s.Pin("a") // two in-flight requests use a

	s.Unpin("a") // one completes — a still has one user, still pinned

	// b is unpinned and LRU-newer than a, but a is still protected, so evicting
	// forces b out.
	if evicted, ok := s.EvictLRU(); !ok || evicted != "b" {
		t.Fatalf("evictLRU with a still pinned: got (%q, %v), want (\"b\", true)", evicted, ok)
	}

	s.Unpin("a") // last user completes — a now evictable
	if evicted, ok := s.EvictLRU(); !ok || evicted != "a" {
		t.Fatalf("evictLRU after a fully unpinned: got (%q, %v), want (\"a\", true)", evicted, ok)
	}
}

// TestResidentSet_EvictLRUEmpty verifies evictLRU on an empty (or fully pinned)
// set reports no victim rather than panicking.
func TestResidentSet_EvictLRUEmpty(t *testing.T) {
	s := newResidentSet(2)
	if evicted, ok := s.EvictLRU(); ok || evicted != "" {
		t.Fatalf("evictLRU on empty set: got (%q, %v), want (\"\", false)", evicted, ok)
	}
}

// TestResidentSet_ZeroCapacityPanics verifies the constructor rejects a
// non-positive capacity (mirrors newCpuTier; capacity is validated > 0 upstream).
func TestResidentSet_ZeroCapacityPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("newResidentSet(0) did not panic")
		}
	}()
	newResidentSet(0)
}

// --- B-3 (#1491): eviction-seam support methods ---

// TestResidentSet_UnpinnedCandidates_RecencyOrder verifies the candidate list the
// eviction seam consumes is exactly the resident ids in LRU→MRU (eviction-priority)
// order, so a policy that picks candidates[0] evicts the least-recently-used id.
func TestResidentSet_UnpinnedCandidates_RecencyOrder(t *testing.T) {
	s := newResidentSet(3)
	s.Store("a")
	s.Store("b")
	s.Store("c") // LRU→MRU: a, b, c
	s.Touch("a") // promote a to MRU: b, c, a

	got := s.UnpinnedCandidates()
	want := []string{"b", "c", "a"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("UnpinnedCandidates() = %v, want %v (LRU→MRU)", got, want)
	}
}

// TestResidentSet_UnpinnedCandidates_ExcludesPinned verifies a pinned adapter is
// never offered as an eviction candidate (INV-L5): the seam cannot select it.
func TestResidentSet_UnpinnedCandidates_ExcludesPinned(t *testing.T) {
	s := newResidentSet(3)
	s.Store("a")
	s.Store("b")
	s.Store("c") // LRU→MRU: a, b, c
	s.Pin("b")   // b is in-flight — must be excluded

	got := s.UnpinnedCandidates()
	want := []string{"a", "c"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("UnpinnedCandidates() with b pinned = %v, want %v", got, want)
	}
}

// TestResidentSet_Evict_RemovesUnpinned verifies Evict removes a named unpinned id
// and reports success, shrinking the resident set.
func TestResidentSet_Evict_RemovesUnpinned(t *testing.T) {
	s := newResidentSet(2)
	s.Store("a")
	s.Store("b")

	if !s.Evict("a") {
		t.Fatalf("Evict(a) = false, want true (a is unpinned)")
	}
	if s.IsResident("a") {
		t.Fatalf("a still resident after Evict(a)")
	}
	if s.Len() != 1 {
		t.Fatalf("len = %d after Evict(a), want 1", s.Len())
	}
}

// TestResidentSet_Evict_RefusesPinned verifies Evict refuses a pinned id (INV-L5)
// — pin-safety is enforced at the data structure, not just by policy discipline.
func TestResidentSet_Evict_RefusesPinned(t *testing.T) {
	s := newResidentSet(2)
	s.Store("a")
	s.Pin("a")

	if s.Evict("a") {
		t.Fatalf("Evict(a) = true, want false (a is pinned)")
	}
	if !s.IsResident("a") {
		t.Fatalf("pinned a was removed by a refused Evict")
	}
}

// TestResidentSet_Evict_AbsentReturnsFalse verifies evicting an id that is not
// resident is a no-op that reports failure rather than panicking.
func TestResidentSet_Evict_AbsentReturnsFalse(t *testing.T) {
	s := newResidentSet(2)
	if s.Evict("nope") {
		t.Fatalf("Evict(nope) = true, want false (absent id)")
	}
}

// TestLRU_ByteIdenticalToEvictLRU is the crux of B-3 (BC-1): routing victim
// selection through the seam's lru policy picks the SAME victim as the resident
// set's former EvictLRU scan, across several recency permutations. For each op
// sequence it builds two independent sets from identical ops, then compares the
// policy's SelectVictim(candidates) against EvictLRU's victim. Equal victims for
// every sequence prove candidates[0] == first-unpinned-in-LRU→MRU == EvictLRU.
func TestLRU_ByteIdenticalToEvictLRU(t *testing.T) {
	pol, err := eviction.New("lru")
	if err != nil {
		t.Fatalf("eviction.New(lru): %v", err)
	}

	// Each apply mutates a fresh set; two identical applies feed the two paths.
	sequences := map[string]func(s *residentSet){
		"plain-fill": func(s *residentSet) {
			s.Store("a")
			s.Store("b")
			s.Store("c")
		},
		"touch-promotes-lru": func(s *residentSet) {
			s.Store("a")
			s.Store("b")
			s.Store("c")
			s.Touch("a") // a → MRU, so b is now the victim
		},
		"one-pinned": func(s *residentSet) {
			s.Store("a")
			s.Store("b")
			s.Store("c")
			s.Pin("a") // a protected, so b is the victim
		},
	}

	for name, apply := range sequences {
		t.Run(name, func(t *testing.T) {
			seamSet := newResidentSet(3)
			apply(seamSet)
			seamVictim, seamOK := pol.SelectVictim(sim.EvictionContext{
				Candidates: seamSet.UnpinnedCandidates(),
			})

			evictSet := newResidentSet(3)
			apply(evictSet)
			evictVictim, evictOK := evictSet.EvictLRU()

			if seamOK != evictOK || seamVictim != evictVictim {
				t.Fatalf("seam vs EvictLRU victim mismatch: seam=(%q,%v) evictLRU=(%q,%v)",
					seamVictim, seamOK, evictVictim, evictOK)
			}
		})
	}
}
