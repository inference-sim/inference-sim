package eviction

import (
	"math"
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

// ---------------------------------------------------------------------------
// B-4 rank-aware policy contract tests (T017). These reference the not-yet-
// existent rankAware type, New("rank-aware"), New("")→lru, and ValidNames();
// they compile-fail (red) until T018 implements the policy.
// ---------------------------------------------------------------------------

// rankOfMap returns an EvictionContext.RankOf closure backed by a static map:
// declared ids report (rank, true); any other id reports (0, false). This mirrors
// the AdapterRegistry-backed accessor the simulator builds, letting the unit tests
// exercise the policy in isolation from the resident set.
func rankOfMap(ranks map[string]int) func(string) (int, bool) {
	return func(id string) (int, bool) {
		r, ok := ranks[id]
		return r, ok
	}
}

// permutations returns every ordering of the input (Heap's algorithm). Used by the
// parametric rank scan; n is small (4 → 24) so the allocation is trivial.
func permutations(in []int) [][]int {
	a := append([]int(nil), in...)
	var out [][]int
	var gen func(k int)
	gen = func(k int) {
		if k == 1 {
			out = append(out, append([]int(nil), a...))
			return
		}
		for i := 0; i < k; i++ {
			gen(k - 1)
			if k%2 == 0 {
				a[i], a[k-1] = a[k-1], a[i]
			} else {
				a[0], a[k-1] = a[k-1], a[0]
			}
		}
	}
	gen(len(a))
	return out
}

// TestRankAware_SelectsLowestRank verifies the headline law (C-1): the victim is the
// unpinned candidate with the lowest declared rank — the cheapest-to-reload adapter —
// regardless of LRU position. Here the LRU choice would be "a" (rank 64), but
// rank-aware must pick "c" (rank 8).
func TestRankAware_SelectsLowestRank(t *testing.T) {
	victim, ok := rankAware{}.SelectVictim(sim.EvictionContext{
		Candidates: []string{"a", "b", "c"}, // LRU→MRU
		RankOf:     rankOfMap(map[string]int{"a": 64, "b": 16, "c": 8}),
	})
	if !ok || victim != "c" {
		t.Fatalf("SelectVictim = (%q, %v), want (\"c\", true) — lowest rank", victim, ok)
	}
}

// TestRankAware_TieBreakByID verifies C-2: when the lowest rank is shared, the
// lexicographically smallest id is evicted, deterministically.
func TestRankAware_TieBreakByID(t *testing.T) {
	victim, ok := rankAware{}.SelectVictim(sim.EvictionContext{
		Candidates: []string{"z", "m", "a"}, // LRU→MRU; all same rank
		RankOf:     rankOfMap(map[string]int{"z": 8, "m": 8, "a": 8}),
	})
	if !ok || victim != "a" {
		t.Fatalf("SelectVictim tie = (%q, %v), want (\"a\", true) — smallest id", victim, ok)
	}
}

// TestRankAware_NoCandidates verifies C-3 no-victim parity: an empty candidate set
// yields ("", false), byte-identical to lru, so the caller starts no load and retries
// on a later step (INV-8).
func TestRankAware_NoCandidates(t *testing.T) {
	victim, ok := rankAware{}.SelectVictim(sim.EvictionContext{
		Candidates: nil,
		RankOf:     rankOfMap(nil),
	})
	if ok || victim != "" {
		t.Fatalf("SelectVictim(nil) = (%q, %v), want (\"\", false)", victim, ok)
	}
}

// TestRankAware_ParametricMonotonicScan enumerates all 4! = 24 assignments of four
// distinct ranks to a fixed 4-element candidate slice (LRU→MRU order held constant,
// only the ranks permuted) and asserts the victim is always argmin(rank ASC, id ASC).
// This pins the exact min-rank law: a "victim ≠ LRU" assertion would pass for a random
// or reversed comparator, so instead we compare against an independent reference argmin.
func TestRankAware_ParametricMonotonicScan(t *testing.T) {
	ids := []string{"a", "b", "c", "d"} // fixed LRU→MRU order
	distinctRanks := []int{8, 16, 32, 64}
	for _, perm := range permutations(distinctRanks) {
		ranks := map[string]int{}
		for i, id := range ids {
			ranks[id] = perm[i]
		}
		// Independent reference argmin over (rank ASC, id ASC).
		want := ids[0]
		for _, id := range ids {
			if ranks[id] < ranks[want] || (ranks[id] == ranks[want] && id < want) {
				want = id
			}
		}
		victim, ok := rankAware{}.SelectVictim(sim.EvictionContext{
			Candidates: append([]string(nil), ids...),
			RankOf:     rankOfMap(ranks),
		})
		if !ok || victim != want {
			t.Fatalf("perm %v: SelectVictim = (%q, %v), want (%q, true)", perm, victim, ok, want)
		}
	}
}

// TestRankAware_Idempotent verifies INV-6 at the policy level: two calls on the same
// immutable context return byte-identical victims (no hidden state, no map-iteration
// nondeterminism leaking into the scan).
func TestRankAware_Idempotent(t *testing.T) {
	ctx := sim.EvictionContext{
		Candidates: []string{"a", "b", "c", "d"},
		RankOf:     rankOfMap(map[string]int{"a": 16, "b": 8, "c": 8, "d": 32}),
	}
	v1, ok1 := rankAware{}.SelectVictim(ctx)
	v2, ok2 := rankAware{}.SelectVictim(ctx)
	if v1 != v2 || ok1 != ok2 {
		t.Fatalf("non-idempotent: (%q,%v) then (%q,%v)", v1, ok1, v2, ok2)
	}
	if v1 != "b" { // b and c tie at rank 8; b is the smaller id
		t.Fatalf("victim = %q, want \"b\" (min rank 8, smaller id)", v1)
	}
}

// TestRankAware_UnregisteredSortsLast verifies the D-4-2 correctness hinge: a candidate
// whose RankOf reports ok=false is treated as rank +∞ (math.MaxInt) and is NEVER evicted
// in preference to a registered adapter — even one with an extreme finite rank
// (math.MaxInt-1). A mis-implementation mapping ok=false→rank 0 would evict the
// unregistered adapter first (the exact opposite); this pins against that.
func TestRankAware_UnregisteredSortsLast(t *testing.T) {
	victim, ok := rankAware{}.SelectVictim(sim.EvictionContext{
		Candidates: []string{"unreg", "reg"},
		RankOf: func(id string) (int, bool) {
			if id == "reg" {
				return math.MaxInt - 1, true
			}
			return 0, false // unreg → sentinel +∞, must sort last
		},
	})
	if !ok || victim != "reg" {
		t.Fatalf("SelectVictim = (%q, %v), want (\"reg\", true) — unregistered must sort last", victim, ok)
	}

	// An all-unregistered context degenerates to the deterministic id tie-break (never a
	// crash, never nondeterministic) — the LoRA-inactive degenerate case (R20).
	victim2, ok2 := rankAware{}.SelectVictim(sim.EvictionContext{
		Candidates: []string{"y", "x", "z"},
		RankOf:     func(string) (int, bool) { return 0, false },
	})
	if !ok2 || victim2 != "x" {
		t.Fatalf("all-unregistered SelectVictim = (%q, %v), want (\"x\", true) — id tie-break", victim2, ok2)
	}
}

// TestNew_RankAwareRegistered verifies rank-aware is constructible by name after B-4.
func TestNew_RankAwareRegistered(t *testing.T) {
	pol, err := New("rank-aware")
	if err != nil {
		t.Fatalf("New(rank-aware): unexpected error %v", err)
	}
	if _, isRankAware := pol.(rankAware); !isRankAware {
		t.Fatalf("New(rank-aware): got %T, want rankAware", pol)
	}
}

// TestNew_EmptyResolvesLRU verifies the canonical empty→lru fallback site (R20): New("")
// resolves to lru, so a library caller passing LoRAConfig{EvictionPolicy:""} never hits
// the "unknown eviction policy" error.
func TestNew_EmptyResolvesLRU(t *testing.T) {
	pol, err := New("")
	if err != nil {
		t.Fatalf("New(\"\"): unexpected error %v", err)
	}
	if _, isLRU := pol.(lru); !isLRU {
		t.Fatalf("New(\"\"): got %T, want lru (empty→lru fallback)", pol)
	}
}

// TestValidNames_SortedContainsPolicies verifies ValidNames() returns the registered
// names sorted and deterministic — the source for the --eviction-policy help text and
// the CLI fail-fast valid-names list.
func TestValidNames_SortedContainsPolicies(t *testing.T) {
	got := ValidNames()
	want := []string{"lru", "rank-aware"}
	if len(got) != len(want) {
		t.Fatalf("ValidNames() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ValidNames() = %v, want %v (sorted)", got, want)
		}
	}
}
