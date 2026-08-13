package sim

import (
	"testing"
)

// These are the B-3 (#1491) integration contract tests for the eviction seam.
// They observe behavior through the exported per-adapter eviction/load counts,
// request completion, and residency — never through the seam's internal shape.
// LoRA is wired into package sim's tests via the blank import in
// lora_import_test.go, which registers the resident-set, registry, cost, and
// eviction-policy construction funcs.
//
// gateTestConfig / runSingleTTFT / newTestRequest / mustNewSimulator /
// newTestSimConfig are shared helpers from cold_load_gate_test.go and
// simulator_test.go.

// injectStaggered injects one cold single-adapter request per id, spaced far
// enough apart that each completes before the next arrives — so residency
// transitions are driven purely by LRU recency, not by concurrent pinning.
func injectStaggered(t *testing.T, cfg SimConfig, ids ...string) *Simulator {
	t.Helper()
	s := mustNewSimulator(t, cfg)
	const gap = 5_000_000 // µs; >> a single request's completion time
	for i, id := range ids {
		req := newTestRequest(id+"-req", int64(i)*gap, 8, 4)
		req.Adapter = id
		s.InjectArrival(req)
	}
	s.Run()
	return s
}

// TestEvictionSeam_LRUVictimSequence verifies that, routed through the seam, a
// capacity-2 instance evicts adapters in LRU order as four distinct cold adapters
// arrive one at a time: A then B fill the two slots, C evicts A (oldest), D evicts
// B. The observable proof is the per-adapter eviction counts and final residency
// (BC-1) — identical to the pre-seam hardcoded LRU.
func TestEvictionSeam_LRUVictimSequence(t *testing.T) {
	cfg := gateTestConfig(2,
		AdapterSpec{ID: "A", Rank: 8},
		AdapterSpec{ID: "B", Rank: 8},
		AdapterSpec{ID: "C", Rank: 8},
		AdapterSpec{ID: "D", Rank: 8},
	)
	s := injectStaggered(t, cfg, "A", "B", "C", "D")

	if s.Metrics.CompletedRequests != 4 {
		t.Fatalf("CompletedRequests = %d, want 4", s.Metrics.CompletedRequests)
	}
	wantEvict := map[string]int64{"A": 1, "B": 1, "C": 0, "D": 0}
	for id, want := range wantEvict {
		if got := s.Metrics.AdapterEvictionCounts[id]; got != want {
			t.Errorf("AdapterEvictionCounts[%s] = %d, want %d", id, got, want)
		}
	}
	// C and D are the last two loaded and never evicted, so they remain resident.
	for _, id := range []string{"C", "D"} {
		if !s.residentAdapters.IsResident(id) {
			t.Errorf("adapter %s should be resident at end (never evicted)", id)
		}
	}
	for _, id := range []string{"A", "B"} {
		if s.residentAdapters.IsResident(id) {
			t.Errorf("adapter %s should have been evicted", id)
		}
	}
}

// TestEvictionSeam_LRUVictimSequence_Reuse verifies recency reordering flows
// through the seam: with sequence A, B, A, C on capacity 2, the second A touches
// it to MRU, so when C needs a slot the victim is B (the least-recently-used),
// NOT A. Evicting A here would prove insertion-order, not recency — this pins the
// LRU semantics through the seam (BC-1).
func TestEvictionSeam_LRUVictimSequence_Reuse(t *testing.T) {
	cfg := gateTestConfig(2,
		AdapterSpec{ID: "A", Rank: 8},
		AdapterSpec{ID: "B", Rank: 8},
		AdapterSpec{ID: "C", Rank: 8},
	)
	s := injectStaggered(t, cfg, "A", "B", "A", "C")

	if s.Metrics.CompletedRequests != 4 {
		t.Fatalf("CompletedRequests = %d, want 4", s.Metrics.CompletedRequests)
	}
	// B is LRU when C loads (A was re-touched to MRU) → B evicted, A survives.
	if got := s.Metrics.AdapterEvictionCounts["B"]; got != 1 {
		t.Errorf("AdapterEvictionCounts[B] = %d, want 1 (B is LRU after A reuse)", got)
	}
	if got := s.Metrics.AdapterEvictionCounts["A"]; got != 0 {
		t.Errorf("AdapterEvictionCounts[A] = %d, want 0 (A touched to MRU, not evicted)", got)
	}
	if !s.residentAdapters.IsResident("A") {
		t.Errorf("A should still be resident (reused → MRU)")
	}
	if s.residentAdapters.IsResident("B") {
		t.Errorf("B should have been evicted for C")
	}
}

// TestEvictionSeam_AllPinnedNoDeadlock verifies BC-4 through the seam: with
// capacity 2 and three distinct cold adapters arriving together, A and B load and
// run concurrently (both pinned). When C reaches the head the set is full and
// every slot is pinned, so the seam returns no victim and starts no load this
// tick — but the simulation does not stall: once A or B completes it unpins a
// slot, C evicts it, loads once, and completes. All three drain (INV-8, INV-11).
func TestEvictionSeam_AllPinnedNoDeadlock(t *testing.T) {
	cfg := gateTestConfig(2,
		AdapterSpec{ID: "A", Rank: 8},
		AdapterSpec{ID: "B", Rank: 8},
		AdapterSpec{ID: "C", Rank: 8},
	)
	s := mustNewSimulator(t, cfg)
	for _, id := range []string{"A", "B", "C"} {
		req := newTestRequest(id+"-req", 0, 8, 4) // all arrive together
		req.Adapter = id
		s.InjectArrival(req)
	}
	s.Run()

	if s.Metrics.CompletedRequests != 3 {
		t.Fatalf("CompletedRequests = %d, want 3 (no deadlock, INV-11)", s.Metrics.CompletedRequests)
	}
	out := s.Metrics.BuildOutput("test-instance", nil)
	for _, id := range []string{"A", "B", "C"} {
		if lc := out.Adapters[id].LoadCount; lc != 1 {
			t.Errorf("adapter %s LoadCount = %d, want 1", id, lc)
		}
	}
	// Exactly one eviction: C displaced whichever of A/B freed a slot first.
	var totalEvict int64
	for _, id := range []string{"A", "B", "C"} {
		totalEvict += s.Metrics.AdapterEvictionCounts[id]
	}
	if totalEvict != 1 {
		t.Errorf("total evictions = %d, want 1 (one slot freed for C)", totalEvict)
	}
}

// TestEvictionSeam_Deterministic verifies the seam preserves byte-identity across
// runs (INV-6): the capacity-2 A,B,C,D scenario yields identical eviction/load
// metrics on two independent runs of the same config.
func TestEvictionSeam_Deterministic(t *testing.T) {
	newCfg := func() SimConfig {
		return gateTestConfig(2,
			AdapterSpec{ID: "A", Rank: 8},
			AdapterSpec{ID: "B", Rank: 8},
			AdapterSpec{ID: "C", Rank: 8},
			AdapterSpec{ID: "D", Rank: 8},
		)
	}
	snapshot := func(s *Simulator) map[string]int64 {
		m := map[string]int64{}
		for id, n := range s.Metrics.AdapterEvictionCounts {
			m["evict:"+id] = n
		}
		for id, n := range s.Metrics.AdapterLoadCounts {
			m["load:"+id] = n
		}
		return m
	}

	s1 := injectStaggered(t, newCfg(), "A", "B", "C", "D")
	s2 := injectStaggered(t, newCfg(), "A", "B", "C", "D")

	m1, m2 := snapshot(s1), snapshot(s2)
	if len(m1) != len(m2) {
		t.Fatalf("metric key count differs: run1=%d run2=%d", len(m1), len(m2))
	}
	for k, v1 := range m1 {
		if v2 := m2[k]; v1 != v2 {
			t.Errorf("metric %q differs across runs: run1=%d run2=%d", k, v1, v2)
		}
	}
}

// TestEvictionContext_RankAccessorMatchesRegistry verifies BC-5: the context the
// simulator hands a policy exposes a rank accessor backed by the declared registry
// — it returns each declared adapter's rank (and reports unregistered / base-model
// ids as absent), the same ranks the registry reports. lru ignores this, but a
// rank-aware policy (B-4) consumes it, so the wiring must be correct now.
func TestEvictionContext_RankAccessorMatchesRegistry(t *testing.T) {
	cfg := gateTestConfig(2,
		AdapterSpec{ID: "a8", Rank: 8},
		AdapterSpec{ID: "a16", Rank: 16},
	)
	s := mustNewSimulator(t, cfg)

	ctx := s.buildEvictionContext()
	if ctx.RankOf == nil {
		t.Fatalf("EvictionContext.RankOf is nil (must be non-nil per contract)")
	}

	cases := []struct {
		id       string
		wantRank int
		wantOK   bool
	}{
		{"a8", 8, true},
		{"a16", 16, true},
		{"", 0, false},          // base model — unregistered
		{"unknown", 0, false},   // never declared
	}
	for _, c := range cases {
		rank, ok := ctx.RankOf(c.id)
		if ok != c.wantOK || (ok && rank != c.wantRank) {
			t.Errorf("RankOf(%q) = (%d, %v), want (%d, %v)", c.id, rank, ok, c.wantRank, c.wantOK)
		}
	}
}
