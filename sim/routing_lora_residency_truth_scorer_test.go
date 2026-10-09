package sim

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newTruthRig is residencyRig holding only lora-residency-truth; each test sets
// the snapshots' ground-truth residency fields directly.
func newTruthRig(t *testing.T, maxLoras int, ids ...string) *residencyRig {
	t.Helper()
	r := newResidencyRig(t, maxLoras, ids...)
	r.p = NewRoutingPolicyWithCache("weighted",
		[]ScorerConfig{{Name: "lora-residency-truth", Weight: 1}}, 16, nil, nil)
	for i := range r.snaps {
		r.snaps[i].ResidentCapacity = maxLoras
	}
	return r
}

func (r *residencyRig) setResidency(id string, order []string, pinned ...string) {
	for i := range r.snaps {
		if r.snaps[i].ID != id {
			continue
		}
		r.snaps[i].ResidentOrder = order
		r.snaps[i].ResidentAdapters = map[string]bool{}
		for _, a := range order {
			r.snaps[i].ResidentAdapters[a] = true
		}
		r.snaps[i].ResidentPinned = nil
		for _, a := range pinned {
			if r.snaps[i].ResidentPinned == nil {
				r.snaps[i].ResidentPinned = map[string]bool{}
			}
			r.snaps[i].ResidentPinned[a] = true
		}
		return
	}
	r.t.Fatalf("unknown instance %s", id)
}

// The eviction victim is the TRUE LRU adapter: i0 would evict x (in demand), i1
// would evict y (no demand), so i1 is cheaper for a new adapter z.
func TestLoRAResidencyTruth_VictimFollowsTrueOrder(t *testing.T) {
	r := newTruthRig(t, 2, "i0", "i1", "i2")
	for _, id := range []string{"x1", "x2", "x3"} { // demand for x, routed elsewhere and settled
		req := r.route(id, "x", "i2", 100)
		r.start(req, "i2", 150)
	}
	r.setResidency("i0", []string{"x", "y"})
	r.setResidency("i1", []string{"y", "x"})
	s := r.score("z", 200)
	assert.Greater(t, s["i1"], s["i0"], "evicting y beats evicting x: %v", s)
}

// A pinned resident adapter cannot be evicted: at capacity 1, an instance whose
// only adapter is pinned is worse for a new adapter than one whose is not.
func TestLoRAResidencyTruth_PinnedBlocks(t *testing.T) {
	r := newTruthRig(t, 1, "i0", "i1")
	r.setResidency("i0", []string{"a"}, "a")
	r.setResidency("i1", []string{"a"})
	s := r.score("b", 100)
	assert.Greater(t, s["i1"], s["i0"], "pinned a blocks b on i0: %v", s)
}

// A routed request not yet started is pending and attracts its adapter; its first
// token settles it, after which only the snapshot's truth counts.
func TestLoRAResidencyTruth_PendingAttractsUntilStarted(t *testing.T) {
	r := newTruthRig(t, 1, "i0", "i1")
	req := r.route("r1", "a", "i0", 100)
	s := r.score("a", 150)
	assert.Greater(t, s["i0"], s["i1"], "pending a on i0: %v", s)

	r.start(req, "i0", 200)
	s = r.score("a", 250)
	assert.Equal(t, s["i0"], s["i1"], "settled, and the snapshots show nothing resident: %v", s)
}

// Completion also settles a pending request (e.g. one that never started).
func TestLoRAResidencyTruth_CompletionSettles(t *testing.T) {
	r := newTruthRig(t, 1, "i0", "i1")
	req := r.route("r1", "a", "i0", 100)
	r.complete(req, "i0", 150)
	s := r.score("a", 200)
	assert.Equal(t, s["i0"], s["i1"], "%v", s)
}

// The router-side estimate is never used: a served adapter that the snapshots do
// not show resident earns nothing (lora-residency would prefer i0 here), and
// ActiveAdapters is ignored.
func TestLoRAResidencyTruth_IgnoresEstimateAndActiveAdapters(t *testing.T) {
	r := newTruthRig(t, 1, "i0", "i1")
	req := r.route("r1", "a", "i0", 100)
	r.start(req, "i0", 200)
	r.complete(req, "i0", 300)
	r.snaps[0].ActiveAdapters = map[string]int{"a": 3}
	s := r.score("a", 400)
	assert.Equal(t, s["i0"], s["i1"], "%v", s)

	r.setResidency("i1", []string{"a"})
	s = r.score("a", 400)
	assert.Greater(t, s["i1"], s["i0"], "truth says a is on i1: %v", s)
}

// A load in progress holds its slot: at capacity 1, an instance loading a is worse
// for b than an idle one, and better for a.
func TestLoRAResidencyTruth_LoadingHoldsSlot(t *testing.T) {
	r := newTruthRig(t, 1, "i0", "i1")
	r.snaps[0].LoadingAdapter = "a"
	s := r.score("b", 100)
	assert.Greater(t, s["i1"], s["i0"], "i0's only slot is being filled by a: %v", s)
	s = r.score("a", 100)
	assert.Greater(t, s["i0"], s["i1"], "a is already loading on i0: %v", s)
}

// Review of PR #60, finding 2: a loading adapter is not resident yet. A request for
// it is pending on the loading instance, which is cheaper than a fresh load but
// dearer than an instance where it is genuinely resident.
func TestLoRAResidencyTruth_LoadingIsPendingNotResident(t *testing.T) {
	r := newTruthRig(t, 2, "loading", "resident", "idle")
	r.snaps[0].LoadingAdapter = "a"
	r.setResidency("resident", []string{"a"})
	s := r.score("a", 100)
	assert.Greater(t, s["resident"], s["loading"], "%v", s)
	assert.Greater(t, s["loading"], s["idle"], "%v", s)
}

// The capacity is ResidentCapacity, refreshed with the truth, not MaxLoras, which
// rides ActiveAdapters' cadence and can be stale or zero.
func TestLoRAResidencyTruth_CapacityFromResidentCapacity(t *testing.T) {
	r := newTruthRig(t, 2, "i0", "i1", "i2")
	r.start(r.route("x1", "x", "i2", 50), "i2", 60) // demand for x, so evicting it costs
	for i := range r.snaps {
		r.snaps[i].MaxLoras = 0 // not yet refreshed
	}
	r.setResidency("i0", []string{"x"})
	r.setResidency("i1", []string{"x"})
	r.snaps[1].ResidentCapacity = 1
	s := r.score("y", 100)
	assert.Greater(t, s["i0"], s["i1"], "i0 has a free slot, i1 must evict x: %v", s)
}

// Review of PR #60, finding 4: pending is kept per (instance, adapter) pair; the pair
// stays pending until its last request settles, and a re-routed ID moves.
func TestLoRAResidencyTruth_PendingPairCounts(t *testing.T) {
	r := newTruthRig(t, 1, "i0", "i1")
	r1 := r.route("r1", "a", "i0", 100)
	r2 := r.route("r2", "a", "i0", 110)
	r.start(r1, "i0", 120)
	s := r.score("a", 130)
	assert.Greater(t, s["i0"], s["i1"], "r2 still pending on i0: %v", s)
	r.complete(r2, "i0", 140)
	s = r.score("a", 150)
	assert.Equal(t, s["i0"], s["i1"], "both settled: %v", s)

	r3 := r.route("r3", "a", "i0", 160)
	r3again := r.route("r3", "a", "i1", 170) // same ID, routed again elsewhere
	r.start(r3again, "i1", 180)
	_ = r3
	s = r.score("a", 190)
	assert.Equal(t, s["i0"], s["i1"], "the re-route moved r3, and its start settled it: %v", s)
}

func TestLoRAResidencyTruth_BaseModelNeutral(t *testing.T) {
	r := newTruthRig(t, 1, "i0", "i1")
	r.setResidency("i0", []string{"a"}, "a")
	s := r.score("", 100)
	assert.Equal(t, s["i0"], s["i1"], "%v", s)
}

// Same policy as lora-residency: when lora-residency's estimate equals the truth,
// the two score identically.
func TestLoRAResidencyTruth_MatchesLoRAResidencyOnSameState(t *testing.T) {
	est := newResidencyRig(t, 2, "i0", "i1")
	tru := newTruthRig(t, 2, "i0", "i1")
	for _, rig := range []*residencyRig{est, tru} {
		for i, step := range []struct{ id, adapter, inst string }{
			{"r1", "a", "i0"}, {"r2", "b", "i0"}, {"r3", "a", "i1"},
		} {
			tick := int64(100 * (i + 1))
			req := rig.route(step.id, step.adapter, step.inst, tick)
			rig.start(req, step.inst, tick+10)
			rig.complete(req, step.inst, tick+20)
		}
		rig.route("r4", "b", "i1", 400) // pending on i1
	}
	tru.setResidency("i0", []string{"a", "b"})
	tru.setResidency("i1", []string{"a"})
	for _, adapter := range []string{"a", "b", "c"} {
		want, got := est.score(adapter, 500), tru.score(adapter, 500)
		require.Len(t, got, 2)
		for id := range want {
			assert.InDelta(t, want[id], got[id], 1e-12, "adapter %s instance %s", adapter, id)
		}
	}
}
