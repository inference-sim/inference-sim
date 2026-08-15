package creation

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// fakeDemand is a deterministic DemandWindow: RequestedSince returns ids in the given
// order, which the tests set to sorted order to match the real accessor's contract.
type fakeDemand struct{ since map[int64][]string }

func (f fakeDemand) LastRequestedAt(string) (int64, bool) { return 0, false }
func (f fakeDemand) RequestedSince(s int64) []string      { return f.since[s] }

// TestKeepWarm_GateBehaviourIsOnDemand pins that keep-warm changes only the TICK. Its
// two instance-scoped methods must stay on-demand's, so a cold miss is still admitted
// and t=0 seeds nothing — the tick is additive, not a replacement gate policy.
func TestKeepWarm_GateBehaviourIsOnDemand(t *testing.T) {
	p, err := New("keep-warm")
	if err != nil {
		t.Fatalf("New(keep-warm): %v", err)
	}
	if got := p.Initial(sim.CreationContext{Assigned: []string{"a1"}}); got != nil {
		t.Errorf("Initial must seed nothing (t=0 belongs to Initial policies), got %v", got)
	}
	if !p.OnResidentMiss(sim.CreationContext{MissedAdapter: "a1"}) {
		t.Error("OnResidentMiss must admit, satisfying the starvation-freedom obligation")
	}
}

// TestKeepWarm_PrefetchesEvictedRecentAdapter is the core behaviour: an adapter
// requested inside the window that is resident nowhere gets one decision.
func TestKeepWarm_PrefetchesEvictedRecentAdapter(t *testing.T) {
	p := mustPeriodic(t, "keep-warm")
	ctx := sim.PeriodicCreationContext{
		Now: 1000, Interval: 1000,
		Instances: []sim.InstanceResidency{
			{ID: "i0", Resident: []string{"a2"}, Unpinned: []string{"a2"}, Capacity: 1},
		},
		Demand: fakeDemand{since: map[int64][]string{0: {"a1"}}},
	}
	got := p.OnTick(ctx)
	want := []sim.PrefetchDecision{{Instance: "i0", Adapter: "a1"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("OnTick = %v, want %v", got, want)
	}
}

// TestKeepWarm_SkipsAdapterResidentAnywhere pins the deliberate scope refusal: if the
// adapter is resident on ANY instance, keep-warm emits nothing, even when requests for
// it may be landing elsewhere. Fixing that mismatch is the routing seam's job.
func TestKeepWarm_SkipsAdapterResidentAnywhere(t *testing.T) {
	p := mustPeriodic(t, "keep-warm")
	ctx := sim.PeriodicCreationContext{
		Now: 1000, Interval: 1000,
		Instances: []sim.InstanceResidency{
			{ID: "i0", Resident: nil, Unpinned: nil, Capacity: 2},
			{ID: "i1", Resident: []string{"a1"}, Unpinned: []string{"a1"}, Capacity: 2},
		},
		Demand: fakeDemand{since: map[int64][]string{0: {"a1"}}},
	}
	if got := p.OnTick(ctx); len(got) != 0 {
		t.Errorf("OnTick = %v, want no decisions (a1 is resident on i1)", got)
	}
}

// TestKeepWarm_SkipsAdapterAlreadyLoading prevents a duplicate charged load for an
// adapter whose load is already in flight somewhere.
func TestKeepWarm_SkipsAdapterAlreadyLoading(t *testing.T) {
	p := mustPeriodic(t, "keep-warm")
	ctx := sim.PeriodicCreationContext{
		Now: 1000, Interval: 1000,
		Instances: []sim.InstanceResidency{
			{ID: "i0", Capacity: 2, Loading: "a1"},
		},
		Demand: fakeDemand{since: map[int64][]string{0: {"a1"}}},
	}
	if got := p.OnTick(ctx); len(got) != 0 {
		t.Errorf("OnTick = %v, want no decisions (a1 is already loading on i0)", got)
	}
}

// TestKeepWarm_PrefersFreeSlotThenEvictableInConstructionOrder pins target selection.
// Determinism (INV-6) requires a total order, and it must be the cluster's construction
// order, never Go's map order.
func TestKeepWarm_PrefersFreeSlotThenEvictableInConstructionOrder(t *testing.T) {
	p := mustPeriodic(t, "keep-warm")
	// i0 is full but evictable; i1 has a free slot. The free slot must win.
	ctx := sim.PeriodicCreationContext{
		Now: 1000, Interval: 1000,
		Instances: []sim.InstanceResidency{
			{ID: "i0", Resident: []string{"a9"}, Unpinned: []string{"a9"}, Capacity: 1},
			{ID: "i1", Resident: []string{"a8"}, Unpinned: []string{"a8"}, Capacity: 2},
		},
		Demand: fakeDemand{since: map[int64][]string{0: {"a1"}}},
	}
	got := p.OnTick(ctx)
	if len(got) != 1 || got[0].Instance != "i1" {
		t.Fatalf("OnTick = %v, want one decision on i1 (the instance with a free slot)", got)
	}

	// Now nobody has a free slot: the first instance in construction order with an
	// evictable adapter wins.
	ctx.Instances = []sim.InstanceResidency{
		{ID: "i0", Resident: []string{"a9"}, Unpinned: nil, Capacity: 1},            // full, all pinned
		{ID: "i1", Resident: []string{"a8"}, Unpinned: []string{"a8"}, Capacity: 1}, // full, evictable
	}
	got = p.OnTick(ctx)
	if len(got) != 1 || got[0].Instance != "i1" {
		t.Fatalf("OnTick = %v, want one decision on i1 (first with an evictable slot)", got)
	}
}

// TestKeepWarm_NoTargetYieldsNoDecision covers the fully-pinned cluster.
func TestKeepWarm_NoTargetYieldsNoDecision(t *testing.T) {
	p := mustPeriodic(t, "keep-warm")
	ctx := sim.PeriodicCreationContext{
		Now: 1000, Interval: 1000,
		Instances: []sim.InstanceResidency{
			{ID: "i0", Resident: []string{"a9"}, Unpinned: nil, Capacity: 1},
		},
		Demand: fakeDemand{since: map[int64][]string{0: {"a1"}}},
	}
	if got := p.OnTick(ctx); len(got) != 0 {
		t.Errorf("OnTick = %v, want no decisions (no free and no evictable slot)", got)
	}
}

// TestKeepWarm_NilDemandIsSafe pins the documented nil-check obligation: Demand may be
// nil when the subsystem is inert, and a policy must not panic.
func TestKeepWarm_NilDemandIsSafe(t *testing.T) {
	p := mustPeriodic(t, "keep-warm")
	if got := p.OnTick(sim.PeriodicCreationContext{Now: 1000, Interval: 1000}); len(got) != 0 {
		t.Errorf("OnTick with nil Demand = %v, want no decisions", got)
	}
}

// TestKeepWarm_Deterministic runs the same context twice and requires identical output
// (INV-6). It is the unit-level catch for a map that slipped into the policy.
func TestKeepWarm_Deterministic(t *testing.T) {
	p := mustPeriodic(t, "keep-warm")
	ctx := sim.PeriodicCreationContext{
		Now: 1000, Interval: 1000,
		Instances: []sim.InstanceResidency{
			{ID: "i0", Capacity: 4},
			{ID: "i1", Capacity: 4},
		},
		Demand: fakeDemand{since: map[int64][]string{0: {"a1", "a2", "a3"}}},
	}
	first := p.OnTick(ctx)
	for i := 0; i < 20; i++ {
		if got := p.OnTick(ctx); !reflect.DeepEqual(got, first) {
			t.Fatalf("OnTick is not deterministic: run %d = %v, first = %v", i, got, first)
		}
	}
}

func mustPeriodic(t *testing.T, name string) sim.PeriodicCreationPolicy {
	t.Helper()
	p, err := New(name)
	if err != nil {
		t.Fatalf("New(%q): %v", name, err)
	}
	tp, ok := p.(sim.PeriodicCreationPolicy)
	if !ok {
		t.Fatalf("policy %q does not implement sim.PeriodicCreationPolicy", name)
	}
	return tp
}
