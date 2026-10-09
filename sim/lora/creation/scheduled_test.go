package creation

import (
	"reflect"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func twoEntrySchedule() []sim.PlacementScheduleEntry {
	return []sim.PlacementScheduleEntry{
		{AtUs: 0, Placement: map[int][]string{0: {"a0", "a1"}, 1: {"a2"}}},
		{AtUs: 10_000_000, Placement: map[int][]string{0: {"a3"}, 1: {"a4", "a5"}}},
	}
}

func scheduledPolicy(t *testing.T, entries []sim.PlacementScheduleEntry) sim.PeriodicCreationPolicy {
	t.Helper()
	p, err := New("scheduled", sim.CreationPolicyConfig{PlacementSchedule: entries})
	if err != nil {
		t.Fatalf(`New("scheduled"): %v`, err)
	}
	tick, ok := p.(sim.PeriodicCreationPolicy)
	if !ok {
		t.Fatalf("scheduled does not implement sim.PeriodicCreationPolicy (%T)", p)
	}
	return tick
}

// ctxAt builds a two-instance context. Both instances start empty with capacity 2.
// ConstructionIndex matches slice position here (no instance is ever skipped in these
// tests), mirroring the common case; TestScheduledKeysByConstructionIndexNotSlicePosition
// below is the one that exercises the case where they diverge.
func ctxAt(now int64, resident0, resident1 []string) sim.PeriodicCreationContext {
	return sim.PeriodicCreationContext{
		Now:      now,
		Interval: 50_000,
		Instances: []sim.InstanceResidency{
			{ID: "i0", ConstructionIndex: 0, Resident: resident0, Capacity: 2},
			{ID: "i1", ConstructionIndex: 1, Resident: resident1, Capacity: 2},
		},
	}
}

// The config must actually reach the policy -- the other half of Task 1's proof.
func TestScheduledReceivesItsScheduleFromTheFactory(t *testing.T) {
	p := scheduledPolicy(t, twoEntrySchedule())
	if got := p.OnTick(ctxAt(0, nil, nil)); len(got) == 0 {
		t.Fatal("a loaded schedule produced no decisions at t=0; the config did not reach the policy")
	}
	bare, err := New("scheduled", sim.CreationPolicyConfig{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if got := bare.(sim.PeriodicCreationPolicy).OnTick(ctxAt(0, nil, nil)); got != nil {
		t.Errorf("an empty schedule must propose nothing, got %#v", got)
	}
}

func TestScheduledSelectsTheLatestEntryAtOrBeforeNow(t *testing.T) {
	p := scheduledPolicy(t, twoEntrySchedule())
	for _, tc := range []struct {
		name string
		now  int64
		want []sim.PrefetchDecision
	}{
		{"at the first entry", 0, []sim.PrefetchDecision{
			{Instance: "i0", Adapter: "a0"}, {Instance: "i0", Adapter: "a1"},
			{Instance: "i1", Adapter: "a2"}}},
		{"between entries still uses the first", 9_999_999, []sim.PrefetchDecision{
			{Instance: "i0", Adapter: "a0"}, {Instance: "i0", Adapter: "a1"},
			{Instance: "i1", Adapter: "a2"}}},
		{"exactly at the second entry switches", 10_000_000, []sim.PrefetchDecision{
			{Instance: "i0", Adapter: "a3"},
			{Instance: "i1", Adapter: "a4"}, {Instance: "i1", Adapter: "a5"}}},
		{"far past the last entry keeps it", 999_000_000, []sim.PrefetchDecision{
			{Instance: "i0", Adapter: "a3"},
			{Instance: "i1", Adapter: "a4"}, {Instance: "i1", Adapter: "a5"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := p.OnTick(ctxAt(tc.now, nil, nil)); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("OnTick(now=%d):\n got %#v\nwant %#v", tc.now, got, tc.want)
			}
		})
	}
}

// A schedule whose first entry is after now applies to nothing yet.
func TestScheduledProposesNothingBeforeItsFirstEntry(t *testing.T) {
	p := scheduledPolicy(t, []sim.PlacementScheduleEntry{
		{AtUs: 5_000_000, Placement: map[int][]string{0: {"a0"}}},
	})
	if got := p.OnTick(ctxAt(4_999_999, nil, nil)); got != nil {
		t.Errorf("want nil before the first entry, got %#v", got)
	}
}

func TestScheduledSkipsWhatIsAlreadyResidentOrLoadingOnTheAssignedInstance(t *testing.T) {
	p := scheduledPolicy(t, twoEntrySchedule())
	// a0 resident on its assigned instance; a1 still missing there.
	got := p.OnTick(ctxAt(0, []string{"a0"}, []string{"a2"}))
	want := []sim.PrefetchDecision{{Instance: "i0", Adapter: "a1"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("resident skip:\n got %#v\nwant %#v", got, want)
	}

	ctx := ctxAt(0, nil, []string{"a2"})
	ctx.Instances[0].Loading = "a0"
	got = p.OnTick(ctx)
	want = []sim.PrefetchDecision{{Instance: "i0", Adapter: "a1"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("loading skip:\n got %#v\nwant %#v", got, want)
	}
}

// THE design decision of §13.3, pinned as a test so a later "optimisation" fails loudly.
func TestScheduledRelocatesAnAdapterResidentOnTheWrongInstance(t *testing.T) {
	p := scheduledPolicy(t, twoEntrySchedule())
	// a0 belongs on i0 but is resident on i1. keep-warm would skip it (resident SOMEWHERE);
	// scheduled must still place it, because the schedule is a partition and a stray replica
	// means the solver's load balance is not realized.
	got := p.OnTick(ctxAt(0, []string{"a1"}, []string{"a2", "a0"}))
	want := []sim.PrefetchDecision{{Instance: "i0", Adapter: "a0"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("relocation:\n got %#v\nwant %#v\n"+
			"scheduled diffs PER INSTANCE, not cluster-wide (design §13.3)", got, want)
	}
}

// INV-6: decisions ordered by (construction index, adapter id), whatever the map's order.
func TestScheduledIsDeterministicAndSorted(t *testing.T) {
	entries := []sim.PlacementScheduleEntry{
		{AtUs: 0, Placement: map[int][]string{1: {"z9", "b2"}, 0: {"m5", "a1"}}},
	}
	want := []sim.PrefetchDecision{
		{Instance: "i0", Adapter: "a1"}, {Instance: "i0", Adapter: "m5"},
		{Instance: "i1", Adapter: "b2"}, {Instance: "i1", Adapter: "z9"},
	}
	for i := 0; i < 20; i++ {
		p := scheduledPolicy(t, entries)
		if got := p.OnTick(ctxAt(0, nil, nil)); !reflect.DeepEqual(got, want) {
			t.Fatalf("iteration %d:\n got %#v\nwant %#v", i, got, want)
		}
	}
}

// OnTick is a pure query: it must not mutate the context it is handed (Principle I),
// AND it must not mutate the schedule the policy was constructed from. The latter is
// the sharper risk: those Placement slices are the policy's own state, shared with
// every instance's policy copy through the same CreationPolicyConfig value, so an
// in-place sort here would corrupt state well outside this one call.
func TestScheduledOnTickDoesNotMutateItsContext(t *testing.T) {
	// The target lists are deliberately NOT pre-sorted (m5 before a1, z9 before b2): a
	// schedule fed in already-sorted order (as twoEntrySchedule's is) would let an
	// in-place sort pass unnoticed, since sorting a sorted slice changes nothing
	// observable. Unsorted input is what makes the copy-vs-sort-in-place distinction
	// detectable at all.
	entries := []sim.PlacementScheduleEntry{
		{AtUs: 0, Placement: map[int][]string{0: {"m5", "a1"}, 1: {"z9", "b2"}}},
	}
	p := scheduledPolicy(t, entries)
	ctx := ctxAt(0, []string{"a9"}, []string{"b9"}) // resident adapters distinct from targets, so nothing is skipped
	before := [][]string{append([]string(nil), ctx.Instances[0].Resident...),
		append([]string(nil), ctx.Instances[1].Resident...)}
	scheduleBefore := cloneSchedule(entries)
	p.OnTick(ctx)
	for i, was := range before {
		if !reflect.DeepEqual(ctx.Instances[i].Resident, was) {
			t.Errorf("instance %d Resident mutated: %#v -> %#v", i, was, ctx.Instances[i].Resident)
		}
	}
	if got := cloneSchedule(entries); !reflect.DeepEqual(got, scheduleBefore) {
		t.Errorf("OnTick mutated the policy's own schedule:\n before %#v\n after  %#v", scheduleBefore, got)
	}
}

// cloneSchedule deep-copies a schedule's Placement slices so a before/after comparison
// can detect an in-place sort on the policy's OWN state, not just on a caller's copy.
func cloneSchedule(entries []sim.PlacementScheduleEntry) []sim.PlacementScheduleEntry {
	out := make([]sim.PlacementScheduleEntry, len(entries))
	for i, e := range entries {
		p := make(map[int][]string, len(e.Placement))
		for k, v := range e.Placement {
			p[k] = append([]string(nil), v...)
		}
		out[i] = sim.PlacementScheduleEntry{AtUs: e.AtUs, Placement: p}
	}
	return out
}

// The bug this guards against: buildContext's routable filter means SLICE POSITION and
// CONSTRUCTION INDEX diverge whenever an earlier instance is skipped as non-routable.
// Construct that divergence directly (construction indices 0 and 2 occupy slice
// positions 0 and 1, as if construction index 1 were skipped) and confirm OnTick keys
// off ConstructionIndex. Keying by slice position instead would look up Placement[1]
// for the second instance -- which is empty here -- and silently drop its decision.
func TestScheduledKeysByConstructionIndexNotSlicePosition(t *testing.T) {
	p := scheduledPolicy(t, []sim.PlacementScheduleEntry{
		{AtUs: 0, Placement: map[int][]string{0: {"a0"}, 2: {"a2"}}},
	})
	ctx := sim.PeriodicCreationContext{
		Now: 0,
		Instances: []sim.InstanceResidency{
			{ID: "i0", ConstructionIndex: 0, Capacity: 2},
			{ID: "i2", ConstructionIndex: 2, Capacity: 2},
		},
	}
	got := p.OnTick(ctx)
	want := []sim.PrefetchDecision{{Instance: "i0", Adapter: "a0"}, {Instance: "i2", Adapter: "a2"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %#v\nwant %#v\n(position-based keying would silently drop i2's decision)", got, want)
	}
}

// Gate behaviour must be pre-placement's, byte for byte.
func TestScheduledGateBehaviourMatchesPrePlacement(t *testing.T) {
	sched, err := New("scheduled", sim.CreationPolicyConfig{PlacementSchedule: twoEntrySchedule()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	pre, err := New("pre-placement", sim.CreationPolicyConfig{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for _, ctx := range []sim.CreationContext{
		{Assigned: []string{"a0", "a1"}},
		{Assigned: nil},
		{MissedAdapter: "a9"},
	} {
		if got, want := sched.Initial(ctx), pre.Initial(ctx); !reflect.DeepEqual(got, want) {
			t.Errorf("Initial(%#v): got %#v, pre-placement gives %#v", ctx, got, want)
		}
		if got, want := sched.OnResidentMiss(ctx), pre.OnResidentMiss(ctx); got != want {
			t.Errorf("OnResidentMiss(%#v): got %v, pre-placement gives %v", ctx, got, want)
		}
	}
}
