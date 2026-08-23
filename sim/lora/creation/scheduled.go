package creation

import (
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// scheduled is Spec 4 Slice B's treatment policy: pre-placement's gate behaviour plus a
// tick that drives residency toward a TIMED placement (design §13.3).
//
// Gate behaviour is pre-placement's exactly, by delegation rather than by duplication —
// t=0 seeding from ctx.Assigned, uncharged (INV-L3), and OnResidentMiss always admits, which
// discharges the starvation-freedom obligation on sim.CreationPolicy. Only OnTick is new.
//
// It is the first registered creation policy that is NOT a stateless empty struct: it owns
// its schedule, built once at construction from --lora-placement-schedule and never mutated,
// so OnTick stays a pure query (Spec 3, Principle I).
//
// Two arms of Spec 4 Slice B are the same policy fed different files — a constant schedule is
// "static + repair", a cohort-tracking one is the oracle re-solve — and a lookahead offset is
// a shift of the timestamps. That is why the schedule is data and why the entry lookup is a
// <= scan rather than now/interval arithmetic: non-uniform schedules must stay expressible.
//
// Unlike keep-warm, scheduled is not capped at one prefetch per tick. keep-warm's pickTarget
// is pure over the unmutated instance list, so every decision it returns names the same
// instance and the cluster's per-instance actuation cap admits one. scheduled's decisions name
// distinct instances by construction, so up to one per instance actuates each tick.
type scheduled struct {
	// schedule is ordered by strictly increasing AtUs, with a non-empty Placement per entry.
	// cmd.parseLoRAPlacementSchedule enforces both, and
	// cluster.ValidateLoRAPlacementSchedule enforces index range, id registration,
	// uniqueness and per-instance capacity. Nil is legal and inert.
	schedule []sim.PlacementScheduleEntry
}

// Initial delegates to pre-placement: the seeded set is the cluster-resolved subset, copied
// defensively, and seeding is uncharged (INV-L3).
func (scheduled) Initial(ctx sim.CreationContext) []string {
	return prePlacement{}.Initial(ctx)
}

// OnResidentMiss delegates to pre-placement: always admit, so the miss path is unchanged and
// starvation-freedom holds trivially.
func (scheduled) OnResidentMiss(ctx sim.CreationContext) bool {
	return prePlacement{}.OnResidentMiss(ctx)
}

// entryAt returns the latest entry with AtUs <= now, and false when none applies yet.
//
// A binary search over a strictly-increasing key rather than now/L arithmetic: the schedule
// is not required to be uniform, which is what lets a caller express a lookahead offset
// (apply epoch k's placement before epoch k begins) without a second flag.
func (s scheduled) entryAt(now int64) (sim.PlacementScheduleEntry, bool) {
	// i is the number of entries with AtUs <= now, so i-1 indexes the latest applicable one.
	i := sort.Search(len(s.schedule), func(k int) bool { return s.schedule[k].AtUs > now })
	if i == 0 {
		return sim.PlacementScheduleEntry{}, false
	}
	return s.schedule[i-1], true
}

// OnTick returns the prefetches that would move residency toward the entry in force at
// ctx.Now. A pure query: it reads the context and the schedule and mutates neither.
//
// The diff is PER INSTANCE, deliberately. An adapter resident on an instance the schedule
// does not assign it to is a stray replica, not a satisfied placement: the schedule is a
// partition, and a stray copy means the solver's load balance is not realized. So unlike
// keep-warm this policy does NOT consult residentOrLoadingAnywhere, and it can pay a charged
// load to relocate an adapter that is already warm elsewhere. That cost is real — under LRU a
// relocation can oscillate — and it is measured by the leaf's churn pair rather than avoided
// here. Changing this to a cluster-wide check would discard the objective under test.
//
// Determinism (INV-6): instances are visited in ctx.Instances order — a subsequence of
// cs.instances order, since buildContext appends in that order and only omits non-routable
// instances — and each entry's adapter list is sorted into a copy, so no map is ranged and
// the returned slice is ordered by (ConstructionIndex, adapter id).
//
// Placement is keyed by CONSTRUCTION index (InstanceResidency.ConstructionIndex), never
// by a range index over ctx.Instances: a non-routable instance is omitted from
// ctx.Instances, so the two indices diverge exactly when that happens, and keying by
// position would silently hand one instance another instance's target set.
//
// That ConstructionIndex is also the key space the schedule file itself was written in
// holds WHILE every instance comes from NewClusterSimulator's construction loop — every run
// without NodePools, and every NodePools run with no deferred instance. See
// sim.InstanceResidency.ConstructionIndex for the exception: a NodePools deferral appends
// the instance at the end of cs.instances instead of at its construction-loop index, which
// REORDERS rather than subsets, and no check here would notice. This experiment's
// configuration does not reach it.
func (s scheduled) OnTick(ctx sim.PeriodicCreationContext) []sim.PrefetchDecision {
	entry, ok := s.entryAt(ctx.Now)
	if !ok {
		return nil
	}
	var out []sim.PrefetchDecision
	for _, inst := range ctx.Instances {
		targets := entry.Placement[inst.ConstructionIndex]
		if len(targets) == 0 {
			continue
		}
		sorted := make([]string, len(targets))
		copy(sorted, targets)
		sort.Strings(sorted)
		for _, adapter := range sorted {
			if inst.Loading == adapter || holdsAdapter(inst, adapter) {
				continue
			}
			out = append(out, sim.PrefetchDecision{Instance: inst.ID, Adapter: adapter})
		}
	}
	return out
}

// holdsAdapter reports whether this ONE instance already has the adapter resident. The
// per-instance counterpart to keep-warm's residentOrLoadingAnywhere, and the difference
// between the two is the design decision documented on OnTick.
func holdsAdapter(inst sim.InstanceResidency, adapter string) bool {
	for _, id := range inst.Resident {
		if id == adapter {
			return true
		}
	}
	return false
}
