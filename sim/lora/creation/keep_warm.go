package creation

import "github.com/inference-sim/inference-sim/sim"

// keepWarm is Spec 3's REFERENCE periodic creation policy. Its purpose is to make the
// tick mechanism observable and to be the contract-test subject — it is deliberately
// NOT the paper's treatment arm (that is Spec 4's solver, issue #40).
//
// Gate behaviour is on-demand's exactly: nothing is seeded at t=0 and every cold miss is
// admitted, so keep-warm is purely additive with respect to the two instance-scoped
// entry points and trivially satisfies the starvation-freedom obligation.
//
// On each tick it prefetches adapters requested since the previous tick that are
// resident nowhere. The window IS the tick interval, so the policy needs no config of
// its own and is stateless.
//
// Two things it deliberately does not do, both to keep the three decisions factored:
// it does not care WHICH instance holds an adapter (routing's concern), and it never
// names an eviction victim (the eviction seam's concern — the cluster calls that seam
// when the chosen target is full).
//
// A property of this implementation, not a refusal: pickTarget is a pure function of
// the unmutated instance list, so every decision OnTick returns in a single call names
// the SAME instance — deterministically, not incidentally. Combined with the cluster's
// per-instance actuation cap (the first surviving decision per instance actuates; later
// ones naming that instance are dropped), keep-warm's prefetch throughput is at most one
// adapter per tick, regardless of cluster size.
//
// Expected to perform WORSE than on-demand in most regimes: a prefetch into a full
// instance evicts a warm adapter at load START, before knowing the prefetched one will
// be used again, at 7-10 ms per attempt under the Spec 2 constants. That is a correct
// result for a policy chosen for simplicity, not a defect — see design §9.
type keepWarm struct{}

func (keepWarm) Initial(sim.CreationContext) []string { return nil }

func (keepWarm) OnResidentMiss(sim.CreationContext) bool { return true }

// OnTick is a pure query: it reads the context and returns decisions, mutating nothing.
// Determinism (INV-6) comes from iterating only the ordered collections the context
// provides — RequestedSince is sorted and Instances is in construction order — so no
// map is ranged here.
func (keepWarm) OnTick(ctx sim.PeriodicCreationContext) []sim.PrefetchDecision {
	if ctx.Demand == nil {
		return nil // inert subsystem; the context documents Demand as nil-able
	}
	var out []sim.PrefetchDecision
	for _, adapter := range ctx.Demand.RequestedSince(ctx.Now - ctx.Interval) {
		if residentOrLoadingAnywhere(ctx.Instances, adapter) {
			continue
		}
		if target, ok := pickTarget(ctx.Instances); ok {
			out = append(out, sim.PrefetchDecision{Instance: target, Adapter: adapter})
		}
	}
	return out
}

// residentOrLoadingAnywhere reports whether the adapter already has, or is about to
// have, residency somewhere in the cluster — in which case a prefetch would be a wasted
// charged load.
func residentOrLoadingAnywhere(instances []sim.InstanceResidency, adapter string) bool {
	for _, inst := range instances {
		if inst.Loading == adapter {
			return true
		}
		for _, id := range inst.Resident {
			if id == adapter {
				return true
			}
		}
	}
	return false
}

// pickTarget selects a prefetch target deterministically: the first instance in
// construction order with a free slot, else the first with an evictable adapter. Free
// slots are preferred because filling one costs no warm residency, whereas evicting
// does. Returns false when the cluster is full and fully pinned.
func pickTarget(instances []sim.InstanceResidency) (string, bool) {
	for _, inst := range instances {
		if len(inst.Resident) < inst.Capacity {
			return inst.ID, true
		}
	}
	for _, inst := range instances {
		if len(inst.Unpinned) > 0 {
			return inst.ID, true
		}
	}
	return "", false
}
