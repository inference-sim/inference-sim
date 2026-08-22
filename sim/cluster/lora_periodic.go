package cluster

import (
	"container/heap"
	"fmt"
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
)

// loraPeriodicPipeline owns the periodic LoRA creation tick (Spec 3), activating the
// LoRAPeriodicTriggerEvent that B-7 reserved. It mirrors autoscalerPipeline's shape — a
// declared interval, a self-scheduling tick, a decide-then-actuate split — because the
// scaffold was explicitly written to that shape.
//
// One deliberate difference from the autoscaler: there is NO separate actuation delay
// and no ScaleActuationEvent analogue. LoadLatency already IS the physical lag (an
// adapter is not resident until now+LoadLatency, 7-10 ms under the Spec 2 constants), so
// a second delay would add a config knob plus a staleness window in which a decision can
// name an instance the autoscaler just scaled away.
type loraPeriodicPipeline struct {
	// policy is the cluster's OWN tick-policy instance, resolved by name AND by config from
	// the same fields each instance's Simulator resolves from. Through Spec 3 every shipped
	// policy was a stateless empty struct, so a separate instance was trivially identical to
	// the instances' own; Spec 4's `scheduled` carries a placement schedule, so the two are
	// identical because they are built from the SAME CreationPolicyConfig, not because the
	// policy is empty. A cluster-scope decision could not be owned by any one instance's
	// policy object in any case.
	policy sim.PeriodicCreationPolicy
	// interval is the tick period in microseconds (always > 0 when the pipeline exists).
	interval int64
	// demand is the cluster-wide per-adapter request-recency window, updated on every
	// cluster arrival and handed to the policy read-only.
	demand *demandWindow
	// registry is the read-only adapter registry, resolved once at construction. Held
	// here rather than reached through an instance: registries are per-instance
	// unexported state built from the same cluster-wide config, so one shared read-only
	// view is both correct and cheaper than reaching into an instance for it.
	registry sim.AdapterRegistry
}

// newLoRAPeriodicPipeline resolves the cluster's tick policy and returns the pipeline,
// or nil when no tick can ever fire. Nil unless ALL THREE hold:
//
//  1. the LoRA subsystem is active — adapters declared AND a capacity set, the same
//     predicate BuildAdapterRegistry and NewSimulator gate their LoRA wiring on;
//  2. LoRAPeriodicIntervalUs > 0;
//  3. the effective creation policy implements the optional sim.PeriodicCreationPolicy.
//
// That three-way condition is what makes INV-PS3' STRUCTURAL rather than a runtime
// check: with no pipeline there is no tick to be inert about — Run() pushes no first
// tick, ClusterArrivalEvent records no demand, and the trigger event's nil guard drops
// any stray one. A gate-only policy (on-demand, pre-placement) is therefore
// byte-identical with the interval set or unset.
//
// The empty policy name resolves to on-demand, matching the default NewSimulator applies
// so the cluster and its instances can never disagree about which policy is in effect.
func newLoRAPeriodicPipeline(config DeploymentConfig) *loraPeriodicPipeline {
	if config.LoRAPeriodicIntervalUs <= 0 {
		return nil
	}
	if !config.HasAdapters() || config.AdapterCapacity == nil {
		// LoRA is inert: no residency exists to prefetch into.
		logrus.Warnf("[lora-periodic] lora_periodic_interval_us=%d set but the LoRA subsystem is inactive (no adapters declared, or adapter_capacity unset); no tick scheduled", config.LoRAPeriodicIntervalUs)
		return nil
	}
	if sim.NewCreationPolicyFunc == nil {
		// sim/lora was never linked, so the seam factories are unregistered and the
		// subsystem is inert everywhere (NewSimulator fails the same check). Warn rather
		// than silently dropping a configured tick (R1).
		logrus.Warnf("[lora-periodic] lora_periodic_interval_us=%d set but the creation-policy registry is unregistered (sim/lora not linked); no tick scheduled", config.LoRAPeriodicIntervalUs)
		return nil
	}
	name := config.CreationPolicy
	if name == "" {
		name = "on-demand"
	}
	policy, err := sim.NewCreationPolicyFunc(name, sim.CreationPolicyConfig{
		PlacementSchedule: config.PlacementSchedule,
	})
	if err != nil {
		// Unreachable in practice — instance construction above resolves the same name
		// and panics first — but the library layer fails fast rather than ticking with
		// no policy (Principle V).
		panic(fmt.Sprintf("ClusterSimulator: periodic creation policy: %v", err))
	}
	tickPolicy, ok := policy.(sim.PeriodicCreationPolicy)
	if !ok {
		return nil // gate-only policy: INV-PS3' holds by construction
	}
	registry, err := sim.BuildAdapterRegistry(config.ToSimConfig())
	if err != nil {
		panic(fmt.Sprintf("ClusterSimulator: periodic creation adapter registry: %v", err))
	}
	if registry == nil {
		// Would make every decision fail validation as "unregistered adapter", i.e. a
		// silently dead tick policy. Refuse to build the pipeline instead (R1).
		logrus.Warnf("[lora-periodic] creation policy %q implements the tick seam but no adapter registry could be built; no tick scheduled", name)
		return nil
	}
	return &loraPeriodicPipeline{
		policy:   tickPolicy,
		interval: config.LoRAPeriodicIntervalUs,
		demand:   newDemandWindow(),
		registry: registry,
	}
}

// tick builds the cluster-scope context, asks the policy for decisions, actuates the
// ones that survive validation, and self-schedules.
func (p *loraPeriodicPipeline) tick(cs *ClusterSimulator, nowUs int64) {
	if p.policy != nil {
		p.actuate(cs, nowUs, p.policy.OnTick(p.buildContext(cs, nowUs)))
	}
	p.scheduleNextTick(cs, nowUs)
}

// buildContext assembles the read-only cluster-scope view. Every collection is ordered
// deterministically (INV-6): instances in cs.instances construction order, and each
// instance's adapter slices sorted — ResidentAdapterIDs and UnpinnedAdapterIDs both
// return LRU→MRU order over freshly-built slices, so sorting in place is safe and is
// what the context's fields promise. No map is ranged here.
func (p *loraPeriodicPipeline) buildContext(cs *ClusterSimulator, nowUs int64) sim.PeriodicCreationContext {
	insts := make([]sim.InstanceResidency, 0, len(cs.instances))
	for _, inst := range cs.instances {
		if !inst.IsRoutable() {
			continue // a decision naming it would be dropped at actuation anyway
		}
		resident := inst.ResidentAdapterIDs()
		unpinned := inst.UnpinnedAdapterIDs()
		sort.Strings(resident)
		sort.Strings(unpinned)
		insts = append(insts, sim.InstanceResidency{
			ID:          string(inst.ID()),
			Resident:    resident,
			Unpinned:    unpinned,
			Capacity:    inst.AdapterCapacity(),
			Loading:     inst.LoadingAdapter(),
			GateBlocked: inst.HasGateBlockedRequest(),
		})
	}
	// Demand is documented as nil-able, so pass a genuine nil interface rather than a
	// typed nil pointer a policy's nil-check would not catch.
	var demand sim.DemandWindow
	if p.demand != nil {
		demand = p.demand
	}
	return sim.PeriodicCreationContext{
		Now:       nowUs,
		Interval:  p.interval,
		Instances: insts,
		Demand:    demand,
		Registry:  p.registry,
	}
}

// actuate applies the surviving decisions. A decision is a REQUEST, not a command: this
// is the single place the demand-priority rule is enforced, so a buggy or adversarial
// policy cannot defeat it. That rule is narrow and is defined precisely at the check
// below — read it there before relying on it.
//
// At most ONE prefetch per instance per tick starts, because loads serialize per
// instance. The cap is per instance, not per tick: decisions are taken in returned
// order, the first SURVIVING decision for each instance actuates, and later decisions
// naming an instance that already started one are DROPPED rather than queued — a queued
// decision would act on stale state the next tick recomputes anyway. Decisions naming
// distinct instances all actuate in the same tick.
//
// That last sentence documents the mechanism's contract, not something exercised today:
// no SHIPPED policy currently emits decisions naming distinct instances in one tick —
// keep-warm included, because its pickTarget names the same instance for every decision
// in a call (see creation.keepWarm's doc comment). Only a test stub does.
//
// This is reachable, not hypothetical: keep-warm's pickTarget re-reads the unmutated
// context on every iteration, so Resident never grows during a single OnTick and several
// requested adapters can all name the same instance.
func (p *loraPeriodicPipeline) actuate(cs *ClusterSimulator, nowUs int64, decisions []sim.PrefetchDecision) {
	if len(decisions) == 0 {
		return
	}
	started := make(map[string]bool, len(decisions))
	for _, d := range decisions {
		if started[d.Instance] {
			continue // one prefetch per instance per tick
		}
		inst := cs.instanceByID(InstanceID(d.Instance))
		if inst == nil || !inst.IsRoutable() {
			logrus.Debugf("[lora-periodic] t=%d dropping decision for unknown/non-routable instance %q", nowUs, d.Instance)
			continue
		}
		if p.registry == nil || !p.registry.Has(d.Adapter) {
			logrus.Debugf("[lora-periodic] t=%d dropping decision for unregistered adapter %q", nowUs, d.Adapter)
			continue
		}
		// Demand priority, stated exactly: never take the load channel from a cold-miss
		// request already at the gate, and never queue behind a load in flight.
		//
		// That is NARROWER than "never step in front of work visible on this instance",
		// and the difference is real, not pedantic: HasGateBlockedRequest is only "the
		// wait-queue HEAD is a cold miss", so a request whose adapter IS resident is
		// unprotected even when it is the head. Its adapter stays unpinned until batch
		// admission takes the pin (recordAdapterResidency), so StartPrefetch's eviction
		// seam may evict exactly that adapter and then hold the channel — turning a warm
		// hit into a cold miss and delaying it by up to one LoadLatency. Liveness is
		// unaffected (the gate re-runs and reloads it, INV-8); the cost is latency.
		//
		// Widening this to "skip any instance with a non-empty wait queue" would suppress
		// prefetching precisely under load. That is a design decision, not a defect fix,
		// so the narrow rule stands — stated honestly here rather than advertised as
		// something stronger than it is.
		//
		// Both halves are kept even though the busy-channel one is currently redundant
		// with StartPrefetch's own refusal (defense in depth): the redundancy is
		// contingent — if StartPrefetch ever queued instead of refusing, both halves would
		// become load-bearing. The gate-blocked half is enforced ONLY here, because the
		// instance's load gate cannot know a prefetch is being considered.
		if inst.LoadingAdapter() != "" || inst.HasGateBlockedRequest() {
			logrus.Debugf("[lora-periodic] t=%d deferring prefetch of %q on %q: load channel busy or a request is gate-blocked", nowUs, d.Adapter, d.Instance)
			continue
		}
		// StartPrefetch owns slot reservation and calls the eviction seam itself, so the
		// tick policy never names a victim.
		if inst.StartPrefetch(nowUs, d.Adapter) {
			started[d.Instance] = true
		}
	}
}

// scheduleNextTick self-schedules, carrying autoscalerPipeline.scheduleNextTick's
// request-bounded termination guard verbatim (autoscaler.go): in a request-bounded run
// (Horizon == math.MaxInt64) it stops ticking once all arrivals are processed and all
// instances are idle. Without it such a run — which is how every lora-control leaf runs
// — ticks forever. Nothing is lost by stopping: with arrivals drained there is no demand
// left to prefetch for. The <= 0 on pendingArrivals matches the autoscaler's safety net
// for a future push site that bypasses pushArrival.
func (p *loraPeriodicPipeline) scheduleNextTick(cs *ClusterSimulator, nowUs int64) {
	if cs.config.Horizon == math.MaxInt64 && cs.pendingArrivals <= 0 {
		var inFlight int
		for _, v := range cs.inFlightRequests {
			inFlight += v
		}
		if inFlight == 0 {
			return // no more work; don't self-schedule
		}
	}
	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &LoRAPeriodicTriggerEvent{At: nowUs + p.interval},
		seqID: cs.nextSeqID(),
	})
}

// demandWindow is the sim.DemandWindow implementation: last-requested-at per adapter,
// updated on every cluster arrival. It is written on the arrival path and read only by
// the tick, so it needs no history beyond the most recent timestamp — keep-warm's window
// IS the tick interval.
//
// Reads are sorted so a policy never sees map order (INV-6); adapter counts are in the
// tens, so the sort is free.
type demandWindow struct{ lastAt map[string]int64 }

func newDemandWindow() *demandWindow { return &demandWindow{lastAt: map[string]int64{}} }

// record stamps the adapter as requested at nowUs. Base-model-only requests (empty
// adapter id) are attributed to no adapter.
func (d *demandWindow) record(adapter string, nowUs int64) {
	if adapter == "" {
		return
	}
	d.lastAt[adapter] = nowUs
}

// LastRequestedAt returns the most recent request time for the adapter, and false when
// it has never been requested.
func (d *demandWindow) LastRequestedAt(adapter string) (int64, bool) {
	at, ok := d.lastAt[adapter]
	return at, ok
}

// RequestedSince returns the adapter ids requested at or after since, sorted.
func (d *demandWindow) RequestedSince(since int64) []string {
	out := make([]string, 0, len(d.lastAt))
	for id, at := range d.lastAt {
		if at >= since {
			out = append(out, id)
		}
	}
	// The map range above only SELECTS; this makes the RESULT ordered (INV-6). Ranging a
	// map to build an ordered slice would be a determinism defect — sorting into one is
	// not.
	sort.Strings(out)
	return out
}

// Compile-time assertion that demandWindow satisfies the read-only accessor the context
// carries, so the policy-facing contract cannot drift from the implementation.
var _ sim.DemandWindow = (*demandWindow)(nil)
