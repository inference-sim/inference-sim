package cluster

import (
	"bytes"
	"encoding/json"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim"
)

// Spec 3 integration tests for the live periodic LoRA creation tick. The scaffold's
// inertness law (lora_periodic_scaffold_test.go) proves a gate-only policy is
// unaffected; these prove the mechanism is actually wired, bounded, deterministic, and
// that actuation — not the policy — is where demand priority is enforced.

// periodicTestAdapters is a registry large enough that adapters compete for the small
// per-instance capacity, so keep-warm always has non-resident adapters to re-warm.
var periodicTestAdapters = []string{"a0", "a1", "a2", "a3", "a4", "a5"}

// newKeepWarmCluster builds a LoRA-ACTIVE two-instance cluster whose creation policy is
// keep-warm (the reference tick policy) and whose periodic interval is intervalUs.
// Capacity 2 against 6 adapters means eviction genuinely happens, which is the regime
// keep-warm exists to act in. The workload carries adapters (zipfianAdapterRequests) —
// with base-model-only requests the demand window would stay empty and every tick would
// correctly return nothing, making a dead tick indistinguishable from a live one.
func newKeepWarmCluster(t *testing.T, intervalUs int64) *ClusterSimulator {
	t.Helper()
	config := loraAffinityTestConfig(2, 2, periodicTestAdapters)
	config.CreationPolicy = "keep-warm"
	config.LoRAPeriodicIntervalUs = intervalUs
	reqs := zipfianAdapterRequests(50, periodicTestAdapters)
	return NewClusterSimulator(config, NewSliceRequestSource(reqs), nil)
}

// runKeepWarm runs a keep-warm cluster at the given interval and returns its aggregated
// metrics as JSON — the same byte-comparison currency the inertness law uses.
func runKeepWarm(t *testing.T, intervalUs int64) []byte {
	t.Helper()
	cs := newKeepWarmCluster(t, intervalUs)
	mustRun(t, cs)
	b, err := json.Marshal(cs.AggregatedMetrics())
	if err != nil {
		t.Fatalf("json marshal: %v", err)
	}
	return b
}

// forceGateBlocked makes inst's wait-queue head a cold prefill for adapter, which is
// exactly the gate-blocked condition HasGateBlockedRequest reports. Enqueues on the
// wrapped Simulator's wait queue directly (same package) rather than through arrival
// events, so no metrics or scheduling side effects contaminate the assertion.
func forceGateBlocked(t *testing.T, inst *InstanceSimulator, adapter string) {
	t.Helper()
	req := newTestRequests(1)[0]
	req.Adapter = adapter
	inst.sim.WaitQ.Enqueue(req)
	if !inst.HasGateBlockedRequest() {
		t.Fatalf("setup: instance %s reports no gate-blocked request after enqueueing a cold %q prefill", inst.ID(), adapter)
	}
}

// TestPeriodicTick_NotInertUnderKeepWarm is the point of the whole spec: with a tick
// policy selected, a positive interval must CHANGE the run. Without this test the
// mechanism could be dead and the suite green — the scaffold's inertness law alone
// cannot tell "correctly inert" from "never wired".
func TestPeriodicTick_NotInertUnderKeepWarm(t *testing.T) {
	mOff := runKeepWarm(t, 0)
	mOn := runKeepWarm(t, 200_000) // 200ms ticks

	if bytes.Equal(mOff, mOn) {
		t.Fatalf("keep-warm with a 200ms interval produced output byte-identical to interval=0 — the tick is not wired:\n%s", mOn)
	}
}

// TestPeriodicTick_ChargesPrefetchCounter pins that the tick's loads are charged and
// attributed, so the new metric is reachable end-to-end rather than only in unit tests.
func TestPeriodicTick_ChargesPrefetchCounter(t *testing.T) {
	cs := newKeepWarmCluster(t, 200_000)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	total := int64(0)
	for _, v := range m.AdapterPrefetchCounts {
		total += v
	}
	t.Logf("prefetches=%v loads=%v evictions=%v", m.AdapterPrefetchCounts, m.AdapterLoadCounts, m.AdapterEvictionCounts)
	if total == 0 {
		t.Error("no prefetch was charged over the run; the tick fired but actuated nothing")
	}
	for id, pre := range m.AdapterPrefetchCounts {
		if load := m.AdapterLoadCounts[id]; pre > load {
			t.Errorf("adapter %q: prefetch %d > load %d — prefetch must be a strict subset", id, pre, load)
		}
	}
}

// TestPeriodicTick_Deterministic is the integration-level INV-6 catch for a map that
// reached an ordered decision inside the context builder — something the policy's own
// unit tests cannot see, because they are handed the ordered context directly.
func TestPeriodicTick_Deterministic(t *testing.T) {
	first := runKeepWarm(t, 200_000)
	for i := 1; i <= 3; i++ {
		if got := runKeepWarm(t, 200_000); !bytes.Equal(first, got) {
			t.Fatalf("run %d differs from run 0 — the tick is not deterministic (INV-6)", i)
		}
	}
}

// TestPeriodicTick_TerminatesInRequestBoundedRun guards the self-scheduling loop. A
// tick without the autoscaler's request-bounded guard (autoscaler.go scheduleNextTick)
// ticks forever, and every lora-control leaf runs request-bounded — newTestDeploymentConfig
// sets Horizon: math.MaxInt64, so that is every test here. The timeout is the assertion.
func TestPeriodicTick_TerminatesInRequestBoundedRun(t *testing.T) {
	done := make(chan struct{})
	go func() {
		defer close(done)
		cs := newKeepWarmCluster(t, 1_000) // 1ms ticks: many ticks, little work each
		mustRun(t, cs)
	}()
	select {
	case <-done:
	case <-time.After(60 * time.Second):
		t.Fatal("request-bounded run with a periodic tick did not terminate — the self-scheduling guard is missing")
	}
}

// TestPeriodicTick_DeferralEnforcedNotTrusted_GateBlocked is the load-bearing half of
// demand priority: an instance holding a cold-miss request at its gate must not have its
// single load channel taken by a prefetch for some other adapter. The guarantee lives in
// ACTUATION, so a buggy or adversarial policy cannot defeat it — mirroring the
// discipline of the load gate calling the eviction seam itself rather than trusting a
// policy's victim choice.
func TestPeriodicTick_DeferralEnforcedNotTrusted_GateBlocked(t *testing.T) {
	cs := newKeepWarmCluster(t, 200_000)
	inst := cs.instances[0]
	forceGateBlocked(t, inst, "a0")

	// Swap only the policy on the real pipeline, so the registry and demand window stay
	// genuine: a hand-built pipeline would have a nil registry and every decision would
	// be dropped as unregistered, passing this test for entirely the wrong reason.
	cs.loraPeriodic.policy = alwaysTargets{instance: string(inst.ID()), adapter: "a1"}
	cs.loraPeriodic.tick(cs, 1_000_000)

	if got := inst.LoadingAdapter(); got != "" {
		t.Errorf("actuation started a prefetch of %q on an instance with a gate-blocked request; demand priority is not enforced at actuation", got)
	}
}

// TestPeriodicTick_DeferralEnforcedNotTrusted_BusyChannel is the same rule's other half:
// a decision naming an instance whose load channel is already in flight is dropped.
// Defense in depth — StartPrefetch refuses a busy channel on its own — so this pins the
// contract rather than the only barrier.
func TestPeriodicTick_DeferralEnforcedNotTrusted_BusyChannel(t *testing.T) {
	cs := newKeepWarmCluster(t, 200_000)
	inst := cs.instances[0]
	if !inst.StartPrefetch(0, "a0") {
		t.Fatal("setup: StartPrefetch(a0) failed on an empty resident set")
	}
	before := inst.LoadingAdapter()

	cs.loraPeriodic.policy = alwaysTargets{instance: string(inst.ID()), adapter: "a1"}
	cs.loraPeriodic.tick(cs, 1_000_000)

	if after := inst.LoadingAdapter(); after != before {
		t.Errorf("actuation started a prefetch on a busy channel: %q -> %q", before, after)
	}
}

// TestPeriodicTick_OnePrefetchPerInstancePerTick pins the per-instance cap, which is
// reachable in practice and not a hypothetical: keep-warm's pickTarget re-reads the
// unmutated context on every iteration, so Resident never grows during one OnTick and
// several requested adapters can all name the same instance. Decisions are taken in
// returned order — the first surviving one per instance actuates, later ones naming that
// instance are DROPPED rather than queued, and a decision on a distinct instance still
// actuates in the same tick.
func TestPeriodicTick_OnePrefetchPerInstancePerTick(t *testing.T) {
	cs := newKeepWarmCluster(t, 200_000)
	i0, i1 := cs.instances[0], cs.instances[1]
	cs.loraPeriodic.policy = fixedDecisions{decisions: []sim.PrefetchDecision{
		{Instance: string(i0.ID()), Adapter: "a1"},
		{Instance: string(i0.ID()), Adapter: "a2"}, // same instance: dropped, not queued
		{Instance: string(i1.ID()), Adapter: "a3"}, // distinct instance: actuates
	}}
	cs.loraPeriodic.tick(cs, 1_000_000)

	if got := i0.LoadingAdapter(); got != "a1" {
		t.Errorf("instance 0 load channel = %q, want %q (the first surviving decision naming it)", got, "a1")
	}
	if got := i1.LoadingAdapter(); got != "a3" {
		t.Errorf("instance 1 load channel = %q, want %q (a decision on a distinct instance must actuate in the same tick)", got, "a3")
	}
}

// TestPeriodicTick_DropsUnknownInstanceAndAdapter pins the rest of the validation: a
// decision is a request, never a command, so a name the cluster cannot resolve is
// dropped rather than trusted or fatal.
func TestPeriodicTick_DropsUnknownInstanceAndAdapter(t *testing.T) {
	cs := newKeepWarmCluster(t, 200_000)

	cs.loraPeriodic.policy = alwaysTargets{instance: "no-such-instance", adapter: "a1"}
	cs.loraPeriodic.tick(cs, 1_000_000) // must not panic

	cs.loraPeriodic.policy = alwaysTargets{instance: string(cs.instances[0].ID()), adapter: "no-such-adapter"}
	cs.loraPeriodic.tick(cs, 2_000_000) // must not panic

	for _, inst := range cs.instances {
		if got := inst.LoadingAdapter(); got != "" {
			t.Errorf("instance %s started a load (%q) from an unresolvable decision", inst.ID(), got)
		}
	}
}

// TestPeriodicTick_NilPolicyIsSafe pins that a pipeline with no policy still
// self-schedules rather than panicking, so a partially-wired pipeline degrades to inert.
func TestPeriodicTick_NilPolicyIsSafe(t *testing.T) {
	cs := newKeepWarmCluster(t, 200_000)
	cs.loraPeriodic.policy = nil
	cs.loraPeriodic.tick(cs, 1_000_000) // must not panic
}

// TestDemandWindow_RequestedSinceIsSortedAndBounded pins the two properties a policy
// depends on: the result is sorted (INV-6 — the map range inside must not leak order)
// and the window is inclusive of `since`.
func TestDemandWindow_RequestedSinceIsSortedAndBounded(t *testing.T) {
	d := newDemandWindow()
	d.record("z", 100)
	d.record("m", 200)
	d.record("a", 300)
	d.record("", 300) // base-model-only: attributed to no adapter

	got := d.RequestedSince(200)
	want := []string{"a", "m"}
	if len(got) != len(want) {
		t.Fatalf("RequestedSince(200) = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("RequestedSince(200) = %v, want %v (sorted)", got, want)
		}
	}
	if at, ok := d.LastRequestedAt("z"); !ok || at != 100 {
		t.Errorf("LastRequestedAt(z) = (%d, %v), want (100, true)", at, ok)
	}
	if _, ok := d.LastRequestedAt(""); ok {
		t.Error("the empty adapter id was recorded; base-model-only requests must be attributed to no adapter")
	}
	if _, ok := d.LastRequestedAt("never"); ok {
		t.Error("LastRequestedAt reported true for an adapter that was never requested")
	}
}

// alwaysTargets is an adversarial policy: it returns the same decision every tick
// regardless of context, ignoring Loading and GateBlocked entirely.
type alwaysTargets struct{ instance, adapter string }

func (a alwaysTargets) OnTick(sim.PeriodicCreationContext) []sim.PrefetchDecision {
	return []sim.PrefetchDecision{{Instance: a.instance, Adapter: a.adapter}}
}

// fixedDecisions returns a fixed decision list every tick, so a test can present an
// exact ordering to actuation.
type fixedDecisions struct{ decisions []sim.PrefetchDecision }

func (f fixedDecisions) OnTick(sim.PeriodicCreationContext) []sim.PrefetchDecision {
	return f.decisions
}
