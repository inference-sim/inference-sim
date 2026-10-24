package cluster

import (
	"reflect"
	"strings"
	"testing"

	"pgregory.net/rapid"

	"github.com/inference-sim/blis-vllm/compat/kvtransfer"
	sim "github.com/inference-sim/inference-sim/sim"
)

// INV-P2-2 (#1922): with contention on, concurrent P/D handoffs share the links they cross.
// The demand is the latency backend's (here a fake shaped like blis-latency-kernel's: the
// prefill node's NIC egress and the decode node's NIC ingress, each at one rate, and a fixed
// floor); the arbitration is the simulator's.

const testNICBytesPerSecond = 1e9

// testDemand is a cross-node handoff: 1000 bytes per token over the sender's egress and the
// receiver's ingress, with a 50 us floor.
func testDemand(tokens int64, from, to InstanceID) kvtransfer.Flow {
	return kvtransfer.Flow{Bytes: tokens * 1000, FloorNs: 50_000, Links: []kvtransfer.Link{
		{ID: "nic/egress/" + string(from), BytesPerSecond: testNICBytesPerSecond},
		{ID: "nic/ingress/" + string(to), BytesPerSecond: testNICBytesPerSecond},
	}}
}

// testDemandPrice is PDTransferTime consistent with testDemand, as the kernel's law requires.
func testDemandPrice(tokens int64, from, to InstanceID) int64 {
	return kvtransfer.AloneTicks(testDemand(tokens, from, to))
}

func contentionConfig(numInstances, prefill, decode int, contention bool) DeploymentConfig {
	cfg := newTestDisaggDeploymentConfig(numInstances, prefill, decode)
	cfg.PDTransferTime = testDemandPrice
	cfg.PDTransferDemand = testDemand
	cfg.PDTransferContention = contention
	return cfg
}

func cloneRequests(reqs []*sim.Request) []*sim.Request {
	out := make([]*sim.Request, len(reqs))
	for i, r := range reqs {
		c := *r
		out[i] = &c
	}
	return out
}

func parentsByID(cs *ClusterSimulator) map[string]*ParentRequest {
	out := map[string]*ParentRequest{}
	for _, p := range cs.ParentRequests() {
		out[p.ID] = p
	}
	return out
}

// Metamorphic: when handoffs never overlap, contention changes nothing -- every handoff is
// alone on its links and so costs exactly its price, and the run is identical to one priced
// without contention.
func TestPDContention_OneTransferAtATimeIsUnchanged(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		n := rapid.IntRange(1, 6).Draw(rt, "n")
		reqs := make([]*sim.Request, n)
		for i := range reqs {
			// 10 s apart: every request finishes long before the next arrives.
			reqs[i] = holdTestRequest(string(rune('a'+i)), i, int64(i)*10_000_000,
				rapid.IntRange(1, 4000).Draw(rt, "in"), rapid.IntRange(1, 8).Draw(rt, "out"))
		}
		on := runHold(rt, contentionConfig(2, 1, 1, true), cloneRequests(reqs)...)
		off := runHold(rt, contentionConfig(2, 1, 1, false), cloneRequests(reqs)...)
		mOn, mOff := on.AggregatedMetrics(), off.AggregatedMetrics()
		if !reflect.DeepEqual(mOn.RequestE2Es, mOff.RequestE2Es) || !reflect.DeepEqual(mOn.RequestTTFTs, mOff.RequestTTFTs) {
			rt.Fatalf("isolated handoffs changed latency under contention:\non %v\noff %v", mOn.RequestE2Es, mOff.RequestE2Es)
		}
		for id, p := range parentsByID(on) {
			if d := p.TransferCompleteTime - p.TransferStartTime; d != p.TransferAloneTicks {
				rt.Fatalf("%s alone took %d ticks, its price is %d", id, d, p.TransferAloneTicks)
			}
		}
	})
}

// Two prefill instances hand off to one decode instance at the same moment: both handoffs
// cross the decode node's NIC ingress, so each takes longer than alone, and the run's mean
// E2E rises over the uncontended run.
func TestPDContention_OverlappingHandoffsShareTheDecodeNIC(t *testing.T) {
	reqs := []*sim.Request{holdTestRequest("a", 1, 0, 4096, 4), holdTestRequest("b", 2, 0, 4096, 4)}
	on := runHold(t, contentionConfig(3, 2, 1, true), cloneRequests(reqs)...)
	off := runHold(t, contentionConfig(3, 2, 1, false), cloneRequests(reqs)...)
	ps := parentsByID(on)
	if ps["a"].PrefillInstanceID == ps["b"].PrefillInstanceID {
		t.Fatalf("the two requests must prefill on different instances to overlap their handoffs")
	}
	if ps["a"].TransferStartTime != ps["b"].TransferStartTime {
		t.Fatalf("handoffs must start together: %d vs %d", ps["a"].TransferStartTime, ps["b"].TransferStartTime)
	}
	for id, p := range ps {
		d := p.TransferCompleteTime - p.TransferStartTime
		if d <= p.TransferAloneTicks {
			t.Errorf("%s: %d ticks on a shared ingress, %d alone; sharing must slow it", id, d, p.TransferAloneTicks)
		}
		// 4096 tokens x 1000 B at 1 GB/s is 4096 us alone; half the ingress each doubles it.
		if want := 2 * int64(4096*1000/1000); d != want {
			t.Errorf("%s: %d ticks, want %d (half of the shared ingress for the whole transfer)", id, d, want)
		}
	}
	e2e := func(cs *ClusterSimulator) float64 {
		var s float64
		for _, v := range cs.AggregatedMetrics().RequestE2Es {
			s += v
		}
		return s
	}
	if e2e(on) <= e2e(off) {
		t.Errorf("total E2E with contention %.0f, without %.0f; contention must raise it", e2e(on), e2e(off))
	}
	if peak := on.PDLinkPeakConcurrent(); peak["nic/ingress/instance_2"] != 2 {
		t.Errorf("peak concurrent handoffs per link = %v, want 2 on the decode ingress", peak)
	}
}

// Handoffs on disjoint links finish exactly as they would alone, however many overlap.
func TestPDContention_DisjointLinksDoNotInteract(t *testing.T) {
	reqs := []*sim.Request{holdTestRequest("a", 1, 0, 4096, 4), holdTestRequest("b", 2, 0, 4096, 4)}
	cs := runHold(t, contentionConfig(4, 2, 2, true), reqs...)
	ps := parentsByID(cs)
	a, b := ps["a"], ps["b"]
	if a.PrefillInstanceID == b.PrefillInstanceID || a.DecodeInstanceID == b.DecodeInstanceID {
		t.Fatalf("the handoffs must cross disjoint links: a %s->%s, b %s->%s",
			a.PrefillInstanceID, a.DecodeInstanceID, b.PrefillInstanceID, b.DecodeInstanceID)
	}
	for id, p := range ps {
		if d := p.TransferCompleteTime - p.TransferStartTime; d != p.TransferAloneTicks {
			t.Errorf("%s on its own links took %d ticks, alone %d", id, d, p.TransferAloneTicks)
		}
	}
}

// Property: under any overlap, no handoff completes before its uncontended price (causality),
// INV-PD-3 holds, every request is conserved (INV-1) -- including at a horizon that cuts
// handoffs mid-flight -- and the run is deterministic (INV-6).
func TestPDContention_NeverFasterThanAloneAndConserves(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		n := rapid.IntRange(1, 10).Draw(rt, "n")
		reqs := make([]*sim.Request, n)
		var at int64
		for i := range reqs {
			at += rapid.Int64Range(0, 3000).Draw(rt, "gap")
			reqs[i] = holdTestRequest(string(rune('a'+i)), i, at, rapid.IntRange(1, 4000).Draw(rt, "in"), rapid.IntRange(1, 6).Draw(rt, "out"))
		}
		prefill := rapid.IntRange(1, 3).Draw(rt, "prefill")
		decode := rapid.IntRange(1, 2).Draw(rt, "decode")
		cfg := contentionConfig(prefill+decode, prefill, decode, true)
		cfg.PDPrefillKVHold = rapid.Bool().Draw(rt, "hold")
		cs := runHold(rt, cfg, cloneRequests(reqs)...)
		for id, p := range parentsByID(cs) {
			if p.TransferCompleteTime > 0 && p.TransferCompleteTime-p.TransferStartTime < p.TransferAloneTicks {
				rt.Fatalf("%s completed in %d ticks, faster than alone (%d)", id, p.TransferCompleteTime-p.TransferStartTime, p.TransferAloneTicks)
			}
		}
		if cs.transfersInitiated != cs.transfersCompleted {
			rt.Fatalf("INV-PD-3: initiated %d != completed %d", cs.transfersInitiated, cs.transfersCompleted)
		}
		if l := newClusterLedger(cs); l.accounted() != n {
			rt.Fatalf("INV-1: injected %d != accounted %d [%s]", n, l.accounted(), l)
		}
		again := runHold(rt, cfg, cloneRequests(reqs)...)
		if !reflect.DeepEqual(cs.AggregatedMetrics().RequestE2Es, again.AggregatedMetrics().RequestE2Es) {
			rt.Fatalf("two identical runs differ")
		}

		// Cut after the last arrival has reached its instance (a cut inside the arrival
		// pipeline is a known INV-1 edge, see Run's post-simulation note), while handoffs and
		// steps may still be in flight.
		cut := cfg
		cut.Horizon = rapid.Int64Range(at+20_000, at+200_000).Draw(rt, "horizon")
		cutRun := runHold(rt, cut, cloneRequests(reqs)...)
		if l := newClusterLedger(cutRun); l.accounted() != n {
			rt.Fatalf("INV-1 at horizon %d: injected %d != accounted %d [%s]", cut.Horizon, n, l.accounted(), l)
		}
	})
}

// Contention is refused without the demand behind each price, and a backend whose demand
// does not cost its own price alone is refused at the first handoff rather than arbitrated.
func TestPDContention_RefusesMissingOrInconsistentDemand(t *testing.T) {
	expectPanic := func(t *testing.T, want string, fn func()) {
		t.Helper()
		defer func() {
			r := recover()
			if r == nil || !strings.Contains(r.(string), want) {
				t.Fatalf("panic = %v, want one naming %q", r, want)
			}
		}()
		fn()
	}
	t.Run("no demand", func(t *testing.T) {
		cfg := contentionConfig(2, 1, 1, true)
		cfg.PDTransferDemand = nil
		expectPanic(t, "PDTransferDemand", func() { NewClusterSimulator(cfg, NewSliceRequestSource(nil), nil) })
	})
	t.Run("demand disagrees with price", func(t *testing.T) {
		cfg := contentionConfig(2, 1, 1, true)
		cfg.PDTransferTime = func(tokens int64, f, g InstanceID) int64 { return testDemandPrice(tokens, f, g) + 1 }
		cs := NewClusterSimulator(cfg, NewSliceRequestSource([]*sim.Request{holdTestRequest("a", 1, 0, 64, 2)}), nil)
		expectPanic(t, "AloneTime law", func() { _ = cs.Run() })
	})
}

// The link arbiter's wakes stay bounded by the events that need them. A wake is scheduled by a
// handoff start, or by a live wake at a tick where a handoff drains or finishes; one superseded
// by a later start does nothing (otherwise it would re-advance and schedule yet another). Here
// every handoff drains at its finish (its 50 us floor is below every duration), so the wakes
// number at most the starts plus the distinct finish ticks, plus one. A burst of 40 handoffs on
// one decode NIC.
func TestPDContention_WakesStayLinearInABurst(t *testing.T) {
	const n = 40
	reqs := make([]*sim.Request, n)
	for i := range reqs {
		reqs[i] = holdTestRequest(string(rune('A'+i)), i, 0, 256, 2)
	}
	cs := runHold(t, contentionConfig(5, 4, 1, true), reqs...)
	if cs.transfersCompleted != n {
		t.Fatalf("fixture: %d of %d handoffs completed", cs.transfersCompleted, n)
	}
	if peak := cs.PDLinkPeakConcurrent()["nic/ingress/instance_4"]; peak < 4 {
		t.Fatalf("fixture: handoffs must overlap on the decode ingress (peak %d)", peak)
	}
	finishes := map[int64]bool{}
	for _, p := range cs.ParentRequests() {
		if p.TransferCompleteTime-p.TransferStartTime <= 50 {
			t.Fatalf("fixture: %s's handoff is within its floor", p.ID)
		}
		finishes[p.TransferCompleteTime] = true
	}
	if bound := int64(n + len(finishes) + 1); cs.pdLinkWakeGen > bound {
		t.Errorf("%d link wakes scheduled for %d starts and %d finish ticks, want at most %d",
			cs.pdLinkWakeGen, n, len(finishes), bound)
	}
}
