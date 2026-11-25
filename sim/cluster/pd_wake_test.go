package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// The requests of #2685's report: three at t=0 and four around 1.3 ms, sharing a 32-token head,
// as TestPrefillKVHold_AmpleCapacityChangesNoLatency builds them.
func wakeCaseRequests() []*sim.Request {
	arrivals := []int64{0, 1321, 0, 1323, 1324, 1325, 0}
	out := make([]*sim.Request, len(arrivals))
	for i, at := range arrivals {
		r := holdTestRequest(string(rune('a'+i)), i, at, 64+16*(i%4), 1+i%5)
		copy(r.InputTokens[:32], make([]sim.TokenID, 32))
		out[i] = r
	}
	return out
}

// A decode request whose pull lands is stepped, though a handoff reaches its engine at the very
// instant the engine's wake for it is due (#2685). At these transfer times the next handoff
// arrives at 3060 µs, the wake's instant, and used to supersede it: the landed request was never
// stepped, every later handoff waited for its decode KV, and the run stalled (SIM-7). With the
// prefill KV hold or without it, every request completes.
func TestPD_LandedDecodeIsSteppedWhenAHandoffArrivesAtItsWake(t *testing.T) {
	for _, hold := range []bool{false, true} {
		for _, transfer := range []int64{338, 640} {
			cs := newTestCluster(holdConfig(hold, transfer), NewSliceRequestSource(wakeCaseRequests()), nil)
			if err := cs.Run(); err != nil {
				t.Fatalf("hold %v, transfer %d µs: %v", hold, transfer, err)
			}
			if got := cs.AggregatedMetrics().CompletedRequests; got != 7 {
				t.Errorf("hold %v, transfer %d µs: completed %d of 7", hold, transfer, got)
			}
		}
	}
}

// A run whose held work the engine defers past the horizon ends at the horizon (#2685). Here the
// decode engine, holding two landed requests, next asks to be woken 1 µs after the horizon; the
// driver schedules no wake past it, so the run drains early, and that used to be reported as a
// stall (SIM-7). The horizon ends it: the two requests are queued at the horizon, and every
// request is accounted for (INV-1).
func TestPD_WorkDeferredPastTheHorizonEndsAtTheHorizon(t *testing.T) {
	cfg := holdConfig(true, 1)
	editPrefill(&cfg, func(c *simConfig) { c.TotalKVBlocks = 13 })
	cfg.Horizon = 18_979
	reqs := []*sim.Request{
		holdTestRequest("a", 0, 14_182, 138, 1),
		holdTestRequest("b", 1, 14_182, 1, 1),
		holdTestRequest("c", 2, 14_182, 1, 1),
	}
	cs := newTestCluster(cfg, NewSliceRequestSource(reqs), nil)
	if err := cs.Run(); err != nil {
		t.Fatalf("the run must end at its horizon, not stall: %v", err)
	}
	if !cs.Instances()[1].WakesPastHorizon() {
		t.Fatal("premise: the decode engine must next ask to be woken past the horizon")
	}
	if cs.Clock() != cfg.Horizon {
		t.Errorf("clock %d, want the horizon %d", cs.Clock(), cfg.Horizon)
	}
	for _, id := range []string{"b", "c"} {
		f, ok := cs.fates.Fate(id)
		if !ok || f.Fate != sim.FateQueuedAtHorizon || f.At != cfg.Horizon {
			t.Errorf("%s: fate %+v (recorded %v), want queued at the horizon %d", id, f, ok, cfg.Horizon)
		}
	}
	if l := newClusterLedger(cs); l.accounted() != len(reqs) {
		t.Errorf("INV-1: injected %d != accounted %d [%s]", len(reqs), l.accounted(), l)
	}
}
