package sim

import (
	"fmt"
	"math"
	"testing"

	"pgregory.net/rapid"
)

// lockstepRanks builds n independent simulators and binds them into one lockstep group.
func lockstepRanks(t interface {
	Helper()
	Fatalf(string, ...any)
}, n int) ([]*Simulator, *LockstepGroup) {
	t.Helper()
	ranks := make([]*Simulator, n)
	for i := range ranks {
		cfg := newTestSimConfig()
		s, err := NewSimulator(cfg, MustNewKVStoreFromConfig(cfg.KVCacheConfig), newFakeLatency())
		if err != nil {
			t.Fatalf("NewSimulator: %v", err)
		}
		ranks[i] = s
	}
	g, err := NewLockstepGroup(ranks)
	if err != nil {
		t.Fatalf("NewLockstepGroup: %v", err)
	}
	return ranks, g
}

// driveGroup runs the ranks' event queues in the cluster's order -- earliest time, ties to
// the lowest rank -- calling check after every event.
func driveGroup(ranks []*Simulator, check func()) {
	for {
		idx, tmin := -1, int64(math.MaxInt64)
		for i, r := range ranks {
			if r.HasPendingEvents() {
				if ts := r.PeekNextEventTime(); ts < tmin {
					idx, tmin = i, ts
				}
			}
		}
		if idx < 0 {
			return
		}
		ranks[idx].ProcessNextEvent()
		check()
	}
}

func lockReq(id string, arrival int64, in, out int) *Request {
	return &Request{ID: id, ArrivalTime: arrival, InputTokens: make([]TokenID, in),
		OutputTokens: make([]TokenID, out), State: StateQueued}
}

// INV-8 for a lockstep group: after every event, while any rank has waiting or running work
// the group's one StepEvent is armed and every rank's step guard points at it. A wave ends
// on a sync step (the first, or a multiple of DPSyncInterval) fewer than DPSyncInterval
// dummy steps after its last request ran, and every request completes (INV-1).
func TestLockstepGroup_WorkConservingAndWaveEnd(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		n := rapid.IntRange(2, 4).Draw(rt, "ranks")
		ranks, g := lockstepRanks(rt, n)
		reqs := rapid.IntRange(1, 12).Draw(rt, "requests")
		for i := 0; i < reqs; i++ {
			k := rapid.IntRange(0, n-1).Draw(rt, "rank")
			at := rapid.Int64Range(0, 200_000).Draw(rt, "arrival")
			in := rapid.IntRange(1, 3000).Draw(rt, "in")
			out := rapid.IntRange(1, 40).Draw(rt, "out")
			ranks[k].InjectArrival(lockReq(fmt.Sprintf("r%d", i), at, in, out))
		}
		waves := 0
		driveGroup(ranks, func() {
			busy := false
			for _, r := range ranks {
				busy = busy || hasWork(r)
			}
			if busy && g.pending == nil {
				rt.Fatalf("INV-8: a rank has work but no group step is armed")
			}
			var armed Event
			if g.pending != nil {
				armed = g.pending
			}
			for i, r := range ranks {
				if r.stepEvent != armed {
					rt.Fatalf("rank %d's step guard does not point at the group's step", i)
				}
			}
			if g.pending == nil && g.lastWave > 0 {
				waves++
				if g.lastWave != 1 && g.lastWave%DPSyncInterval != 0 {
					rt.Fatalf("a wave ended after %d steps, not on a sync step", g.lastWave)
				}
				if g.lastWaveIdle >= DPSyncInterval {
					rt.Fatalf("a wave ran %d dummy steps past its last request", g.lastWaveIdle)
				}
				g.lastWave = 0
			}
		})
		done := 0
		for _, r := range ranks {
			done += r.Metrics.CompletedRequests
		}
		if done != reqs || waves == 0 {
			rt.Fatalf("completed %d of %d requests over %d waves", done, reqs, waves)
		}
	})
}

// A request queued on any rank at the step's own tick joins that step: two identical
// requests arriving together on different ranks get identical TTFTs, whichever rank serves
// which (the group's step sits in rank 0's queue, which a time tie processes first).
func TestLockstepGroup_SameTickArrivalsJoinTheSameStep(t *testing.T) {
	ttfts := func(order [2]int) [2]float64 {
		ranks, _ := lockstepRanks(t, 2)
		ranks[order[0]].InjectArrival(lockReq("a", 0, 512, 8))
		ranks[order[1]].InjectArrival(lockReq("b", 0, 512, 8))
		driveGroup(ranks, func() {})
		return [2]float64{ranks[order[0]].Metrics.RequestTTFTs["a"], ranks[order[1]].Metrics.RequestTTFTs["b"]}
	}
	ab, ba := ttfts([2]int{0, 1}), ttfts([2]int{1, 0})
	if ab[0] != ab[1] || ab != ba {
		t.Errorf("identical same-tick requests got TTFTs %v (a on rank 0) and %v (a on rank 1); want all equal", ab, ba)
	}
}

// A rank that only ran dummy steps keeps the clock of its last real event, so the wave's
// trailing dummy steps do not stretch SimEndedTime.
func TestLockstepGroup_DummyStepsDoNotAdvanceAnIdleLeader(t *testing.T) {
	ranks, _ := lockstepRanks(t, 2)
	ranks[1].InjectArrival(lockReq("only", 0, 256, 16))
	driveGroup(ranks, func() {})
	for _, r := range ranks {
		r.Finalize()
	}
	if got := ranks[0].Metrics.SimEndedTime; got != 0 {
		t.Errorf("the leader served nothing but ended at %d; dummy steps advanced its clock", got)
	}
	if ranks[1].Metrics.SimEndedTime <= 0 {
		t.Error("the serving rank must end at its last real event")
	}
}
