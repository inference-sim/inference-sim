package sim

import "testing"

func walk(t *testing.T, l *FateLedger, id string, states ...LifecycleState) {
	t.Helper()
	for _, s := range states {
		if err := l.Advance(id, s); err != nil {
			t.Fatalf("Advance(%s, %s): %v", id, s, err)
		}
	}
}

// SIM-8: a P/D request passes prefill, the wait for decode KV, the transfer and decoding in that
// order, a preemption returns it to the stage it left, and a move that skips a stage is refused
// and leaves the state as it was.
func TestLifecycle_PDPathAndPreemption(t *testing.T) {
	l := NewFateLedger()
	mustOK(t, l.Create("r"), l.Arrive("r"))
	walk(t, l, "r", LifecyclePrefillQueued, LifecyclePrefilling, LifecyclePreempted, LifecyclePrefilling,
		LifecycleWaitingDecodeKV, LifecycleTransferring, LifecycleDecodeQueued, LifecycleDecoding,
		LifecyclePreempted, LifecycleDecoding)
	if err := l.Advance("r", LifecyclePrefillQueued); err == nil {
		t.Error("decoding -> prefill_queued was accepted")
	}
	if got, _ := l.State("r"); got != LifecycleDecoding {
		t.Errorf("state after a refused move = %s, want decoding", got)
	}

	// A colocated request runs and is preempted; it never enters a P/D state.
	mustOK(t, l.Create("c"), l.Arrive("c"))
	walk(t, l, "c", LifecycleRunning, LifecyclePreempted, LifecycleRunning)
	for _, bad := range []LifecycleState{LifecycleTransferring, LifecycleWaitingDecodeKV, LifecycleDecodeQueued} {
		if err := l.Advance("c", bad); err == nil {
			t.Errorf("running -> %s was accepted", bad)
		}
	}
	// A created request cannot skip its arrival, and an ended request cannot move.
	mustOK(t, l.Create("n"))
	if err := l.Advance("n", LifecycleRunning); err == nil {
		t.Error("created -> running was accepted")
	}
	mustOK(t, l.End("r", FateCompleted, "", 5))
	if err := l.Advance("r", LifecycleDecoding); err == nil {
		t.Error("an ended request moved")
	}
}

// SIM-8: the horizon's fate is read from the state: running and moving stages are in flight,
// every waiting state is queued, and each held state may end in any fate but not started.
func TestLifecycle_HorizonFateIsReadFromTheState(t *testing.T) {
	want := map[LifecycleState]Fate{
		LifecycleArrived: FateQueuedAtHorizon, LifecycleRunning: FateInFlightAtHorizon,
		LifecyclePrefillQueued: FateQueuedAtHorizon, LifecyclePrefilling: FateInFlightAtHorizon,
		LifecycleWaitingDecodeKV: FateQueuedAtHorizon, LifecycleTransferring: FateInFlightAtHorizon,
		LifecycleDecodeQueued: FateQueuedAtHorizon, LifecycleDecoding: FateInFlightAtHorizon,
		LifecyclePreempted: FateQueuedAtHorizon,
	}
	for _, s := range LifecycleStates() {
		f, ok := HorizonFate(s)
		if !IsHeld(s) {
			if ok {
				t.Errorf("state %s is not held yet has horizon fate %s", s, f)
			}
			continue
		}
		if !ok || f != want[s] {
			t.Errorf("HorizonFate(%s) = %s,%v, want %s", s, f, ok, want[s])
		}
		for _, fate := range Fates() {
			if got, exp := CanEnd(s, fate), fate != FateNotStarted; got != exp {
				t.Errorf("CanEnd(%s, %s) = %v, want %v", s, fate, got, exp)
			}
		}
	}
}
