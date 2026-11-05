package sim

import (
	"errors"
	"strings"
	"testing"
)

// The state machine's edge table (SIM-8): a created request may only end not started; an
// arrived request may end in any other fate; an ended request has no edge.
func TestFateLedger_TransitionTable(t *testing.T) {
	for _, s := range []LifecycleState{LifecycleCreated, LifecycleArrived, LifecycleEnded} {
		for _, f := range Fates() {
			want := (s == LifecycleCreated && f == FateNotStarted) ||
				(s == LifecycleArrived && f != FateNotStarted)
			if got := CanEnd(s, f); got != want {
				t.Errorf("CanEnd(%s, %s) = %v, want %v", s, f, got, want)
			}
		}
	}
	// Every edge the ledger accepts is in the table: drive each state to each fate.
	for _, f := range Fates() {
		for _, arrive := range []bool{false, true} {
			l := NewFateLedger()
			if err := l.Create("r"); err != nil {
				t.Fatal(err)
			}
			from := LifecycleCreated
			if arrive {
				if err := l.Arrive("r"); err != nil {
					t.Fatal(err)
				}
				from = LifecycleArrived
			}
			reason := ""
			if f == FateDropped {
				reason = DropRouting
			}
			err := l.End("r", f, reason, 7)
			if (err == nil) != CanEnd(from, f) {
				t.Errorf("End from %s to %s: err %v, table says %v", from, f, err, CanEnd(from, f))
			}
		}
	}
}

// DEV-9: a fate is written once. The second write fails and names both fates.
func TestFateLedger_SecondWriteFails(t *testing.T) {
	l := NewFateLedger()
	mustOK(t, l.Create("r"), l.Arrive("r"), l.End("r", FateCompleted, "", 10))
	err := l.End("r", FateTimedOut, "", 20)
	if err == nil || !strings.Contains(err.Error(), "written once") || !strings.Contains(err.Error(), "completed") {
		t.Fatalf("second End: err %v, want a written-once error naming the first fate", err)
	}
	if got, _ := l.Fate("r"); got.Fate != FateCompleted || got.At != 10 {
		t.Errorf("fate after the refused write = %+v, want the first", got)
	}
}

// SIM-8: a request identifier is never reused within a run.
func TestFateLedger_IDReuseFails(t *testing.T) {
	l := NewFateLedger()
	mustOK(t, l.Create("r"))
	if err := l.Create("r"); err == nil || !strings.Contains(err.Error(), "never reused") {
		t.Fatalf("second Create: err %v, want an identifier-reuse error", err)
	}
}

// A drop states its reason; no other fate has one.
func TestFateLedger_DropNeedsAReason(t *testing.T) {
	l := NewFateLedger()
	mustOK(t, l.Create("a"), l.Arrive("a"), l.Create("b"), l.Arrive("b"))
	if err := l.End("a", FateDropped, "", 1); err == nil {
		t.Error("a drop without a reason was accepted")
	}
	if err := l.End("b", FateCompleted, DropRouting, 1); err == nil {
		t.Error("a completion with a drop reason was accepted")
	}
}

// DEV-9's arrival-count check: created == fates written, and the check names a request
// without a fate (the known-bad input: a request the code forgot to end).
func TestFateLedger_CheckCountsCreatedAgainstFates(t *testing.T) {
	l := NewFateLedger()
	mustOK(t, l.Create("a"), l.Arrive("a"), l.End("a", FateCompleted, "", 1), l.Create("lost"), l.Arrive("lost"))
	err := l.Check()
	if err == nil || !strings.Contains(err.Error(), "lost") {
		t.Fatalf("Check: err %v, want a conservation error naming the request without a fate", err)
	}
	mustOK(t, l.End("lost", FateQueuedAtHorizon, "", 2))
	if err := l.Check(); err != nil {
		t.Fatalf("Check after every request has a fate: %v", err)
	}
	tally := l.Tally()
	if tally.Created != 2 || tally.Total() != 2 || tally.ByFate[FateCompleted] != 1 || tally.ByFate[FateQueuedAtHorizon] != 1 {
		t.Errorf("tally = %+v", tally)
	}
}

func TestFateLedger_UnknownRequest(t *testing.T) {
	l := NewFateLedger()
	if err := l.Arrive("x"); err == nil {
		t.Error("Arrive of a request never created was accepted")
	}
	if err := l.End("x", FateCompleted, "", 0); err == nil {
		t.Error("End of a request never created was accepted")
	}
}

func TestErrNoProgressIsWrapped(t *testing.T) {
	d := &EngineDriver{cfg: EngineDriverConfig{StallSteps: 2}, reqs: map[string]*driverReq{"r": {}}}
	for i := 0; i < 3; i++ {
		d.noteStep(int64(i))
	}
	if !errors.Is(d.StallErr(), ErrNoProgress) {
		t.Fatalf("StallErr after 3 stalled steps at limit 2 = %v, want ErrNoProgress", d.StallErr())
	}
	d2 := &EngineDriver{cfg: EngineDriverConfig{StallSteps: 2}}
	for i := 0; i < 3; i++ {
		d2.noteStep(int64(i))
		d2.noteProgress()
	}
	if d2.StallErr() != nil {
		t.Fatalf("StallErr with progress every step = %v, want nil", d2.StallErr())
	}
}

func mustOK(t *testing.T, errs ...error) {
	t.Helper()
	for _, err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
}
