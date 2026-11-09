package sim

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
)

// HeadIsNoop is true exactly when the next event is one ProcessNextEvent drops without running
// (#2072): a live request's timeout is work, and the same timeout is a no-op once the request
// completed.
func TestEngineDriver_HeadIsNoopMatchesDroppedEvents(t *testing.T) {
	d := newFIFODriver(t, enginetest.FIFOConfig{MaxRunning: 4, TokenBudget: 64, StepBaseUs: 100, StepPerTokenUs: 1}, 1<<40)
	req := &Request{ID: "r0", InputTokens: tokens(10), OutputTokens: tokens(4), Deadline: 1_000_000}
	d.InjectArrival(req)
	sawLiveTimeout, sawNoop := false, false
	for d.HasPendingEvents() {
		if d.PeekNextEventTime() == req.Deadline && req.State != StateCompleted {
			sawLiveTimeout = true
		}
		noop := d.HeadIsNoop()
		sawNoop = sawNoop || noop
		clock := d.Clock
		ev := d.ProcessNextEvent()
		if skipped := ev.(interface{ Skipped() bool }).Skipped(); skipped != noop {
			t.Fatalf("event at %d: HeadIsNoop = %v, but it was skipped = %v", ev.Timestamp(), noop, skipped)
		}
		if noop && d.Clock != clock {
			t.Fatalf("a no-op event at %d moved the clock from %d to %d", ev.Timestamp(), clock, d.Clock)
		}
	}
	if !sawNoop {
		t.Fatal("the completed request's timeout never reached the head as a no-op")
	}
	if sawLiveTimeout || req.State != StateCompleted || d.Metrics.TimedOutRequests != 0 {
		t.Fatalf("state %v, timed out %d: want the request completed and its timeout dropped", req.State, d.Metrics.TimedOutRequests)
	}

	// A deadline the request cannot meet: its timeout is live work, not a no-op.
	d = newFIFODriver(t, enginetest.FIFOConfig{MaxRunning: 4, TokenBudget: 64, StepBaseUs: 100, StepPerTokenUs: 1}, 1<<40)
	late := &Request{ID: "r1", InputTokens: tokens(10), OutputTokens: tokens(40), Deadline: 300}
	d.InjectArrival(late)
	for d.HasPendingEvents() {
		if d.PeekNextEventTime() == late.Deadline && late.State != StateCompleted && late.State != StateTimedOut && d.HeadIsNoop() {
			t.Fatal("a live request's timeout reads as a no-op")
		}
		d.ProcessNextEvent()
	}
	if late.State != StateTimedOut {
		t.Fatalf("state %v, want timed out", late.State)
	}
}
