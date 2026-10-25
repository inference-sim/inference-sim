package sim

import (
	"sort"
	"testing"
)

// timedDeferKV defers each request in readyAt until the clock reaches that tick, like
// a secondary-tier fetch whose transfer lands at a known completion time. It reports
// that tick through NextDeferralWake and records the clock at each first admission.
type timedDeferKV struct {
	*fakeDeferKV
	clock   int64
	readyAt map[string]int64
	waiting map[string]bool
	admitAt map[string]int64
}

func newTimedDeferKV(readyAt map[string]int64) *timedDeferKV {
	return &timedDeferKV{
		fakeDeferKV: newFakeDeferKV(),
		readyAt:     readyAt,
		waiting:     map[string]bool{},
		admitAt:     map[string]int64{},
	}
}

func (f *timedDeferKV) SetClock(c int64) { f.clock = c }

func (f *timedDeferKV) AllocateKVBlocks(req *Request, _, _ int64, _ []int64) bool {
	if t, ok := f.readyAt[req.ID]; ok && f.clock < t {
		f.waiting[req.ID] = true
		return false
	}
	delete(f.waiting, req.ID)
	if _, seen := f.admitAt[req.ID]; !seen {
		f.admitAt[req.ID] = f.clock
	}
	return true
}

func (f *timedDeferKV) PollDeferred(now int64) []string {
	f.polls++
	out := make([]string, 0, len(f.waiting))
	for id := range f.waiting {
		if now < f.readyAt[id] {
			out = append(out, id)
		}
	}
	sort.Strings(out)
	return out
}

func (f *timedDeferKV) IsDeferred(id string) bool { return f.waiting[id] && f.clock < f.readyAt[id] }

func (f *timedDeferKV) ClearDeferred(id string) { delete(f.waiting, id) }

func (f *timedDeferKV) NextDeferralWake(now int64) (int64, bool) {
	best, ok := int64(0), false
	for id := range f.waiting {
		if t := f.readyAt[id]; t > now && (!ok || t < best) {
			best, ok = t, true
		}
	}
	return best, ok
}

var _ DeferrableKVStore = (*timedDeferKV)(nil)

func runIdleWake(t *testing.T, kv *timedDeferKV, reqs ...*Request) *Simulator {
	t.Helper()
	s, err := NewSimulator(newTestSimConfig(), kv, &fixedOverheadModel{})
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}
	for _, r := range reqs {
		s.InjectArrival(r)
	}
	s.Run()
	return s
}

func arriving(id string, at int64) *Request {
	r := deferReq(id)
	r.ArrivalTime = at
	return r
}

// While every waiting request is deferred and nothing runs, the simulator must sleep
// until the next transfer completion, not step 1 tick at a time (an empty step costs
// 1 tick, so a 1 s fetch used to take ~1,000,000 steps). Admission still happens at
// exactly the completion tick, the same tick the 1-tick crawl reached.
func TestIdleWake_AllDeferredSkipsToCompletion(t *testing.T) {
	const ready = int64(1_000_000) // 1 s of simulated time
	kv := newTimedDeferKV(map[string]int64{"A": ready})
	s := runIdleWake(t, kv, arriving("A", 0))

	if got := s.Metrics.CompletedRequests; got != 1 {
		t.Fatalf("the deferred request must complete (INV-1, INV-8), completed=%d", got)
	}
	if got := kv.admitAt["A"]; got != ready {
		t.Fatalf("A must be admitted exactly at its completion tick %d, got %d", ready, got)
	}
	if s.StepCountForTest() > 100 {
		t.Fatalf("an idle wait must not cost a step per tick: %d steps for a %d-tick wait", s.StepCountForTest(), ready)
	}
}

// A request arriving while the simulator sleeps toward a far completion must be
// served at its arrival tick (work conservation, INV-8), not held until the wake;
// the superseded wake must not start a second step chain.
func TestIdleWake_ArrivalPreemptsWake(t *testing.T) {
	const ready = int64(1_000_000)
	kv := newTimedDeferKV(map[string]int64{"A": ready})
	s := runIdleWake(t, kv, arriving("A", 0), arriving("B", 500))

	if got := s.Metrics.CompletedRequests; got != 2 {
		t.Fatalf("both requests must complete (INV-1), completed=%d", got)
	}
	if got := kv.admitAt["B"]; got != 500 {
		t.Fatalf("B arrived at tick 500 during A's idle wait and must be admitted then, got %d", got)
	}
	if got := s.IdleWakesSuperseded(); got != 1 {
		t.Errorf("B's arrival must supersede exactly one pending wake, got %d", got)
	}
	if got := kv.admitAt["A"]; got != ready {
		t.Fatalf("A must still be admitted at its completion tick %d, got %d", ready, got)
	}
	if s.StepCountForTest() > 100 {
		t.Fatalf("idle waits around B must not crawl: %d steps", s.StepCountForTest())
	}
}

// TimeoutEvent is the one idle-scheduling site that keeps the pre-#1886
// `stepEvent == nil` guard instead of calling scheduleStepBy (sim/event.go explains why:
// a step at the timeout tick re-creates the #1096 same-deadline cascade). So a queued
// request timing out during an all-deferred wait leaves the pending wake where it is, and
// the question is whether that can push a sibling deferral's admission later than the
// pre-#1886 1-tick crawl would have.
//
// It cannot, and the law is stated as a comparison rather than a constant: the same run
// with and without a mid-wait timeout must admit the surviving request on the same tick.
// A times out at 500,000 while the wake sits at its own ready tick 1,000,000; that wake is
// now stale, costing one empty poll there, and B is still admitted at 2,000,000.
func TestIdleWake_TimeoutDuringWaitDoesNotDelayAdmission(t *testing.T) {
	const (
		readyA = int64(1_000_000)
		readyB = int64(2_000_000)
	)
	ready := map[string]int64{"A": readyA, "B": readyB}

	control := newTimedDeferKV(ready)
	runIdleWake(t, control, arriving("A", 0), arriving("B", 0))

	timeout := newTimedDeferKV(ready)
	a := arriving("A", 0)
	a.Deadline = 500_000 // mid-wait, before A's own fetch lands
	ts := runIdleWake(t, timeout, a, arriving("B", 0))

	if got := ts.Metrics.TimedOutRequests; got != 1 {
		t.Fatalf("A must time out during the wait, otherwise this tests nothing: timedOut=%d", got)
	}
	if got := timeout.admitAt["B"]; got != control.admitAt["B"] {
		t.Errorf("B was admitted at %d with A timing out mid-wait but %d without: a timeout during an "+
			"idle wait must not move admission (INV-8)", got, control.admitAt["B"])
	}
	if got := timeout.admitAt["B"]; got != readyB {
		t.Errorf("B must still be admitted at its own completion tick %d, got %d", readyB, got)
	}
	if got := ts.Metrics.CompletedRequests; got != 1 {
		t.Errorf("B must complete and A must not (INV-1): completed=%d", got)
	}
	// The stale wake is one empty poll, not a crawl back to the 1-tick cadence.
	if ts.StepCountForTest() > 100 {
		t.Errorf("a timeout during the wait must not restart per-tick stepping: %d steps", ts.StepCountForTest())
	}
}

// A store with deferrals but no known wake (NextDeferralWake ok=false) keeps the
// original 1-tick re-poll, so stores that do not report wakes are unchanged.
func TestIdleWake_NoWakeKeepsRepoll(t *testing.T) {
	kv := newFakeDeferKV()
	kv.pending["A"] = true
	s, err := NewSimulator(newTestSimConfig(), kv, &fixedOverheadModel{})
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}
	s.InjectArrival(arriving("A", 0))
	s.Horizon = 50
	s.Run()
	if kv.polls < 40 {
		t.Fatalf("without a reported wake the deferred request must be re-polled every tick, polls=%d", kv.polls)
	}
}

// newIdleWakeSim builds a Simulator whose WaitQ holds ids, with every id in deferred
// marked as waiting on a fetch that lands at readyAt[id]. It bypasses arrival/admission
// so a test can put the scheduler in an exact queue state and inspect one decision.
func newIdleWakeSim(t *testing.T, readyAt map[string]int64, deferred []string, ids ...string) (*Simulator, *timedDeferKV) {
	t.Helper()
	kv := newTimedDeferKV(readyAt)
	s, err := NewSimulator(newTestSimConfig(), kv, &fixedOverheadModel{})
	if err != nil {
		t.Fatalf("NewSimulator: %v", err)
	}
	for _, id := range ids {
		s.WaitQ.Enqueue(deferReq(id))
	}
	for _, id := range deferred {
		kv.waiting[id] = true
	}
	return s, kv
}

// A single NOT-deferred request in the WaitQ must veto the sleep entirely: an empty step
// would admit it, so skipping ahead would strand runnable work (INV-8). idleDeferralWake
// enforces this by requiring EVERY waiting request to be deferred, and the position of the
// odd one out must not matter.
//
// The decision is driven directly rather than through Run() because the state is
// defense-in-depth, not reachable from a whole run: for a step to leave the batch empty
// while a non-deferred request is still queued, that request has to be blocked by capacity
// — and with nothing running there is nothing to free it, so no choice of sleep length
// changes the outcome. An end-to-end variant of this test passes even with the guard
// weakened to "any deferred", which is what makes it worth saying so here.
func TestIdleWake_MixedQueueDoesNotSleep(t *testing.T) {
	ready := map[string]int64{"A": 1000, "B": 1000}

	for _, tc := range []struct {
		name  string
		order []string
	}{
		{"non-deferred last", []string{"A", "B"}},
		{"non-deferred first", []string{"B", "A"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Only A is deferred; B is runnable.
			s, _ := newIdleWakeSim(t, ready, []string{"A"}, tc.order...)
			if wake, ok := s.IdleDeferralWake(0); ok {
				t.Fatalf("a runnable request is waiting: must not sleep, got wake=(%d, %v)", wake, ok)
			}
		})
	}

	// Contrast, so the cases above are not vacuous: once every waiting request is
	// deferred, the same queue does report a wake.
	s, _ := newIdleWakeSim(t, ready, []string{"A", "B"}, "A", "B")
	if wake, ok := s.IdleDeferralWake(0); !ok || wake != 1000 {
		t.Fatalf("all-deferred queue must report its earliest wake: got (%d, %v), want (1000, true)", wake, ok)
	}
}

// The wake only ever moves the next step LATER, never earlier: scheduleNextStep takes it
// just when wake > now+currStepAdvance. A wake at or before the normal next step must
// leave both the tick and the idleWake flag alone — the flag is what makes a step
// eligible for supersession by scheduleStepBy, so mislabelling an ordinary step as an
// idle wake would let a later arrival cancel a step that was already due.
func TestIdleWake_WakeNotLaterThanNextStepIsNotIdle(t *testing.T) {
	const now = int64(0)

	for _, tc := range []struct {
		name     string
		advance  int64
		readyAt  int64
		wantTime int64
		wantIdle bool
	}{
		{"wake far in the future: sleep to it", 1, 1_000_000, 1_000_000, true},
		{"wake exactly at the next step: no sleep", 1, now + 1, now + 1, false},
		{"wake before the next step: no sleep", 1000, now + 1, now + 1000, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, _ := newIdleWakeSim(t, map[string]int64{"A": tc.readyAt}, []string{"A"}, "A")

			// Empty batch (remaining == nil) with pending WaitQ work: the idle-wake branch.
			s.scheduleNextStep(now, tc.advance, nil)

			step, ok := s.stepEvent.(*StepEvent)
			if !ok {
				t.Fatalf("a step must stay pending while a request waits (INV-8), got %T", s.stepEvent)
			}
			if step.time != tc.wantTime {
				t.Errorf("step time = %d, want %d", step.time, tc.wantTime)
			}
			if step.idleWake != tc.wantIdle {
				t.Errorf("idleWake = %v, want %v", step.idleWake, tc.wantIdle)
			}
		})
	}
}

// Same inputs give the same admissions and step count (INV-6).
func TestIdleWake_Deterministic(t *testing.T) {
	run := func() (int, int64, int64) {
		kv := newTimedDeferKV(map[string]int64{"A": 750_000, "C": 1_200_000})
		s := runIdleWake(t, kv, arriving("A", 0), arriving("B", 500), arriving("C", 600))
		return s.StepCountForTest(), kv.admitAt["A"], kv.admitAt["C"]
	}
	s1, a1, c1 := run()
	s2, a2, c2 := run()
	if s1 != s2 || a1 != a2 || c1 != c2 {
		t.Fatalf("idle wakes must be deterministic: steps %d/%d, A %d/%d, C %d/%d", s1, s2, a1, a2, c1, c2)
	}
}
