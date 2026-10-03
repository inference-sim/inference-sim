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
	if s.stepCount > 100 {
		t.Fatalf("an idle wait must not cost a step per tick: %d steps for a %d-tick wait", s.stepCount, ready)
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
	if got := kv.admitAt["A"]; got != ready {
		t.Fatalf("A must still be admitted at its completion tick %d, got %d", ready, got)
	}
	if s.stepCount > 100 {
		t.Fatalf("idle waits around B must not crawl: %d steps", s.stepCount)
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

// Same inputs give the same admissions and step count (INV-6).
func TestIdleWake_Deterministic(t *testing.T) {
	run := func() (int, int64, int64) {
		kv := newTimedDeferKV(map[string]int64{"A": 750_000, "C": 1_200_000})
		s := runIdleWake(t, kv, arriving("A", 0), arriving("B", 500), arriving("C", 600))
		return s.stepCount, kv.admitAt["A"], kv.admitAt["C"]
	}
	s1, a1, c1 := run()
	s2, a2, c2 := run()
	if s1 != s2 || a1 != a2 || c1 != c2 {
		t.Fatalf("idle wakes must be deterministic: steps %d/%d, A %d/%d, C %d/%d", s1, s2, a1, a2, c1, c2)
	}
}
