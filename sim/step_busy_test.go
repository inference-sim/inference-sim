package sim

import (
	"math"
	"testing"

	"pgregory.net/rapid"
)

// The engine runs one forward pass at a time. A step whose requests all finish empties the
// running batch at the step's START (processCompletions runs inside Step), so before the #1944
// fix an arrival during that step started the next step at the arrival, overlapping the step
// still executing: every P/D prefill sub-request finishes in its one prefill step, so a
// prefill pool ran its steps in parallel and its throughput was unbounded. Property: for
// requests that each finish in one step, no request is scheduled before the previous one
// departs.
func TestStep_NoStepStartsBeforeThePreviousEnds(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		n := rapid.IntRange(2, 12).Draw(rt, "n")
		cfg := SimConfig{
			Horizon:             math.MaxInt64,
			KVCacheConfig:       NewKVCacheConfig(100000, 16, 0, 0, 0, 0),
			BatchConfig:         NewBatchConfig(1, 100000, 0), // one request per step
			ModelHardwareConfig: NewModelHardwareConfig(testModelConfig(), "m", "H100", 1, 1, false, 0),
		}
		s, err := NewSimulator(cfg, MustNewKVStoreFromConfig(cfg.KVCacheConfig), fakeLatencyFor(cfg))
		if err != nil {
			rt.Fatal(err)
		}
		reqs := make([]*Request, n)
		var at int64
		for i := range reqs {
			at += rapid.Int64Range(0, 3000).Draw(rt, "gap")
			reqs[i] = &Request{ID: string(rune('a' + i)), ArrivalTime: at, State: StateQueued,
				InputTokens: make([]TokenID, rapid.IntRange(1, 4000).Draw(rt, "in")), OutputTokens: make([]TokenID, 1), MaxOutputLen: 1}
			for j := range reqs[i].InputTokens {
				reqs[i].InputTokens[j] = TokenID(i*10_000 + j + 1)
			}
		}
		injectRequests(s, reqs)
		s.Run()
		var prevDeparture int64
		for _, r := range reqs {
			if r.State != StateCompleted {
				rt.Fatalf("%s did not complete", r.ID)
			}
			scheduled := r.ArrivalTime + s.Metrics.RequestSchedulingDelays[r.ID]
			if scheduled < prevDeparture {
				rt.Fatalf("%s scheduled at %d, before the previous step ended at %d", r.ID, scheduled, prevDeparture)
			}
			prevDeparture = r.DepartureTime
		}
	})
}

// drainedSimulator runs one request through a step that empties the batch and returns the
// simulator with that step's end (busyUntil) still ahead of the time the site tests use.
func drainedSimulator(t *testing.T) (*Simulator, int64) {
	t.Helper()
	cfg := SimConfig{
		Horizon:             math.MaxInt64,
		KVCacheConfig:       NewKVCacheConfig(100000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(8, 100000, 0),
		ModelHardwareConfig: NewModelHardwareConfig(testModelConfig(), "m", "H100", 1, 1, false, 0),
	}
	s, err := NewSimulator(cfg, MustNewKVStoreFromConfig(cfg.KVCacheConfig), fakeLatencyFor(cfg))
	if err != nil {
		t.Fatal(err)
	}
	a := &Request{ID: "a", InputTokens: make([]TokenID, 2000), OutputTokens: make([]TokenID, 1), MaxOutputLen: 1, State: StateQueued}
	injectRequests(s, []*Request{a})
	// Stop right after the step that finishes a: the clock still reads that step's start.
	for a.State != StateCompleted {
		if !s.HasPendingEvents() {
			t.Fatal("fixture: a never completed")
		}
		s.ProcessNextEvent()
	}
	if s.stepEvent != nil || s.busyUntil != a.DepartureTime || s.CurrentClock() >= a.DepartureTime-1 {
		t.Fatalf("fixture: want an idle simulator mid-step (clock %d) after a draining step ending at %d",
			s.CurrentClock(), a.DepartureTime)
	}
	return s, a.DepartureTime
}

func stepRequest(id string) *Request {
	return &Request{ID: id, InputTokens: make([]TokenID, 16), OutputTokens: make([]TokenID, 2), MaxOutputLen: 2, State: StateQueued}
}

// Each of the four sites that start a step on an idle instance waits for the step still
// executing (#1944): new work at a time before that step ends is scheduled at its end.
func TestStep_EverySiteWaitsForTheExecutingStep(t *testing.T) {
	for _, site := range []struct {
		name string
		fire func(s *Simulator, at int64)
	}{
		{"arrival (QueuedEvent)", func(s *Simulator, at int64) {
			(&QueuedEvent{time: at, Request: stepRequest("b")}).Execute(s)
		}},
		{"timeout of a waiting request", func(s *Simulator, at int64) {
			c, d := stepRequest("c"), stepRequest("d")
			s.WaitQ.Enqueue(c)
			s.WaitQ.Enqueue(d)
			(&TimeoutEvent{time: at, Request: c}).Execute(s)
		}},
		{"ScheduleStepIfIdle", func(s *Simulator, at int64) {
			s.WaitQ.Enqueue(stepRequest("e"))
			s.ScheduleStepIfIdle(at)
		}},
		{"injected decode sub-request", func(s *Simulator, at int64) {
			r := stepRequest("f")
			r.IsDecodeSubRequest = true
			r.ProgressIndex = r.InputLen()
			s.EnqueueDecodeSubRequest(r, at)
		}},
	} {
		t.Run(site.name, func(t *testing.T) {
			s, busy := drainedSimulator(t)
			site.fire(s, busy-1)
			if s.stepEvent == nil {
				t.Fatal("no step was scheduled for the new work")
			}
			if got := s.stepEvent.Timestamp(); got != busy {
				t.Errorf("step scheduled at %d, before the executing step ends at %d", got, busy)
			}
		})
	}
}
