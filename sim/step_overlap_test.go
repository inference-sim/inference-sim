package sim

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim/internal/testutil"
)

// stepSpan is one executed engine step: [start, end).
type stepSpan struct{ start, end int64 }

// recordingLatency wraps a LatencyModel and remembers the step time it last
// priced, so a test can reconstruct each step's span from the model's own output
// rather than from simulator internals.
type recordingLatency struct {
	LatencyModel
	last int64
}

func (r *recordingLatency) StepTime(batch []*Request) int64 {
	r.last = r.LatencyModel.StepTime(batch)
	return r.last
}

// runAndRecordSteps drives s one event at a time and records the span of every
// StepEvent that executed a batch (stepCount advanced). Single-tier KV adds no
// transfer latency, so a step's duration is max(1, the priced step time).
func runAndRecordSteps(s *Simulator, lat *recordingLatency) []stepSpan {
	var spans []stepSpan
	for s.HasPendingEvents() {
		before := s.stepCount
		lat.last = 0
		ev := s.ProcessNextEvent()
		if _, ok := ev.(*StepEvent); ok && s.stepCount > before {
			spans = append(spans, stepSpan{start: ev.Timestamp(), end: ev.Timestamp() + max(1, lat.last)})
		}
	}
	return spans
}

// TestStep_NoOverlap_RequestArrivingDuringDrainingStep verifies the engine runs one
// step at a time (issue #1964):
//
// GIVEN an engine whose steps drain it (the step that finishes the last running
// request leaves the engine idle while the step is still executing)
// WHEN later requests arrive at various offsets relative to that draining step
// THEN no step starts before the previous step ends (steps never overlap in
// simulated time), AND a request's first token is produced no earlier than
// max(arrival, end of the step executing when it arrived) + one step.
//
// The fixed-step cases pin the arrival-offset boundaries; the token-cost case
// varies step durations across consecutive steps (chunked prefill then decode,
// different prompt sizes) so the laws are checked against real step ends rather
// than an assumed constant.
func TestStep_NoOverlap_RequestArrivingDuringDrainingStep(t *testing.T) {
	const stepTicks = int64(1000)
	type reqShape struct{ in, out int }
	cases := []struct {
		name      string
		lat       LatencyModel
		threshold int64 // long-prefill (chunking) threshold; 0 = off
		interval  int64 // arrival spacing
		shapes    []reqShape
	}{
		{"fixed: arrival mid-step", &fixedStepModel{stepTime: stepTicks}, 0, 700, nil},
		{"fixed: arrival one tick after step start", &fixedStepModel{stepTime: stepTicks}, 0, 1, nil},
		{"fixed: arrival one tick before step end", &fixedStepModel{stepTime: stepTicks}, 0, stepTicks - 1, nil},
		{"fixed: arrival exactly at step end", &fixedStepModel{stepTime: stepTicks}, 0, stepTicks, nil},
		{"fixed: arrival after engine idle", &fixedStepModel{stepTime: stepTicks}, 0, 2 * stepTicks, nil},
		{
			"token-cost: variable steps, chunked prefill then decode",
			newFakeLatencyWith(testutil.FakeLatency{BaseTicks: 50, PerScheduledTokenTicks: 1}),
			64, 120,
			[]reqShape{{200, 3}, {30, 1}, {90, 2}, {10, 1}, {300, 4}, {50, 1}, {130, 2}, {20, 1}},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			shapes := tc.shapes
			if shapes == nil {
				// Each request needs exactly one step, so every step drains the engine.
				shapes = []reqShape{{10, 1}, {10, 1}, {10, 1}, {10, 1}, {10, 1}, {10, 1}}
			}
			cfg := newTestSimConfig()
			cfg.BatchConfig = NewBatchConfig(256, 2048, tc.threshold)
			lat := &recordingLatency{LatencyModel: tc.lat}
			cfg.LatencyModel = lat
			s := mustNewSimulator(t, cfg)
			for i, sh := range shapes {
				s.InjectArrival(&Request{
					ID:           fmt.Sprintf("req-%d", i),
					ArrivalTime:  int64(i) * tc.interval,
					InputTokens:  make([]TokenID, sh.in),
					OutputTokens: make([]TokenID, sh.out),
					State:        StateQueued,
				})
			}
			spans := runAndRecordSteps(s, lat)

			if s.Metrics.CompletedRequests != len(shapes) {
				t.Fatalf("CompletedRequests = %d, want %d", s.Metrics.CompletedRequests, len(shapes))
			}
			if len(spans) < 2 {
				t.Fatalf("non-vacuity: only %d executed steps recorded", len(spans))
			}
			if tc.shapes != nil {
				distinct := map[int64]bool{}
				for _, sp := range spans {
					distinct[sp.end-sp.start] = true
				}
				if len(distinct) < 3 {
					t.Fatalf("non-vacuity: only %d distinct step durations; the variable-duration case "+
						"no longer varies them", len(distinct))
				}
			}

			// Law 1: one step at a time.
			for k := 1; k < len(spans); k++ {
				if spans[k].start < spans[k-1].end {
					t.Errorf("step overlap: step %d starts at %d, before step %d ends at %d",
						k, spans[k].start, k-1, spans[k-1].end)
				}
			}

			// Law 2 (causality): a request cannot be served by a step that began
			// before the engine finished the step it arrived during.
			for k := range shapes {
				id := fmt.Sprintf("req-%d", k)
				arrival := int64(k) * tc.interval
				var busyUntil int64
				for _, sp := range spans {
					if sp.start < arrival && sp.end > busyUntil {
						busyUntil = sp.end
					}
				}
				start := max(arrival, busyUntil)
				firstStep := int64(-1)
				for _, sp := range spans {
					if sp.start >= start {
						firstStep = sp.end - sp.start
						break
					}
				}
				firstToken := arrival + int64(s.Metrics.RequestTTFTs[id])
				if firstStep < 0 || firstToken < start+firstStep {
					t.Errorf("%s: first token at %d, want >= %d (arrival %d, engine busy until %d)",
						id, firstToken, start+firstStep, arrival, busyUntil)
				}
			}
		})
	}
}

// TestScheduleStepIfIdle_StaleTime_NeverSchedulesInThePast verifies the Clock floor
// in nextStepStart (INV-3):
// GIVEN an idle engine whose clock has advanced past its last step's end
// WHEN ScheduleStepIfIdle is called with a time behind both the clock and busyUntil
// THEN the step is scheduled at the current clock, not in the past.
func TestScheduleStepIfIdle_StaleTime_NeverSchedulesInThePast(t *testing.T) {
	cfg := newTestSimConfig()
	cfg.LatencyModel = &fixedStepModel{stepTime: 100}
	s := mustNewSimulator(t, cfg)
	s.InjectArrival(&Request{ID: "a", ArrivalTime: 0, InputTokens: make([]TokenID, 10),
		OutputTokens: make([]TokenID, 1), State: StateQueued})
	// A late request too large for the KV cache is dropped at enqueue: it advances
	// the clock well past the last step's end (busyUntil) without running a step.
	s.InjectArrival(&Request{ID: "huge", ArrivalTime: 5000, InputTokens: make([]TokenID, 10000*16+1),
		OutputTokens: make([]TokenID, 1), State: StateQueued})
	s.Run()
	if s.busyUntil != 100 || s.Clock < 5000 {
		t.Fatalf("setup: want busyUntil 100 and clock >= 5000, got busyUntil %d clock %d", s.busyUntil, s.Clock)
	}
	clock := s.Clock
	s.WaitQ.Enqueue(&Request{ID: "b", ArrivalTime: clock, InputTokens: make([]TokenID, 10),
		OutputTokens: make([]TokenID, 1), State: StateQueued})
	s.ScheduleStepIfIdle(50)
	if !s.HasPendingEvents() {
		t.Fatal("ScheduleStepIfIdle scheduled no step for a non-empty wait queue")
	}
	if ev := s.ProcessNextEvent(); ev.Timestamp() < clock {
		t.Errorf("step scheduled at %d, behind the clock %d (INV-3)", ev.Timestamp(), clock)
	}
}
