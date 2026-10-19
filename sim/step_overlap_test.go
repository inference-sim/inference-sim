package sim

import (
	"fmt"
	"testing"
)

// stepSpan is one executed engine step: [start, end).
type stepSpan struct{ start, end int64 }

// runAndRecordSteps drives s one event at a time and records every StepEvent that
// executed a batch (stepCount advanced), pricing each at the fixed step time.
func runAndRecordSteps(s *Simulator, stepTicks int64) []stepSpan {
	var spans []stepSpan
	for s.HasPendingEvents() {
		before := s.stepCount
		ev := s.ProcessNextEvent()
		if _, ok := ev.(*StepEvent); ok && s.stepCount > before {
			spans = append(spans, stepSpan{start: ev.Timestamp(), end: ev.Timestamp() + stepTicks})
		}
	}
	return spans
}

// TestStep_NoOverlap_RequestArrivingDuringDrainingStep verifies the engine runs one
// step at a time (issue #1964; INV-5 causality):
//
// GIVEN an engine whose every step drains it (each request has one output token, so
// the step that computes its prefill also completes it and leaves the engine idle)
// WHEN later requests arrive at various offsets relative to that draining step
// THEN no step starts before the previous step ends (steps never overlap in
// simulated time), AND a request's first token is produced no earlier than
// max(arrival, end of the step executing when it arrived) + one step.
func TestStep_NoOverlap_RequestArrivingDuringDrainingStep(t *testing.T) {
	const stepTicks = int64(1000)
	cases := []struct {
		name     string
		interval int64 // arrival spacing
	}{
		{"arrival mid-step", 700},
		{"arrival one tick after step start", 1},
		{"arrival one tick before step end", stepTicks - 1},
		{"arrival exactly at step end", stepTicks},
		{"arrival after engine idle", 2 * stepTicks},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := newTestSimConfig()
			cfg.LatencyModel = &fixedStepModel{stepTime: stepTicks}
			s := mustNewSimulator(t, cfg)
			const n = 6
			for i := 0; i < n; i++ {
				s.InjectArrival(&Request{
					ID:           fmt.Sprintf("req-%d", i),
					ArrivalTime:  int64(i) * tc.interval,
					InputTokens:  make([]TokenID, 10),
					OutputTokens: make([]TokenID, 1),
					State:        StateQueued,
				})
			}
			spans := runAndRecordSteps(s, stepTicks)

			if s.Metrics.CompletedRequests != n {
				t.Fatalf("CompletedRequests = %d, want %d", s.Metrics.CompletedRequests, n)
			}
			if len(spans) < 2 {
				t.Fatalf("non-vacuity: only %d executed steps recorded", len(spans))
			}

			// Law 1: one step at a time.
			for k := 1; k < len(spans); k++ {
				if spans[k].start < spans[k-1].end {
					t.Errorf("step overlap: step %d starts at %d, before step %d ends at %d",
						k, spans[k].start, k-1, spans[k-1].end)
				}
			}

			// Law 2 (INV-5): a request cannot be served by a step that began before
			// the engine finished the step it arrived during.
			for k := 0; k < n; k++ {
				id := fmt.Sprintf("req-%d", k)
				arrival := int64(k) * tc.interval
				var busyUntil int64
				for _, sp := range spans {
					if sp.start < arrival && sp.end > busyUntil {
						busyUntil = sp.end
					}
				}
				firstToken := arrival + int64(s.Metrics.RequestTTFTs[id])
				if want := max(arrival, busyUntil) + stepTicks; firstToken < want {
					t.Errorf("%s: first token at %d, want >= %d (arrival %d, engine busy until %d)",
						id, firstToken, want, arrival, busyUntil)
				}
			}
		})
	}
}
