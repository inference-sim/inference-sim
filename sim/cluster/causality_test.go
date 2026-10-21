package cluster

import "testing"

// TestINV5_Causality_DispatchChain asserts INV-5 on the cluster path:
//
//	arrival_time <= dispatch_time <= schedule_time <= completion_time
//
// Clock frames, all in ticks (microseconds) on the shared cluster clock:
//   - arrival:    req.ArrivalTime.
//   - dispatch:   req.GatewayDispatchTime, stamped by the cluster when the gateway
//     dispatches the request.
//   - schedule:   ArrivalTime + Metrics.RequestSchedulingDelays[id]; the stored
//     delay is `now - req.ArrivalTime` at batch formation, so it is arrival-relative.
//   - completion: Metrics.RequestCompletionTimes[id], absolute.
//
// Two legs: the gateway dispatches on arrival, or holds every request for a while
// first (as llm-d flow control may), so the dispatch link is non-trivial. The fake
// gateway never sets GatewayEnqueueTime, and the test must not read that absent
// timestamp as a zero that happens to satisfy a comparison.
func TestINV5_Causality_DispatchChain(t *testing.T) {
	for _, tc := range []struct {
		name   string
		holdUs int64
	}{
		{name: "dispatch on arrival", holdUs: 0},
		{name: "dispatch held 5ms", holdUs: 5000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assertINV5DispatchChain(t, tc.holdUs)
		})
	}
}

func assertINV5DispatchChain(t *testing.T, holdUs int64) {
	t.Helper()
	config := newTestDeploymentConfig(1)
	requests := newTestRequests(5)
	for i, req := range requests {
		req.ArrivalTime = int64(i)*100 + 100
	}

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
	cs.SetGateway(&fakeGateway{holdUs: holdUs})
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	checked := 0
	for _, req := range requests {
		if req.GatewayEnqueueTime != 0 {
			t.Errorf("request %s carries a gateway enqueue timestamp (%d) the fake gateway never sets",
				req.ID, req.GatewayEnqueueTime)
		}
		delay, hasDelay := m.RequestSchedulingDelays[req.ID]
		completedAt, hasCompletion := m.RequestCompletionTimes[req.ID]
		if !hasDelay || !hasCompletion {
			continue
		}
		checked++
		scheduledAt := float64(req.ArrivalTime) + float64(delay)
		if req.GatewayDispatchTime != req.ArrivalTime+holdUs {
			t.Errorf("request %s dispatched at %d, want arrival %d + hold %d",
				req.ID, req.GatewayDispatchTime, req.ArrivalTime, holdUs)
		}
		if req.ArrivalTime > req.GatewayDispatchTime {
			t.Errorf("INV-5 arrival->dispatch: request %s arrived at %d but was dispatched at %d",
				req.ID, req.ArrivalTime, req.GatewayDispatchTime)
		}
		if float64(req.GatewayDispatchTime) > scheduledAt {
			t.Errorf("INV-5 dispatch->schedule: request %s dispatched at %d but scheduled at %.0f (arrival=%d delay=%d)",
				req.ID, req.GatewayDispatchTime, scheduledAt, req.ArrivalTime, delay)
		}
		if scheduledAt > completedAt {
			t.Errorf("INV-5 schedule->completion: request %s scheduled at %.0f but completed at %.0f",
				req.ID, scheduledAt, completedAt)
		}
	}
	if checked != m.CompletedRequests {
		t.Errorf("checked %d requests but %d completed — a completed request is missing a scheduling or completion entry",
			checked, m.CompletedRequests)
	}
	if checked == 0 {
		t.Fatal("no request completed — the fixture proves nothing about causality")
	}
}
