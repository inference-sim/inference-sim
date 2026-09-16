package cluster

import "testing"

// TestINV5_Causality_FullChain asserts INV-5 as stated:
//
//	arrival_time <= enqueue_time <= schedule_time <= completion_time
//
// Every existing test that cites INV-5 asserts `TTFT >= 0` and `E2E >= TTFT`
// instead. Both of those are measured from arrival, so together they only say the
// last token follows the first — the enqueue and schedule links of the chain were
// asserted by nothing (issue #1720, Finding 2).
//
// Clock frames, all in ticks (microseconds) on the shared cluster clock:
//   - arrival:    req.ArrivalTime, never re-stamped as a request moves between
//     instances.
//   - enqueue:    req.GatewayEnqueueTime, assigned from the router state clock in
//     FlowControlAdmission.Admit.
//   - dispatch:   req.GatewayDispatchTime, the instant the gateway released the
//     request. Asserted explicitly rather than folded into the
//     instance-side delay, so the gateway link is measured, not inferred.
//   - schedule:   ArrivalTime + Metrics.RequestSchedulingDelays[id]; the stored
//     delay is `now - req.ArrivalTime` at batch formation, so it is
//     arrival-relative.
//   - completion: Metrics.RequestCompletionTimes[id], absolute.
func TestINV5_Causality_FullChain(t *testing.T) {
	config := newTestDeploymentConfig(1)
	config.FlowControlEnabled = true
	config.FlowControlDetector = "concurrency"
	config.FlowControlDispatchOrder = "fifo"
	config.FlowControlMaxConcurrency = 1 // force real gateway queueing

	// Arrivals start at a non-zero tick and are staggered. GatewayEnqueueTime is
	// taken from the clock and is *reset to 0* for rejected and shed requests, so a
	// request admitted at tick 0 would be indistinguishable from one that never
	// reached the gateway. Starting at 100 keeps "0" meaning "no gateway timestamp".
	requests := newTestRequests(8)
	for i, req := range requests {
		req.ArrivalTime = int64(i)*100 + 100
	}

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()

	// Per-link counters. A request legitimately absent from a link (no gateway
	// timestamp, or no scheduling/completion entry) is skipped rather than compared
	// against a zero value, so each link needs its own non-vacuity floor below.
	var enqueueChecked, dispatchChecked, scheduleChecked, completionChecked int

	for _, req := range requests {
		// arrival -> enqueue
		if req.GatewayEnqueueTime != 0 {
			enqueueChecked++
			if req.ArrivalTime > req.GatewayEnqueueTime {
				t.Errorf("INV-5 arrival->enqueue: request %s arrived at %d but was enqueued at %d",
					req.ID, req.ArrivalTime, req.GatewayEnqueueTime)
			}
		}

		// enqueue -> dispatch
		if req.GatewayEnqueueTime != 0 && req.GatewayDispatchTime != 0 {
			dispatchChecked++
			if req.GatewayEnqueueTime > req.GatewayDispatchTime {
				t.Errorf("INV-5 enqueue->dispatch: request %s enqueued at %d but dispatched at %d",
					req.ID, req.GatewayEnqueueTime, req.GatewayDispatchTime)
			}
		}

		// dispatch -> schedule. Two-value read: the scheduling-delay key is deleted
		// and only conditionally re-added for PD parent requests, so a missing key is
		// a real state and a bare index would silently read 0 and invert the test.
		delay, hasDelay := m.RequestSchedulingDelays[req.ID]
		scheduledAt := float64(req.ArrivalTime) + float64(delay)
		if hasDelay && req.GatewayDispatchTime != 0 {
			scheduleChecked++
			if float64(req.GatewayDispatchTime) > scheduledAt {
				t.Errorf("INV-5 dispatch->schedule: request %s dispatched at %d but scheduled at %.0f (arrival=%d delay=%d)",
					req.ID, req.GatewayDispatchTime, scheduledAt, req.ArrivalTime, delay)
			}
		}

		// schedule -> completion. Compared in float64 because completion times are
		// float ticks; truncating to int64 can turn an exact-equality case into a
		// spurious failure.
		completedAt, hasCompletion := m.RequestCompletionTimes[req.ID]
		if hasDelay && hasCompletion {
			completionChecked++
			if scheduledAt > completedAt {
				t.Errorf("INV-5 schedule->completion: request %s scheduled at %.0f but completed at %.0f",
					req.ID, scheduledAt, completedAt)
			}
		}
	}

	// Non-vacuity, per link. A single global counter would be wrong: with
	// concurrency capped, requests left in the queue at the horizon and TTL-expired
	// ones never get a dispatch stamp, and anything not completed has no completion
	// entry. Each floor subtracts exactly the buckets its link cannot reach.
	shed := cs.GatewayQueueShed()
	rejected := cs.GatewayQueueRejected()
	depth := cs.GatewayQueueDepth()
	expired := cs.GatewayExpired()

	if want := len(requests) - rejected; enqueueChecked != want {
		t.Errorf("INV-5 arrival->enqueue was checked for %d requests, want %d — the fixture stopped exercising the gateway, so this link proves nothing",
			enqueueChecked, want)
	}
	if want := len(requests) - shed - rejected - depth - expired; dispatchChecked != want {
		t.Errorf("INV-5 enqueue->dispatch was checked for %d requests, want %d (shed=%d rejected=%d stillQueued=%d expired=%d)",
			dispatchChecked, want, shed, rejected, depth, expired)
	}
	if completionChecked != m.CompletedRequests {
		t.Errorf("INV-5 schedule->completion was checked for %d requests but %d completed — a completed request is missing a scheduling or completion entry",
			completionChecked, m.CompletedRequests)
	}
	if scheduleChecked == 0 {
		t.Error("INV-5 dispatch->schedule was never checked — no request had both a dispatch timestamp and a scheduling delay")
	}

	// The gateway must actually have held something, or the two middle links reduce
	// to the trivial pass-through case.
	held := false
	for _, req := range requests {
		if req.GatewayDispatchTime > req.GatewayEnqueueTime && req.GatewayEnqueueTime != 0 {
			held = true
			break
		}
	}
	if !held {
		t.Error("no request waited in the gateway queue — with FlowControlMaxConcurrency=1 at least one should have, so the enqueue->dispatch link is vacuous")
	}
}

// TestINV5_Causality_SingleInstancePath covers the path the test above cannot: a
// cluster with no gateway queue, where GatewayEnqueueTime is unset for every
// request. The chain then reduces to arrival <= schedule <= completion, which must
// still hold — and the test must not read the absent enqueue timestamp as a zero
// that happens to satisfy the comparison.
func TestINV5_Causality_SingleInstancePath(t *testing.T) {
	config := newTestDeploymentConfig(1)
	requests := newTestRequests(5)
	for i, req := range requests {
		req.ArrivalTime = int64(i)*100 + 100
	}

	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	checked := 0
	for _, req := range requests {
		if req.GatewayEnqueueTime != 0 {
			t.Errorf("request %s carries a gateway enqueue timestamp (%d) with flow control disabled",
				req.ID, req.GatewayEnqueueTime)
		}
		delay, hasDelay := m.RequestSchedulingDelays[req.ID]
		completedAt, hasCompletion := m.RequestCompletionTimes[req.ID]
		if !hasDelay || !hasCompletion {
			continue
		}
		checked++
		scheduledAt := float64(req.ArrivalTime) + float64(delay)
		if delay < 0 {
			t.Errorf("INV-5 arrival->schedule: request %s has a negative scheduling delay (%d)", req.ID, delay)
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
