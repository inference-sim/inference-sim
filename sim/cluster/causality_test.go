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
	// Two fixtures. The first drains cleanly, so every subtrahend in the
	// non-vacuity floors below is zero. The second is deliberately saturated with a
	// bounded queue and a short horizon, so requests are shed and left queued at the
	// horizon — that is what makes the floors' arithmetic load-bearing rather than
	// coincidentally equal to len(requests).
	for _, tc := range []struct {
		name      string
		configure func(*DeploymentConfig)
		requests  int
		// Each leg declares what it must exercise, so a config change that quietly
		// turns a leg into a duplicate of another one fails rather than passing.
		requireHeld     bool // a request waited in the queue and was then dispatched
		requireRejected bool // the bounded queue turned requests away
		requireExpired  bool // TTL removed requests from the queue
	}{
		{
			name: "concurrency-gated, drains",
			configure: func(cfg *DeploymentConfig) {
				cfg.FlowControlMaxConcurrency = 1
			},
			requests:    8,
			requireHeld: true,
		},
		{
			// Measured on this fixture: 20 gateway-queue rejections and 2 requests
			// still queued at the horizon, so both of those subtrahends are non-zero
			// and the floors below are genuinely arithmetic rather than
			// coincidentally equal to len(requests).
			name: "saturated, rejects and leaves requests queued",
			configure: func(cfg *DeploymentConfig) {
				cfg.FlowControlMaxConcurrency = 1
				cfg.FlowControlDispatchOrder = "priority"
				cfg.FlowControlMaxQueueDepth = 3
				cfg.Horizon = 200_000
			},
			requests:        24,
			requireRejected: true,
		},
		{
			// Adds TTL expiry on top, so the expired subtrahend is exercised too. shed
			// stays zero in all three legs: shedding needs a displaceable
			// lower-priority victim, which no integration fixture in this package
			// reliably produces — it is covered at unit level in
			// TestGatewayQueue_CriticalityProtection_NonSheddableNeverEvicted. The
			// subtrahend is kept because a request shed from the queue does have its
			// enqueue timestamp cleared, so omitting it would be wrong the day such a
			// fixture exists.
			name: "saturated with TTL expiry",
			configure: func(cfg *DeploymentConfig) {
				cfg.FlowControlMaxConcurrency = 1
				cfg.FlowControlDispatchOrder = "priority"
				cfg.FlowControlMaxQueueDepth = 3
				cfg.FlowControlRequestTTL = 2000
				cfg.Horizon = 200_000
			},
			requests:       24,
			requireExpired: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			assertINV5FullChain(t, tc.configure, tc.requests, tc.requireHeld, tc.requireRejected, tc.requireExpired)
		})
	}
}

func assertINV5FullChain(t *testing.T, configure func(*DeploymentConfig), numRequests int, requireHeld, requireRejected, requireExpired bool) {
	t.Helper()

	config := newTestDeploymentConfig(1)
	config.FlowControlEnabled = true
	config.FlowControlDetector = "concurrency"
	config.FlowControlDispatchOrder = "fifo"
	configure(&config)

	// Arrivals start at a non-zero tick and are staggered. GatewayEnqueueTime is
	// taken from the clock and is *reset to 0* for rejected and shed requests, so a
	// request admitted at tick 0 would be indistinguishable from one that never
	// reached the gateway. Starting at 100 keeps "0" meaning "no gateway timestamp".
	requests := newTestRequests(numRequests)
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

	// A shed victim also has its enqueue timestamp reset to 0 (flow_control_admission.go),
	// so it is skipped by the loop above and must be subtracted here too — not only
	// rejected requests.
	if want := len(requests) - rejected - shed; enqueueChecked != want {
		t.Errorf("INV-5 arrival->enqueue was checked for %d requests, want %d (rejected=%d shed=%d) — the fixture stopped exercising the gateway, so this link proves nothing",
			enqueueChecked, want, rejected, shed)
	}
	if want := len(requests) - shed - rejected - depth - expired; dispatchChecked != want {
		t.Errorf("INV-5 enqueue->dispatch was checked for %d requests, want %d (shed=%d rejected=%d stillQueued=%d expired=%d)",
			dispatchChecked, want, shed, rejected, depth, expired)
	}
	// Without this, the two completion-side floors below are satisfied by zero: a config
	// change that stopped any request completing would leave dispatch->schedule and
	// schedule->completion entirely unchecked while the test still passed.
	if m.CompletedRequests == 0 {
		t.Error("no request completed — the dispatch->schedule and schedule->completion links prove nothing in this leg")
	}
	if completionChecked != m.CompletedRequests {
		t.Errorf("INV-5 schedule->completion was checked for %d requests but %d completed — a completed request is missing a scheduling or completion entry",
			completionChecked, m.CompletedRequests)
	}
	// Every completed request was necessarily dispatched and then scheduled, so this
	// link must cover at least the completed set. An exact count is not derivable — a
	// request can be scheduled and still be running at the horizon — but this is
	// stronger than "> 0", which would pass while checking 1 of 24.
	if scheduleChecked < completionChecked {
		t.Errorf("INV-5 dispatch->schedule was checked for %d requests but %d completed, so a completed request had no dispatch timestamp or no scheduling delay",
			scheduleChecked, completionChecked)
	}
	if scheduleChecked == 0 {
		t.Error("INV-5 dispatch->schedule was never checked — no request had both a dispatch timestamp and a scheduling delay")
	}

	// Per-leg expectations. Each subtrahend above is only meaningful if some leg
	// makes it non-zero; asserting that here means a config change cannot quietly
	// reduce every leg to the same trivial case.
	if requireHeld {
		held := false
		for _, req := range requests {
			if req.GatewayDispatchTime > req.GatewayEnqueueTime && req.GatewayEnqueueTime != 0 {
				held = true
				break
			}
		}
		if !held {
			t.Error("no request waited in the gateway queue and was then dispatched, so the enqueue->dispatch link is vacuous in this leg")
		}
	}
	if requireRejected && rejected == 0 {
		t.Error("no request was turned away by the bounded queue, so the rejected subtrahend in the floors above is untested")
	}
	if requireExpired && expired == 0 {
		t.Error("no request expired, so the expired subtrahend in the floors above is untested")
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
