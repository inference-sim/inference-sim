package cluster

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// newDeferredTestRequests creates n requests with the given SLOClass,
// arriving every 10µs starting at t=0, with 50 input tokens and 20 output tokens.
func newDeferredTestRequests(n int, sloClass string) []*sim.Request {
	reqs := make([]*sim.Request, n)
	for i := range reqs {
		reqs[i] = &sim.Request{
			ID:           fmt.Sprintf("req_%s_%d", sloClass, i),
			ArrivalTime:  int64(i) * 10,
			SLOClass:     sloClass,
			InputTokens:  make([]int, 50),
			OutputTokens: make([]int, 20),
			State:        sim.StateQueued,
		}
	}
	return reqs
}

// T002 — BC-D1: Batch and Background requests are deferred (not rejected) when cluster is busy.
// Table-driven over both SLO classes so that removing either branch from the intercept
// (cluster_event.go:127) would be caught.
// Uses a 1-instance cluster overloaded with standard requests arriving densely.
// After Run(), the deferred request must not be rejected AND must reach a terminal
// state (completed or deferred-at-horizon) — confirming deferral actually occurred.
func TestDeferredQueue_BatchDeferredWhenBusy(t *testing.T) {
	for _, sloClass := range []string{"batch", "background"} {
		t.Run(sloClass, func(t *testing.T) {
			// Create a mix: many standard requests (to keep cluster busy) + 1 deferred-tier request
			var requests []*sim.Request
			// Standard requests arriving very densely to ensure cluster stays busy
			for i := 0; i < 30; i++ {
				requests = append(requests, &sim.Request{
					ID:           fmt.Sprintf("std_%d", i),
					ArrivalTime:  int64(i) * 5,
					SLOClass:     "standard",
					InputTokens:  make([]int, 100),
					OutputTokens: make([]int, 50),
					State:        sim.StateQueued,
				})
			}
			// Deferred-tier request arrives early, when cluster is definitely busy
			deferredReq := &sim.Request{
				ID:           sloClass + "_0",
				ArrivalTime:  50, // arrives while cluster is saturated
				SLOClass:     sloClass,
				InputTokens:  make([]int, 50),
				OutputTokens: make([]int, 20),
				State:        sim.StateQueued,
			}
			requests = append(requests, deferredReq)

			cfg := newTestDeploymentConfig(1)
			// Short horizon guarantees standard requests are still running when the
			// deferred-tier request arrives and the cluster never becomes idle before
			// horizon. This makes DeferredQueueLen() > 0 the falsifiable assertion:
			// without the deferral intercept the request would be admitted normally
			// by AlwaysAdmit and NOT appear in the deferred queue.
			cfg.Horizon = 200
			cs := NewClusterSimulator(cfg, requests, nil)
			mustRun(t, cs)

			// Core contract 1: deferred-tier request must NEVER be rejected by admission.
			if cs.RejectedRequests() > 0 {
				t.Errorf("%s request should not be rejected (it should be deferred); got RejectedRequests=%d", sloClass, cs.RejectedRequests())
			}
			// Core contract 2: the request must actually be in the deferred queue at
			// horizon — confirming the intercept fired. Removing the deferral intercept
			// would cause AlwaysAdmit to admit the request normally; it would end up in
			// the wait queue or running, and DeferredQueueLen() would be 0.
			if cs.DeferredQueueLen() == 0 {
				t.Errorf("%s request should remain in deferred queue at short horizon (intercept must have fired), got DeferredQueueLen=0", sloClass)
			}
		})
	}
}

// T003 — BC-D2: Batch request admitted normally when cluster is idle.
// Also covers "background" SLO class (I3): both classes must be admitted,
// not deferred, when isBusy() returns false.
func TestDeferredQueue_BatchAdmittedWhenIdle(t *testing.T) {
	for _, sloClass := range []string{"batch", "background"} {
		t.Run(sloClass, func(t *testing.T) {
			requests := newDeferredTestRequests(1, sloClass)
			cfg := newTestDeploymentConfig(1)
			cs := NewClusterSimulator(cfg, requests, nil)
			mustRun(t, cs)

			if cs.RejectedRequests() > 0 {
				t.Errorf("%s request should be admitted when cluster idle, got RejectedRequests=%d", sloClass, cs.RejectedRequests())
			}
			if cs.DeferredQueueLen() != 0 {
				t.Errorf("%s: deferred queue should be empty after idle-cluster run, got DeferredQueueLen=%d", sloClass, cs.DeferredQueueLen())
			}
			m := cs.AggregatedMetrics()
			if m.CompletedRequests != 1 {
				t.Errorf("%s request should complete when admitted to idle cluster, got CompletedRequests=%d", sloClass, m.CompletedRequests)
			}
		})
	}
}

// T004 — BC-D3: Deferred requests are promoted and complete once the cluster becomes idle.
// Table-driven over both SLO classes so that a class-filtered bug in promoteDeferred()
// (e.g., one that silently discards "background" entries) would be caught.
func TestDeferredQueue_DeferredPromotedAfterIdle(t *testing.T) {
	for _, sloClass := range []string{"batch", "background"} {
		t.Run(sloClass, func(t *testing.T) {
			// 5 standard requests complete first, then 5 deferred-tier requests should be promoted
			var requests []*sim.Request
			for i := 0; i < 5; i++ {
				requests = append(requests, &sim.Request{
					ID:           fmt.Sprintf("std_%d", i),
					ArrivalTime:  int64(i) * 100,
					SLOClass:     "standard",
					InputTokens:  make([]int, 30),
					OutputTokens: make([]int, 10),
					State:        sim.StateQueued,
				})
			}
			for i := 0; i < 5; i++ {
				requests = append(requests, &sim.Request{
					ID:           fmt.Sprintf("%s_%d", sloClass, i),
					ArrivalTime:  int64(i) * 100, // arrive same time as standard — will be deferred
					SLOClass:     sloClass,
					InputTokens:  make([]int, 20),
					OutputTokens: make([]int, 5),
					State:        sim.StateQueued,
				})
			}

			cfg := newTestDeploymentConfig(1)
			cs := NewClusterSimulator(cfg, requests, nil)
			mustRun(t, cs)

			// All requests must be accounted for — none silently lost (INV-1 extended)
			m := cs.AggregatedMetrics()
			total := m.CompletedRequests + m.StillQueued + m.StillRunning + m.DroppedUnservable +
				m.TimedOutRequests + cs.DeferredQueueLen() + cs.RejectedRequests() + cs.RoutingRejections()
			if total != 10 {
				t.Errorf("conservation: completed(%d)+queued(%d)+running(%d)+dropped(%d)+timedout(%d)+deferred(%d)+rejected(%d)+routingRejected(%d)=%d, want 10",
					m.CompletedRequests, m.StillQueued, m.StillRunning, m.DroppedUnservable,
					m.TimedOutRequests, cs.DeferredQueueLen(), cs.RejectedRequests(), cs.RoutingRejections(), total)
			}
			if cs.RejectedRequests() > 0 {
				t.Errorf("%s requests should not be rejected, got RejectedRequests=%d", sloClass, cs.RejectedRequests())
			}
			// Promotion must have fired: all 10 requests should complete.
			// Without this, a deleted promoteDeferred() still satisfies conservation as 5+5=10.
			if m.CompletedRequests != 10 {
				t.Errorf("all 10 requests should complete after deferred promotion, got CompletedRequests=%d", m.CompletedRequests)
			}
			// All promoted requests must have left the deferred queue (no partial-promotion bug).
			if cs.DeferredQueueLen() != 0 {
				t.Errorf("deferred queue must be empty after full promotion, got DeferredQueueLen=%d", cs.DeferredQueueLen())
			}
		})
	}
}

// T005 — BC-D4: Standard requests are not crowded out by Batch traffic.
// Two runs: run A has only standard requests, run B has standard + batch.
// All 20 standard requests must complete in run A; run B must also complete
// at least 20 requests (standard requests are not crowded out by batch).
func TestDeferredQueue_RealTimeNotCrowdedOut(t *testing.T) {
	makeStandardRequests := func() []*sim.Request {
		reqs := make([]*sim.Request, 20)
		for i := range reqs {
			reqs[i] = &sim.Request{
				ID:           fmt.Sprintf("std_%d", i),
				ArrivalTime:  int64(i) * 50,
				SLOClass:     "standard",
				InputTokens:  make([]int, 40),
				OutputTokens: make([]int, 15),
				State:        sim.StateQueued,
			}
		}
		return reqs
	}

	// Run A: standard requests only
	reqsA := makeStandardRequests()
	cfgA := newTestDeploymentConfig(2)
	csA := NewClusterSimulator(cfgA, reqsA, nil)
	mustRun(t, csA)

	// Run B: same standard requests + batch requests
	reqsB := makeStandardRequests()
	for i := 0; i < 10; i++ {
		reqsB = append(reqsB, &sim.Request{
			ID:           fmt.Sprintf("batch_%d", i),
			ArrivalTime:  int64(i) * 30,
			SLOClass:     "batch",
			InputTokens:  make([]int, 30),
			OutputTokens: make([]int, 10),
			State:        sim.StateQueued,
		})
	}
	cfgB := newTestDeploymentConfig(2)
	csB := NewClusterSimulator(cfgB, reqsB, nil)
	mustRun(t, csB)

	mA := csA.AggregatedMetrics()
	mB := csB.AggregatedMetrics()

	// All 20 standard-only requests should complete in run A (no other competing traffic)
	if mA.CompletedRequests != 20 {
		t.Errorf("run A: expected all 20 standard requests to complete, got CompletedRequests=%d", mA.CompletedRequests)
	}
	// Run B must complete at least 20 requests — standard requests must not be crowded out
	// by batch traffic; deferred batch requests must not consume standard queue slots
	if mB.CompletedRequests < 20 {
		t.Errorf("run B: expected at least 20 completions (batch traffic should not crowd out standard), got CompletedRequests=%d", mB.CompletedRequests)
	}
}

// T006 — BC-D5 / INV-1: Request conservation holds with deferred queue at horizon.
func TestDeferredQueue_INV1_Conservation(t *testing.T) {
	// Use a short horizon to force some batch requests to remain deferred
	const numRequests = 15
	var requests []*sim.Request

	// Standard requests fill the queue
	for i := 0; i < 10; i++ {
		requests = append(requests, &sim.Request{
			ID:           fmt.Sprintf("std_%d", i),
			ArrivalTime:  int64(i) * 20,
			SLOClass:     "standard",
			InputTokens:  make([]int, 80),
			OutputTokens: make([]int, 40),
			State:        sim.StateQueued,
		})
	}
	// Batch requests — all arrive when cluster will be busy
	for i := 0; i < 5; i++ {
		requests = append(requests, &sim.Request{
			ID:           fmt.Sprintf("batch_%d", i),
			ArrivalTime:  int64(i) * 20,
			SLOClass:     "batch",
			InputTokens:  make([]int, 50),
			OutputTokens: make([]int, 25),
			State:        sim.StateQueued,
		})
	}

	// Short horizon: cuts off before batch requests are promoted.
	// Horizon is deliberately small to guarantee at least one batch request remains deferred.
	cfg := newTestDeploymentConfig(1)
	cfg.Horizon = 500 // 0.5ms — short enough to cut off before all batch requests are promoted
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	// The test exists to verify the deferred-at-horizon code path; if nothing is deferred the
	// extended INV-1 formula is never exercised.
	if cs.DeferredQueueLen() == 0 {
		t.Fatalf("expected at least one batch request to remain deferred at horizon (reduce Horizon further if this fails), got DeferredQueueLen=0")
	}
	// INV-1 extended: injected == completed + still_running + still_queued + dropped + timed_out + rejected + routing_rejected + deferred
	conservation := m.CompletedRequests + m.StillQueued + m.StillRunning + m.DroppedUnservable +
		m.TimedOutRequests + cs.RejectedRequests() + cs.RoutingRejections() + cs.DeferredQueueLen()
	if conservation != numRequests {
		t.Errorf("INV-1 violated: completed(%d)+queued(%d)+running(%d)+dropped(%d)+timedout(%d)+rejected(%d)+routingRejected(%d)+deferred(%d)=%d, want %d",
			m.CompletedRequests, m.StillQueued, m.StillRunning, m.DroppedUnservable,
			m.TimedOutRequests, cs.RejectedRequests(), cs.RoutingRejections(), cs.DeferredQueueLen(), conservation, numRequests)
	}
}

// T007 — BC-D7: Idle cluster (no competing work) — Batch/Background admitted normally, not deferred.
func TestDeferredQueue_IdleClusterAdmitsNormally(t *testing.T) {
	// Idle cluster: only batch requests, no standard traffic to keep it busy.
	// isBusy() returns false → deferral intercept does not fire.
	requests := newDeferredTestRequests(5, "batch")
	cfg := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	if cs.DeferredQueueLen() != 0 {
		t.Errorf("idle cluster: expected DeferredQueueLen=0 (no deferral when not busy), got %d", cs.DeferredQueueLen())
	}
	if cs.RejectedRequests() > 0 {
		t.Errorf("idle cluster: batch requests should not be rejected, got RejectedRequests=%d", cs.RejectedRequests())
	}
	m := cs.AggregatedMetrics()
	if m.CompletedRequests != 5 {
		t.Errorf("idle cluster: all 5 batch requests should complete, got CompletedRequests=%d", m.CompletedRequests)
	}
}

// T008 — DeferredQueueLen() panics before Run().
func TestDeferredQueue_DeferredQueueLenPanicsBeforeRun(t *testing.T) {
	cfg := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(cfg, nil, nil)

	defer func() {
		r := recover()
		if r == nil {
			t.Error("expected DeferredQueueLen() to panic before Run(), got no panic")
		}
	}()
	_ = cs.DeferredQueueLen() // should panic
}

// TestDeferredQueue_StandardSLONotSerialized asserts BC-2: standard-class requests
// are NOT serialized by the deferred queue.
//
// Setup: 10 requests with SLOClass "standard" arriving every 10µs — dense enough
// that most requests are in-flight simultaneously. The deferred queue intercept fires
// only for "batch"/"background"; standard requests bypass it entirely.
//
// Bound derivation (plan Section F):
//   Default config: BetaCoeffs=[1000,10,5], AlphaCoeffs=[100,1,100].
//   Non-serialized: QueueingTime=150µs + batched prefill(500 tokens)=6000µs → ~6.2ms mean TTFT.
//   Serialized:     each request waits for all predecessors (~21.75ms each) → ~100ms mean TTFT.
//   Bound 15ms splits the two cases with ~2.4× margin each side.
//
// Regression guard for issue #965.
func TestDeferredQueue_StandardSLONotSerialized(t *testing.T) {
	requests := newDeferredTestRequests(10, "standard")
	cfg := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	if m.CompletedRequests != 10 {
		t.Fatalf("completed %d requests, want 10", m.CompletedRequests)
	}

	ttftMeanMs := float64(m.TTFTSum) / float64(m.CompletedRequests) / 1000.0
	const boundMs = 15.0
	if ttftMeanMs >= boundMs {
		t.Errorf("mean TTFT %.2fms >= bound %.1fms: standard requests are being serialized by the deferred queue (regression: issue #965)",
			ttftMeanMs, boundMs)
	}
}

// TestDeferredQueue_BatchSLOIsSerializedAboveBound asserts BC-3: batch-class requests
// ARE serialized by the deferred queue. This is the guard-validity companion to
// TestDeferredQueue_StandardSLONotSerialized — it confirms the 15ms bound is a real
// discriminator, not a vacuous pass.
func TestDeferredQueue_BatchSLOIsSerializedAboveBound(t *testing.T) {
	requests := newDeferredTestRequests(10, "batch")
	cfg := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	if m.CompletedRequests != 10 {
		t.Fatalf("completed %d requests, want 10", m.CompletedRequests)
	}

	ttftMeanMs := float64(m.TTFTSum) / float64(m.CompletedRequests) / 1000.0
	const boundMs = 15.0
	if ttftMeanMs < boundMs {
		t.Errorf("mean TTFT %.2fms < bound %.1fms: batch requests are NOT being serialized — deferred queue may be broken",
			ttftMeanMs, boundMs)
	}
}
