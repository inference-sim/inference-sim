package cluster

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// newBatchTestRequests creates n requests with the given SLOClass,
// arriving every 10µs starting at t=0, with 50 input tokens and 20 output tokens.
func newBatchTestRequests(n int, sloClass string) []*sim.Request {
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

// TestAlwaysAdmit_BatchNotDeferred (BC-1, BC-4) verifies that batch and background
// requests are admitted normally under always-admit (the default), even when the
// cluster is busy. They must NOT be silently deferred.
//
// Setup: 30 standard requests (keep cluster busy) + 1 batch/background request.
// Before this PR, the batch/background request would be deferred. After this PR,
// it flows through Admit() and is admitted.
func TestAlwaysAdmit_BatchNotDeferred(t *testing.T) {
	for _, sloClass := range []string{"batch", "background"} {
		t.Run(sloClass, func(t *testing.T) {
			var requests []*sim.Request
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
			requests = append(requests, &sim.Request{
				ID:           sloClass + "_0",
				ArrivalTime:  50,
				SLOClass:     sloClass,
				InputTokens:  make([]int, 50),
				OutputTokens: make([]int, 20),
				State:        sim.StateQueued,
			})

			cfg := newTestDeploymentConfig(1)
			cs := NewClusterSimulator(cfg, requests, nil)
			mustRun(t, cs)

			// BC-1: batch/background must not be rejected
			if cs.RejectedRequests() > 0 {
				t.Errorf("%s request should be admitted (always-admit), got RejectedRequests=%d", sloClass, cs.RejectedRequests())
			}
			// BC-4: deferred queue must be empty — no pre-admission intercept
			if cs.DeferredQueueLen() != 0 {
				t.Errorf("%s request should NOT be deferred (intercept removed), got DeferredQueueLen=%d", sloClass, cs.DeferredQueueLen())
			}
			// Verify request actually completes (not just non-rejected)
			m := cs.AggregatedMetrics()
			if m.CompletedRequests < 31 {
				t.Errorf("%s request should complete (31 total: 30 standard + 1 %s), got CompletedRequests=%d", sloClass, sloClass, m.CompletedRequests)
			}
		})
	}
}

// TestTierShed_RejectsBatchUnderOverload (BC-2) verifies that tier-shed rejects
// batch/background requests when the cluster is overloaded and their priority is
// below MinAdmitPriority.
//
// Setup: tier-shed with MinAdmitPriority=2 (rejects batch=1, background=0).
// Dense standard traffic to create overload. Batch request arrives under overload.
func TestTierShed_RejectsBatchUnderOverload(t *testing.T) {
	for _, tc := range []struct {
		sloClass string
		priority int // batch=1, background=0
	}{
		{"batch", 1},
		{"background", 0},
	} {
		t.Run(tc.sloClass, func(t *testing.T) {
			var requests []*sim.Request
			// Dense standard traffic to trigger overload
			for i := 0; i < 50; i++ {
				requests = append(requests, &sim.Request{
					ID:           fmt.Sprintf("std_%d", i),
					ArrivalTime:  int64(i) * 2,
					SLOClass:     "standard",
					InputTokens:  make([]int, 200),
					OutputTokens: make([]int, 100),
					State:        sim.StateQueued,
				})
			}
			// Low-priority request arrives while cluster is saturated
			requests = append(requests, &sim.Request{
				ID:           tc.sloClass + "_0",
				ArrivalTime:  10,
				SLOClass:     tc.sloClass,
				InputTokens:  make([]int, 50),
				OutputTokens: make([]int, 20),
				State:        sim.StateQueued,
			})

			cfg := newTestDeploymentConfig(1)
			cfg.AdmissionPolicy = "tier-shed"
			cfg.TierShedThreshold = 1 // any load triggers overload
			cfg.TierShedMinPriority = 2  // rejects batch(1) and background(0)
			cs := NewClusterSimulator(cfg, requests, nil)
			mustRun(t, cs)

			// BC-2: tier-shed must reject exactly the low-priority request under overload
			if cs.RejectedRequests() != 1 {
				t.Errorf("tier-shed should reject exactly 1 %s request (priority=%d < min=2) under overload, got RejectedRequests=%d", tc.sloClass, tc.priority, cs.RejectedRequests())
			}
			// BC-4: deferred queue must be empty — no pre-admission intercept
			if cs.DeferredQueueLen() != 0 {
				t.Errorf("deferred queue should be empty (intercept removed), got DeferredQueueLen=%d", cs.DeferredQueueLen())
			}
		})
	}
}

// TestINV1_NoDeferredTerm (BC-3) verifies INV-1 conservation holds without the
// deferred_horizon_interrupted term. Uses mixed SLO-class traffic.
func TestINV1_NoDeferredTerm(t *testing.T) {
	var requests []*sim.Request
	// Mix of standard and batch traffic
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

	cfg := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	// BC-3: INV-1 without deferred term
	injected := len(requests) - cs.RejectedRequests()
	accounted := m.CompletedRequests + m.StillQueued + m.StillRunning +
		m.DroppedUnservable + m.TimedOutRequests + cs.RoutingRejections()
	if injected != accounted {
		t.Errorf("INV-1: injected=%d != accounted=%d (completed=%d queued=%d running=%d dropped=%d timedout=%d routingRejected=%d)",
			injected, accounted,
			m.CompletedRequests, m.StillQueued, m.StillRunning,
			m.DroppedUnservable, m.TimedOutRequests, cs.RoutingRejections())
	}
	// BC-4: deferred queue must be empty
	if cs.DeferredQueueLen() != 0 {
		t.Errorf("deferred queue should be empty (intercept removed), got DeferredQueueLen=%d", cs.DeferredQueueLen())
	}
}

// TestDeferredQueueInfraExists (BC-5) verifies the deferred queue infrastructure
// is still callable (preserved for #899).
func TestDeferredQueueInfraExists(t *testing.T) {
	cfg := newTestDeploymentConfig(1)
	requests := newBatchTestRequests(1, "batch")
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	// BC-5: DeferredQueueLen() must still be callable
	dql := cs.DeferredQueueLen()
	if dql != 0 {
		t.Errorf("expected DeferredQueueLen=0 (nothing feeds the queue now), got %d", dql)
	}
}

// TestBatchRequestsNotSerialized verifies that batch requests are NOT serialized
// after the intercept removal — they flow through admission like standard requests.
// This is a regression guard for issue #965.
func TestBatchRequestsNotSerialized(t *testing.T) {
	requests := newBatchTestRequests(10, "batch")
	cfg := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	if m.CompletedRequests != 10 {
		t.Fatalf("completed %d requests, want 10", m.CompletedRequests)
	}

	// After intercept removal, batch requests should have similar TTFT to standard
	// (not serialized). Using the same 15ms bound from the old test.
	ttftMeanMs := float64(m.TTFTSum) / float64(m.CompletedRequests) / 1000.0
	const boundMs = 15.0
	if ttftMeanMs >= boundMs {
		t.Errorf("mean TTFT %.2fms >= bound %.1fms: batch requests are being serialized (regression: #965)", ttftMeanMs, boundMs)
	}
}
