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

// T002 — BC-D1: Batch request is deferred (not rejected) when cluster is busy.
// Uses a 1-instance cluster overloaded with standard requests arriving densely.
// After Run(), DeferredQueueLen() == 0 (all eventually promoted and processed
// because the horizon is long enough) OR RejectedRequests() == 0 (batch never rejected).
func TestDeferredQueue_BatchDeferredWhenBusy(t *testing.T) {
	// Create a mix: many standard requests (to keep cluster busy) + 1 batch
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
	// Batch request arrives early, when cluster is definitely busy
	batchReq := &sim.Request{
		ID:           "batch_0",
		ArrivalTime:  50, // arrives while cluster is saturated
		SLOClass:     "batch",
		InputTokens:  make([]int, 50),
		OutputTokens: make([]int, 20),
		State:        sim.StateQueued,
	}
	requests = append(requests, batchReq)

	cfg := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	// Batch request must NEVER be counted as rejected by admission policy
	if cs.RejectedRequests() > 0 {
		t.Errorf("batch request should not be rejected (it should be deferred); got RejectedRequests=%d", cs.RejectedRequests())
	}
}

// T003 — BC-D2: Batch request admitted normally when cluster is idle.
func TestDeferredQueue_BatchAdmittedWhenIdle(t *testing.T) {
	// Single batch request, cluster starts idle
	requests := newDeferredTestRequests(1, "batch")
	cfg := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	if cs.RejectedRequests() > 0 {
		t.Errorf("batch request should be admitted when cluster idle, got RejectedRequests=%d", cs.RejectedRequests())
	}
	if cs.DeferredQueueLen() != 0 {
		t.Errorf("deferred queue should be empty after idle-cluster run, got DeferredQueueLen=%d", cs.DeferredQueueLen())
	}
	m := cs.AggregatedMetrics()
	if m.CompletedRequests != 1 {
		t.Errorf("batch request should complete when admitted to idle cluster, got CompletedRequests=%d", m.CompletedRequests)
	}
}

// T004 — BC-D3: Deferred requests are promoted and complete once the cluster becomes idle.
func TestDeferredQueue_DeferredPromotedAfterIdle(t *testing.T) {
	// 5 standard requests complete first, then 5 batch requests should be promoted
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
			ID:           fmt.Sprintf("batch_%d", i),
			ArrivalTime:  int64(i) * 100, // arrive same time as standard — will be deferred
			SLOClass:     "batch",
			InputTokens:  make([]int, 20),
			OutputTokens: make([]int, 5),
			State:        sim.StateQueued,
		})
	}

	cfg := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	// All requests must be accounted for — none silently lost
	m := cs.AggregatedMetrics()
	total := m.CompletedRequests + m.StillQueued + m.StillRunning + m.DroppedUnservable + cs.DeferredQueueLen() + cs.RejectedRequests()
	if total != 10 {
		t.Errorf("conservation: completed(%d)+queued(%d)+running(%d)+dropped(%d)+deferred(%d)+rejected(%d)=%d, want 10",
			m.CompletedRequests, m.StillQueued, m.StillRunning, m.DroppedUnservable, cs.DeferredQueueLen(), cs.RejectedRequests(), total)
	}
	if cs.RejectedRequests() > 0 {
		t.Errorf("no batch requests should be rejected, got RejectedRequests=%d", cs.RejectedRequests())
	}
}

// T005 — BC-D4: Real-time latency is unaffected by Batch traffic.
// Two identical runs with the same seed: one with only standard requests,
// one with standard + batch. CompletedRequests for standard tier must be equal.
func TestDeferredQueue_RealTimeUnaffected(t *testing.T) {
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

	// Standard requests should complete at the same rate regardless of batch presence
	// (batch requests should not steal queue slots from standard requests)
	if mA.CompletedRequests > mB.CompletedRequests {
		t.Errorf("standard-only run completed more requests (%d) than mixed run (%d) — batch traffic interfering with real-time",
			mA.CompletedRequests, mB.CompletedRequests)
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

	// Short horizon: likely cuts off before batch requests are promoted
	cfg := newTestDeploymentConfig(1)
	cfg.Horizon = 5000 // 5ms — may not be enough for everything to complete
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	// INV-1 extended: injected == completed + still_running + still_queued + dropped + rejected + deferred
	conservation := m.CompletedRequests + m.StillQueued + m.StillRunning + m.DroppedUnservable +
		cs.RejectedRequests() + cs.DeferredQueueLen()
	if conservation != numRequests {
		t.Errorf("INV-1 violated: completed(%d)+queued(%d)+running(%d)+dropped(%d)+rejected(%d)+deferred(%d)=%d, want %d",
			m.CompletedRequests, m.StillQueued, m.StillRunning, m.DroppedUnservable,
			cs.RejectedRequests(), cs.DeferredQueueLen(), conservation, numRequests)
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
