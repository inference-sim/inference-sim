package sim

import (
	"container/heap"
	"testing"
)

// TestEventQueue_SameTimestamp_PriorityOrder verifies BC-12: at equal timestamps,
// events fire in priority order (lower priority number first).
// StepEvent (priority 2) must fire before TimeoutEvent (priority 5).
func TestEventQueue_SameTimestamp_PriorityOrder(t *testing.T) {
	eq := &EventQueue{}
	heap.Init(eq)

	// Push events at the same timestamp in reverse priority order
	tick := int64(1000)
	var seq int64

	// Push TimeoutEvent first (priority 5), then StepEvent (priority 2)
	seq++
	heap.Push(eq, eventEntry{event: &TimeoutEvent{time: tick, Request: &Request{ID: "r1"}}, seqID: seq})
	seq++
	heap.Push(eq, eventEntry{event: &StepEvent{time: tick}, seqID: seq})
	seq++
	heap.Push(eq, eventEntry{event: &ArrivalEvent{time: tick, Request: &Request{ID: "r2"}}, seqID: seq})

	// Pop and verify priority order: Arrival(0) < Step(2) < Timeout(5)
	e1 := heap.Pop(eq).(eventEntry)
	e2 := heap.Pop(eq).(eventEntry)
	e3 := heap.Pop(eq).(eventEntry)

	if e1.event.Priority() != PriorityArrival {
		t.Errorf("first event: got priority %d, want %d (Arrival)", e1.event.Priority(), PriorityArrival)
	}
	if e2.event.Priority() != PriorityStep {
		t.Errorf("second event: got priority %d, want %d (Step)", e2.event.Priority(), PriorityStep)
	}
	if e3.event.Priority() != PriorityTimeout {
		t.Errorf("third event: got priority %d, want %d (Timeout)", e3.event.Priority(), PriorityTimeout)
	}
}

// TestEventQueue_SeqID_BreaksTies verifies that seqID breaks ties within
// same-type same-timestamp events for INV-6 determinism.
func TestEventQueue_SeqID_BreaksTies(t *testing.T) {
	eq := &EventQueue{}
	heap.Init(eq)

	tick := int64(2000)
	// Two ArrivalEvents at same tick — seqID determines order
	heap.Push(eq, eventEntry{event: &ArrivalEvent{time: tick, Request: &Request{ID: "first"}}, seqID: 1})
	heap.Push(eq, eventEntry{event: &ArrivalEvent{time: tick, Request: &Request{ID: "second"}}, seqID: 2})

	e1 := heap.Pop(eq).(eventEntry)
	e2 := heap.Pop(eq).(eventEntry)

	if e1.seqID != 1 || e2.seqID != 2 {
		t.Errorf("seqID ordering: got %d then %d, want 1 then 2", e1.seqID, e2.seqID)
	}
}

// TestTimeout_QueuedRequest_TimesOut verifies BC-1: a queued request
// transitions to StateTimedOut when its deadline passes.
func TestTimeout_QueuedRequest_TimesOut(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(1, 2048, 0), // max 1 running request — forces queuing
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{0, 0, 0}), // zero alpha = no queueing delay
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test", "H100", 1, "blackbox", 0),
	}
	sim := mustNewSimulator(t, cfg)

	// Two requests: r1 arrives at 0 (will run), r2 arrives at 0 with short deadline (will timeout while queued)
	r1 := &Request{ID: "r1", ArrivalTime: 0, InputTokens: make([]int, 10), OutputTokens: make([]int, 100), State: StateQueued, MaxOutputLen: 100}
	r2 := &Request{ID: "r2", ArrivalTime: 0, InputTokens: make([]int, 10), OutputTokens: make([]int, 100), State: StateQueued, MaxOutputLen: 100, Deadline: 5000} // timeout at tick 5000

	sim.InjectArrival(r1)
	sim.InjectArrival(r2)
	sim.Run()

	// r2 should have timed out (batch size 1 means r1 runs, r2 waits, r2's deadline passes)
	if r2.State != StateTimedOut {
		t.Errorf("r2 state: got %s, want %s", r2.State, StateTimedOut)
	}
	if sim.Metrics.TimedOutRequests != 1 {
		t.Errorf("TimedOutRequests: got %d, want 1", sim.Metrics.TimedOutRequests)
	}
	// r1 should have completed
	if r1.State != StateCompleted {
		t.Errorf("r1 state: got %s, want %s", r1.State, StateCompleted)
	}
}

// TestTimeout_CompletedRequest_NoOp verifies BC-3: a TimeoutEvent for an
// already-completed request is a no-op.
func TestTimeout_CompletedRequest_NoOp(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test", "H100", 1, "blackbox", 0),
	}
	sim := mustNewSimulator(t, cfg)

	// One request with a deadline far in the future — will complete normally
	r1 := &Request{ID: "r1", ArrivalTime: 0, InputTokens: make([]int, 10), OutputTokens: make([]int, 5), State: StateQueued, MaxOutputLen: 5, Deadline: 999_999}

	sim.InjectArrival(r1)
	sim.Run()

	// Request should complete, timeout should be no-op
	if r1.State != StateCompleted {
		t.Errorf("r1 state: got %s, want %s", r1.State, StateCompleted)
	}
	if sim.Metrics.TimedOutRequests != 0 {
		t.Errorf("TimedOutRequests: got %d, want 0 (timeout should be no-op)", sim.Metrics.TimedOutRequests)
	}
	if sim.Metrics.CompletedRequests != 1 {
		t.Errorf("CompletedRequests: got %d, want 1", sim.Metrics.CompletedRequests)
	}
}

// TestTimeout_CompletionWinsAtEqualTimestamp verifies BC-12: when a StepEvent
// and TimeoutEvent fire at the same tick, the step event fires first (priority ordering).
func TestTimeout_CompletionWinsAtEqualTimestamp(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 0, 0}, []float64{0, 0, 0}), // step time = beta0 = 1000µs, no per-token cost
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test", "H100", 1, "blackbox", 0),
	}
	sim := mustNewSimulator(t, cfg)

	// Request with 1 input token, 1 output token. Step time = 1000µs.
	// Prefill step at t=0 completes at t=1000. Decode step at t=1000 completes at t=2000.
	// Set deadline = 2000 (same tick as completion).
	r1 := &Request{ID: "r1", ArrivalTime: 0, InputTokens: make([]int, 1), OutputTokens: make([]int, 1), State: StateQueued, MaxOutputLen: 1, Deadline: 2000}

	sim.InjectArrival(r1)
	sim.Run()

	// With priority ordering, StepEvent (priority 2) fires before TimeoutEvent (priority 5).
	// The request should complete, not time out.
	if r1.State != StateCompleted {
		t.Errorf("BC-12: r1 state: got %s, want %s (completion should win at equal tick)", r1.State, StateCompleted)
	}
	if sim.Metrics.TimedOutRequests != 0 {
		t.Errorf("BC-12: TimedOutRequests: got %d, want 0", sim.Metrics.TimedOutRequests)
	}
}

// TestWaitQueue_Remove verifies that Remove() correctly removes a request
// from the middle of the queue.
func TestWaitQueue_Remove(t *testing.T) {
	wq := &WaitQueue{}
	r1 := &Request{ID: "r1"}
	r2 := &Request{ID: "r2"}
	r3 := &Request{ID: "r3"}
	wq.Enqueue(r1)
	wq.Enqueue(r2)
	wq.Enqueue(r3)

	if wq.Len() != 3 {
		t.Fatalf("queue length: got %d, want 3", wq.Len())
	}

	// Remove middle element
	found := wq.Remove(r2)
	if !found {
		t.Error("Remove(r2): got false, want true")
	}
	if wq.Len() != 2 {
		t.Errorf("queue length after remove: got %d, want 2", wq.Len())
	}

	// Verify r1 and r3 remain in order
	if wq.Peek() != r1 {
		t.Errorf("peek after remove: got %s, want r1", wq.Peek().ID)
	}

	// Remove non-existent element
	found = wq.Remove(&Request{ID: "r4"})
	if found {
		t.Error("Remove(non-existent): got true, want false")
	}
}
