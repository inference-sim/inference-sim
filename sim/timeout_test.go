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
