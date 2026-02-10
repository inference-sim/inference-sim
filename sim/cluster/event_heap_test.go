package cluster

import (
	"testing"
)

// TestEventHeap_TimestampOrdering tests that events are processed in timestamp order
func TestEventHeap_TimestampOrdering(t *testing.T) {
	h := NewEventHeap()

	// Add events with different timestamps in random order
	e1 := NewRequestArrivalEvent(100, &Request{ID: "r1"}, 1)
	e2 := NewRequestArrivalEvent(50, &Request{ID: "r2"}, 2)
	e3 := NewRequestArrivalEvent(150, &Request{ID: "r3"}, 3)

	h.Schedule(e1)
	h.Schedule(e2)
	h.Schedule(e3)

	// Should be popped in timestamp order: 50, 100, 150
	first := h.PopNext()
	if first.Timestamp() != 50 {
		t.Errorf("First event timestamp = %d, want 50", first.Timestamp())
	}

	second := h.PopNext()
	if second.Timestamp() != 100 {
		t.Errorf("Second event timestamp = %d, want 100", second.Timestamp())
	}

	third := h.PopNext()
	if third.Timestamp() != 150 {
		t.Errorf("Third event timestamp = %d, want 150", third.Timestamp())
	}

	if h.Len() != 0 {
		t.Errorf("Heap should be empty, len = %d", h.Len())
	}
}

// TestEventHeap_TypePriorityOrdering tests same-timestamp events use type priority
func TestEventHeap_TypePriorityOrdering(t *testing.T) {
	h := NewEventHeap()

	// Add events at same timestamp with different types
	// RequestArrival (priority 1) should come before InstanceStep (priority 3)
	eStep := NewInstanceStepEvent(100, "inst1", 1)
	eArrival := NewRequestArrivalEvent(100, &Request{ID: "r1"}, 2)

	// Add in reverse priority order
	h.Schedule(eStep)
	h.Schedule(eArrival)

	// Arrival (priority 1) should come first
	first := h.PopNext()
	if first.Type() != EventTypeRequestArrival {
		t.Errorf("First event type = %s, want RequestArrival", first.Type())
	}

	second := h.PopNext()
	if second.Type() != EventTypeInstanceStep {
		t.Errorf("Second event type = %s, want InstanceStep", second.Type())
	}
}

// TestEventHeap_EventIDOrdering tests same-timestamp same-type events use EventID
func TestEventHeap_EventIDOrdering(t *testing.T) {
	h := NewEventHeap()

	// Add multiple events of same type at same timestamp
	e1 := NewInstanceStepEvent(100, "inst1", 1)
	e2 := NewInstanceStepEvent(100, "inst2", 2)
	e3 := NewInstanceStepEvent(100, "inst3", 3)

	// Store event IDs (they should be increasing)
	id1 := e1.EventID()
	id2 := e2.EventID()
	id3 := e3.EventID()

	// Add in non-increasing order
	h.Schedule(e3)
	h.Schedule(e1)
	h.Schedule(e2)

	// Should be popped in EventID order
	first := h.PopNext()
	if first.EventID() != id1 {
		t.Errorf("First event ID = %d, want %d", first.EventID(), id1)
	}

	second := h.PopNext()
	if second.EventID() != id2 {
		t.Errorf("Second event ID = %d, want %d", second.EventID(), id2)
	}

	third := h.PopNext()
	if third.EventID() != id3 {
		t.Errorf("Third event ID = %d, want %d", third.EventID(), id3)
	}
}

// TestEventHeap_DeterministicOrdering tests that ordering is deterministic regardless of insertion order
func TestEventHeap_DeterministicOrdering(t *testing.T) {
	// Create events at same timestamp with different types
	eArrival := NewRequestArrivalEvent(100, &Request{ID: "r1"}, 1)
	eRoute := NewRouteDecisionEvent(100, &Request{ID: "r1"}, "inst1", 2)
	eStep := NewInstanceStepEvent(100, "inst1", 3)
	eCompleted := NewRequestCompletedEvent(100, &Request{ID: "r1"}, "inst1", 4)

	// Test 1: Add in priority order
	h1 := NewEventHeap()
	h1.Schedule(eArrival)
	h1.Schedule(eRoute)
	h1.Schedule(eStep)
	h1.Schedule(eCompleted)

	// Test 2: Add in reverse priority order
	h2 := NewEventHeap()
	h2.Schedule(eCompleted)
	h2.Schedule(eStep)
	h2.Schedule(eRoute)
	h2.Schedule(eArrival)

	// Both should produce same order
	order1 := []EventType{}
	for h1.Len() > 0 {
		order1 = append(order1, h1.PopNext().Type())
	}

	order2 := []EventType{}
	for h2.Len() > 0 {
		order2 = append(order2, h2.PopNext().Type())
	}

	if len(order1) != len(order2) {
		t.Fatalf("Order lengths differ: %d vs %d", len(order1), len(order2))
	}

	for i := range order1 {
		if order1[i] != order2[i] {
			t.Errorf("Order differs at position %d: %s vs %s", i, order1[i], order2[i])
		}
	}

	// Verify expected order (based on priorities: 1, 2, 3, 4)
	expected := []EventType{
		EventTypeRequestArrival,
		EventTypeRouteDecision,
		EventTypeInstanceStep,
		EventTypeRequestCompleted,
	}

	for i := range expected {
		if order1[i] != expected[i] {
			t.Errorf("Position %d: got %s, want %s", i, order1[i], expected[i])
		}
	}
}

// TestEventHeap_ComplexOrdering tests comprehensive ordering with all criteria
func TestEventHeap_ComplexOrdering(t *testing.T) {
	h := NewEventHeap()

	// Scenario: mix of timestamps, types, and IDs
	// t=50: Arrival
	// t=100: Arrival, Route, Step (should be in type priority order)
	// t=100: Two Steps (should be in EventID order)
	// t=200: Completed

	e1 := NewRequestArrivalEvent(50, &Request{ID: "r1"}, 1)
	e2 := NewInstanceStepEvent(100, "inst1", 2)
	e3 := NewRequestArrivalEvent(100, &Request{ID: "r2"}, 3)
	e4 := NewRouteDecisionEvent(100, &Request{ID: "r2"}, "inst1", 4)
	e5 := NewInstanceStepEvent(100, "inst2", 5)
	e6 := NewRequestCompletedEvent(200, &Request{ID: "r1"}, "inst1", 6)

	// Add in random order
	h.Schedule(e6)
	h.Schedule(e2)
	h.Schedule(e4)
	h.Schedule(e1)
	h.Schedule(e5)
	h.Schedule(e3)

	// Expected order:
	// 1. e1 (t=50, Arrival)
	// 2. e3 (t=100, Arrival, priority 1)
	// 3. e4 (t=100, Route, priority 2)
	// 4. e2 (t=100, Step, lower EventID)
	// 5. e5 (t=100, Step, higher EventID)
	// 6. e6 (t=200, Completed)

	events := []Event{}
	for h.Len() > 0 {
		events = append(events, h.PopNext())
	}

	if len(events) != 6 {
		t.Fatalf("Expected 6 events, got %d", len(events))
	}

	// Verify order
	if events[0].Timestamp() != 50 {
		t.Errorf("Event 0: timestamp = %d, want 50", events[0].Timestamp())
	}

	if events[1].Type() != EventTypeRequestArrival || events[1].Timestamp() != 100 {
		t.Errorf("Event 1: type = %s, timestamp = %d, want Arrival at 100", events[1].Type(), events[1].Timestamp())
	}

	if events[2].Type() != EventTypeRouteDecision || events[2].Timestamp() != 100 {
		t.Errorf("Event 2: type = %s, timestamp = %d, want RouteDecision at 100", events[2].Type(), events[2].Timestamp())
	}

	if events[3].Type() != EventTypeInstanceStep || events[3].Timestamp() != 100 {
		t.Errorf("Event 3: type = %s, timestamp = %d, want InstanceStep at 100", events[3].Type(), events[3].Timestamp())
	}

	if events[4].Type() != EventTypeInstanceStep || events[4].Timestamp() != 100 {
		t.Errorf("Event 4: type = %s, timestamp = %d, want InstanceStep at 100", events[4].Type(), events[4].Timestamp())
	}

	// Verify EventID ordering for the two Step events
	if events[3].EventID() >= events[4].EventID() {
		t.Errorf("Step events not in EventID order: %d >= %d", events[3].EventID(), events[4].EventID())
	}

	if events[5].Timestamp() != 200 {
		t.Errorf("Event 5: timestamp = %d, want 200", events[5].Timestamp())
	}
}

// TestEventHeap_Peek tests Peek without removing
func TestEventHeap_Peek(t *testing.T) {
	h := NewEventHeap()

	if h.Peek() != nil {
		t.Error("Peek on empty heap should return nil")
	}

	e1 := NewRequestArrivalEvent(100, &Request{ID: "r1"}, 1)
	e2 := NewRequestArrivalEvent(50, &Request{ID: "r2"}, 2)

	h.Schedule(e1)
	h.Schedule(e2)

	// Peek should return lowest timestamp without removing
	peeked := h.Peek()
	if peeked.Timestamp() != 50 {
		t.Errorf("Peek timestamp = %d, want 50", peeked.Timestamp())
	}

	if h.Len() != 2 {
		t.Errorf("Peek should not remove event, len = %d, want 2", h.Len())
	}

	// PopNext should return same event
	popped := h.PopNext()
	if popped.Timestamp() != 50 {
		t.Errorf("PopNext timestamp = %d, want 50", popped.Timestamp())
	}

	if h.Len() != 1 {
		t.Errorf("After PopNext, len = %d, want 1", h.Len())
	}
}

// TestEventHeap_EmptyOperations tests operations on empty heap
func TestEventHeap_EmptyOperations(t *testing.T) {
	h := NewEventHeap()

	if h.Len() != 0 {
		t.Errorf("New heap len = %d, want 0", h.Len())
	}

	if h.Peek() != nil {
		t.Error("Peek on empty heap should return nil")
	}

	if h.PopNext() != nil {
		t.Error("PopNext on empty heap should return nil")
	}
}

// TestEventHeap_BC6_AllTypePriorities verifies all event type priorities
func TestEventHeap_BC6_AllTypePriorities(t *testing.T) {
	// Verify EventTypePriority map is complete
	requiredTypes := []EventType{
		EventTypeRequestArrival,
		EventTypeRouteDecision,
		EventTypeInstanceStep,
		EventTypeRequestCompleted,
	}

	for _, et := range requiredTypes {
		if _, ok := EventTypePriority[et]; !ok {
			t.Errorf("EventTypePriority missing entry for %s", et)
		}
	}

	// Verify priorities are strictly increasing
	priorities := []int{
		EventTypePriority[EventTypeRequestArrival],
		EventTypePriority[EventTypeRouteDecision],
		EventTypePriority[EventTypeInstanceStep],
		EventTypePriority[EventTypeRequestCompleted],
	}

	for i := 1; i < len(priorities); i++ {
		if priorities[i] <= priorities[i-1] {
			t.Errorf("Priority[%d] = %d not greater than Priority[%d] = %d", i, priorities[i], i-1, priorities[i-1])
		}
	}
}
