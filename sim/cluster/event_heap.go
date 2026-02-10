package cluster

import "container/heap"

// EventHeap implements a priority queue with deterministic ordering
// Ordering: timestamp → type priority → event ID
type EventHeap struct {
	events []Event
}

// NewEventHeap creates a new event heap
func NewEventHeap() *EventHeap {
	h := &EventHeap{
		events: make([]Event, 0),
	}
	heap.Init(h)
	return h
}

// Len implements heap.Interface
func (h *EventHeap) Len() int {
	return len(h.events)
}

// Less implements heap.Interface with deterministic ordering
// Order by: timestamp → type priority → event ID
func (h *EventHeap) Less(i, j int) bool {
	ei, ej := h.events[i], h.events[j]

	// Primary: timestamp (lower first)
	if ei.Timestamp() != ej.Timestamp() {
		return ei.Timestamp() < ej.Timestamp()
	}

	// Secondary: type priority (lower priority value = processed first)
	priI := EventTypePriority[ei.Type()]
	priJ := EventTypePriority[ej.Type()]
	if priI != priJ {
		return priI < priJ
	}

	// Tertiary: event ID (lower first, deterministic tie-breaker)
	return ei.EventID() < ej.EventID()
}

// Swap implements heap.Interface
func (h *EventHeap) Swap(i, j int) {
	h.events[i], h.events[j] = h.events[j], h.events[i]
}

// Push implements heap.Interface
func (h *EventHeap) Push(x interface{}) {
	h.events = append(h.events, x.(Event))
}

// Pop implements heap.Interface
func (h *EventHeap) Pop() interface{} {
	old := h.events
	n := len(old)
	item := old[n-1]
	h.events = old[0 : n-1]
	return item
}

// Schedule adds an event to the heap
func (h *EventHeap) Schedule(e Event) {
	heap.Push(h, e)
}

// PopNext removes and returns the next event
func (h *EventHeap) PopNext() Event {
	if h.Len() == 0 {
		return nil
	}
	return heap.Pop(h).(Event)
}

// Peek returns the next event without removing it
func (h *EventHeap) Peek() Event {
	if h.Len() == 0 {
		return nil
	}
	return h.events[0]
}
