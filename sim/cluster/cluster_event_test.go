package cluster

import (
	"container/heap"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestClusterEventQueue_Ordering verifies BC-4:
// GIVEN a ClusterEventQueue with events at various timestamps, priorities, and seqIDs
// WHEN events are popped from the heap
// THEN they come out ordered by (Timestamp, Priority, seqID)
func TestClusterEventQueue_Ordering(t *testing.T) {
	type eventSpec struct {
		timestamp int64
		priority  int
		seqID     int64
	}

	tests := []struct {
		name     string
		events   []eventSpec
		expected []eventSpec // expected pop order
	}{
		{
			name: "different timestamps",
			events: []eventSpec{
				{timestamp: 300, priority: 0, seqID: 0},
				{timestamp: 100, priority: 0, seqID: 1},
				{timestamp: 200, priority: 0, seqID: 2},
			},
			expected: []eventSpec{
				{timestamp: 100, priority: 0, seqID: 1},
				{timestamp: 200, priority: 0, seqID: 2},
				{timestamp: 300, priority: 0, seqID: 0},
			},
		},
		{
			name: "same timestamp different priorities",
			events: []eventSpec{
				{timestamp: 100, priority: 2, seqID: 0},
				{timestamp: 100, priority: 0, seqID: 1},
				{timestamp: 100, priority: 1, seqID: 2},
			},
			expected: []eventSpec{
				{timestamp: 100, priority: 0, seqID: 1},
				{timestamp: 100, priority: 1, seqID: 2},
				{timestamp: 100, priority: 2, seqID: 0},
			},
		},
		{
			name: "same timestamp same priority different seqIDs",
			events: []eventSpec{
				{timestamp: 100, priority: 1, seqID: 3},
				{timestamp: 100, priority: 1, seqID: 1},
				{timestamp: 100, priority: 1, seqID: 2},
			},
			expected: []eventSpec{
				{timestamp: 100, priority: 1, seqID: 1},
				{timestamp: 100, priority: 1, seqID: 2},
				{timestamp: 100, priority: 1, seqID: 3},
			},
		},
		{
			name: "full pipeline ordering at same timestamp",
			events: []eventSpec{
				{timestamp: 100, priority: 2, seqID: 2}, // Routing
				{timestamp: 100, priority: 1, seqID: 1}, // Admission
				{timestamp: 100, priority: 0, seqID: 0}, // Arrival
			},
			expected: []eventSpec{
				{timestamp: 100, priority: 0, seqID: 0}, // Arrival first
				{timestamp: 100, priority: 1, seqID: 1}, // Admission second
				{timestamp: 100, priority: 2, seqID: 2}, // Routing last
			},
		},
		{
			name: "mixed timestamps and priorities",
			events: []eventSpec{
				{timestamp: 200, priority: 0, seqID: 3},
				{timestamp: 100, priority: 2, seqID: 0},
				{timestamp: 100, priority: 0, seqID: 1},
				{timestamp: 200, priority: 1, seqID: 4},
				{timestamp: 100, priority: 1, seqID: 2},
			},
			expected: []eventSpec{
				{timestamp: 100, priority: 0, seqID: 1},
				{timestamp: 100, priority: 1, seqID: 2},
				{timestamp: 100, priority: 2, seqID: 0},
				{timestamp: 200, priority: 0, seqID: 3},
				{timestamp: 200, priority: 1, seqID: 4},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			q := &ClusterEventQueue{}
			heap.Init(q)

			for _, e := range tc.events {
				var event ClusterEvent
				switch e.priority {
				case 0:
					event = &ClusterArrivalEvent{time: e.timestamp, request: &sim.Request{}}
				case 1:
					event = &AdmissionDecisionEvent{time: e.timestamp, request: &sim.Request{}}
				case 2:
					event = &RoutingDecisionEvent{time: e.timestamp, request: &sim.Request{}}
				}
				heap.Push(q, clusterEventEntry{event: event, seqID: e.seqID})
			}

			for i, exp := range tc.expected {
				entry := heap.Pop(q).(clusterEventEntry)
				if entry.event.Timestamp() != exp.timestamp {
					t.Errorf("pop %d: timestamp = %d, want %d", i, entry.event.Timestamp(), exp.timestamp)
				}
				if entry.event.Priority() != exp.priority {
					t.Errorf("pop %d: priority = %d, want %d", i, entry.event.Priority(), exp.priority)
				}
				if entry.seqID != exp.seqID {
					t.Errorf("pop %d: seqID = %d, want %d", i, entry.seqID, exp.seqID)
				}
			}

			if q.Len() != 0 {
				t.Errorf("queue should be empty after popping all events, got %d remaining", q.Len())
			}
		})
	}
}

// TestClusterEventPriorities verifies that each event type returns the correct priority.
func TestClusterEventPriorities(t *testing.T) {
	tests := []struct {
		name     string
		event    ClusterEvent
		wantPrio int
	}{
		{"ClusterArrivalEvent", &ClusterArrivalEvent{time: 0}, 0},
		{"AdmissionDecisionEvent", &AdmissionDecisionEvent{time: 0}, 1},
		{"RoutingDecisionEvent", &RoutingDecisionEvent{time: 0}, 2},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.event.Priority(); got != tc.wantPrio {
				t.Errorf("Priority() = %d, want %d", got, tc.wantPrio)
			}
		})
	}
}

// TestClusterEventTimestamps verifies that each event type returns its configured timestamp.
func TestClusterEventTimestamps(t *testing.T) {
	tests := []struct {
		name      string
		event     ClusterEvent
		wantTime  int64
	}{
		{"ClusterArrivalEvent", &ClusterArrivalEvent{time: 42}, 42},
		{"AdmissionDecisionEvent", &AdmissionDecisionEvent{time: 99}, 99},
		{"RoutingDecisionEvent", &RoutingDecisionEvent{time: 1000}, 1000},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.event.Timestamp(); got != tc.wantTime {
				t.Errorf("Timestamp() = %d, want %d", got, tc.wantTime)
			}
		})
	}
}
