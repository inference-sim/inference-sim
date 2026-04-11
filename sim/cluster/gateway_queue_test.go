package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestGatewayQueue_FIFO_DequeueOrder(t *testing.T) {
	q := NewGatewayQueue("fifo", 0, nil) // unlimited
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "critical"}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "background"}, 2)
	q.Enqueue(&sim.Request{ID: "r3", SLOClass: "standard"}, 3)

	// FIFO: order by seqID regardless of priority
	got1 := q.Dequeue()
	got2 := q.Dequeue()
	got3 := q.Dequeue()
	if got1.ID != "r1" || got2.ID != "r2" || got3.ID != "r3" {
		t.Errorf("FIFO order: got %s, %s, %s; want r1, r2, r3", got1.ID, got2.ID, got3.ID)
	}
	if q.Len() != 0 {
		t.Errorf("queue should be empty, got %d", q.Len())
	}
}

func TestGatewayQueue_Priority_DequeueOrder(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil) // unlimited
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "background"}, 1) // priority -3
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "critical"}, 2)   // priority 4
	q.Enqueue(&sim.Request{ID: "r3", SLOClass: "standard"}, 3)   // priority 3

	// Priority: highest priority first, then seqID
	got1 := q.Dequeue()
	got2 := q.Dequeue()
	got3 := q.Dequeue()
	if got1.ID != "r2" {
		t.Errorf("expected critical (r2) first, got %s", got1.ID)
	}
	if got2.ID != "r3" {
		t.Errorf("expected standard (r3) second, got %s", got2.ID)
	}
	if got3.ID != "r1" {
		t.Errorf("expected background (r1) third, got %s", got3.ID)
	}
}

func TestGatewayQueue_Priority_SamePriority_FIFO(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard"}, 10)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard"}, 20)
	q.Enqueue(&sim.Request{ID: "r3", SLOClass: "standard"}, 30)

	// Within same priority, FIFO by seqID
	got1 := q.Dequeue()
	got2 := q.Dequeue()
	got3 := q.Dequeue()
	if got1.ID != "r1" || got2.ID != "r2" || got3.ID != "r3" {
		t.Errorf("same-priority FIFO: got %s, %s, %s; want r1, r2, r3", got1.ID, got2.ID, got3.ID)
	}
}

func TestGatewayQueue_CapacityShed_LowestPriority(t *testing.T) {
	q := NewGatewayQueue("priority", 2, nil) // max 2
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard"}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "critical"}, 2)

	// Queue full. Enqueue a sheddable request — it should be shed (lower than both)
	shed := q.Enqueue(&sim.Request{ID: "r3", SLOClass: "sheddable"}, 3)
	if !shed {
		t.Error("expected sheddable request to be shed")
	}
	if q.Len() != 2 {
		t.Errorf("queue should have 2 items, got %d", q.Len())
	}
	if q.ShedCount() != 1 {
		t.Errorf("shed count should be 1, got %d", q.ShedCount())
	}

	// Enqueue a critical request — should displace the lowest (standard r1)
	shed = q.Enqueue(&sim.Request{ID: "r4", SLOClass: "critical"}, 4)
	if shed {
		t.Error("expected critical request to NOT be shed")
	}
	if q.Len() != 2 {
		t.Errorf("queue should have 2 items, got %d", q.Len())
	}
	if q.ShedCount() != 2 {
		t.Errorf("shed count should be 2, got %d", q.ShedCount())
	}

	// Dequeue should give r2 then r4 (both critical, r2 has lower seqID)
	got1 := q.Dequeue()
	got2 := q.Dequeue()
	if got1.ID != "r2" || got2.ID != "r4" {
		t.Errorf("got %s, %s; want r2, r4", got1.ID, got2.ID)
	}
}

func TestGatewayQueue_DequeueEmpty_ReturnsNil(t *testing.T) {
	q := NewGatewayQueue("fifo", 0, nil)
	if got := q.Dequeue(); got != nil {
		t.Errorf("expected nil from empty queue, got %v", got)
	}
}

func TestGatewayQueue_InvalidDispatchOrder_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for invalid dispatch order")
		}
	}()
	NewGatewayQueue("bogus", 0, nil)
}

func TestGatewayQueue_NegativeMaxDepth_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for negative maxDepth")
		}
	}()
	NewGatewayQueue("fifo", -1, nil)
}
