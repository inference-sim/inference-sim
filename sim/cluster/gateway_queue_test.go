package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestGatewayQueue_FIFO_DequeueOrder(t *testing.T) {
	q := NewGatewayQueue("fifo", 0, nil) // unlimited
	_, _ = q.Enqueue(&sim.Request{ID: "r1", SLOClass: "critical"}, 1)
	_, _ = q.Enqueue(&sim.Request{ID: "r2", SLOClass: "background"}, 2)
	_, _ = q.Enqueue(&sim.Request{ID: "r3", SLOClass: "standard"}, 3)

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
	_, _ = q.Enqueue(&sim.Request{ID: "r1", SLOClass: "background"}, 1) // priority -3
	_, _ = q.Enqueue(&sim.Request{ID: "r2", SLOClass: "critical"}, 2)   // priority 4
	_, _ = q.Enqueue(&sim.Request{ID: "r3", SLOClass: "standard"}, 3)   // priority 3

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
	_, _ = q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard"}, 10)
	_, _ = q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard"}, 20)
	_, _ = q.Enqueue(&sim.Request{ID: "r3", SLOClass: "standard"}, 30)

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
	_, _ = q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard"}, 1)
	_, _ = q.Enqueue(&sim.Request{ID: "r2", SLOClass: "critical"}, 2)

	// Queue full with non-sheddable only. Sheddable request → rejected (no sheddable victim in queue).
	outcome3, victim3 := q.Enqueue(&sim.Request{ID: "r3", SLOClass: "sheddable"}, 3)
	if outcome3 != Rejected {
		t.Errorf("expected Rejected for sheddable at full non-sheddable queue, got %v", outcome3)
	}
	if victim3 != nil {
		t.Error("expected nil victim for Rejected outcome")
	}
	if q.RejectedCount() != 1 {
		t.Errorf("rejected count should be 1, got %d", q.RejectedCount())
	}

	// Another critical → also rejected (non-sheddable cannot displace non-sheddable).
	outcome4, victim4 := q.Enqueue(&sim.Request{ID: "r4", SLOClass: "critical"}, 4)
	if outcome4 != Rejected {
		t.Errorf("expected Rejected — non-sheddable cannot displace non-sheddable, got %v", outcome4)
	}
	if victim4 != nil {
		t.Error("expected nil victim")
	}
	if q.RejectedCount() != 2 {
		t.Errorf("rejected count should be 2, got %d", q.RejectedCount())
	}

	// Dequeue: r2 (critical, higher priority) then r1 (standard) — original entries unchanged.
	got1 := q.Dequeue()
	got2 := q.Dequeue()
	if got1.ID != "r2" || got2.ID != "r1" {
		t.Errorf("got %s, %s; want r2, r1", got1.ID, got2.ID)
	}
}

func TestGatewayQueue_CriticalityProtection_NonSheddableNeverEvicted(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 2, pm)

	outcome1, victim1 := q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard"}, 1)
	if outcome1 != Enqueued || victim1 != nil {
		t.Errorf("r1: expected Enqueued/nil, got %v/%v", outcome1, victim1)
	}

	outcome2, victim2 := q.Enqueue(&sim.Request{ID: "r2", SLOClass: "critical"}, 2)
	if outcome2 != Enqueued || victim2 != nil {
		t.Errorf("r2: expected Enqueued/nil, got %v/%v", outcome2, victim2)
	}

	// Queue full with only non-sheddable entries. New request should be rejected.
	outcome3, victim3 := q.Enqueue(&sim.Request{ID: "r3", SLOClass: "critical"}, 3)
	if outcome3 != Rejected {
		t.Errorf("r3 should be rejected — no sheddable victim, got %v", outcome3)
	}
	if victim3 != nil {
		t.Error("no victim expected when rejected")
	}
	if q.Len() != 2 {
		t.Errorf("queue should still have 2 entries, got %d", q.Len())
	}
	if q.RejectedCount() != 1 {
		t.Errorf("expected 1 rejection, got %d", q.RejectedCount())
	}
}

func TestGatewayQueue_CriticalityProtection_SheddableEvicted(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 2, pm)

	_, _ = q.Enqueue(&sim.Request{ID: "r1", SLOClass: "batch"}, 1)    // priority=-1 (sheddable)
	_, _ = q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard"}, 2) // priority=3

	// Queue full. New standard request should evict the sheddable batch entry.
	outcome, victim := q.Enqueue(&sim.Request{ID: "r3", SLOClass: "standard"}, 3)
	if outcome != ShedVictim {
		t.Errorf("should shed a victim, got %v", outcome)
	}
	if victim == nil || victim.ID != "r1" {
		t.Errorf("batch request (r1) should be the victim, got %v", victim)
	}
	if q.Len() != 2 {
		t.Errorf("queue depth should be unchanged at 2, got %d", q.Len())
	}
	if q.ShedCount() != 1 {
		t.Errorf("expected 1 shed, got %d", q.ShedCount())
	}
}

func TestGatewayQueue_CriticalityProtection_LowerSheddableDoesNotEvictHigherSheddable(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 2, pm)

	_, _ = q.Enqueue(&sim.Request{ID: "r1", SLOClass: "batch"}, 1)    // priority=-1
	_, _ = q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard"}, 2) // priority=3

	// Queue full. Incoming background (-3) is lower than queued batch (-1). Should be rejected.
	outcome, victim := q.Enqueue(&sim.Request{ID: "r3", SLOClass: "background"}, 3)
	if outcome != Rejected {
		t.Errorf("lower-priority sheddable should be rejected, got %v", outcome)
	}
	if victim != nil {
		t.Error("no victim expected")
	}
	if q.RejectedCount() != 1 {
		t.Errorf("expected 1 rejection, got %d", q.RejectedCount())
	}
	if q.Len() != 2 {
		t.Errorf("queue should still have 2 entries, got %d", q.Len())
	}
}

// TestGatewayQueue_CriticalityProtection_EqualPrioritySheddableTieBreak verifies that
// when incoming and victim have equal sheddable priority, FIFO (earlier seqID wins) applies.
func TestGatewayQueue_CriticalityProtection_EqualPrioritySheddableTieBreak(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 2, pm)

	_, _ = q.Enqueue(&sim.Request{ID: "r1", SLOClass: "batch"}, 10) // priority=-1, seqID=10
	_, _ = q.Enqueue(&sim.Request{ID: "r2", SLOClass: "batch"}, 20) // priority=-1, seqID=20

	// Earlier seqID (5 < 20) should displace the latest-seqID victim (r2, seqID=20).
	outcome, victim := q.Enqueue(&sim.Request{ID: "r3", SLOClass: "batch"}, 5)
	if outcome != ShedVictim {
		t.Errorf("earlier seqID should displace, got outcome %v", outcome)
	}
	if victim == nil || victim.ID != "r2" {
		t.Errorf("expected victim r2 (highest seqID), got %v", victim)
	}

	// Later seqID (25 > all queued) should be rejected — cannot displace earlier entries.
	outcome, victim = q.Enqueue(&sim.Request{ID: "r4", SLOClass: "batch"}, 25)
	if outcome != Rejected {
		t.Errorf("later seqID should be rejected, got outcome %v", outcome)
	}
	if victim != nil {
		t.Error("no victim expected for rejection")
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
