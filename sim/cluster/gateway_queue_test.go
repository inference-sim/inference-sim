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

// TestGatewayQueue_CriticalityProtection_HigherSheddableDisplacesLowerSheddable verifies that
// a sheddable incoming with higher priority displaces a lower-priority sheddable victim.
func TestGatewayQueue_CriticalityProtection_HigherSheddableDisplacesLowerSheddable(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 2, pm)

	_, _ = q.Enqueue(&sim.Request{ID: "r1", SLOClass: "background"}, 1) // priority=-3 (sheddable)
	_, _ = q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard"}, 2)   // priority=3 (non-sheddable)

	// Queue full. Incoming batch (priority=-1, sheddable) should displace background (priority=-3, sheddable).
	outcome, victim := q.Enqueue(&sim.Request{ID: "r3", SLOClass: "batch"}, 3)
	if outcome != ShedVictim {
		t.Errorf("higher-priority sheddable should displace lower-priority sheddable, got %v", outcome)
	}
	if victim == nil || victim.ID != "r1" {
		t.Errorf("background (r1) should be the victim, got %v", victim)
	}
	if q.Len() != 2 {
		t.Errorf("queue depth should remain 2, got %d", q.Len())
	}
	if q.ShedCount() != 1 {
		t.Errorf("expected 1 shed, got %d", q.ShedCount())
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

// --- Per-band queue tests (Task 1: BC-2, BC-3, BC-4, BC-5) ---

func TestGatewayQueue_PerBand_FlowKeyMapping(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	r1 := &sim.Request{ID: "r1", TenantID: "tenant-A", SLOClass: "standard"} // priority 3
	r2 := &sim.Request{ID: "r2", TenantID: "tenant-B", SLOClass: "standard"} // priority 3
	r3 := &sim.Request{ID: "r3", TenantID: "", SLOClass: "critical"}         // priority 4, default tenant

	q.Enqueue(r1, 1)
	q.Enqueue(r2, 2)
	q.Enqueue(r3, 3)

	// Band 4 (critical) dispatched first
	got := q.Dequeue()
	if got.ID != "r3" {
		t.Errorf("expected r3 (critical band 4), got %s", got.ID)
	}
	// Band 3: r1 and r2 in separate flows. Global-strict: earliest seqID wins.
	got = q.Dequeue()
	if got.ID != "r1" {
		t.Errorf("expected r1 (earliest in band 3), got %s", got.ID)
	}
	got = q.Dequeue()
	if got.ID != "r2" {
		t.Errorf("expected r2, got %s", got.ID)
	}
}

func TestGatewayQueue_PerBand_DefaultTenantID(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.Enqueue(&sim.Request{ID: "r1", TenantID: "", SLOClass: "standard"}, 1)
	if q.LenByBand(sim.DefaultSLOPriorityMap().Priority("standard")) != 1 {
		t.Errorf("expected 1 request in standard band")
	}
}

func TestGatewayQueue_PerBand_DefaultTenantIDCollision(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.Enqueue(&sim.Request{ID: "r1", TenantID: "", SLOClass: "standard"}, 1)
	q.Enqueue(&sim.Request{ID: "r2", TenantID: "default", SLOClass: "standard"}, 2)

	got1 := q.Dequeue()
	got2 := q.Dequeue()
	if got1.ID != "r1" || got2.ID != "r2" {
		t.Errorf("got %s, %s; want r1, r2 (same flow, FIFO by seqID)", got1.ID, got2.ID)
	}
}

func TestGatewayQueue_PerBand_DispatchOrder(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.Enqueue(&sim.Request{ID: "d", SLOClass: "batch"}, 1)    // band -1
	q.Enqueue(&sim.Request{ID: "b", SLOClass: "standard"}, 2) // band 3
	q.Enqueue(&sim.Request{ID: "a", SLOClass: "critical"}, 3) // band 4
	q.Enqueue(&sim.Request{ID: "c", SLOClass: "standard"}, 4) // band 3

	expected := []string{"a", "b", "c", "d"}
	for i, exp := range expected {
		got := q.Dequeue()
		if got.ID != exp {
			t.Errorf("position %d: got %s, want %s", i, got.ID, exp)
		}
	}
	if q.Dequeue() != nil {
		t.Error("expected nil from empty queue")
	}
}

func TestGatewayQueue_PerBand_GlobalStrictFairness(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.Enqueue(&sim.Request{ID: "b1", TenantID: "B", SLOClass: "standard"}, 10)
	q.Enqueue(&sim.Request{ID: "b2", TenantID: "B", SLOClass: "standard"}, 20)
	q.Enqueue(&sim.Request{ID: "c1", TenantID: "C", SLOClass: "standard"}, 5) // earlier seqID

	got := q.Dequeue()
	if got.ID != "c1" {
		t.Errorf("expected c1 (earliest head), got %s", got.ID)
	}
	got = q.Dequeue()
	if got.ID != "b1" {
		t.Errorf("expected b1, got %s", got.ID)
	}
	got = q.Dequeue()
	if got.ID != "b2" {
		t.Errorf("expected b2, got %s", got.ID)
	}
}

func TestGatewayQueue_PerBand_LenByBand(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("priority", 0, pm)
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "critical"}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard"}, 2)
	q.Enqueue(&sim.Request{ID: "r3", SLOClass: "standard"}, 3)
	q.Enqueue(&sim.Request{ID: "r4", SLOClass: "background"}, 4)

	if q.LenByBand(pm.Priority("critical")) != 1 {
		t.Errorf("critical band wrong")
	}
	if q.LenByBand(pm.Priority("standard")) != 2 {
		t.Errorf("standard band wrong")
	}
	if q.LenByBand(pm.Priority("background")) != 1 {
		t.Errorf("background band wrong")
	}
	if q.LenByBand(99) != 0 {
		t.Errorf("nonexistent band should be 0")
	}
	if q.Len() != 4 {
		t.Errorf("total: got %d", q.Len())
	}
}

func TestGatewayQueue_PerBand_CustomSLOPriorities(t *testing.T) {
	pm := sim.NewSLOPriorityMap(map[string]int{"batch": 0})
	q := NewGatewayQueue("priority", 0, pm)
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "batch"}, 1)    // band 0 (custom)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard"}, 2) // band 3

	got := q.Dequeue()
	if got.ID != "r2" {
		t.Errorf("expected standard (band 3) first, got %s", got.ID)
	}
	got = q.Dequeue()
	if got.ID != "r1" {
		t.Errorf("expected batch (band 0) second, got %s", got.ID)
	}
}

// --- Per-band capacity tests (Task 2: BC-6, BC-11, BC-12) ---

func TestGatewayQueue_PerBand_BandCapacityRejection(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.SetPerBandCapacity(2)

	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard"}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard"}, 2)

	// Third standard -> rejected (band full, no sheddable to evict within band)
	outcome, victim := q.Enqueue(&sim.Request{ID: "r3", SLOClass: "standard"}, 3)
	if outcome != Rejected {
		t.Errorf("expected Rejected, got %v", outcome)
	}
	if victim != nil {
		t.Error("expected nil victim")
	}

	// Different band still accepts
	outcome, _ = q.Enqueue(&sim.Request{ID: "r4", SLOClass: "batch"}, 4)
	if outcome != Enqueued {
		t.Errorf("expected Enqueued in different band, got %v", outcome)
	}
}

func TestGatewayQueue_PerBand_BandCapacitySamePriorityEviction(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.SetPerBandCapacity(2)

	q.Enqueue(&sim.Request{ID: "r1", TenantID: "A", SLOClass: "batch"}, 1)
	q.Enqueue(&sim.Request{ID: "r2", TenantID: "B", SLOClass: "batch"}, 2)

	// Band -1 full. Incoming batch with later seqID -> rejected (cannot displace same-priority)
	outcome, _ := q.Enqueue(&sim.Request{ID: "r3", TenantID: "C", SLOClass: "batch"}, 3)
	if outcome != Rejected {
		t.Errorf("later seqID same-priority should be rejected, got %v", outcome)
	}
}

func TestGatewayQueue_PerBand_GlobalMaxDepthStillWorks(t *testing.T) {
	q := NewGatewayQueue("priority", 2, nil)
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard"}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "critical"}, 2)

	// Queue full. Sheddable -> rejected (no sheddable victims)
	outcome, _ := q.Enqueue(&sim.Request{ID: "r3", SLOClass: "sheddable"}, 3)
	if outcome != Rejected {
		t.Errorf("expected Rejected, got %v", outcome)
	}
	if q.Len() != 2 {
		t.Errorf("expected 2, got %d", q.Len())
	}
}

func TestGatewayQueue_PerBand_CapacityCheckOrder(t *testing.T) {
	q := NewGatewayQueue("priority", 100, nil)
	q.SetPerBandCapacity(2)

	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard"}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard"}, 2)

	// Per-band cap reached (2), global not (2 < 100)
	outcome, _ := q.Enqueue(&sim.Request{ID: "r3", SLOClass: "standard"}, 3)
	if outcome != Rejected {
		t.Errorf("per-band cap should reject, got %v", outcome)
	}
}
