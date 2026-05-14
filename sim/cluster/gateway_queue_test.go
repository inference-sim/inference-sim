package cluster

import (
	"fmt"
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
	q := NewGatewayQueue("priority", 0, nil)                            // unlimited
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
	q.SetSheddingEnabled(true)

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
	q.SetSheddingEnabled(true)

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
	q.SetSheddingEnabled(true)

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

func TestGatewayQueue_SheddingDisabled_RejectsInsteadOfShed(t *testing.T) {
	q := NewGatewayQueue("priority", 2, nil) // maxDepth=2, sheddingEnabled=false (default)
	_, _ = q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard"}, 1)
	_, _ = q.Enqueue(&sim.Request{ID: "r2", SLOClass: "sheddable"}, 2)

	outcome, victim := q.Enqueue(&sim.Request{ID: "r3", SLOClass: "critical"}, 3)
	if outcome != Rejected {
		t.Errorf("shedding disabled: expected Rejected, got %v", outcome)
	}
	if victim != nil {
		t.Error("shedding disabled: expected nil victim")
	}
	if q.ShedCount() != 0 {
		t.Errorf("shedding disabled: ShedCount should be 0, got %d", q.ShedCount())
	}
	if q.RejectedCount() != 1 {
		t.Errorf("shedding disabled: RejectedCount should be 1, got %d", q.RejectedCount())
	}
}

func TestGatewayQueue_SheddingEnabled_ShedsVictim(t *testing.T) {
	q := NewGatewayQueue("priority", 2, nil)
	q.SetSheddingEnabled(true)
	_, _ = q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard"}, 1)
	_, _ = q.Enqueue(&sim.Request{ID: "r2", SLOClass: "sheddable"}, 2)

	outcome, victim := q.Enqueue(&sim.Request{ID: "r3", SLOClass: "critical"}, 3)
	if outcome != ShedVictim {
		t.Errorf("shedding enabled: expected ShedVictim, got %v", outcome)
	}
	if victim == nil || victim.ID != "r2" {
		t.Errorf("shedding enabled: expected victim r2, got %v", victim)
	}
	if q.ShedCount() != 1 {
		t.Errorf("shedding enabled: ShedCount should be 1, got %d", q.ShedCount())
	}
}

func TestGatewayQueue_SheddingEnabled_DefaultFalse(t *testing.T) {
	q := NewGatewayQueue("fifo", 10, nil)
	for i := 0; i < 10; i++ {
		q.Enqueue(&sim.Request{ID: fmt.Sprintf("r%d", i), SLOClass: "sheddable"}, int64(i))
	}
	outcome, _ := q.Enqueue(&sim.Request{ID: "overflow", SLOClass: "critical"}, 100)
	if outcome != Rejected {
		t.Errorf("default queue should reject (not shed) when full, got %v", outcome)
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

func TestGatewayQueue_DequeueGated_HoLBlocking(t *testing.T) {
	// Tests verify per-band HoL blocking matching GIE's dispatchCycle behavior:
	// - Ceilings computed by band POSITION (not non-empty count)
	// - Ceiling check happens BEFORE emptiness check
	// - If saturation >= ceiling, halt entire dispatch (prevents priority inversion)

	// Case 1: 2 bands, low saturation — dispatches highest-priority item.
	q := NewGatewayQueue("priority", 0, nil)
	q.SetUsageLimitThreshold(0.5)
	q.Enqueue(&sim.Request{ID: "critical1", SLOClass: "critical"}, 1)
	q.Enqueue(&sim.Request{ID: "batch1", SLOClass: "batch"}, 2)
	// 2 bands [critical(4), batch(-1)]. Ceilings by position: [1.0, 0.5].
	// sat=0.3 < critical ceiling 1.0 → dispatches critical1.
	req := q.DequeueGated(0.3)
	if req == nil || req.ID != "critical1" {
		t.Fatalf("expected critical1 at sat=0.3, got %v", req)
	}

	// Case 2: Full saturation blocks top band → nothing dispatches.
	q.Enqueue(&sim.Request{ID: "critical2", SLOClass: "critical"}, 3)
	// sat=1.0 >= critical ceiling 1.0 → HoL halt immediately.
	req = q.DequeueGated(1.0)
	if req != nil {
		t.Fatalf("expected nil at sat=1.0, got %v", req.ID)
	}
	if q.Len() != 2 {
		t.Fatalf("expected 2 items still in queue, got %d", q.Len())
	}

	// Case 3: HoL blocks lower band while higher band dispatches.
	// After dequeuing critical, batch remains but band structure is preserved.
	// 2 bands still [critical(4), batch(-1)]. Ceilings: [1.0, 0.5].
	// Dequeue critical2 at sat=0.3:
	req = q.DequeueGated(0.3)
	if req == nil || req.ID != "critical2" {
		t.Fatalf("expected critical2 at sat=0.3, got %v", req)
	}
	// Now critical band empty, batch has batch1. Bands still [4, -1].
	// sat=0.6: critical ceiling=1.0 > 0.6 → passes, band empty → skip.
	//          batch ceiling=0.5 < 0.6 → HoL halt! batch1 stays queued.
	req = q.DequeueGated(0.6)
	if req != nil {
		t.Fatalf("expected nil at sat=0.6 (batch ceiling=0.5 < 0.6), got %v", req.ID)
	}
	if q.Len() != 1 {
		t.Fatalf("expected 1 item (batch1) still queued, got %d", q.Len())
	}
	// sat=0.4 < batch ceiling 0.5 → batch1 dispatches.
	req = q.DequeueGated(0.4)
	if req == nil || req.ID != "batch1" {
		t.Fatalf("expected batch1 at sat=0.4, got %v", req)
	}

	// Case 4: 3 bands with intermediate HoL blocking.
	// critical(4), standard(3), batch(-1). numBands=3. Ceilings: [1.0, 0.75, 0.5].
	q3 := NewGatewayQueue("priority", 0, nil)
	q3.SetUsageLimitThreshold(0.5)
	q3.Enqueue(&sim.Request{ID: "c1", SLOClass: "critical"}, 1)
	q3.Enqueue(&sim.Request{ID: "s1", SLOClass: "standard"}, 2)
	q3.Enqueue(&sim.Request{ID: "b1", SLOClass: "batch"}, 3)

	// sat=0.76: critical ceiling=1.0 > 0.76 → dispatches c1.
	req = q3.DequeueGated(0.76)
	if req == nil || req.ID != "c1" {
		t.Fatalf("expected c1 at sat=0.76, got %v", req)
	}
	// Bands still [4, 3, -1]. Ceilings: [1.0, 0.75, 0.5].
	// sat=0.76: critical ceiling=1.0 > 0.76 → passes (empty, skip).
	//           standard ceiling=0.75 < 0.76 → HoL halt! s1 and b1 stay queued.
	req = q3.DequeueGated(0.76)
	if req != nil {
		t.Fatalf("expected nil at sat=0.76 (standard ceiling=0.75 blocks), got %v", req.ID)
	}
	if q3.Len() != 2 {
		t.Fatalf("expected 2 items still queued, got %d", q3.Len())
	}
	// sat=0.4: all ceilings pass → dispatches s1 (highest non-empty).
	req = q3.DequeueGated(0.4)
	if req == nil || req.ID != "s1" {
		t.Fatalf("expected s1 at sat=0.4, got %v", req)
	}
	// sat=0.4 < batch ceiling 0.5 → dispatches b1.
	req = q3.DequeueGated(0.4)
	if req == nil || req.ID != "b1" {
		t.Fatalf("expected b1 at sat=0.4, got %v", req)
	}

	// Case 5: HoL effect — low-priority items starved under sustained high saturation.
	q5 := NewGatewayQueue("priority", 0, nil)
	q5.SetUsageLimitThreshold(0.5)
	q5.Enqueue(&sim.Request{ID: "s1", SLOClass: "standard"}, 1)
	q5.Enqueue(&sim.Request{ID: "b1", SLOClass: "batch"}, 2)
	// 2 bands [standard(3), batch(-1)]. Ceilings: [1.0, 0.5].
	// sat=0.6: standard ceiling=1.0 > 0.6 → dispatches s1.
	req = q5.DequeueGated(0.6)
	if req == nil || req.ID != "s1" {
		t.Fatalf("expected s1 at sat=0.6, got %v", req)
	}
	// Refill standard. Bands [3, -1] persist. Ceilings: [1.0, 0.5].
	q5.Enqueue(&sim.Request{ID: "s2", SLOClass: "standard"}, 3)
	// sat=0.6: standard ceiling=1.0 > 0.6 → dispatches s2.
	req = q5.DequeueGated(0.6)
	if req == nil || req.ID != "s2" {
		t.Fatalf("expected s2 at sat=0.6, got %v", req)
	}
	// Standard empty, batch has b1. sat=0.6 >= batch ceiling 0.5 → HoL halt.
	// Batch b1 is starved as long as saturation > 0.5!
	req = q5.DequeueGated(0.6)
	if req != nil {
		t.Fatalf("expected nil — batch starved at sat=0.6 (ceiling=0.5), got %v", req.ID)
	}
	// Saturation drops → batch finally dispatches.
	req = q5.DequeueGated(0.4)
	if req == nil || req.ID != "b1" {
		t.Fatalf("expected b1 at sat=0.4, got %v", req)
	}
}

func TestGatewayQueue_DequeueGated_SingleBand_NoCeiling(t *testing.T) {
	// Single band: ceiling is always 1.0 regardless of threshold.
	q := NewGatewayQueue("priority", 0, nil)
	q.SetUsageLimitThreshold(0.5)

	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "critical"}, 1)

	// At saturation=0.9: should still dispatch (single band → ceiling=1.0).
	req := q.DequeueGated(0.9)
	if req == nil || req.ID != "r1" {
		t.Fatalf("expected r1 at sat=0.9 with single band, got %v", req)
	}
}

func TestGatewayQueue_DequeueGated_DefaultThreshold_NoHoL(t *testing.T) {
	// Default threshold=1.0: linear interpolation gives all ceilings=1.0.
	// Formula: ceiling = 1.0 - i*(1.0-1.0)/(N-1) = 1.0 for all i.
	// Equivalent to old behavior — no HoL blocking.
	q := NewGatewayQueue("priority", 0, nil)
	// No SetUsageLimitThreshold call — uses default 1.0.

	q.Enqueue(&sim.Request{ID: "critical1", SLOClass: "critical"}, 1)
	q.Enqueue(&sim.Request{ID: "batch1", SLOClass: "batch"}, 2)

	// At saturation=0.99: all ceilings=1.0 → both dispatch.
	req := q.DequeueGated(0.99)
	if req == nil || req.ID != "critical1" {
		t.Fatalf("expected critical1, got %v", req)
	}
	req = q.DequeueGated(0.99)
	if req == nil || req.ID != "batch1" {
		t.Fatalf("expected batch1, got %v", req)
	}
}

// BC-5: No SetFairnessPolicy call → GlobalStrict default (backward compat).
func TestGatewayQueue_DefaultFairnessIsGlobalStrict(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.Enqueue(&sim.Request{ID: "b1", TenantID: "B", SLOClass: "standard"}, 10)
	q.Enqueue(&sim.Request{ID: "c1", TenantID: "C", SLOClass: "standard"}, 5)
	got := q.Dequeue()
	if got.ID != "c1" {
		t.Errorf("BC-5: default must be global-strict (earliest seqID c1), got %s", got.ID)
	}
}

// BC-2: RoundRobin cycles through tenants in sorted key order.
func TestGatewayQueue_RoundRobin_CyclesTenants(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.SetFairnessPolicy(NewRoundRobinPolicy())

	// All same priority (standard=3), three tenants with equal depth (3 each).
	q.Enqueue(&sim.Request{ID: "a1", SLOClass: "standard", TenantID: "A"}, 1)
	q.Enqueue(&sim.Request{ID: "a2", SLOClass: "standard", TenantID: "A"}, 2)
	q.Enqueue(&sim.Request{ID: "a3", SLOClass: "standard", TenantID: "A"}, 3)
	q.Enqueue(&sim.Request{ID: "b1", SLOClass: "standard", TenantID: "B"}, 4)
	q.Enqueue(&sim.Request{ID: "b2", SLOClass: "standard", TenantID: "B"}, 5)
	q.Enqueue(&sim.Request{ID: "b3", SLOClass: "standard", TenantID: "B"}, 6)
	q.Enqueue(&sim.Request{ID: "c1", SLOClass: "standard", TenantID: "C"}, 7)
	q.Enqueue(&sim.Request{ID: "c2", SLOClass: "standard", TenantID: "C"}, 8)
	q.Enqueue(&sim.Request{ID: "c3", SLOClass: "standard", TenantID: "C"}, 9)

	// Round-robin in sorted key order: A→B→C→A→B→C→A→B→C
	expected := []string{"a1", "b1", "c1", "a2", "b2", "c2", "a3", "b3", "c3"}
	for i, want := range expected {
		got := q.Dequeue()
		if got == nil || got.ID != want {
			t.Fatalf("dequeue %d: want %s, got %v", i, want, got)
		}
	}
	if q.Len() != 0 {
		t.Errorf("queue should be empty after draining, got Len=%d", q.Len())
	}
}

// BC-3: RoundRobin skips empty flows and advances cursor.
func TestGatewayQueue_RoundRobin_SkipsEmpty(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.SetFairnessPolicy(NewRoundRobinPolicy())

	q.Enqueue(&sim.Request{ID: "a1", SLOClass: "standard", TenantID: "A"}, 1)
	// B has no requests (will be empty after first cycle)
	q.Enqueue(&sim.Request{ID: "c1", SLOClass: "standard", TenantID: "C"}, 2)

	// First cycle: cursor unset → starts at A (sorted first)
	got := q.Dequeue()
	if got.ID != "a1" {
		t.Fatalf("want a1, got %s", got.ID)
	}
	// Cursor at A, advance to B — but B doesn't exist, so skip to C
	got = q.Dequeue()
	if got.ID != "c1" {
		t.Fatalf("want c1, got %s", got.ID)
	}
}

// BC-4: RoundRobin handles removed flow (cursor points to nonexistent tenant).
func TestGatewayQueue_RoundRobin_RemovedFlow(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	rr := NewRoundRobinPolicy()
	q.SetFairnessPolicy(rr)

	q.Enqueue(&sim.Request{ID: "a1", SLOClass: "standard", TenantID: "A"}, 1)
	q.Enqueue(&sim.Request{ID: "b1", SLOClass: "standard", TenantID: "B"}, 2)

	// Dequeue A (cursor now at A)
	got := q.Dequeue()
	if got.ID != "a1" {
		t.Fatalf("want a1, got %s", got.ID)
	}
	// Dequeue B (cursor now at B)
	got = q.Dequeue()
	if got.ID != "b1" {
		t.Fatalf("want b1, got %s", got.ID)
	}

	// Now enqueue only C — cursor points to B which no longer exists.
	q.Enqueue(&sim.Request{ID: "c1", SLOClass: "standard", TenantID: "C"}, 3)
	q.Enqueue(&sim.Request{ID: "d1", SLOClass: "standard", TenantID: "D"}, 4)

	// Cursor at B (removed) → wraps to start → picks C (sorted first)
	got = q.Dequeue()
	if got.ID != "c1" {
		t.Fatalf("want c1 (wrap to start after removed cursor), got %s", got.ID)
	}
	got = q.Dequeue()
	if got.ID != "d1" {
		t.Fatalf("want d1, got %s", got.ID)
	}
}

// RoundRobin with single tenant degenerates to FIFO within that tenant.
func TestGatewayQueue_RoundRobin_SingleTenant(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.SetFairnessPolicy(NewRoundRobinPolicy())

	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard", TenantID: "A"}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard", TenantID: "A"}, 2)
	q.Enqueue(&sim.Request{ID: "r3", SLOClass: "standard", TenantID: "A"}, 3)

	for i, want := range []string{"r1", "r2", "r3"} {
		got := q.Dequeue()
		if got == nil || got.ID != want {
			t.Fatalf("dequeue %d: want %s, got %v", i, want, got)
		}
	}
	if q.Len() != 0 {
		t.Errorf("queue should be empty after draining, got Len=%d", q.Len())
	}
}

// FIFO dispatch-order ignores fairness policy (flat global scan).
func TestGatewayQueue_FIFO_IgnoresFairnessPolicy(t *testing.T) {
	q := NewGatewayQueue("fifo", 0, nil)
	q.SetFairnessPolicy(NewRoundRobinPolicy()) // should have no effect

	// With RoundRobin in priority mode, order would be a1,b1,a2,b2.
	// In FIFO mode, order must be strictly by seqID regardless of tenant.
	q.Enqueue(&sim.Request{ID: "a1", SLOClass: "standard", TenantID: "A"}, 1)
	q.Enqueue(&sim.Request{ID: "a2", SLOClass: "standard", TenantID: "A"}, 2)
	q.Enqueue(&sim.Request{ID: "b1", SLOClass: "standard", TenantID: "B"}, 3)
	q.Enqueue(&sim.Request{ID: "b2", SLOClass: "standard", TenantID: "B"}, 4)

	expected := []string{"a1", "a2", "b1", "b2"} // strict seqID order
	for i, want := range expected {
		got := q.Dequeue()
		if got == nil || got.ID != want {
			t.Fatalf("dequeue %d: want %s, got %v", i, want, got)
		}
	}
}

// TestGatewayQueue_RoundRobin_BandCursorIsolation verifies that each priority
// band maintains its own independent cursor. Draining one band must not affect
// the cursor position in another band.
func TestGatewayQueue_RoundRobin_BandCursorIsolation(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.SetFairnessPolicy(NewRoundRobinPolicy())

	// Critical band: A then B. Dequeue once → picks A, cursor ends at A.
	q.Enqueue(&sim.Request{ID: "cA", SLOClass: "critical", TenantID: "A"}, 1)
	q.Enqueue(&sim.Request{ID: "cB", SLOClass: "critical", TenantID: "B"}, 2)

	// Standard band: A and B. If cursors were shared, cursor="A" from critical
	// would advance standard to B first. With isolated cursors, standard starts
	// fresh and picks A first.
	q.Enqueue(&sim.Request{ID: "sA", SLOClass: "standard", TenantID: "A"}, 3)
	q.Enqueue(&sim.Request{ID: "sB", SLOClass: "standard", TenantID: "B"}, 4)

	// Drain critical: picks cA (cursor now "A" in critical band).
	got := q.Dequeue()
	if got.ID != "cA" {
		t.Fatalf("want cA, got %s", got.ID)
	}
	// Next dequeue: critical still has cB, picks it.
	got = q.Dequeue()
	if got.ID != "cB" {
		t.Fatalf("want cB, got %s", got.ID)
	}
	// Critical drained. Standard band: with isolated cursors, fresh start → picks A.
	// With shared cursor ("B" from critical), would pick A too (wraps). So we need
	// a different setup: drain critical with cursor ending at A, then standard
	// must still pick A (fresh cursor), not B (shared cursor advance).
	// Reset and use asymmetric setup:
	q2 := NewGatewayQueue("priority", 0, nil)
	q2.SetFairnessPolicy(NewRoundRobinPolicy())

	// Critical: only A. Dequeue → cursor ends at "A" in critical band.
	q2.Enqueue(&sim.Request{ID: "cA1", SLOClass: "critical", TenantID: "A"}, 1)
	// Standard: A and B.
	q2.Enqueue(&sim.Request{ID: "sA1", SLOClass: "standard", TenantID: "A"}, 2)
	q2.Enqueue(&sim.Request{ID: "sB1", SLOClass: "standard", TenantID: "B"}, 3)

	got = q2.Dequeue() // critical → cA1, cursor["critical"] = "A"
	if got.ID != "cA1" {
		t.Fatalf("want cA1, got %s", got.ID)
	}
	// Standard band: if cursors shared, cursor="A" → startIndex=(0+1)%2=1 → picks B.
	// If cursors isolated (correct), no cursor for standard → startIndex=0 → picks A.
	got = q2.Dequeue()
	if got.ID != "sA1" {
		t.Fatalf("band cursor isolation: want sA1 (fresh cursor), got %s (shared cursor leak)", got.ID)
	}
	got = q2.Dequeue()
	if got.ID != "sB1" {
		t.Fatalf("want sB1, got %s", got.ID)
	}
	if q2.Len() != 0 {
		t.Errorf("queue should be empty, got Len=%d", q2.Len())
	}
}

// BC-6: RoundRobin works through DequeueGated code path.
func TestGatewayQueue_RoundRobin_WithGatedDispatch(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	q.SetFairnessPolicy(NewRoundRobinPolicy())

	// Two bands: critical (4) and standard (3), each with 2 tenants.
	q.Enqueue(&sim.Request{ID: "cA", SLOClass: "critical", TenantID: "A"}, 1)
	q.Enqueue(&sim.Request{ID: "cB", SLOClass: "critical", TenantID: "B"}, 2)
	q.Enqueue(&sim.Request{ID: "sA", SLOClass: "standard", TenantID: "A"}, 3)
	q.Enqueue(&sim.Request{ID: "sB", SLOClass: "standard", TenantID: "B"}, 4)

	// Saturation 0 → all bands dispatch. Critical band first, round-robin within.
	got := q.DequeueGated(0)
	if got.ID != "cA" {
		t.Fatalf("want cA, got %s", got.ID)
	}
	got = q.DequeueGated(0)
	if got.ID != "cB" {
		t.Fatalf("want cB, got %s", got.ID)
	}
	// Critical band empty → standard band, round-robin within.
	got = q.DequeueGated(0)
	if got.ID != "sA" {
		t.Fatalf("want sA, got %s", got.ID)
	}
	got = q.DequeueGated(0)
	if got.ID != "sB" {
		t.Fatalf("want sB, got %s", got.ID)
	}
}

func TestGatewayQueue_RemoveByRequestID(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("fifo", 0, pm)

	r1 := &sim.Request{ID: "r1", SLOClass: "standard", TenantID: "t1"}
	r2 := &sim.Request{ID: "r2", SLOClass: "standard", TenantID: "t1"}
	r3 := &sim.Request{ID: "r3", SLOClass: "batch", TenantID: "t2"}
	q.Enqueue(r1, 1)
	q.Enqueue(r2, 2)
	q.Enqueue(r3, 3)

	if q.Len() != 3 {
		t.Fatalf("expected 3, got %d", q.Len())
	}

	// Remove middle request from same flow as r1.
	got := q.RemoveByRequestID("r2")
	if got != r2 {
		t.Fatalf("expected r2, got %v", got)
	}
	if q.Len() != 2 {
		t.Fatalf("expected 2 after remove, got %d", q.Len())
	}

	// Remove non-existent → no-op, returns nil.
	got = q.RemoveByRequestID("r999")
	if got != nil {
		t.Fatalf("expected nil for missing ID, got %v", got)
	}
	if q.Len() != 2 {
		t.Fatalf("expected 2 unchanged, got %d", q.Len())
	}

	// Remaining requests dequeue in correct order.
	d1 := q.Dequeue()
	d2 := q.Dequeue()
	if d1.ID != "r1" || d2.ID != "r3" {
		t.Fatalf("expected r1, r3; got %s, %s", d1.ID, d2.ID)
	}
	if q.Len() != 0 {
		t.Fatalf("expected 0, got %d", q.Len())
	}
}

func TestGatewayQueue_RemoveByRequestID_HeadRemoval(t *testing.T) {
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("fifo", 0, pm)

	r1 := &sim.Request{ID: "r1", SLOClass: "standard", TenantID: "t1"}
	r2 := &sim.Request{ID: "r2", SLOClass: "standard", TenantID: "t1"}
	q.Enqueue(r1, 1)
	q.Enqueue(r2, 2)

	// Remove head of flow.
	got := q.RemoveByRequestID("r1")
	if got != r1 {
		t.Fatalf("expected r1, got %v", got)
	}
	if q.Len() != 1 {
		t.Fatalf("expected 1, got %d", q.Len())
	}

	d := q.Dequeue()
	if d.ID != "r2" {
		t.Fatalf("expected r2, got %s", d.ID)
	}
}

func TestGatewayQueue_RemoveByRequestID_WithShed(t *testing.T) {
	// Verify that shed victims are cleaned from the index, so subsequent
	// RemoveByRequestID for the victim returns nil (no double-counting).
	pm := sim.DefaultSLOPriorityMap()
	q := NewGatewayQueue("fifo", 2, pm) // maxDepth=2
	q.SetSheddingEnabled(true)

	r1 := &sim.Request{ID: "r1", SLOClass: "sheddable", TenantID: "t1"}
	r2 := &sim.Request{ID: "r2", SLOClass: "standard", TenantID: "t1"}
	q.Enqueue(r1, 1)
	q.Enqueue(r2, 2)

	// Enqueue a third (higher priority) → should shed r1.
	r3 := &sim.Request{ID: "r3", SLOClass: "critical", TenantID: "t1"}
	outcome, victim := q.Enqueue(r3, 3)
	if outcome != ShedVictim || victim.ID != "r1" {
		t.Fatalf("expected ShedVictim/r1, got %v/%v", outcome, victim)
	}

	// r1 was shed — RemoveByRequestID should return nil.
	got := q.RemoveByRequestID("r1")
	if got != nil {
		t.Fatalf("expected nil for shed request, got %v", got)
	}
}

// --- SLO-deadline ordering tests (BC-1 through BC-5, BC-9) ---

func TestGatewayQueue_SLODeadline_EarliestFirst(t *testing.T) {
	q := NewGatewayQueue("slo-deadline", 0, nil)
	// R1: long SLO target → late deadline. R2: short SLO target → early deadline.
	// Same flow (tenant), same priority (standard).
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard", TenantID: "t1", GatewayEnqueueTime: 100, SLOTargetUs: 500000}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard", TenantID: "t1", GatewayEnqueueTime: 200, SLOTargetUs: 100000}, 2)

	// R2 deadline = 200+100000 = 100200. R1 deadline = 100+500000 = 500100. R2 first.
	r := q.Dequeue()
	if r.ID != "r2" {
		t.Fatalf("expected r2 (earliest deadline), got %s", r.ID)
	}
	r = q.Dequeue()
	if r.ID != "r1" {
		t.Fatalf("expected r1, got %s", r.ID)
	}
}

func TestGatewayQueue_SLODeadline_NoTarget_FarFuture(t *testing.T) {
	q := NewGatewayQueue("slo-deadline", 0, nil)
	// R1 has SLO target. R2 has none (0 → far-future → sorts to back).
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard", TenantID: "t1", GatewayEnqueueTime: 100, SLOTargetUs: 200000}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard", TenantID: "t1", GatewayEnqueueTime: 100, SLOTargetUs: 0}, 2)

	r := q.Dequeue()
	if r.ID != "r1" {
		t.Fatalf("expected r1 (has target), got %s", r.ID)
	}
	r = q.Dequeue()
	if r.ID != "r2" {
		t.Fatalf("expected r2, got %s", r.ID)
	}
}

func TestGatewayQueue_SLODeadline_EqualDeadline_FCFS(t *testing.T) {
	q := NewGatewayQueue("slo-deadline", 0, nil)
	// Same deadline, R1 enqueued first (lower seqID) → R1 dispatches first.
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard", TenantID: "t1", GatewayEnqueueTime: 100, SLOTargetUs: 200000}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard", TenantID: "t1", GatewayEnqueueTime: 100, SLOTargetUs: 200000}, 2)

	r := q.Dequeue()
	if r.ID != "r1" {
		t.Fatalf("expected r1 (lower seqID), got %s", r.ID)
	}
}

func TestGatewayQueue_SLODeadline_WithinFlowOnly(t *testing.T) {
	q := NewGatewayQueue("slo-deadline", 0, nil)
	// Two flows. Flow t1 head has seqID=1, flow t2 head has seqID=2.
	// GlobalStrictPolicy picks t1 (earliest seqID head).
	// t2's request has earlier deadline, but fairness picks t1 first.
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard", TenantID: "t1", GatewayEnqueueTime: 100, SLOTargetUs: 500000}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard", TenantID: "t2", GatewayEnqueueTime: 100, SLOTargetUs: 100000}, 2)

	r := q.Dequeue()
	if r.ID != "r1" {
		t.Fatalf("expected r1 (fairness picks flow t1 by seqID), got %s", r.ID)
	}
}

func TestGatewayQueue_SLODeadline_PerTierFallback(t *testing.T) {
	q := NewGatewayQueue("slo-deadline", 0, nil)
	q.SetSLOTargets(map[string]int64{"critical": 100000})
	// R1: no per-request target but SLOClass=critical → fallback to 100000.
	// R2: no per-request target, SLOClass=standard → no fallback → far-future.
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "critical", TenantID: "t1", GatewayEnqueueTime: 100, SLOTargetUs: 0}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard", TenantID: "t1", GatewayEnqueueTime: 100, SLOTargetUs: 0}, 2)

	r := q.Dequeue()
	if r.ID != "r1" {
		t.Fatalf("expected r1 (per-tier fallback), got %s", r.ID)
	}
}

func TestGatewayQueue_SLODeadline_PerRequestOverridesTier(t *testing.T) {
	q := NewGatewayQueue("slo-deadline", 0, nil)
	q.SetSLOTargets(map[string]int64{"standard": 500000})
	// R1: per-request target 100000 (overrides tier). R2: tier fallback 500000.
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard", TenantID: "t1", GatewayEnqueueTime: 100, SLOTargetUs: 100000}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard", TenantID: "t1", GatewayEnqueueTime: 100, SLOTargetUs: 0}, 2)

	r := q.Dequeue()
	if r.ID != "r1" {
		t.Fatalf("expected r1 (per-request overrides tier), got %s", r.ID)
	}
}

func TestGatewayQueue_SLODeadline_NoTargets_DegeneratesToFCFS(t *testing.T) {
	q := NewGatewayQueue("slo-deadline", 0, nil)
	// No slo-targets set, no per-request targets → all deadlines are MaxInt64 → FCFS by seqID.
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard", TenantID: "t1", GatewayEnqueueTime: 100}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard", TenantID: "t1", GatewayEnqueueTime: 200}, 2)
	q.Enqueue(&sim.Request{ID: "r3", SLOClass: "standard", TenantID: "t1", GatewayEnqueueTime: 50}, 3)

	ids := []string{q.Dequeue().ID, q.Dequeue().ID, q.Dequeue().ID}
	expected := []string{"r1", "r2", "r3"}
	for i, id := range ids {
		if id != expected[i] {
			t.Fatalf("position %d: expected %s, got %s (all deadlines MaxInt64 → FCFS by seqID)", i, expected[i], id)
		}
	}
}

func TestGatewayQueue_SLODeadline_DequeueGated(t *testing.T) {
	q := NewGatewayQueue("slo-deadline", 0, nil)
	q.SetUsageLimitThreshold(0.5)
	// Two requests in same flow, different deadlines.
	q.Enqueue(&sim.Request{ID: "r1", SLOClass: "standard", TenantID: "t1", GatewayEnqueueTime: 100, SLOTargetUs: 500000}, 1)
	q.Enqueue(&sim.Request{ID: "r2", SLOClass: "standard", TenantID: "t1", GatewayEnqueueTime: 100, SLOTargetUs: 100000}, 2)

	// Low saturation → should dispatch, and r2 (earlier deadline) first.
	r := q.DequeueGated(0.3)
	if r == nil {
		t.Fatal("expected non-nil from DequeueGated at low saturation")
	}
	if r.ID != "r2" {
		t.Fatalf("expected r2 (earliest deadline via DequeueGated), got %s", r.ID)
	}

	// Saturation >= ceiling (1.0 for single band) → should block.
	r = q.DequeueGated(1.0)
	if r != nil {
		t.Fatalf("expected nil from DequeueGated at saturation >= ceiling, got %s", r.ID)
	}
}

func TestGatewayQueue_SLODeadline_EmptyQueue(t *testing.T) {
	q := NewGatewayQueue("slo-deadline", 0, nil)
	r := q.Dequeue()
	if r != nil {
		t.Fatalf("expected nil from empty slo-deadline queue, got %v", r)
	}
}
