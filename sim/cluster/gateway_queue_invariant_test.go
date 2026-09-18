// gateway_queue_invariant_test.go — companion invariant tests for the two
// enforcement-anchored gateway-queue invariants promoted by #1772.
//
// INV-16 (counter consistency) and INV-17 (shed victim is the flow tail) are both
// enforced in production by panics that fire only once the damage has been done: a
// desynced counter is caught when a dequeue comes up empty, and a non-tail shed is
// caught inside removeEntryByIndex. These tests assert the properties DIRECTLY —
// by walking the band/flow structure and comparing it against the counters, and by
// asserting the shed is a truncation — so a regression fails on the property rather
// than on whichever unrelated test happens to drain a queue next.
package cluster

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// walkQueueEntries counts the entries actually held in the band/flow structure and
// returns the total plus a per-band-priority breakdown. It is deliberately independent
// of q.totalLen and band.totalLen — comparing a counter against itself proves nothing.
func walkQueueEntries(q *GatewayQueue) (total int, perBand map[int]int) {
	perBand = make(map[int]int, len(q.bands))
	for _, band := range q.bands {
		for _, flow := range band.flows {
			perBand[band.priority] += len(flow.requests)
			total += len(flow.requests)
		}
	}
	return total, perBand
}

// assertINV16 checks the full counter-consistency statement at one point in time:
// the walked entry count equals q.totalLen, every band's walked count equals its own
// band.totalLen, and requestIndex holds exactly one location per held entry.
func assertINV16(t *testing.T, q *GatewayQueue, after string) {
	t.Helper()
	walked, perBand := walkQueueEntries(q)
	if walked != q.totalLen {
		t.Errorf("INV-16 violated after %s: walked %d entries across bands/flows but totalLen=%d",
			after, walked, q.totalLen)
	}
	for _, band := range q.bands {
		if perBand[band.priority] != band.totalLen {
			t.Errorf("INV-16 violated after %s: band priority=%d holds %d entries but band.totalLen=%d",
				after, band.priority, perBand[band.priority], band.totalLen)
		}
	}
	if len(q.requestIndex) != walked {
		t.Errorf("INV-16 violated after %s: requestIndex has %d locations but %d entries are held",
			after, len(q.requestIndex), walked)
	}
	// An empty flow must not be left behind: it would make a band look non-empty to a
	// walk that trusted len(band.flows) rather than the per-flow lengths.
	for _, band := range q.bands {
		for tid, flow := range band.flows {
			if len(flow.requests) == 0 {
				t.Errorf("INV-16 violated after %s: band priority=%d retains empty flow %q",
					after, band.priority, tid)
			}
		}
	}
}

// TestINV16_CounterConsistency_AcrossMutations drives a mixed sequence of every
// mutation the queue supports — enqueue across several bands and tenants, plain
// dequeue, gated dequeue, capacity shedding, and TTL removal by ID — asserting the
// counter identity after each step. A counter that drifts by one fails here rather
// than surfacing later as a panic in an unrelated test.
func TestINV16_CounterConsistency_AcrossMutations(t *testing.T) {
	q := NewGatewayQueue("priority", 6, nil)
	q.SetSheddingEnabled(true)

	assertINV16(t, q, "construction")

	// Fill across three priority bands and two tenants, so both counter levels and
	// the multi-flow-per-band case are exercised.
	enqueues := []struct {
		id, class, tenant string
	}{
		{"a1", "sheddable", "t1"},
		{"a2", "sheddable", "t1"},
		{"a3", "sheddable", "t2"},
		{"b1", "standard", "t1"},
		{"b2", "standard", "t2"},
		{"c1", "critical", "t1"},
	}
	for _, e := range enqueues {
		outcome, _ := q.Enqueue(&sim.Request{ID: e.id, SLOClass: e.class, TenantID: e.tenant}, int64(len(q.requestIndex)+1))
		if outcome != Enqueued {
			t.Fatalf("setup: enqueue %s returned %v, want Enqueued", e.id, outcome)
		}
		assertINV16(t, q, "enqueue "+e.id)
	}
	if q.Len() != len(enqueues) {
		t.Fatalf("setup: Len()=%d, want %d", q.Len(), len(enqueues))
	}

	// Shed: the queue is at maxDepth, so a critical arrival displaces a sheddable entry.
	outcome, victim := q.Enqueue(&sim.Request{ID: "c2", SLOClass: "critical", TenantID: "t1"}, 100)
	if outcome != ShedVictim || victim == nil {
		t.Fatalf("expected ShedVictim with a victim at maxDepth, got %v/%v", outcome, victim)
	}
	assertINV16(t, q, "shed on enqueue c2")

	// TTL removal by ID (arbitrary position, unlike shedding).
	if removed := q.RemoveByRequestID("b1"); removed == nil {
		t.Fatal("RemoveByRequestID(b1) returned nil, want the request")
	}
	assertINV16(t, q, "RemoveByRequestID b1")

	// A miss must not touch any counter.
	if removed := q.RemoveByRequestID("not-present"); removed != nil {
		t.Fatalf("RemoveByRequestID(not-present) returned %v, want nil", removed)
	}
	assertINV16(t, q, "RemoveByRequestID miss")

	// Drain through both dequeue entry points, alternating, down to empty.
	for i := 0; q.Len() > 0; i++ {
		var got *sim.Request
		if i%2 == 0 {
			got = q.Dequeue()
		} else {
			got = q.DequeueGated(0.0)
		}
		if got == nil {
			t.Fatalf("dequeue %d returned nil with Len()=%d", i, q.Len())
		}
		assertINV16(t, q, "dequeue")
	}
	assertINV16(t, q, "drain to empty")
	if _, perBand := walkQueueEntries(q); len(perBand) != 0 {
		t.Errorf("after draining, walk found entries in %d bands, want 0", len(perBand))
	}
}

// TestINV16_PositiveCounterAlwaysDequeues asserts the corollary the production panics
// check: a dequeue that finds the counter positive returns an entry. It drains a
// multi-band, multi-flow queue through Dequeue and through DequeueGated, and the
// counter must reach zero exactly when the queue is exhausted — never a nil return
// with a positive counter (the panic), and never a positive counter with nothing left.
func TestINV16_PositiveCounterAlwaysDequeues(t *testing.T) {
	for _, mode := range []string{"fifo", "priority", "slo-deadline"} {
		t.Run(mode, func(t *testing.T) {
			for _, gated := range []bool{false, true} {
				name := "Dequeue"
				if gated {
					name = "DequeueGated"
				}
				t.Run(name, func(t *testing.T) {
					q := NewGatewayQueue(mode, 0, nil)
					classes := []string{"critical", "standard", "sheddable", "batch", "background"}
					seq := int64(0)
					for _, class := range classes {
						for _, tenant := range []string{"t1", "t2"} {
							seq++
							q.Enqueue(&sim.Request{
								ID:       class + "-" + tenant,
								SLOClass: class,
								TenantID: tenant,
							}, seq)
						}
					}
					want := len(classes) * 2

					drained := 0
					for q.totalLen > 0 {
						var got *sim.Request
						if gated {
							got = q.DequeueGated(0.0)
						} else {
							got = q.Dequeue()
						}
						if got == nil {
							t.Fatalf("INV-16 violated: totalLen=%d (positive) but dequeue returned nil after %d drains",
								q.totalLen, drained)
						}
						drained++
						if drained > want {
							t.Fatalf("INV-16 violated: drained %d entries from a queue of %d — counter over-counted",
								drained, want)
						}
					}
					if drained != want {
						t.Errorf("INV-16 violated: counter reached zero after %d drains but %d entries were enqueued",
							drained, want)
					}
					// Empty queue: both entry points return nil rather than panicking.
					if got := q.Dequeue(); got != nil {
						t.Errorf("Dequeue on empty queue returned %v, want nil", got)
					}
					if got := q.DequeueGated(0.0); got != nil {
						t.Errorf("DequeueGated on empty queue returned %v, want nil", got)
					}
				})
			}
		})
	}
}

// TestINV17_ShedVictimIsFlowTail asserts that shedding a flow is a TRUNCATION: the
// victim is the highest-seqID entry, the surviving entries are the original prefix in
// the original order, and index 0 — the head Dequeue and the fairness policies read —
// is untouched. A victim-selection change that preferred an older entry would reorder
// the flow's FIFO without moving any counter, which is the failure this pins.
func TestINV17_ShedVictimIsFlowTail(t *testing.T) {
	q := NewGatewayQueue("priority", 3, nil)
	q.SetSheddingEnabled(true)

	// One sheddable flow (same class + tenant ⇒ same band, same flow), appended in
	// seqID order so the tail is unambiguous.
	for i, id := range []string{"s1", "s2", "s3"} {
		outcome, _ := q.Enqueue(&sim.Request{ID: id, SLOClass: "sheddable", TenantID: "t1"}, int64(i+1))
		if outcome != Enqueued {
			t.Fatalf("setup: enqueue %s returned %v", id, outcome)
		}
	}

	// A critical arrival at maxDepth displaces the sheddable tail.
	outcome, victim := q.Enqueue(&sim.Request{ID: "c1", SLOClass: "critical", TenantID: "t1"}, 10)
	if outcome != ShedVictim {
		t.Fatalf("expected ShedVictim, got %v", outcome)
	}
	if victim == nil || victim.ID != "s3" {
		t.Fatalf("INV-17 violated: shed victim = %v, want the flow tail s3", victim)
	}

	// The survivors are the original prefix, in order, with the head intact.
	sheddablePriority := sim.DefaultSLOPriorityMap().Priority("sheddable")
	var flow *flowQueue
	for _, band := range q.bands {
		if band.priority == sheddablePriority {
			flow = band.flows["t1"]
		}
	}
	if flow == nil {
		t.Fatal("sheddable flow disappeared after shedding")
	}
	gotIDs := make([]string, 0, len(flow.requests))
	for _, e := range flow.requests {
		gotIDs = append(gotIDs, e.request.ID)
	}
	if len(gotIDs) != 2 || gotIDs[0] != "s1" || gotIDs[1] != "s2" {
		t.Errorf("INV-17 violated: surviving flow = %v, want [s1 s2] (a truncation preserving the head)", gotIDs)
	}
	// The shed entry must also be gone from the ID index (INV-16's companion clause).
	if _, stillIndexed := q.requestIndex["s3"]; stillIndexed {
		t.Error("shed victim s3 is still present in requestIndex")
	}
	assertINV16(t, q, "shed of flow tail")
}

// TestINV17_NonTailRemovalPanics asserts the guard is live, not dead code: removing a
// non-tail index panics rather than silently shifting the flow. Without this, the
// truncation in removeEntryByIndex would be an unasserted assumption.
func TestINV17_NonTailRemovalPanics(t *testing.T) {
	q := NewGatewayQueue("priority", 0, nil)
	for i, id := range []string{"s1", "s2", "s3"} {
		q.Enqueue(&sim.Request{ID: id, SLOClass: "sheddable", TenantID: "t1"}, int64(i+1))
	}
	sheddablePriority := sim.DefaultSLOPriorityMap().Priority("sheddable")
	var band *priorityBand
	for _, b := range q.bands {
		if b.priority == sheddablePriority {
			band = b
		}
	}
	if band == nil {
		t.Fatal("sheddable band not created")
	}
	flow := band.flows["t1"]

	for _, idx := range []int{0, 1} {
		t.Run(fmt.Sprintf("idx%d", idx), func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("INV-17 guard did not fire: removeEntryByIndex(idx=%d) on a 3-entry flow should panic", idx)
				}
			}()
			q.removeEntryByIndex(flow, band, idx)
		})
	}

	// The tail index is accepted — the guard rejects only non-tail removal.
	q.removeEntryByIndex(flow, band, len(flow.requests)-1)
	assertINV16(t, q, "tail removal")
}
