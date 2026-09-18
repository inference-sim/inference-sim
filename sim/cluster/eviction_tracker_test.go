package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestEvictionTracker_TrackAndPop(t *testing.T) {
	tracker := NewEvictionTracker()
	pm := sim.DefaultSLOPriorityMap()

	r1 := &sim.Request{ID: "r1", SLOClass: "sheddable", GatewayDispatchTime: 100}
	r2 := &sim.Request{ID: "r2", SLOClass: "background", GatewayDispatchTime: 200}
	r3 := &sim.Request{ID: "r3", SLOClass: "batch", GatewayDispatchTime: 150}

	tracker.Track(r1, "inst-0", pm) // priority -2
	tracker.Track(r2, "inst-1", pm) // priority -3 (lowest — evict first)
	tracker.Track(r3, "inst-0", pm) // priority -1

	if tracker.Len() != 3 {
		t.Fatalf("expected Len()=3, got %d", tracker.Len())
	}

	// Pop: lowest priority first (background=-3)
	victim, instID := tracker.Pop()
	if victim == nil || victim.ID != "r2" {
		t.Fatalf("expected r2 (background, priority -3), got %v", victim)
	}
	if instID != "inst-1" {
		t.Fatalf("expected inst-1, got %s", instID)
	}

	// Next: sheddable=-2
	victim, instID = tracker.Pop()
	if victim == nil || victim.ID != "r1" {
		t.Fatalf("expected r1 (sheddable, priority -2), got %v", victim)
	}
	if instID != "inst-0" {
		t.Fatalf("expected inst-0, got %s", instID)
	}

	if tracker.Len() != 1 {
		t.Fatalf("expected Len()=1, got %d", tracker.Len())
	}
}

func TestEvictionTracker_SamePriority_NewestDispatchFirst(t *testing.T) {
	tracker := NewEvictionTracker()
	pm := sim.DefaultSLOPriorityMap()

	r1 := &sim.Request{ID: "r1", SLOClass: "sheddable", GatewayDispatchTime: 100}
	r2 := &sim.Request{ID: "r2", SLOClass: "sheddable", GatewayDispatchTime: 200}

	tracker.Track(r1, "inst-0", pm)
	tracker.Track(r2, "inst-0", pm)

	// Same priority → newest dispatch time evicted first
	victim, _ := tracker.Pop()
	if victim == nil || victim.ID != "r2" {
		t.Fatalf("expected r2 (newest dispatch time 200), got %v", victim)
	}
}

func TestEvictionTracker_Untrack(t *testing.T) {
	tracker := NewEvictionTracker()
	pm := sim.DefaultSLOPriorityMap()

	r1 := &sim.Request{ID: "r1", SLOClass: "sheddable", GatewayDispatchTime: 100}
	r2 := &sim.Request{ID: "r2", SLOClass: "sheddable", GatewayDispatchTime: 200}

	tracker.Track(r1, "inst-0", pm)
	tracker.Track(r2, "inst-0", pm)

	tracker.Untrack("r2")

	if tracker.Len() != 1 {
		t.Fatalf("expected Len()=1 after untrack, got %d", tracker.Len())
	}

	victim, _ := tracker.Pop()
	if victim == nil || victim.ID != "r1" {
		t.Fatalf("expected r1 after r2 untracked, got %v", victim)
	}
}

func TestEvictionTracker_NonSheddableNotTracked(t *testing.T) {
	tracker := NewEvictionTracker()
	pm := sim.DefaultSLOPriorityMap()

	r1 := &sim.Request{ID: "r1", SLOClass: "critical", GatewayDispatchTime: 100}
	r2 := &sim.Request{ID: "r2", SLOClass: "standard", GatewayDispatchTime: 200}

	tracker.Track(r1, "inst-0", pm)
	tracker.Track(r2, "inst-0", pm)

	if tracker.Len() != 0 {
		t.Fatalf("expected Len()=0 for non-sheddable requests, got %d", tracker.Len())
	}
}

func TestEvictionTracker_PopEmpty(t *testing.T) {
	tracker := NewEvictionTracker()
	victim, instID := tracker.Pop()
	if victim != nil {
		t.Fatalf("expected nil from empty tracker, got %v", victim)
	}
	if instID != "" {
		t.Fatalf("expected empty instID, got %s", instID)
	}
}

func TestEvictionTracker_UntrackUnknown(t *testing.T) {
	tracker := NewEvictionTracker()
	// Should not panic
	tracker.Untrack("nonexistent")
	if tracker.Len() != 0 {
		t.Fatalf("expected Len()=0, got %d", tracker.Len())
	}
}
