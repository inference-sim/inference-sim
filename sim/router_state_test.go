package sim

import "testing"

func TestRouterState_FieldAccess(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 2, KVUtilization: 0.3, FreeKVBlocks: 100},
		{ID: "instance_1", QueueDepth: 3, BatchSize: 1, KVUtilization: 0.5, FreeKVBlocks: 80},
	}
	state := &RouterState{
		Snapshots: snapshots,
		Clock:     42000,
	}

	if len(state.Snapshots) != 2 {
		t.Errorf("expected 2 snapshots, got %d", len(state.Snapshots))
	}
	if state.Clock != 42000 {
		t.Errorf("expected clock 42000, got %d", state.Clock)
	}
	if state.Snapshots[0].ID != "instance_0" {
		t.Errorf("expected first snapshot ID 'instance_0', got %q", state.Snapshots[0].ID)
	}
}
