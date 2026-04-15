package cluster

import (
	"testing"
)

// TestPreemptionCount_Accessor_SurfacesMetric verifies BC-1:
// PreemptionCount() returns the instance's cumulative preemption count.
func TestPreemptionCount_Accessor_SurfacesMetric(t *testing.T) {
	inst := newTestInstance("inst_0", 100)
	inst.sim.Metrics.PreemptionCount = 7

	if got := inst.PreemptionCount(); got != 7 {
		t.Errorf("PreemptionCount() = %d, want 7", got)
	}
}

// TestPreemptionCount_Snapshot_AlwaysImmediate verifies BC-2:
// Snapshot() injects PreemptionCount unconditionally regardless of ObservabilityConfig.
func TestPreemptionCount_Snapshot_AlwaysImmediate(t *testing.T) {
	inst := newTestInstance("inst_0", 100)
	inst.sim.Metrics.PreemptionCount = 5

	instances := map[InstanceID]*InstanceSimulator{"inst_0": inst}
	provider := NewCachedSnapshotProvider(instances, DefaultObservabilityConfig())

	snap := provider.Snapshot("inst_0", 0)
	if snap.PreemptionCount != 5 {
		t.Errorf("Snapshot PreemptionCount = %d, want 5", snap.PreemptionCount)
	}

	// Advance count — must reflect on next call regardless of clock
	inst.sim.Metrics.PreemptionCount = 12
	snap2 := provider.Snapshot("inst_0", 0) // same clock, Periodic fields would be stale
	if snap2.PreemptionCount != 12 {
		t.Errorf("Snapshot PreemptionCount after increment = %d, want 12 (must be Immediate)", snap2.PreemptionCount)
	}
}
