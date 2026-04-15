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
// Uses a Periodic config for other fields so the test proves PreemptionCount stays
// Immediate even when the other fields would be stale.
func TestPreemptionCount_Snapshot_AlwaysImmediate(t *testing.T) {
	inst := newTestInstance("inst_0", 100)
	inst.sim.Metrics.PreemptionCount = 5

	instances := map[InstanceID]*InstanceSimulator{"inst_0": inst}

	// All other fields are Periodic with a large interval — they won't refresh on same clock.
	// PreemptionCount is Immediate, so it must always update.
	config := ObservabilityConfig{
		QueueDepth:      FieldConfig{Mode: Periodic, Interval: 1_000_000},
		BatchSize:       FieldConfig{Mode: Periodic, Interval: 1_000_000},
		KVUtilization:   FieldConfig{Mode: Periodic, Interval: 1_000_000},
		PreemptionCount: FieldConfig{Mode: Immediate},
	}
	provider := NewCachedSnapshotProvider(instances, config)

	snap := provider.Snapshot("inst_0", 0)
	if snap.PreemptionCount != 5 {
		t.Errorf("Snapshot PreemptionCount = %d, want 5", snap.PreemptionCount)
	}

	// Advance count — must reflect on next call at same clock (Periodic fields would be stale)
	inst.sim.Metrics.PreemptionCount = 12
	snap2 := provider.Snapshot("inst_0", 0) // same clock, Periodic interval not elapsed
	if snap2.PreemptionCount != 12 {
		t.Errorf("Snapshot PreemptionCount after increment = %d, want 12 (must be Immediate)", snap2.PreemptionCount)
	}
}

// TestPreemptionCount_RefreshAll_SnapshotRecovery verifies BC-3:
// After RefreshAll(), the next Snapshot() call returns the correct PreemptionCount.
func TestPreemptionCount_RefreshAll_SnapshotRecovery(t *testing.T) {
	inst := newTestInstance("inst_0", 100)
	inst.sim.Metrics.PreemptionCount = 5

	instances := map[InstanceID]*InstanceSimulator{"inst_0": inst}
	provider := NewCachedSnapshotProvider(instances, DefaultObservabilityConfig())

	provider.RefreshAll(0)

	snap := provider.Snapshot("inst_0", 0)
	if snap.PreemptionCount != 5 {
		t.Errorf("Snapshot PreemptionCount after RefreshAll = %d, want 5", snap.PreemptionCount)
	}
}

// TestPreemptionCount_AddInstance_SnapshotReadsLive verifies BC-4:
// After AddInstance(), the first Snapshot() returns the live PreemptionCount (not zero).
func TestPreemptionCount_AddInstance_SnapshotReadsLive(t *testing.T) {
	inst := newTestInstance("inst_0", 100)
	inst.sim.Metrics.PreemptionCount = 3

	provider := NewCachedSnapshotProvider(map[InstanceID]*InstanceSimulator{}, DefaultObservabilityConfig())
	provider.AddInstance("inst_0", inst)

	snap := provider.Snapshot("inst_0", 0)
	if snap.PreemptionCount != 3 {
		t.Errorf("Snapshot PreemptionCount after AddInstance = %d, want 3", snap.PreemptionCount)
	}
}
