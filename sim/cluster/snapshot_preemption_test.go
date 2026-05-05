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
// When ObservabilityConfig.PreemptionCount is configured as Immediate,
// Snapshot() re-reads it on every call — even at the same clock tick
// where other Periodic fields would not refresh.
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

	// Advance count — must reflect on next call at same clock (Periodic interval not elapsed)
	inst.sim.Metrics.PreemptionCount = 12
	snap2 := provider.Snapshot("inst_0", 0)
	if snap2.PreemptionCount != 12 {
		t.Errorf("Snapshot PreemptionCount after increment = %d, want 12 (must be Immediate)", snap2.PreemptionCount)
	}
}

// TestPreemptionCount_Snapshot_Periodic_GoesStale verifies BC-2b:
// When ObservabilityConfig.PreemptionCount is configured as Periodic,
// Snapshot() returns the cached value until the interval elapses.
func TestPreemptionCount_Snapshot_Periodic_GoesStale(t *testing.T) {
	inst := newTestInstance("inst_0", 100)
	inst.sim.Metrics.PreemptionCount = 5

	instances := map[InstanceID]*InstanceSimulator{"inst_0": inst}
	config := newObservabilityConfig(1_000_000, 0) // all fields Periodic with 1s interval; no cache delay
	provider := NewCachedSnapshotProvider(instances, config)

	// Advance clock to interval boundary — triggers the first read (1_000_000 - 0 >= 1_000_000)
	snap1 := provider.Snapshot("inst_0", 1_000_000)
	if snap1.PreemptionCount != 5 {
		t.Errorf("first Periodic read Snapshot PreemptionCount = %d, want 5", snap1.PreemptionCount)
	}

	// Advance counter — interval has not elapsed yet (same clock), so value should be stale
	inst.sim.Metrics.PreemptionCount = 12
	snap2 := provider.Snapshot("inst_0", 1_000_000)
	if snap2.PreemptionCount != 5 {
		t.Errorf("Periodic stale Snapshot PreemptionCount = %d, want 5 (interval not elapsed)", snap2.PreemptionCount)
	}

	// Advance clock past the next interval boundary — should now refresh
	snap3 := provider.Snapshot("inst_0", 2_000_000)
	if snap3.PreemptionCount != 12 {
		t.Errorf("Snapshot PreemptionCount after interval elapsed = %d, want 12", snap3.PreemptionCount)
	}
}

// TestPreemptionCount_RefreshAll_SnapshotRecovery verifies BC-3:
// RefreshAll() writes the live PreemptionCount into the cache, and
// Snapshot() returns it even when ObservabilityConfig.PreemptionCount is OnDemand.
func TestPreemptionCount_RefreshAll_SnapshotRecovery(t *testing.T) {
	inst := newTestInstance("inst_0", 100)
	inst.sim.Metrics.PreemptionCount = 5

	instances := map[InstanceID]*InstanceSimulator{"inst_0": inst}
	config := ObservabilityConfig{
		QueueDepth:      FieldConfig{Mode: Immediate},
		BatchSize:       FieldConfig{Mode: Immediate},
		KVUtilization:   FieldConfig{Mode: Immediate},
		PreemptionCount: FieldConfig{Mode: OnDemand}, // only updated via RefreshAll
	}
	provider := NewCachedSnapshotProvider(instances, config)

	// Before RefreshAll: OnDemand field starts at zero (never read yet)
	snap0 := provider.Snapshot("inst_0", 0)
	if snap0.PreemptionCount != 0 {
		t.Errorf("OnDemand PreemptionCount before RefreshAll = %d, want 0", snap0.PreemptionCount)
	}

	// After RefreshAll: live value should be written to cache
	provider.RefreshAll(0)
	snap1 := provider.Snapshot("inst_0", 0)
	if snap1.PreemptionCount != 5 {
		t.Errorf("Snapshot PreemptionCount after RefreshAll = %d, want 5", snap1.PreemptionCount)
	}

	// Advance counter without RefreshAll — OnDemand stays stale
	inst.sim.Metrics.PreemptionCount = 10
	snap2 := provider.Snapshot("inst_0", 0)
	if snap2.PreemptionCount != 5 {
		t.Errorf("OnDemand Snapshot PreemptionCount without RefreshAll = %d, want 5 (stale)", snap2.PreemptionCount)
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
