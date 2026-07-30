package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// These tests cover B-2's D7 freshness override (#1490): route-to-holder pins the
// ResidentAdapters snapshot field to Immediate freshness regardless of the global
// --snapshot-refresh-interval, because holder membership drives an all-or-nothing
// candidate restriction where a stale resident set is a correctness bug (INV-PS1), not
// a quality regression. All assertions are on Snapshot() OUTPUT (never .Mode reflection
// for the wiring tests), so they survive a rewrite of the freshness internals (DV-4).

// hugeInterval is larger than any short test run's sim-clock span, so a Periodic field
// is NEVER refreshed during the run (clock - 0 < hugeInterval at every Snapshot call).
// This makes the clock-0 probe robust to run duration: a Periodic ResidentAdapters
// stays nil, while an Immediate (D7-pinned) one refreshes to the live resident set.
const hugeInterval = int64(1) << 60

// runLoRAClusterWithResident builds and runs a 1-instance cluster whose requests all
// target adapter_x (capacity 8 ⇒ never evicted), so after the run the sole instance
// holds adapter_x. routingPolicy and interval are caller-controlled to exercise the
// D7 construction conditional (§5.4) and freshness override (§5.3). Mirrors the fixture
// in TestBuildRouterState_PopulatesResidentAdapters.
func runLoRAClusterWithResident(t *testing.T, routingPolicy string, interval int64) (*ClusterSimulator, *InstanceSimulator) {
	t.Helper()
	config := newTestDeploymentConfig(1)
	config.RoutingPolicy = routingPolicy
	config.SnapshotRefreshInterval = interval
	capVal := 8
	base, bw, fp := 1000.0, 2.0e6, 2.0e6
	config.LoRAConfig = sim.LoRAConfig{
		AdapterCapacity:       &capVal,
		LoadBaseLatencyUs:     &base,
		LoadBandwidthBytesUs:  &bw,
		FootprintBytesPerRank: &fp,
		Adapters:              []sim.AdapterSpec{{ID: "adapter_x", Rank: 8}},
	}
	requests := newTestRequests(3)
	for _, r := range requests {
		r.Adapter = "adapter_x"
	}
	cs := NewClusterSimulator(config, NewSliceRequestSource(requests), nil)
	mustRun(t, cs)
	instances := cs.Instances()
	require.Len(t, instances, 1)
	require.Contains(t, instances[0].ResidentAdapterIDs(), "adapter_x",
		"precondition: adapter_x must be resident after the run")
	return cs, instances[0]
}

// T2 / BC-7: PinResidentAdaptersImmediate sets ResidentAdapters to Immediate and leaves
// every other field's config untouched. Method-level contract test — asserting on .Mode
// is appropriate here because setting that mode IS the method's whole observable job.
func TestPinResidentAdaptersImmediate_OnlyChangesResidentAdapters(t *testing.T) {
	before := newObservabilityConfig(hugeInterval, 0) // Prometheus fields + ResidentAdapters Periodic
	require.Equal(t, Periodic, before.ResidentAdapters.Mode, "precondition: ResidentAdapters starts Periodic")

	after := before
	after.PinResidentAdaptersImmediate()

	assert.Equal(t, Immediate, after.ResidentAdapters.Mode, "ResidentAdapters must become Immediate")
	assert.Equal(t, before.QueueDepth, after.QueueDepth, "QueueDepth must be unchanged")
	assert.Equal(t, before.BatchSize, after.BatchSize, "BatchSize must be unchanged")
	assert.Equal(t, before.KVUtilization, after.KVUtilization, "KVUtilization must be unchanged")
	assert.Equal(t, before.CacheBlocks, after.CacheBlocks, "CacheBlocks must be unchanged")
	assert.Equal(t, before.PreemptionCount, after.PreemptionCount, "PreemptionCount must be unchanged")
}

// T1 / BC-9 (INV-6 no-op) + the §5.4 construction conditional: with a Periodic global
// interval, a non-route-to-holder cluster leaves ResidentAdapters Periodic (stale nil at
// a sub-interval clock — the no-op direction), while a route-to-holder cluster pins it
// Immediate (live adapter_x at clock 0 — the D7 direction). Behavioral: asserts on the
// cluster provider's Snapshot() output, not on config internals.
func TestClusterD7_PinsResidentAdaptersOnlyForRouteToHolder(t *testing.T) {
	// Control (BC-9 no-op): weighted leaves ResidentAdapters Periodic ⇒ stale nil.
	csW, instW := runLoRAClusterWithResident(t, "weighted", hugeInterval)
	snapW := csW.snapshotProvider.Snapshot(instW.ID(), 0)
	assert.Nil(t, snapW.ResidentAdapters,
		"weighted (no D7): ResidentAdapters Periodic ⇒ stale nil at clock 0 (INV-6 no-op)")

	// Pinned (D7): route-to-holder refreshes ResidentAdapters on every call (Immediate).
	csR, instR := runLoRAClusterWithResident(t, "route-to-holder", hugeInterval)
	snapR := csR.snapshotProvider.Snapshot(instR.ID(), 0)
	assert.True(t, snapR.ResidentAdapters["adapter_x"],
		"route-to-holder (D7): ResidentAdapters Immediate ⇒ live adapter_x at clock 0")
}

// T9 / BC-8 (D7 override, behavioral): a FRESH provider (lastRefresh 0) with the D7 pin
// refreshes ResidentAdapters at clock 0 (Immediate); an identical provider WITHOUT the
// pin does not (Periodic stale at clock 0). Exercises the REAL PinResidentAdaptersImmediate
// + newObservabilityConfig via the proven clock-0 idiom (see
// TestCachedSnapshotProvider_PeriodicInterval); asserts on Snapshot() output (DV-4).
func TestD7FreshnessOverride_ImmediateVsPeriodicAtClockZero(t *testing.T) {
	_, inst := runLoRAClusterWithResident(t, "weighted", hugeInterval) // any policy; we only need a resident instance
	instMap := map[InstanceID]*InstanceSimulator{inst.ID(): inst}

	// Pinned provider (mimics route-to-holder construction, §5.3/§5.4).
	cfgImm := newObservabilityConfig(50_000, 0)
	cfgImm.PinResidentAdaptersImmediate()
	pImm := NewCachedSnapshotProvider(instMap, cfgImm)

	// Control provider (Periodic, no pin).
	cfgPer := newObservabilityConfig(50_000, 0)
	pPer := NewCachedSnapshotProvider(instMap, cfgPer)

	assert.True(t, pImm.Snapshot(inst.ID(), 0).ResidentAdapters["adapter_x"],
		"D7 pin: ResidentAdapters refreshed at clock 0 (Immediate)")
	assert.Nil(t, pPer.Snapshot(inst.ID(), 0).ResidentAdapters,
		"no pin: ResidentAdapters stale nil at clock 0 (Periodic)")
}
