package cluster

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inference-sim/inference-sim/sim"
)

// newTestInstance creates a minimal InstanceSimulator for snapshot tests.
func newTestInstance(id InstanceID, totalKVBlocks int64) *InstanceSimulator {
	cfg := sim.SimConfig{
		Horizon:             1000000,
		Seed:                42,
		KVCacheConfig:       sim.NewKVCacheConfig(totalKVBlocks, 16, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test", "H100", 1, "blackbox", 0),
	}
	return NewInstanceSimulator(id, cfg)
}

// TestSnapshot_Immutability verifies BC-5, NC-2:
// GIVEN a snapshot taken from an instance
// WHEN the instance state subsequently changes
// THEN the snapshot values remain unchanged (value-type semantics)
func TestSnapshot_Immutability(t *testing.T) {
	inst := newTestInstance("snap-test", 100)

	// Inject a request to change instance state
	req := &sim.Request{
		ID:           "req_0",
		ArrivalTime:  0,
		InputTokens:  make([]int, 50),
		OutputTokens: make([]int, 10),
		State:        sim.StateQueued,
	}
	inst.InjectRequest(req)

	instances := map[InstanceID]*InstanceSimulator{"snap-test": inst}
	provider := NewCachedSnapshotProvider(instances, DefaultObservabilityConfig())

	snap1 := provider.Snapshot("snap-test", 0)
	snap1QD := snap1.QueueDepth

	// Now inject another request to change state
	req2 := &sim.Request{
		ID:           "req_1",
		ArrivalTime:  100,
		InputTokens:  make([]int, 30),
		OutputTokens: make([]int, 5),
		State:        sim.StateQueued,
	}
	inst.InjectRequest(req2)

	// Take a new snapshot — should reflect new state
	_ = provider.Snapshot("snap-test", 100)

	// snap1 should NOT have changed (value-type semantics)
	if snap1.QueueDepth != snap1QD {
		t.Errorf("snap1.QueueDepth changed from %d to %d — value semantics violated", snap1QD, snap1.QueueDepth)
	}

	// Verify ID is correct string type
	if snap1.ID != "snap-test" {
		t.Errorf("snap1.ID = %q, want %q", snap1.ID, "snap-test")
	}
}

// TestCachedSnapshotProvider_RefreshBehavior verifies BC-6:
// GIVEN a CachedSnapshotProvider with mixed Immediate/Periodic/OnDemand fields
// WHEN Snapshot() is called at different clock times
// THEN Immediate re-reads every time, Periodic respects interval, OnDemand only via RefreshAll
//
// Uses TotalKvCapacityTokens (always non-zero from instance, zero in default snapshot) as the
// Periodic observable to avoid vacuous 0==0 assertions.
func TestCachedSnapshotProvider_RefreshBehavior(t *testing.T) {
	inst := newTestInstance("refresh-test", 100) // 100 blocks × 16 tokens/block = 1600

	instances := map[InstanceID]*InstanceSimulator{"refresh-test": inst}

	config := ObservabilityConfig{
		QueueDepth:    FieldConfig{Mode: Immediate},
		BatchSize:     FieldConfig{Mode: OnDemand},
		KVUtilization: FieldConfig{Mode: Periodic, Interval: 1000},
		CacheBlocks:   FieldConfig{Mode: Immediate},
	}
	provider := NewCachedSnapshotProvider(instances, config)

	// Verify live TotalKvCapacityTokens is non-zero so Periodic assertions are non-vacuous.
	liveCapacity := inst.TotalKvCapacityTokens()
	require.Greater(t, liveCapacity, int64(0),
		"need non-zero TotalKvCapacityTokens to distinguish stale from live")

	// First snapshot at clock=0: Periodic NOT refreshed (0-0 < 1000).
	snap0 := provider.Snapshot("refresh-test", 0)

	// Immediate field (QueueDepth) always reads live (QueueDepth=0 for idle instance).
	assert.Equal(t, 0, snap0.QueueDepth, "Immediate QueueDepth at clock=0")

	// Periodic field (KVUtilization group) NOT refreshed — TotalKvCapacityTokens is default 0.
	assert.Equal(t, int64(0), snap0.TotalKvCapacityTokens,
		"Periodic TotalKvCapacityTokens at clock=0 should be 0 (not yet refreshed)")

	// OnDemand field (BatchSize) NOT refreshed — stays at default 0.
	assert.Equal(t, 0, snap0.BatchSize,
		"OnDemand BatchSize at clock=0 should be 0 (never refreshed)")

	// Snapshot at clock=500 — Periodic should NOT refresh (500-0 < 1000).
	snap500 := provider.Snapshot("refresh-test", 500)
	assert.Equal(t, int64(0), snap500.TotalKvCapacityTokens,
		"Periodic TotalKvCapacityTokens at clock=500 should still be 0 (interval not elapsed)")

	// Snapshot at clock=1000 — Periodic should refresh (1000-0 >= 1000).
	snap1000 := provider.Snapshot("refresh-test", 1000)
	assert.Equal(t, liveCapacity, snap1000.TotalKvCapacityTokens,
		"Periodic TotalKvCapacityTokens at clock=1000 should match live value (interval elapsed)")

	// OnDemand (BatchSize) should still be 0 — not refreshed by Snapshot().
	assert.Equal(t, 0, snap1000.BatchSize,
		"OnDemand BatchSize at clock=1000 should still be 0 (only RefreshAll updates it)")

	// After RefreshAll, OnDemand fields should be updated.
	provider.RefreshAll(2000)
	snapAfterRefresh := provider.Snapshot("refresh-test", 2000)
	assert.Equal(t, liveCapacity, snapAfterRefresh.TotalKvCapacityTokens,
		"TotalKvCapacityTokens after RefreshAll should match live value")
	// BatchSize reads live after RefreshAll (0 for idle instance — but now it was actually read).
	assert.Equal(t, 0, snapAfterRefresh.BatchSize,
		"BatchSize after RefreshAll should reflect live state (0 for idle instance)")
}

// TestCachedSnapshotProvider_PeriodicInterval verifies that Periodic mode
// only refreshes when the configured interval has elapsed, using TotalKvCapacityTokens
// as the observable signal (constant non-zero value from instance, zero in default snapshot).
func TestCachedSnapshotProvider_PeriodicInterval(t *testing.T) {
	inst := newTestInstance("periodic-test", 100) // 100 blocks × 16 tokens/block = 1600
	instances := map[InstanceID]*InstanceSimulator{"periodic-test": inst}

	config := ObservabilityConfig{
		QueueDepth:    FieldConfig{Mode: Immediate},
		BatchSize:     FieldConfig{Mode: Immediate},
		KVUtilization: FieldConfig{Mode: Periodic, Interval: 100},
		CacheBlocks:   FieldConfig{Mode: Immediate},
	}
	provider := NewCachedSnapshotProvider(instances, config)

	// Verify the live value is non-zero so our assertions are non-vacuous.
	liveCapacity := inst.TotalKvCapacityTokens()
	require.Greater(t, liveCapacity, int64(0),
		"need non-zero TotalKvCapacityTokens to distinguish stale from live")

	// Initial Periodic snapshot at clock=0: lastRefresh=0, 0-0=0 < 100, so NOT refreshed.
	// TotalKvCapacityTokens stays at default 0.
	snap0 := provider.Snapshot("periodic-test", 0)
	assert.Equal(t, int64(0), snap0.TotalKvCapacityTokens,
		"TotalKvCapacityTokens at clock=0 should be 0 (not yet refreshed)")

	// At clock=99: should NOT refresh (99-0 < 100) — still sees stale default 0.
	snap99 := provider.Snapshot("periodic-test", 99)
	assert.Equal(t, int64(0), snap99.TotalKvCapacityTokens,
		"TotalKvCapacityTokens at clock=99 should still be 0 (interval not elapsed)")

	// At clock=100: should refresh (100-0 >= 100) — reads live TotalKvCapacityTokens.
	snap100 := provider.Snapshot("periodic-test", 100)
	assert.Equal(t, liveCapacity, snap100.TotalKvCapacityTokens,
		"TotalKvCapacityTokens at clock=100 should match live value (interval elapsed)")
}

// TestSnapshotProvider_DefaultConfig_AllImmediate verifies BC-7:
// GIVEN DefaultObservabilityConfig()
// THEN all fields are configured as Immediate mode
func TestSnapshotProvider_DefaultConfig_AllImmediate(t *testing.T) {
	config := DefaultObservabilityConfig()

	tests := []struct {
		name string
		fc   FieldConfig
	}{
		{"QueueDepth", config.QueueDepth},
		{"BatchSize", config.BatchSize},
		{"KVUtilization", config.KVUtilization},
		{"CacheBlocks", config.CacheBlocks},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.fc.Mode != Immediate {
				t.Errorf("Mode = %d, want Immediate (%d)", tc.fc.Mode, Immediate)
			}
		})
	}
}

// TestNewObservabilityConfig_ZeroAndNegativeInterval_AllImmediate verifies BC-2 and EC-1:
// GIVEN zero or negative refresh intervals
// WHEN newObservabilityConfig is called
// THEN all fields use Immediate mode (backward-compatible default)
func TestNewObservabilityConfig_ZeroAndNegativeInterval_AllImmediate(t *testing.T) {
	for _, interval := range []int64{0, -1, -100} {
		t.Run(fmt.Sprintf("interval=%d", interval), func(t *testing.T) {
			config := newObservabilityConfig(interval, 0)
			for _, f := range []struct {
				name string
				fc   FieldConfig
			}{
				{"QueueDepth", config.QueueDepth},
				{"BatchSize", config.BatchSize},
				{"KVUtilization", config.KVUtilization},
				{"CacheBlocks", config.CacheBlocks},
			} {
				if f.fc.Mode != Immediate {
					t.Errorf("%s: Mode = %d, want Immediate (%d)", f.name, f.fc.Mode, Immediate)
				}
			}
		})
	}
}

// TestNewObservabilityConfig_NonZeroInterval_AllFieldsPeriodic verifies BC-1:
// GIVEN a non-zero refresh interval
// WHEN newObservabilityConfig is called
// THEN all three fields (QueueDepth, BatchSize, KVUtilization) use Periodic mode
// with the same interval.
func TestNewObservabilityConfig_NonZeroInterval_AllFieldsPeriodic(t *testing.T) {
	config := newObservabilityConfig(5000, 0) // 5ms

	fields := []struct {
		name string
		fc   FieldConfig
	}{
		{"QueueDepth", config.QueueDepth},
		{"BatchSize", config.BatchSize},
		{"KVUtilization", config.KVUtilization},
	}
	for _, f := range fields {
		t.Run(f.name, func(t *testing.T) {
			if f.fc.Mode != Periodic {
				t.Errorf("Mode = %d, want Periodic (%d)", f.fc.Mode, Periodic)
			}
			if f.fc.Interval != 5000 {
				t.Errorf("Interval = %d, want 5000", f.fc.Interval)
			}
		})
	}
}

// TestCachedSnapshotProvider_AddInstance verifies that AddInstance dynamically
// registers a new instance so that subsequent Snapshot calls return a valid snapshot,
// and panics when called again with the same ID.
func TestCachedSnapshotProvider_AddInstance(t *testing.T) {
	// GIVEN a CachedSnapshotProvider initialized with one instance
	inst := newTestInstance("existing", 100)
	instances := map[InstanceID]*InstanceSimulator{"existing": inst}
	provider := NewCachedSnapshotProvider(instances, DefaultObservabilityConfig())

	// WHEN AddInstance is called with a new ID
	newInst := newTestInstance("new-inst", 64)
	provider.AddInstance("new-inst", newInst)

	// THEN subsequent Snapshot calls return a valid (non-zero ID) snapshot for the new instance
	snap := provider.Snapshot("new-inst", 0)
	if snap.ID != "new-inst" {
		t.Errorf("Snapshot after AddInstance: ID = %q, want %q", snap.ID, "new-inst")
	}

	// WHEN AddInstance is called again with the same ID
	// THEN it panics
	panicked := false
	func() {
		defer func() {
			if r := recover(); r != nil {
				panicked = true
			}
		}()
		provider.AddInstance("new-inst", newInst)
	}()
	if !panicked {
		t.Error("AddInstance with duplicate ID did not panic")
	}
}

// TestCachedSnapshotProvider_ImmediateAlwaysReadsLive verifies Immediate mode
// re-reads from the instance on every Snapshot() call.
func TestCachedSnapshotProvider_ImmediateAlwaysReadsLive(t *testing.T) {
	inst := newTestInstance("imm-test", 100)
	instances := map[InstanceID]*InstanceSimulator{"imm-test": inst}
	provider := NewCachedSnapshotProvider(instances, DefaultObservabilityConfig())

	// Snapshot before any state change
	snap1 := provider.Snapshot("imm-test", 0)
	if snap1.FreeKVBlocks != 100 {
		t.Errorf("initial FreeKVBlocks = %d, want 100", snap1.FreeKVBlocks)
	}

	// Snapshot at a later time — still reflects current state
	snap2 := provider.Snapshot("imm-test", 1000)
	if snap2.FreeKVBlocks != 100 {
		t.Errorf("FreeKVBlocks at clock=1000 = %d, want 100", snap2.FreeKVBlocks)
	}
}

// --- Task 1 tests (BC-1, BC-2, BC-3) ---

func TestObservabilityConfig_CacheBlocks_DefaultImmediate(t *testing.T) {
	// BC-3: When cache delay is 0, CacheBlocks uses Immediate mode.
	config := newObservabilityConfig(0, 0)
	if config.CacheBlocks.Mode != Immediate {
		t.Errorf("CacheBlocks.Mode = %d, want Immediate (%d)", config.CacheBlocks.Mode, Immediate)
	}
}

func TestObservabilityConfig_CacheBlocks_Periodic(t *testing.T) {
	// BC-1: When cache delay > 0, CacheBlocks uses Periodic mode with given interval.
	config := newObservabilityConfig(0, 50_000)
	if config.CacheBlocks.Mode != Periodic {
		t.Errorf("CacheBlocks.Mode = %d, want Periodic (%d)", config.CacheBlocks.Mode, Periodic)
	}
	if config.CacheBlocks.Interval != 50_000 {
		t.Errorf("CacheBlocks.Interval = %d, want 50000", config.CacheBlocks.Interval)
	}
}

func TestObservabilityConfig_CacheBlocks_IndependentOfSnapshot(t *testing.T) {
	// BC-1: CacheBlocks interval is independent of snapshot refresh interval.
	config := newObservabilityConfig(10_000, 50_000)
	if config.QueueDepth.Mode != Periodic {
		t.Errorf("QueueDepth.Mode = %d, want Periodic", config.QueueDepth.Mode)
	}
	if config.QueueDepth.Interval != 10_000 {
		t.Errorf("QueueDepth.Interval = %d, want 10000", config.QueueDepth.Interval)
	}
	if config.CacheBlocks.Mode != Periodic {
		t.Errorf("CacheBlocks.Mode = %d, want Periodic", config.CacheBlocks.Mode)
	}
	if config.CacheBlocks.Interval != 50_000 {
		t.Errorf("CacheBlocks.Interval = %d, want 50000", config.CacheBlocks.Interval)
	}
}

// TestCachedSnapshotProvider_DualTimers_IndependentRefresh verifies that
// SnapshotRefreshInterval and CacheSignalDelay fire on independent schedules.
// Snapshot() refreshes KVUtilization-group fields on one timer; RefreshCacheIfNeeded
// refreshes cache block snapshots on a separate timer.
//
// Uses TotalKvCapacityTokens (constant non-zero from instance, zero in default snapshot)
// for the scalar timer, and CacheQuery for the cache timer.
func TestCachedSnapshotProvider_DualTimers_IndependentRefresh(t *testing.T) {
	cfg := newTestSimConfig()
	cfg.Horizon = 10_000_000
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-0", cfg)
	instances := map[InstanceID]*InstanceSimulator{"inst-0": inst}

	// Snapshot interval = 200µs, cache interval = 500µs.
	obsConfig := newObservabilityConfig(200, 500)
	provider := NewCachedSnapshotProvider(instances, obsConfig)

	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}

	// Run a request to populate cache blocks.
	req := &sim.Request{
		ID: "r1", ArrivalTime: 0, InputTokens: tokens,
		OutputTokens: []int{100}, State: sim.StateQueued,
	}
	inst.InjectRequest(req)
	inst.Run()
	require.Greater(t, inst.GetCachedBlockCount(tokens), 0, "live cache must have blocks")

	// Verify live TotalKvCapacityTokens is non-zero so scalar assertions are non-vacuous.
	liveCapacity := inst.TotalKvCapacityTokens()
	require.Greater(t, liveCapacity, int64(0),
		"need non-zero TotalKvCapacityTokens to distinguish stale from live")

	// At clock=200: snapshot interval elapsed (200-0 >= 200), cache interval NOT (200-0 < 500).
	// KVUtilization-group (TotalKvCapacityTokens) should refresh; cache blocks should NOT.
	provider.RefreshCacheIfNeeded(200)
	snap200 := provider.Snapshot("inst-0", 200)
	assert.Equal(t, liveCapacity, snap200.TotalKvCapacityTokens,
		"TotalKvCapacityTokens should refresh at clock=200 (snapshot interval elapsed)")
	assert.Equal(t, 0, provider.CacheQuery("inst-0", tokens),
		"cache blocks should NOT refresh at clock=200 (cache interval not elapsed)")

	// At clock=499: snapshot fires again (499-200 >= 200), cache still NOT (499-0 < 500).
	provider.RefreshCacheIfNeeded(499)
	assert.Equal(t, 0, provider.CacheQuery("inst-0", tokens),
		"cache blocks should NOT refresh at clock=499 (cache interval not elapsed)")

	// At clock=500: cache interval elapsed (500-0 >= 500), cache blocks should refresh.
	provider.RefreshCacheIfNeeded(500)
	assert.Greater(t, provider.CacheQuery("inst-0", tokens), 0,
		"cache blocks should refresh at clock=500 (cache interval elapsed)")
}

// --- Task 2 tests (BC-1, BC-3, BC-4) ---

func TestCachedSnapshotProvider_CacheQuery_StaleUntilRefresh(t *testing.T) {
	// BC-1: Cache queries return stale data until refresh interval elapses.
	cfg := newTestSimConfig()
	cfg.Horizon = 10_000_000
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-0", cfg)
	instances := map[InstanceID]*InstanceSimulator{"inst-0": inst}

	obsConfig := newObservabilityConfig(0, 1000) // cache delay 1000µs
	provider := NewCachedSnapshotProvider(instances, obsConfig)

	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}

	// Initial snapshot — empty cache
	assert.Equal(t, 0, provider.CacheQuery("inst-0", tokens))

	// Populate cache via request
	req := &sim.Request{
		ID: "r1", ArrivalTime: 0, InputTokens: tokens,
		OutputTokens: []int{100}, State: sim.StateQueued,
	}
	inst.InjectRequest(req)
	inst.Run()
	require.Greater(t, inst.GetCachedBlockCount(tokens), 0)

	// Before refresh interval: still stale
	provider.RefreshCacheIfNeeded(500) // 500 < 1000
	assert.Equal(t, 0, provider.CacheQuery("inst-0", tokens))

	// After refresh interval: sees new data
	provider.RefreshCacheIfNeeded(1000) // 1000 >= 1000
	assert.Greater(t, provider.CacheQuery("inst-0", tokens), 0)
}

func TestCachedSnapshotProvider_CacheQuery_OracleMode(t *testing.T) {
	// BC-3: When CacheBlocks is Immediate, queries return live data.
	cfg := newTestSimConfig()
	cfg.Horizon = 10_000_000
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-0", cfg)
	instances := map[InstanceID]*InstanceSimulator{"inst-0": inst}

	obsConfig := newObservabilityConfig(0, 0) // oracle mode
	provider := NewCachedSnapshotProvider(instances, obsConfig)

	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}
	req := &sim.Request{
		ID: "r1", ArrivalTime: 0, InputTokens: tokens,
		OutputTokens: []int{100}, State: sim.StateQueued,
	}
	inst.InjectRequest(req)
	inst.Run()

	// Oracle mode: sees live data immediately without refresh
	assert.Greater(t, provider.CacheQuery("inst-0", tokens), 0)
}

func TestCachedSnapshotProvider_AddRemoveCacheInstance(t *testing.T) {
	// BC-4: Dynamic instance add/remove for cache queries.
	obsConfig := newObservabilityConfig(0, 1000)
	provider := NewCachedSnapshotProvider(nil, obsConfig)

	cfg := newTestSimConfig()
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-new", cfg)

	// AddInstance registers for scalar snapshots; AddCacheInstance for cache tracking.
	provider.AddInstance("inst-new", inst)
	provider.AddCacheInstance("inst-new", inst)
	tokens := []int{1, 2, 3, 4}
	assert.Equal(t, 0, provider.CacheQuery("inst-new", tokens)) // empty cache

	provider.RemoveCacheInstance("inst-new")
	assert.Equal(t, 0, provider.CacheQuery("inst-new", tokens)) // returns 0 for unknown
}

func TestCachedSnapshotProvider_IsStaleCacheMode(t *testing.T) {
	// IsStaleCacheMode gates the stale vs oracle dispatch in registerInstanceCacheQueryFn.
	assert.False(t, NewCachedSnapshotProvider(nil, newObservabilityConfig(0, 0)).IsStaleCacheMode(),
		"oracle mode (cacheDelay=0) should return false")
	assert.True(t, NewCachedSnapshotProvider(nil, newObservabilityConfig(0, 1000)).IsStaleCacheMode(),
		"stale mode (cacheDelay>0) should return true")
}

// --- Ported tests from StaleCacheIndex (stale cache management on CachedSnapshotProvider) ---

func TestCachedSnapshotProvider_AddCacheInstance_HonorsStaleBoundary(t *testing.T) {
	// Ported from TestStaleCacheIndex_AddInstance_HonorsStaleBoundary.
	// The stale-mode deferred-instance path: AddCacheInstance registers an instance
	// with a frozen snapshot. The resulting closure must honor stale semantics — snapshot
	// frozen at registration time, updated only by RefreshCacheIfNeeded.

	cfg := newTestSimConfig()
	cfg.Horizon = 10_000_000
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-deferred", cfg)

	obsConfig := newObservabilityConfig(0, 1000) // stale mode, interval=1000
	provider := NewCachedSnapshotProvider(nil, obsConfig)
	provider.AddInstance("inst-deferred", inst)
	provider.AddCacheInstance("inst-deferred", inst) // snapshot taken here: cache empty

	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}
	cqf := provider.BuildCacheQueryFn()

	// Populate cache after registration
	req := &sim.Request{
		ID: "r1", ArrivalTime: 0, InputTokens: tokens,
		OutputTokens: []int{100}, State: sim.StateQueued,
	}
	inst.InjectRequest(req)
	inst.Run()
	require.Greater(t, inst.GetCachedBlockCount(tokens), 0, "live cache must have blocks")

	// Before RefreshCacheIfNeeded: stale snapshot is empty — returns 0 (not live state)
	assert.Equal(t, 0, cqf["inst-deferred"](tokens),
		"stale semantics: snapshot was empty at AddCacheInstance time, must not see live blocks yet")

	// After RefreshCacheIfNeeded fires
	provider.RefreshCacheIfNeeded(1000)

	// The snapshot is updated and the closure sees the populated cache
	assert.Greater(t, cqf["inst-deferred"](tokens), 0,
		"after RefreshCacheIfNeeded: snapshot updated, closure sees populated cache")
}

func TestCachedSnapshotProvider_RefreshCacheIfNeeded_BoundaryAtIntervalMinusOne(t *testing.T) {
	// Ported from TestStaleCacheIndex_RefreshIfNeeded_BoundaryAtIntervalMinusOne.
	cfg := newTestSimConfig()
	cfg.Horizon = 10_000_000
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-0", cfg)

	instances := map[InstanceID]*InstanceSimulator{"inst-0": inst}
	obsConfig := newObservabilityConfig(0, 1000) // stale mode, interval=1000
	provider := NewCachedSnapshotProvider(instances, obsConfig)

	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}

	// Populate cache
	req := &sim.Request{
		ID: "r1", ArrivalTime: 0, InputTokens: tokens,
		OutputTokens: []int{100}, State: sim.StateQueued,
	}
	inst.InjectRequest(req)
	inst.Run()
	require.Greater(t, inst.GetCachedBlockCount(tokens), 0, "live cache must have blocks")

	// At clock = interval - 1 = 999: NOT refreshed (999 - 0 = 999 < 1000)
	provider.RefreshCacheIfNeeded(999)
	assert.Equal(t, 0, provider.CacheQuery("inst-0", tokens),
		"snapshot must NOT be refreshed at clock=interval-1 (strict < boundary)")

	// At clock = interval = 1000: IS refreshed (1000 - 0 = 1000 >= 1000)
	provider.RefreshCacheIfNeeded(1000)
	assert.Greater(t, provider.CacheQuery("inst-0", tokens), 0,
		"snapshot must be refreshed at clock=interval (>= threshold)")
}

func TestCachedSnapshotProvider_AddCacheInstance_DuplicateID_Panics(t *testing.T) {
	// Ported from TestStaleCacheIndex_AddInstance_DuplicateID_Panics.
	cfg := newTestSimConfig()
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-0", cfg)

	instances := map[InstanceID]*InstanceSimulator{"inst-0": inst}
	obsConfig := newObservabilityConfig(0, 1000)
	provider := NewCachedSnapshotProvider(instances, obsConfig)

	defer func() {
		r := recover()
		assert.NotNil(t, r, "expected panic for duplicate instance ID")
		assert.Contains(t, fmt.Sprintf("%v", r), "already registered")
	}()
	provider.AddCacheInstance("inst-0", inst)
}

func TestCachedSnapshotProvider_BuildCacheQueryFn_DelegatesToStale(t *testing.T) {
	// Ported from TestStaleCacheIndex_BuildCacheQueryFn_DelegatesToStale.
	cfg := newTestSimConfig()
	cfg.Horizon = 10_000_000
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-0", cfg)

	instances := map[InstanceID]*InstanceSimulator{"inst-0": inst}
	obsConfig := newObservabilityConfig(0, 1000)
	provider := NewCachedSnapshotProvider(instances, obsConfig)

	cqf := provider.BuildCacheQueryFn()
	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}

	// Initially returns 0 (empty cache in snapshot)
	assert.Equal(t, 0, cqf["inst-0"](tokens))

	// Populate cache
	req := &sim.Request{
		ID: "r1", ArrivalTime: 0, InputTokens: tokens,
		OutputTokens: []int{100}, State: sim.StateQueued,
	}
	inst.InjectRequest(req)
	inst.Run()

	// Still stale before refresh
	assert.Equal(t, 0, cqf["inst-0"](tokens), "should be stale before refresh")

	// After refresh, the SAME closure sees the new data
	provider.RefreshCacheIfNeeded(1000)
	assert.Greater(t, cqf["inst-0"](tokens), 0, "should see blocks after refresh via same closure")
}

func TestCachedSnapshotProvider_RemoveCacheInstance_Basic(t *testing.T) {
	// Ported from TestStaleCacheIndex_RemoveInstance.
	cfg := newTestSimConfig()
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-0", cfg)

	instances := map[InstanceID]*InstanceSimulator{"inst-0": inst}
	obsConfig := newObservabilityConfig(0, 1000)
	provider := NewCachedSnapshotProvider(instances, obsConfig)

	tokens := []int{1, 2, 3, 4}

	// Sanity: instance is queryable
	assert.Equal(t, 0, provider.CacheQuery("inst-0", tokens))

	// Remove the instance
	provider.RemoveCacheInstance("inst-0")

	// BuildCacheQueryFn no longer includes the instance
	cqf := provider.BuildCacheQueryFn()
	_, exists := cqf["inst-0"]
	assert.False(t, exists, "removed instance should not appear in BuildCacheQueryFn")

	// Refresh should not panic (no instances to snapshot)
	provider.RefreshCacheIfNeeded(2000)

	// CacheQuery for the removed instance returns 0
	assert.Equal(t, 0, provider.CacheQuery("inst-0", tokens), "query for removed instance should return 0")
}

func TestCachedSnapshotProvider_RemoveCacheInstance_Idempotent(t *testing.T) {
	// Ported from TestStaleCacheIndex_RemoveInstance_Idempotent.
	obsConfig := newObservabilityConfig(0, 1000)
	provider := NewCachedSnapshotProvider(nil, obsConfig)

	// Remove a non-existent instance — should not panic (no-op)
	provider.RemoveCacheInstance("nonexistent")
}

// --- Default value test ---

func TestDefaultCacheSignalDelay_Is50ms(t *testing.T) {
	// BC-2: Default cache signal delay is 50ms (50,000 µs).
	assert.Equal(t, int64(50_000), DefaultCacheSignalDelay)
}

// --- Integration tests (unchanged behavioral contracts) ---

func TestCluster_CacheSignalDelay_StaleRouting(t *testing.T) {
	// GIVEN two identical clusters — one oracle (delay=0), one stale (delay=very large)
	// — both using precise-prefix-cache as the sole scorer with shared-prefix requests.
	//
	// With oracle mode, r2 should see r1's cached blocks and prefer the same instance.
	// With stale mode (delay > r2 arrival), r2 cannot see r1's cached blocks because
	// the snapshot hasn't refreshed yet, so routing decisions may differ.
	makeConfig := func(delay int64) DeploymentConfig {
		return DeploymentConfig{
			SimConfig: sim.SimConfig{
				Horizon:             10_000_000,
				Seed:                42,
				KVCacheConfig:       sim.NewKVCacheConfig(100, 4, 0, 0, 0, 0),
				BatchConfig:         sim.NewBatchConfig(10, 2048, 0),
				LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
				ModelHardwareConfig: sim.ModelHardwareConfig{Backend: "blackbox"},
			},
			NumInstances:     2,
			CacheSignalDelay: delay,
			RoutingPolicy:    "weighted",
			RoutingScorerConfigs: []sim.ScorerConfig{
				{Name: "precise-prefix-cache", Weight: 1.0},
			},
		}
	}

	tokens := make([]int, 16) // 4 blocks of size 4
	for i := range tokens {
		tokens[i] = i + 1
	}

	// Generate N requests: r0 warms the cache, r1..r(N-1) share the same prefix.
	numRequests := 10
	makeRequests := func() []*sim.Request {
		reqs := make([]*sim.Request, numRequests)
		for i := 0; i < numRequests; i++ {
			reqs[i] = &sim.Request{
				ID:           fmt.Sprintf("r%d", i),
				ArrivalTime:  int64(i) * 50_000, // 50ms apart
				InputTokens:  tokens,
				OutputTokens: []int{1},
				State:        sim.StateQueued,
			}
		}
		return reqs
	}

	// Oracle mode: delay=0
	csOracle := NewClusterSimulator(makeConfig(0), makeRequests(), nil)
	require.NoError(t, csOracle.Run())
	oraclePerInst := csOracle.PerInstanceMetrics()

	// Stale mode: delay=50s (much larger than total workload span)
	csStale := NewClusterSimulator(makeConfig(50_000_000), makeRequests(), nil)
	require.NoError(t, csStale.Run())
	stalePerInst := csStale.PerInstanceMetrics()

	// Both must complete all requests.
	oracleAgg := csOracle.aggregateMetrics()
	staleAgg := csStale.aggregateMetrics()
	assert.Equal(t, numRequests, oracleAgg.CompletedRequests, "oracle: all requests should complete")
	assert.Equal(t, numRequests, staleAgg.CompletedRequests, "stale: all requests should complete")

	// Oracle mode: precise-prefix-cache strongly attracts all requests to the instance
	// that cached the first request's prefix.
	oracleMax := oraclePerInst[0].CompletedRequests
	if oraclePerInst[1].CompletedRequests > oracleMax {
		oracleMax = oraclePerInst[1].CompletedRequests
	}
	staleMax := stalePerInst[0].CompletedRequests
	if stalePerInst[1].CompletedRequests > staleMax {
		staleMax = stalePerInst[1].CompletedRequests
	}

	// Oracle concentrates more than stale (stale spreads due to ties).
	assert.Greater(t, oracleMax, staleMax,
		"oracle mode should concentrate more requests on cache-warm instance (%d) "+
			"than stale mode (%d) — stale mode sees ties and spreads requests",
		oracleMax, staleMax)
}

func TestCluster_CacheSignalDelay_Zero_OracleBehavior(t *testing.T) {
	// GIVEN a cluster with cache-signal-delay = 0 (oracle mode)
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             5_000_000,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(100, 4, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
			ModelHardwareConfig: sim.ModelHardwareConfig{Backend: "blackbox"},
		},
		NumInstances:  2,
		RoutingPolicy: "weighted",
		RoutingScorerConfigs: []sim.ScorerConfig{
			{Name: "precise-prefix-cache", Weight: 1.0},
		},
	}

	tokens := make([]int, 16)
	for i := range tokens {
		tokens[i] = i + 1
	}

	requests := []*sim.Request{
		{ID: "r1", ArrivalTime: 0, InputTokens: tokens, OutputTokens: []int{1}, State: sim.StateQueued},
		{ID: "r2", ArrivalTime: 100_000, InputTokens: tokens, OutputTokens: []int{1}, State: sim.StateQueued},
	}

	cs := NewClusterSimulator(config, requests, nil)
	err := cs.Run()
	require.NoError(t, err)

	// Backward-compatibility smoke test
	m := cs.aggregateMetrics()
	assert.Greater(t, m.CompletedRequests, 0, "requests should complete")
}
