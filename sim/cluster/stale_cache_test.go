package cluster

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/inference-sim/inference-sim/sim"
)

func TestStaleCacheIndex_StaleUntilRefresh(t *testing.T) {
	// GIVEN a StaleCacheIndex with one instance and interval=1000
	cfg := newTestSimConfig()
	cfg.Horizon = 10_000_000
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-0", cfg)

	instances := map[InstanceID]*InstanceSimulator{"inst-0": inst}
	idx := NewStaleCacheIndex(instances, 1000)

	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}

	// Initial snapshot at clock=0 — empty cache
	assert.Equal(t, 0, idx.Query("inst-0", tokens), "initial snapshot should see 0 blocks")

	// Inject and run a request to populate cache
	req := &sim.Request{
		ID:           "r1",
		ArrivalTime:  0,
		InputTokens:  tokens,
		OutputTokens: []int{100},
		State:        sim.StateQueued,
	}
	inst.InjectRequest(req)
	inst.Run()

	// Live query confirms blocks exist
	require.Greater(t, inst.GetCachedBlockCount(tokens), 0)

	// WHEN we query the stale index before refresh interval elapses
	idx.RefreshIfNeeded(500) // clock=500 < interval=1000
	// THEN stale index still returns 0
	assert.Equal(t, 0, idx.Query("inst-0", tokens), "stale: should NOT see blocks before refresh")

	// WHEN we query after refresh interval elapses
	idx.RefreshIfNeeded(1000) // clock=1000 >= interval=1000
	// THEN stale index sees the blocks
	assert.Greater(t, idx.Query("inst-0", tokens), 0, "after refresh: should see blocks")
}

func TestStaleCacheIndex_AddInstance(t *testing.T) {
	// GIVEN an empty StaleCacheIndex
	idx := NewStaleCacheIndex(nil, 1000)

	// WHEN we add an instance
	cfg := newTestSimConfig()
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-new", cfg)
	idx.AddInstance("inst-new", inst)

	// THEN it's queryable (returns 0 for empty cache)
	tokens := []int{1, 2, 3, 4}
	assert.Equal(t, 0, idx.Query("inst-new", tokens))
}

func TestStaleCacheIndex_RefreshIfNeeded_BoundaryAtIntervalMinusOne(t *testing.T) {
	// GIVEN a StaleCacheIndex with interval=1000 and a populated cache
	cfg := newTestSimConfig()
	cfg.Horizon = 10_000_000
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-0", cfg)

	instances := map[InstanceID]*InstanceSimulator{"inst-0": inst}
	idx := NewStaleCacheIndex(instances, 1000) // lastRefresh=0

	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}

	// Populate cache
	req := &sim.Request{
		ID: "r1", ArrivalTime: 0, InputTokens: tokens,
		OutputTokens: []int{100}, State: sim.StateQueued,
	}
	inst.InjectRequest(req)
	inst.Run()
	require.Greater(t, inst.GetCachedBlockCount(tokens), 0, "live cache must have blocks")

	// WHEN RefreshIfNeeded is called at exactly clock = interval - 1 = 999
	// THEN the snapshot is NOT refreshed (999 - 0 = 999 < 1000, strict < boundary)
	idx.RefreshIfNeeded(999)
	assert.Equal(t, 0, idx.Query("inst-0", tokens),
		"snapshot must NOT be refreshed at clock=interval-1 (strict < boundary)")

	// WHEN RefreshIfNeeded is called at clock = interval = 1000
	// THEN the snapshot IS refreshed (1000 - 0 = 1000 >= 1000)
	idx.RefreshIfNeeded(1000)
	assert.Greater(t, idx.Query("inst-0", tokens), 0,
		"snapshot must be refreshed at clock=interval (>= threshold)")
}

func TestStaleCacheIndex_AddInstance_DuplicateID_Panics(t *testing.T) {
	// GIVEN a StaleCacheIndex with instance "inst-0" already registered
	cfg := newTestSimConfig()
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-0", cfg)
	instances := map[InstanceID]*InstanceSimulator{"inst-0": inst}
	idx := NewStaleCacheIndex(instances, 1000)

	// WHEN AddInstance is called with the same ID again
	// THEN it panics with a message containing "already registered"
	defer func() {
		r := recover()
		assert.NotNil(t, r, "expected panic for duplicate instance ID")
		assert.Contains(t, fmt.Sprintf("%v", r), "already registered")
	}()
	idx.AddInstance("inst-0", inst)
}

func TestStaleCacheIndex_BuildCacheQueryFn_DelegatesToStale(t *testing.T) {
	// GIVEN a StaleCacheIndex with one instance
	cfg := newTestSimConfig()
	cfg.Horizon = 10_000_000
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-0", cfg)

	instances := map[InstanceID]*InstanceSimulator{"inst-0": inst}
	idx := NewStaleCacheIndex(instances, 1000)

	// Build the cacheQueryFn map
	cqf := idx.BuildCacheQueryFn()

	tokens := []int{1, 2, 3, 4, 5, 6, 7, 8}

	// Initially returns 0 (empty cache in snapshot)
	assert.Equal(t, 0, cqf["inst-0"](tokens))

	// Populate cache
	req := &sim.Request{
		ID:           "r1",
		ArrivalTime:  0,
		InputTokens:  tokens,
		OutputTokens: []int{100},
		State:        sim.StateQueued,
	}
	inst.InjectRequest(req)
	inst.Run()

	// Still stale before refresh
	assert.Equal(t, 0, cqf["inst-0"](tokens), "should be stale before refresh")

	// After refresh, the SAME closure sees the new data (reads staleFns at call time)
	idx.RefreshIfNeeded(1000)
	assert.Greater(t, cqf["inst-0"](tokens), 0, "should see blocks after refresh via same closure")
}

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
	// With oracle, r1+ see r0's cached blocks on inst-X and concentrate there.
	// With stale (delay >> inter-arrival), r1+ see no cache anywhere → ties → spread.
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
	// that cached the first request's prefix. The max-loaded instance should have most requests.
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

func TestStaleCacheIndex_RemoveInstance(t *testing.T) {
	// GIVEN a StaleCacheIndex with one instance
	cfg := newTestSimConfig()
	cfg.TotalKVBlocks = 100
	cfg.BlockSizeTokens = 4
	inst := NewInstanceSimulator("inst-0", cfg)

	instances := map[InstanceID]*InstanceSimulator{"inst-0": inst}
	idx := NewStaleCacheIndex(instances, 1000)

	tokens := []int{1, 2, 3, 4}

	// Sanity: instance is queryable
	assert.Equal(t, 0, idx.Query("inst-0", tokens))

	// WHEN we remove the instance
	idx.RemoveInstance("inst-0")

	// THEN BuildCacheQueryFn no longer includes the instance
	cqf := idx.BuildCacheQueryFn()
	_, exists := cqf["inst-0"]
	assert.False(t, exists, "removed instance should not appear in BuildCacheQueryFn")

	// AND refresh should not panic (no instances to snapshot)
	idx.RefreshIfNeeded(2000)

	// AND Query for the removed instance returns 0 (warn-and-return-0 path)
	assert.Equal(t, 0, idx.Query("inst-0", tokens), "query for removed instance should return 0")
}

func TestStaleCacheIndex_RemoveInstance_Idempotent(t *testing.T) {
	// GIVEN an empty StaleCacheIndex
	idx := NewStaleCacheIndex(nil, 1000)

	// WHEN we remove a non-existent instance
	// THEN it should not panic (no-op)
	idx.RemoveInstance("nonexistent")
}

func TestNewStaleCacheIndex_ZeroInterval_Panics(t *testing.T) {
	defer func() {
		r := recover()
		assert.NotNil(t, r, "expected panic for interval=0")
		assert.Contains(t, r, "interval must be > 0")
	}()
	NewStaleCacheIndex(nil, 0)
}

func TestNewStaleCacheIndex_NegativeInterval_Panics(t *testing.T) {
	defer func() {
		r := recover()
		assert.NotNil(t, r, "expected panic for negative interval")
		assert.Contains(t, r, "interval must be > 0")
	}()
	NewStaleCacheIndex(nil, -100)
}
