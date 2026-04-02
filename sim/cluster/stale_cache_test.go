package cluster

import (
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
	// GIVEN a 2-instance cluster with cache-signal-delay > 0 and precise-prefix-cache scorer
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             5_000_000,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(100, 4, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
			ModelHardwareConfig: sim.ModelHardwareConfig{Backend: "blackbox"},
		},
		NumInstances:     2,
		CacheSignalDelay: 1_000_000, // 1 second — very large to ensure staleness
		RoutingPolicy:    "weighted",
		RoutingScorerConfigs: []sim.ScorerConfig{
			{Name: "precise-prefix-cache", Weight: 1.0},
		},
	}

	tokens := make([]int, 16) // 4 blocks of size 4
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

	// Integration test: wiring doesn't panic and produces valid results.
	m := cs.aggregateMetrics()
	assert.Greater(t, m.CompletedRequests, 0, "requests should complete")
}

func TestCluster_CacheSignalDelay_Zero_OracleBehavior(t *testing.T) {
	// GIVEN a cluster with cache-signal-delay = 0 (default)
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
