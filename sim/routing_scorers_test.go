package sim

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseScorerConfigs_ValidInput(t *testing.T) {
	configs, err := ParseScorerConfigs("queue-depth:2,kv-utilization:3,load-balance:1")
	require.NoError(t, err)
	assert.Len(t, configs, 3)
	assert.Equal(t, "queue-depth", configs[0].Name)
	assert.Equal(t, 2.0, configs[0].Weight)
	assert.Equal(t, "kv-utilization", configs[1].Name)
	assert.Equal(t, 3.0, configs[1].Weight)
	assert.Equal(t, "load-balance", configs[2].Name)
	assert.Equal(t, 1.0, configs[2].Weight)
}

func TestParseScorerConfigs_EmptyString_ReturnsNil(t *testing.T) {
	configs, err := ParseScorerConfigs("")
	require.NoError(t, err)
	assert.Nil(t, configs)
}

func TestParseScorerConfigs_InvalidInput(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"unknown scorer", "unknown-scorer:1"},
		{"missing weight", "queue-depth"},
		{"negative weight", "queue-depth:-1"},
		{"zero weight", "queue-depth:0"},
		{"NaN weight", "queue-depth:NaN"},
		{"Inf weight", "queue-depth:Inf"},
		{"non-numeric weight", "queue-depth:abc"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseScorerConfigs(tt.input)
			assert.Error(t, err)
		})
	}
}

func TestIsValidScorer_KnownNames(t *testing.T) {
	assert.True(t, IsValidScorer("queue-depth"))
	assert.True(t, IsValidScorer("kv-utilization"))
	assert.True(t, IsValidScorer("load-balance"))
	assert.False(t, IsValidScorer("unknown"))
	assert.False(t, IsValidScorer(""))
}

func TestValidScorerNames_Sorted(t *testing.T) {
	names := ValidScorerNames()
	assert.GreaterOrEqual(t, len(names), 4, "at least 4 scorers expected")
	for i := 1; i < len(names); i++ {
		assert.True(t, names[i-1] < names[i], "names must be sorted")
	}
}

func TestDefaultScorerConfigs_ReturnsThreeScorers(t *testing.T) {
	configs := DefaultScorerConfigs()
	assert.Len(t, configs, 3)
	for _, c := range configs {
		assert.True(t, IsValidScorer(c.Name), "default scorer %q must be valid", c.Name)
		assert.True(t, c.Weight > 0, "default weight must be positive")
	}
}

func TestNormalizeScorerWeights_PreservesRatio(t *testing.T) {
	configs := []ScorerConfig{
		{Name: "queue-depth", Weight: 3.0},
		{Name: "load-balance", Weight: 2.0},
	}
	weights := normalizeScorerWeights(configs)
	assert.InDelta(t, 0.6, weights[0], 0.001)
	assert.InDelta(t, 0.4, weights[1], 0.001)
	assert.InDelta(t, 1.0, weights[0]+weights[1], 0.001)
}

func TestParseScorerConfigs_WhitespaceHandling(t *testing.T) {
	configs, err := ParseScorerConfigs(" queue-depth : 2 , load-balance : 1 ")
	require.NoError(t, err)
	assert.Len(t, configs, 2)
	assert.Equal(t, "queue-depth", configs[0].Name)
	assert.Equal(t, 2.0, configs[0].Weight)
}

func TestParseScorerConfigs_DuplicateScorer_Rejected(t *testing.T) {
	_, err := ParseScorerConfigs("queue-depth:2,queue-depth:3")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate scorer")
}

func TestParseScorerConfigs_SingleScorer(t *testing.T) {
	configs, err := ParseScorerConfigs("load-balance:1")
	require.NoError(t, err)
	assert.Len(t, configs, 1)
	assert.Equal(t, "load-balance", configs[0].Name)
}

// === Invariant Tests ===

// TestLoadBalanceOnly_EquivalentToLeastLoaded verifies BC-17-5:
// weighted with load-balance:1 must select the same instance as least-loaded
// for every request, because argmax(1/(1+load)) = argmin(load).
func TestLoadBalanceOnly_EquivalentToLeastLoaded(t *testing.T) {
	loadBalanceOnly := NewRoutingPolicy("weighted", []ScorerConfig{{Name: "load-balance", Weight: 1.0}}, 16)
	leastLoaded := NewRoutingPolicy("least-loaded", nil, 16)

	testCases := [][]RoutingSnapshot{
		{
			{ID: "a", QueueDepth: 10, BatchSize: 2},
			{ID: "b", QueueDepth: 3, BatchSize: 1},
			{ID: "c", QueueDepth: 7, BatchSize: 0},
		},
		{
			{ID: "a", QueueDepth: 5, BatchSize: 5, PendingRequests: 3},
			{ID: "b", QueueDepth: 5, BatchSize: 5, PendingRequests: 0},
		},
		{
			{ID: "a", QueueDepth: 0, BatchSize: 0},
			{ID: "b", QueueDepth: 0, BatchSize: 0},
			{ID: "c", QueueDepth: 0, BatchSize: 0},
		},
		{
			{ID: "a", QueueDepth: 100, BatchSize: 50, PendingRequests: 25},
			{ID: "b", QueueDepth: 1, BatchSize: 0, PendingRequests: 0},
		},
	}

	for i, snapshots := range testCases {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			req := &Request{ID: fmt.Sprintf("req_%d", i)}
			state := &RouterState{Snapshots: snapshots, Clock: 1000}

			wDecision := loadBalanceOnly.Route(req, state)
			llDecision := leastLoaded.Route(req, state)

			assert.Equal(t, llDecision.TargetInstance, wDecision.TargetInstance,
				"load-balance-only weighted must select same instance as least-loaded")
		})
	}
}

// === Scorer Behavioral Tests (BC-17-1, BC-17-7, BC-17-9) ===

func TestScoreQueueDepth_MinMaxNormalization(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 10, BatchSize: 0, PendingRequests: 0}, // highest load
		{ID: "b", QueueDepth: 5, BatchSize: 0, PendingRequests: 0},  // middle load
		{ID: "c", QueueDepth: 0, BatchSize: 0, PendingRequests: 0},  // lowest load
	}
	scores := scoreQueueDepth(nil, snapshots)
	// Monotonicity: lower load → higher score
	assert.Greater(t, scores["c"], scores["b"], "lowest load should score highest")
	assert.Greater(t, scores["b"], scores["a"], "middle load should score higher than highest load")
	// Boundary: max load scores 0, min load scores 1
	assert.Equal(t, 0.0, scores["a"], "highest load should score 0.0")
	assert.Equal(t, 1.0, scores["c"], "lowest load should score 1.0")
}

func TestScoreQueueDepth_UniformLoad_AllScoreOne(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 5, BatchSize: 3},
		{ID: "b", QueueDepth: 5, BatchSize: 3},
	}
	scores := scoreQueueDepth(nil, snapshots)
	assert.Equal(t, 1.0, scores["a"])
	assert.Equal(t, 1.0, scores["b"])
}

func TestScoreQueueDepth_IncludesPendingRequests(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 0, PendingRequests: 5}, // load=5
		{ID: "b", QueueDepth: 5, PendingRequests: 0}, // load=5
		{ID: "c", QueueDepth: 0, PendingRequests: 0}, // load=0 → best
	}
	scores := scoreQueueDepth(nil, snapshots)
	assert.Equal(t, scores["a"], scores["b"], "equal effective load → equal score")
	assert.Greater(t, scores["c"], scores["a"], "lower load → higher score")
}

func TestScoreKVUtilization_InverseUtilization(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", KVUtilization: 0.0}, // lowest utilization
		{ID: "b", KVUtilization: 0.5}, // middle
		{ID: "c", KVUtilization: 1.0}, // highest utilization
	}
	scores := scoreKVUtilization(nil, snapshots)
	// Monotonicity: lower utilization → higher score
	assert.Greater(t, scores["a"], scores["b"], "lower utilization should score higher")
	assert.Greater(t, scores["b"], scores["c"], "middle should score higher than highest utilization")
	// Boundaries
	assert.Equal(t, 1.0, scores["a"], "zero utilization should score 1.0")
	assert.Equal(t, 0.0, scores["c"], "full utilization should score 0.0")
}

func TestScoreLoadBalance_InverseTransform(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 0},  // zero load
		{ID: "b", QueueDepth: 9},  // high load
		{ID: "c", QueueDepth: 99}, // very high load
	}
	scores := scoreLoadBalance(nil, snapshots)
	// Monotonicity: lower load → higher score
	assert.Greater(t, scores["a"], scores["b"], "lower load should score higher")
	assert.Greater(t, scores["b"], scores["c"], "middle load should score higher than very high load")
	// Boundary: zero load scores 1.0 (max possible)
	assert.Equal(t, 1.0, scores["a"], "zero load should score 1.0")
	// All scores positive (inverse transform never reaches 0)
	assert.Greater(t, scores["c"], 0.0, "score should always be positive")
}

func TestAllScorers_ReturnScoreForEveryInstance(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 1, KVUtilization: 0.3},
		{ID: "b", QueueDepth: 2, KVUtilization: 0.7},
		{ID: "c", QueueDepth: 0, KVUtilization: 0.0},
	}
	scorerFns := []struct {
		name string
		fn   scorerFunc
	}{
		{"queue-depth", scoreQueueDepth},
		{"kv-utilization", scoreKVUtilization},
		{"load-balance", scoreLoadBalance},
	}
	for _, sf := range scorerFns {
		t.Run(sf.name, func(t *testing.T) {
			scores := sf.fn(nil, snapshots)
			// INV-2: score for every instance
			assert.Len(t, scores, len(snapshots))
			for _, snap := range snapshots {
				score, ok := scores[snap.ID]
				assert.True(t, ok, "missing score for %s", snap.ID)
				// INV-1: score in [0,1]
				assert.GreaterOrEqual(t, score, 0.0, "score below 0 for %s", snap.ID)
				assert.LessOrEqual(t, score, 1.0, "score above 1 for %s", snap.ID)
				// BC-17-9: no NaN/Inf
				assert.False(t, math.IsNaN(score), "NaN score for %s", snap.ID)
				assert.False(t, math.IsInf(score, 0), "Inf score for %s", snap.ID)
			}
		})
	}
}
