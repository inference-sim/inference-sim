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
	assert.True(t, IsValidScorer("active-requests"))
	assert.True(t, IsValidScorer("running-requests"))
	assert.True(t, IsValidScorer("load-aware"))
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
	loadBalanceOnly := NewRoutingPolicy("weighted", []ScorerConfig{{Name: "load-balance", Weight: 1.0}}, 16, nil)
	leastLoaded := NewRoutingPolicy("least-loaded", nil, 16, nil)

	testCases := [][]RoutingSnapshot{
		{
			{ID: "a", QueueDepth: 10, BatchSize: 2},
			{ID: "b", QueueDepth: 3, BatchSize: 1},
			{ID: "c", QueueDepth: 7, BatchSize: 0},
		},
		{
			{ID: "a", QueueDepth: 5, BatchSize: 5, InFlightRequests: 3},
			{ID: "b", QueueDepth: 5, BatchSize: 5, InFlightRequests: 0},
		},
		{
			{ID: "a", QueueDepth: 0, BatchSize: 0},
			{ID: "b", QueueDepth: 0, BatchSize: 0},
			{ID: "c", QueueDepth: 0, BatchSize: 0},
		},
		{
			{ID: "a", QueueDepth: 100, BatchSize: 50, InFlightRequests: 25},
			{ID: "b", QueueDepth: 1, BatchSize: 0, InFlightRequests: 0},
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
		{ID: "a", QueueDepth: 10, BatchSize: 0, InFlightRequests: 0}, // highest depth
		{ID: "b", QueueDepth: 5, BatchSize: 0, InFlightRequests: 0},  // middle depth
		{ID: "c", QueueDepth: 0, BatchSize: 0, InFlightRequests: 0},  // lowest depth
	}
	scores := scoreQueueDepth(nil, snapshots)
	// Monotonicity: lower depth → higher score
	assert.Greater(t, scores["c"], scores["b"], "lowest depth should score highest")
	assert.Greater(t, scores["b"], scores["a"], "middle depth should score higher than highest")
	// Boundary: max depth scores 0, min depth scores 1, midpoint scores 0.5
	assert.Equal(t, 0.0, scores["a"], "highest depth should score 0.0")
	assert.Equal(t, 0.5, scores["b"], "midpoint depth should score 0.5")
	assert.Equal(t, 1.0, scores["c"], "lowest depth should score 1.0")
}

func TestScoreQueueDepth_UniformLoad_AllScoreOne(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 5},
		{ID: "b", QueueDepth: 5},
	}
	scores := scoreQueueDepth(nil, snapshots)
	assert.Equal(t, 1.0, scores["a"])
	assert.Equal(t, 1.0, scores["b"])
}

func TestScoreQueueDepth_SingleInstance_ScoresOne(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 42},
	}
	scores := scoreQueueDepth(nil, snapshots)
	assert.Equal(t, 1.0, scores["a"], "single instance always scores 1.0")
}

func TestScoreQueueDepth_AllZeroDepth_AllScoreOne(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 0},
		{ID: "b", QueueDepth: 0},
	}
	scores := scoreQueueDepth(nil, snapshots)
	assert.Equal(t, 1.0, scores["a"], "all-zero queue depths should score 1.0")
	assert.Equal(t, 1.0, scores["b"], "all-zero queue depths should score 1.0")
}

// TestScoreQueueDepth_IgnoresBatchSizeAndInFlight verifies that scores depend
// only on QueueDepth — BatchSize and InFlightRequests are ignored (GIE parity).
func TestScoreQueueDepth_IgnoresBatchSizeAndInFlight(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 10, BatchSize: 0, InFlightRequests: 0},   // highest QueueDepth
		{ID: "b", QueueDepth: 0, BatchSize: 100, InFlightRequests: 50}, // lowest QueueDepth, large batch+inflight
	}
	scores := scoreQueueDepth(nil, snapshots)
	assert.Equal(t, 1.0, scores["b"], "lowest QueueDepth should score 1.0 regardless of BatchSize/InFlightRequests")
	assert.Equal(t, 0.0, scores["a"], "highest QueueDepth should score 0.0 regardless of BatchSize/InFlightRequests")
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

// === active-requests scorer tests (BC-1, BC-2) ===

func TestScoreActiveRequests_AllZero_AllScoreOne(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", InFlightRequests: 0},
		{ID: "b", InFlightRequests: 0},
		{ID: "c", InFlightRequests: 0},
	}
	scores := scoreActiveRequests(nil, snapshots)
	for _, snap := range snapshots {
		assert.Equal(t, 1.0, scores[snap.ID], "zero in-flight should score 1.0")
	}
}

func TestScoreActiveRequests_Varied_MonotonicAndBoundary(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", InFlightRequests: 0},  // zero → 1.0
		{ID: "b", InFlightRequests: 5},  // middle
		{ID: "c", InFlightRequests: 10}, // max → 0.0
	}
	scores := scoreActiveRequests(nil, snapshots)
	// BC-1: zero in-flight always scores 1.0
	assert.Equal(t, 1.0, scores["a"], "zero in-flight should score 1.0")
	// BC-2: max count scores 0.0
	assert.Equal(t, 0.0, scores["c"], "max in-flight should score 0.0")
	// BC-2: monotonicity — fewer in-flight → higher score
	assert.Greater(t, scores["a"], scores["b"], "fewer in-flight should score higher")
	assert.Greater(t, scores["b"], scores["c"], "fewer in-flight should score higher")
}

func TestScoreActiveRequests_AllEqual_NonZero(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", InFlightRequests: 5},
		{ID: "b", InFlightRequests: 5},
	}
	scores := scoreActiveRequests(nil, snapshots)
	// When all equal and non-zero: (max-count)/max = (5-5)/5 = 0.0 for all
	assert.Equal(t, 0.0, scores["a"], "all at max should score 0.0")
	assert.Equal(t, 0.0, scores["b"], "all at max should score 0.0")
	assert.False(t, math.IsNaN(scores["a"]), "score must not be NaN")
}

func TestScoreActiveRequests_SingleNonZeroInstance_ScoresZero(t *testing.T) {
	snapshots := []RoutingSnapshot{{ID: "a", InFlightRequests: 5}}
	scores := scoreActiveRequests(nil, snapshots)
	assert.Equal(t, 0.0, scores["a"], "single non-zero instance: max-only normalization → 0.0")
	assert.False(t, math.IsNaN(scores["a"]))
}

// === running-requests scorer tests (BC-3, BC-4) ===

func TestScoreRunningRequests_AllEqual_AllScoreOne(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", BatchSize: 5},
		{ID: "b", BatchSize: 5},
	}
	scores := scoreRunningRequests(nil, snapshots)
	assert.Equal(t, 1.0, scores["a"])
	assert.Equal(t, 1.0, scores["b"])
}

func TestScoreRunningRequests_Varied_MinMaxNormalization(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", BatchSize: 0},  // min → 1.0
		{ID: "b", BatchSize: 5},  // middle
		{ID: "c", BatchSize: 10}, // max → 0.0
	}
	scores := scoreRunningRequests(nil, snapshots)
	// BC-4: min batch scores 1.0, max scores 0.0
	assert.Equal(t, 1.0, scores["a"], "min batch should score 1.0")
	assert.Equal(t, 0.0, scores["c"], "max batch should score 0.0")
	// BC-4: monotonicity
	assert.Greater(t, scores["a"], scores["b"], "smaller batch should score higher")
	assert.Greater(t, scores["b"], scores["c"], "smaller batch should score higher")
	// BC-4: proportional — middle at 5/10 should score 0.5
	assert.InDelta(t, 0.5, scores["b"], 0.001, "mid-point should score ~0.5")
}

func TestScoreRunningRequests_AllZero_AllScoreOne(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", BatchSize: 0},
		{ID: "b", BatchSize: 0},
	}
	scores := scoreRunningRequests(nil, snapshots)
	assert.Equal(t, 1.0, scores["a"])
	assert.Equal(t, 1.0, scores["b"])
}

// === load-aware scorer tests (BC-5, BC-6, BC-7) ===

func TestScoreLoadAware_EmptyQueue_ScoresHalf(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 0},
		{ID: "b", QueueDepth: 0},
	}
	scores := scoreLoadAware(nil, snapshots)
	assert.Equal(t, 0.5, scores["a"], "empty queue should score 0.5")
	assert.Equal(t, 0.5, scores["b"], "empty queue should score 0.5")
}

func TestScoreLoadAware_PartialQueue_LinearDecrease(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 0},   // empty → 0.5
		{ID: "b", QueueDepth: 64},  // half threshold → 0.25
		{ID: "c", QueueDepth: 128}, // at threshold → 0.0
	}
	scores := scoreLoadAware(nil, snapshots)
	// BC-5: empty = 0.5
	assert.Equal(t, 0.5, scores["a"], "empty queue should score 0.5")
	// BC-6: half threshold
	assert.InDelta(t, 0.25, scores["b"], 0.001, "half-threshold should score ~0.25")
	// BC-6: at threshold
	assert.Equal(t, 0.0, scores["c"], "at-threshold should score 0.0")
	// Monotonicity
	assert.Greater(t, scores["a"], scores["b"], "less queue depth should score higher")
	assert.Greater(t, scores["b"], scores["c"], "less queue depth should score higher")
}

func TestScoreLoadAware_AboveThreshold_ScoresZero(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 200}, // well above threshold
		{ID: "b", QueueDepth: 128}, // exactly at threshold
	}
	scores := scoreLoadAware(nil, snapshots)
	// BC-7: above threshold → clamped → 0.0
	assert.Equal(t, 0.0, scores["a"], "above threshold should score 0.0")
	assert.Equal(t, 0.0, scores["b"], "at threshold should score 0.0")
}

func TestScoreLoadAware_ScoreRange_MaxIsHalf(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 0},
		{ID: "b", QueueDepth: 50},
		{ID: "c", QueueDepth: 200},
	}
	scores := scoreLoadAware(nil, snapshots)
	for _, snap := range snapshots {
		assert.LessOrEqual(t, scores[snap.ID], 0.5, "load-aware max score should be 0.5")
		assert.GreaterOrEqual(t, scores[snap.ID], 0.0, "load-aware min score should be 0.0")
	}
}

func TestAllScorers_ReturnScoreForEveryInstance(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 1, KVUtilization: 0.3},
		{ID: "b", QueueDepth: 2, KVUtilization: 0.7},
		{ID: "c", QueueDepth: 0, KVUtilization: 0.0},
	}
	cacheQueryFn := cacheQueryFn{
		"a": func(tokens []int) int { return 2 },
		"b": func(tokens []int) int { return 0 },
		"c": func(tokens []int) int { return 1 },
	}
	precisePrefixScorer, _ := newPrecisePrefixCacheScorer(cacheQueryFn)
	noHitLRUScorer, _ := newNoHitLRUScorer(cacheQueryFn)
	scorerFns := []struct {
		name string
		fn   scorerFunc
	}{
		{"queue-depth", scoreQueueDepth},
		{"kv-utilization", scoreKVUtilization},
		{"load-balance", scoreLoadBalance},
		{"active-requests", scoreActiveRequests},
		{"running-requests", scoreRunningRequests},
		{"load-aware", scoreLoadAware},
		{"precise-prefix-cache", precisePrefixScorer},
		{"no-hit-lru", noHitLRUScorer},
	}
	req := &Request{ID: "r1", InputTokens: []int{1, 2, 3}}
	for _, sf := range scorerFns {
		t.Run(sf.name, func(t *testing.T) {
			scores := sf.fn(req, snapshots)
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
	// Verify nil-request path for all stateless scorers.
	// These scorers ignore the request parameter; this confirms they don't panic on nil.
	for _, sf := range scorerFns[:6] {
		t.Run(sf.name+"/nil-request", func(t *testing.T) {
			scores := sf.fn(nil, snapshots)
			assert.Len(t, scores, len(snapshots))
		})
	}
}

// TestNewScorerFactory_PrecisePrefixAndNoHitLRU verifies BC-6: factory chain
// correctly wires precise-prefix-cache and no-hit-lru scorers via NewRoutingPolicyWithCache.
func TestNewScorerFactory_PrecisePrefixAndNoHitLRU(t *testing.T) {
	cacheQueryFn := cacheQueryFn{
		"a": func(tokens []int) int { return 5 },
		"b": func(tokens []int) int { return 0 },
	}
	policy := NewRoutingPolicyWithCache("weighted", []ScorerConfig{
		{Name: "precise-prefix-cache", Weight: 1.0},
	}, 16, nil, cacheQueryFn)

	req := &Request{ID: "r1", InputTokens: []int{1, 2, 3}}
	state := &RouterState{
		Snapshots: []RoutingSnapshot{{ID: "a"}, {ID: "b"}},
		Clock:     1000,
	}
	decision := policy.Route(req, state)
	// Instance "a" has 5 cached blocks, "b" has 0 → "a" should win
	assert.Equal(t, "a", decision.TargetInstance, "precise-prefix-cache should prefer instance with more cached blocks")
}

// === vllm-dp scorer tests ===

func TestScoreVLLMDP_BasicFormula(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 10, BatchSize: 5},  // 10×4 + 5 = 45
		{ID: "b", QueueDepth: 5, BatchSize: 10},  // 5×4 + 10 = 30
		{ID: "c", QueueDepth: 2, BatchSize: 2},   // 2×4 + 2 = 10 (min)
	}
	scores := scoreVLLMDP(nil, snapshots)

	// c has lowest raw score (10) → should score 1.0
	assert.Equal(t, 1.0, scores["c"], "lowest load should score 1.0")
	// a has highest raw score (45) → should score 0.0
	assert.Equal(t, 0.0, scores["a"], "highest load should score 0.0")
	// b is midpoint: (45-30)/(45-10) = 15/35 ≈ 0.428
	assert.InDelta(t, 0.428, scores["b"], 0.01, "midpoint should score ~0.43")
}
