package sim

import (
	"fmt"
	"math"
	"sort"
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
		"a": func(tokens []TokenID) int { return 2 },
		"b": func(tokens []TokenID) int { return 0 },
		"c": func(tokens []TokenID) int { return 1 },
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
		{"vllm-dp", scoreVLLMDP},
		{"precise-prefix-cache", precisePrefixScorer},
		{"no-hit-lru", noHitLRUScorer},
	}
	req := &Request{ID: "r1", InputTokens:  []TokenID{1, 2, 3}}
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
	// Indices 0-6: queue-depth, kv-utilization, load-balance, active-requests,
	// running-requests, load-aware, vllm-dp (all use _ *Request).
	for _, sf := range scorerFns[:7] {
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
		"a": func(tokens []TokenID) int { return 5 },
		"b": func(tokens []TokenID) int { return 0 },
	}
	policy := NewRoutingPolicyWithCache("weighted", []ScorerConfig{
		{Name: "precise-prefix-cache", Weight: 1.0},
	}, 16, nil, cacheQueryFn)

	req := &Request{ID: "r1", InputTokens:  []TokenID{1, 2, 3}}
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

func TestScoreVLLMDP_AllEqual(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 5, BatchSize: 3}, // 5×4 + 3 = 23
		{ID: "b", QueueDepth: 5, BatchSize: 3}, // 5×4 + 3 = 23
	}
	scores := scoreVLLMDP(nil, snapshots)
	assert.Equal(t, 1.0, scores["a"], "all equal loads should score 1.0")
	assert.Equal(t, 1.0, scores["b"], "all equal loads should score 1.0")
}

func TestScoreVLLMDP_MonotonicityAndBoundaries(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 0, BatchSize: 0},  // 0 (min)
		{ID: "b", QueueDepth: 3, BatchSize: 2},  // 14
		{ID: "c", QueueDepth: 5, BatchSize: 10}, // 30 (max)
	}
	scores := scoreVLLMDP(nil, snapshots)

	// Boundaries
	assert.Equal(t, 1.0, scores["a"], "min load should score 1.0")
	assert.Equal(t, 0.0, scores["c"], "max load should score 0.0")

	// Monotonicity: lower load → higher score
	assert.Greater(t, scores["a"], scores["b"], "lower load should score higher")
	assert.Greater(t, scores["b"], scores["c"], "lower load should score higher")

	// No NaN/Inf (BC-17-9)
	for id, score := range scores {
		assert.False(t, math.IsNaN(score), "score for %s must not be NaN", id)
		assert.False(t, math.IsInf(score, 0), "score for %s must not be Inf", id)
	}
}

func TestScoreVLLMDP_SingleInstance(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 42, BatchSize: 17}, // 42×4 + 17 = 185
	}
	scores := scoreVLLMDP(nil, snapshots)
	assert.Equal(t, 1.0, scores["a"], "single instance always scores 1.0")
	assert.False(t, math.IsNaN(scores["a"]), "score must not be NaN")
}

func TestScoreVLLMDP_WeightEquivalence(t *testing.T) {
	// BC-VLLM-1: The 4:1 weighting means +1 QueueDepth ≡ -4 BatchSize in routing preference.
	// Two instances with QD_x=QD_y+1 and BS_x=BS_y-4 should have equal raw scores.
	snapshots := []RoutingSnapshot{
		{ID: "x", QueueDepth: 3, BatchSize: 10}, // 3×4 + 10 = 22
		{ID: "y", QueueDepth: 2, BatchSize: 14}, // 2×4 + 14 = 22 (equal raw score)
		{ID: "z", QueueDepth: 5, BatchSize: 2},  // 5×4 + 2 = 22 (also equal)
	}
	scores := scoreVLLMDP(nil, snapshots)

	// All three instances have equal raw scores (22), so all should score 1.0 (tie)
	assert.Equal(t, 1.0, scores["x"], "BC-VLLM-1: equal raw score should score 1.0")
	assert.Equal(t, 1.0, scores["y"], "BC-VLLM-1: equal raw score should score 1.0")
	assert.Equal(t, 1.0, scores["z"], "BC-VLLM-1: equal raw score should score 1.0")

	// Verify the 4:1 law with different values: QD+1 and BS-4 should preserve score equality
	snapshots2 := []RoutingSnapshot{
		{ID: "a", QueueDepth: 10, BatchSize: 8}, // 10×4 + 8 = 48
		{ID: "b", QueueDepth: 11, BatchSize: 4}, // 11×4 + 4 = 48 (QD+1, BS-4)
		{ID: "c", QueueDepth: 0, BatchSize: 100}, // 0×4 + 100 = 100 (different)
	}
	scores2 := scoreVLLMDP(nil, snapshots2)

	// a and b have equal raw scores → should have equal normalized scores
	assert.InDelta(t, scores2["a"], scores2["b"], 1e-9, "BC-VLLM-1: +1 QD = -4 BS in routing preference")
	// c has higher load → lower score
	assert.Less(t, scores2["c"], scores2["a"], "higher load should score lower")
}

func TestScoreVLLMDP_PileOnInPeriodicMode(t *testing.T) {
	// Documents a known difference from real vLLM: vLLM's DPLBAsyncMPClient
	// speculatively increments the cached waiting count after routing
	// (core_client.py:1225), spreading subsequent requests within the 100ms
	// coordinator update window. BLIS does not model this — with a stale
	// snapshot, N requests in the same window all route to the same instance.

	// Stale snapshot: instance A looks empty, B and C are loaded.
	snapshot := []RoutingSnapshot{
		{ID: "a", QueueDepth: 0, BatchSize: 0},  // raw=0, scores 1.0
		{ID: "b", QueueDepth: 2, BatchSize: 3},  // raw=11, scores lower
		{ID: "c", QueueDepth: 5, BatchSize: 1},  // raw=21, scores lowest
	}

	// All three requests in the same snapshot window see the same stale counts.
	// BLIS routes all three to "a" — vLLM would spread them after the first.
	for i := 0; i < 3; i++ {
		scores := scoreVLLMDP(nil, snapshot) // snapshot not updated between calls
		assert.Equal(t, 1.0, scores["a"],
			"request %d: BLIS routes to 'a' (known divergence from vLLM's speculative increment)", i)
		// In vLLM, only the first request would pick "a"; subsequent ones would
		// see incremented counts and potentially route to "b" or "c".
	}
}

// --- lora-affinity scorer (PR6, #1469) -------------------------------------
//
// Contract (contracts/routing-snapshot.md, spec US4):
//   raw(instance)   = 1.0 if req.Adapter ∈ instance.ResidentAdapters else 0.0
//   score(instance) = minMaxNormalize(raw over candidate instances)  (llm-d parity)

func residentSetOf(ids ...string) map[string]bool {
	m := make(map[string]bool, len(ids))
	for _, id := range ids {
		m[id] = true
	}
	return m
}

// TestScoreLoRAAffinity_WarmScoresHigherThanCold is US4 scenario 1: an instance
// that already holds req.Adapter must outscore one that does not, all else equal.
func TestScoreLoRAAffinity_WarmScoresHigherThanCold(t *testing.T) {
	req := &Request{Adapter: "sql-lora"}
	snapshots := []RoutingSnapshot{
		{ID: "warm", ResidentAdapters: residentSetOf("sql-lora", "other")},
		{ID: "cold", ResidentAdapters: residentSetOf("other")},
	}
	scores := scoreLoRAAffinity(req, snapshots)
	assert.Greater(t, scores["warm"], scores["cold"], "warm instance must outscore cold")
	// min-max over {1.0, 0.0}: warm→1.0, cold→0.0.
	assert.Equal(t, 1.0, scores["warm"])
	assert.Equal(t, 0.0, scores["cold"])
}

// TestScoreLoRAAffinity_EmptyAdapterNeutral: base-model requests (no adapter)
// must not bias routing — every instance scores equally (neutral).
func TestScoreLoRAAffinity_EmptyAdapterNeutral(t *testing.T) {
	req := &Request{Adapter: ""}
	snapshots := []RoutingSnapshot{
		{ID: "a", ResidentAdapters: residentSetOf("sql-lora")},
		{ID: "b", ResidentAdapters: residentSetOf("other")},
		{ID: "c"}, // nil ResidentAdapters
	}
	scores := scoreLoRAAffinity(req, snapshots)
	assert.Equal(t, 1.0, scores["a"])
	assert.Equal(t, 1.0, scores["b"])
	assert.Equal(t, 1.0, scores["c"])
}

// TestScoreLoRAAffinity_NilResidentAdaptersNeutral: the zero value (nil set) on
// every instance yields all-cold ⇒ min-max of all-equal ⇒ all score 1.0. This is
// the no-op default that keeps routing unchanged when the LoRA subsystem is inert.
func TestScoreLoRAAffinity_NilResidentAdaptersNeutral(t *testing.T) {
	req := &Request{Adapter: "sql-lora"}
	snapshots := []RoutingSnapshot{{ID: "a"}, {ID: "b"}}
	scores := scoreLoRAAffinity(req, snapshots)
	assert.Equal(t, 1.0, scores["a"], "all-cold ⇒ neutral")
	assert.Equal(t, 1.0, scores["b"], "all-cold ⇒ neutral")
}

// TestScoreLoRAAffinity_AllWarmNeutral: when every candidate already holds the
// adapter, min-max of all-equal raw scores ⇒ all 1.0 (no differentiation).
func TestScoreLoRAAffinity_AllWarmNeutral(t *testing.T) {
	req := &Request{Adapter: "sql-lora"}
	snapshots := []RoutingSnapshot{
		{ID: "a", ResidentAdapters: residentSetOf("sql-lora")},
		{ID: "b", ResidentAdapters: residentSetOf("sql-lora")},
	}
	scores := scoreLoRAAffinity(req, snapshots)
	assert.Equal(t, 1.0, scores["a"])
	assert.Equal(t, 1.0, scores["b"])
}

// TestScoreLoRAAffinity_SingleInstance always scores 1.0 (min == max).
func TestScoreLoRAAffinity_SingleInstance(t *testing.T) {
	req := &Request{Adapter: "sql-lora"}
	scores := scoreLoRAAffinity(req, []RoutingSnapshot{{ID: "a"}})
	assert.Equal(t, 1.0, scores["a"], "single instance always scores 1.0")
}

// TestScoreLoRAAffinity_IgnoresOutputTokens enforces INV-9 (oracle knowledge
// boundary): the scorer reads only Adapter/ResidentAdapters, never OutputTokens.
// Scores must be identical whether OutputTokens is set or zero.
func TestScoreLoRAAffinity_IgnoresOutputTokens(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "warm", ResidentAdapters: residentSetOf("sql-lora")},
		{ID: "cold"},
	}
	base := scoreLoRAAffinity(&Request{Adapter: "sql-lora"}, snapshots)
	withOut := scoreLoRAAffinity(&Request{Adapter: "sql-lora", OutputTokens: make([]TokenID, 4096)}, snapshots)
	assert.Equal(t, base, withOut, "OutputTokens must not influence routing scores (INV-9)")
}

// TestLoRAAffinity_Registered verifies the scorer is a recognized name (R8) and
// resolves to a stateless scorerFunc (nil observer) via newScorerWithObserver.
func TestLoRAAffinity_Registered(t *testing.T) {
	assert.True(t, IsValidScorer("lora-affinity"), "lora-affinity must be a valid scorer name")
	assert.Contains(t, ValidScorerNames(), "lora-affinity")
	scorer, observer := newScorerWithObserver("lora-affinity", 16, nil)
	require.NotNil(t, scorer, "lora-affinity must resolve to a scorerFunc")
	assert.Nil(t, observer, "lora-affinity is stateless (no observer)")
}

// TestLoRAAffinity_NotInProfile_RoutingUnchanged is US4 scenario 3 / INV-6: with
// the scorer absent from the weighted profile, an adapter request routes exactly
// as it would with no LoRA awareness. We assert the composite decision is
// identical with and without ResidentAdapters populated, because the default
// profile never consults them.
func TestLoRAAffinity_NotInProfile_RoutingUnchanged(t *testing.T) {
	configs := DefaultScorerConfigs() // precise-prefix-cache, queue-depth, kv-utilization
	policyWithout := NewRoutingPolicy("weighted", configs, 16, nil)
	req := &Request{Adapter: "sql-lora", InputTokens: make([]TokenID, 32)}

	// Two snapshot sets identical except for ResidentAdapters (which the default
	// profile ignores). Same routing decision proves the adapter field is inert.
	bare := []RoutingSnapshot{
		{ID: "a", QueueDepth: 5, KVUtilization: 0.5},
		{ID: "b", QueueDepth: 1, KVUtilization: 0.1},
	}
	withResident := []RoutingSnapshot{
		{ID: "a", QueueDepth: 5, KVUtilization: 0.5, ResidentAdapters: residentSetOf("sql-lora")},
		{ID: "b", QueueDepth: 1, KVUtilization: 0.1},
	}
	dBare := policyWithout.Route(req, &RouterState{Snapshots: bare})
	dResident := policyWithout.Route(req, &RouterState{Snapshots: withResident})
	assert.Equal(t, dBare.TargetInstance, dResident.TargetInstance,
		"default profile must ignore ResidentAdapters (INV-6)")
}

// === B-1: scorer registry contract tests (#1489) ===

const registryBlockSize = 16

// registryTestFixture returns the shared (req, snapshots, cacheFn) fixture for
// BC-1. The cacheFn keys MUST match the snapshot IDs so the two cache-backed
// scorers (precise-prefix-cache, no-hit-lru) exercise their real
// min-max-normalized path rather than the missing-key scores[snap.ID]=1.0
// default branch — which would hollow out BC-1's byte-identity guard for exactly
// the two param-sensitive scorers.
func registryTestFixture(t *testing.T) (*Request, []RoutingSnapshot, cacheQueryFn) {
	t.Helper()
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 10, BatchSize: 2, KVUtilization: 0.5},
		{ID: "b", QueueDepth: 5, BatchSize: 3, KVUtilization: 0.8},
	}
	cacheFn := cacheQueryFn{
		"a": func([]TokenID) int { return 4 },
		"b": func([]TokenID) int { return 1 },
	}
	// Self-enforcing guard: fail loudly at setup if a future edit adds a snapshot
	// ID or drops a cacheFn key, rather than silently degrading to the 1.0 path.
	require.Equal(t, len(cacheFn), len(snapshots))
	for _, snap := range snapshots {
		require.Contains(t, cacheFn, snap.ID)
	}
	req := &Request{ID: "r1", InputTokens: []TokenID{1, 2, 3, 4, 5, 6, 7, 8}}
	return req, snapshots, cacheFn
}

// cloneRegistry makes a shallow copy of the scorer registry. scorerConstructor
// is a func value (immutable), so a shallow copy fully isolates the clone; the
// registry-mutating tests mutate the clone and t.Cleanup restores the original.
func cloneRegistry(src map[string]scorerConstructor) map[string]scorerConstructor {
	dst := make(map[string]scorerConstructor, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

// stubScorerCtor is a minimal constructor used by the registration guard tests.
func stubScorerCtor(_ int, _ cacheQueryFn) (scorerFunc, observerFunc) {
	return scoreQueueDepth, nil
}

// TestScorerRegistry_ReproducesSwitchSelection (BC-1): the registry must map
// each built-in name to the same underlying scorer the pre-conversion switch
// did. Equality against a reference built from the same constructor/func tests
// the name→func WIRING (a mis-wire, e.g. queue-depth→scoreKVUtilization, would
// diverge). Observer is non-nil for exactly {prefix-affinity, no-hit-lru}.
func TestScorerRegistry_ReproducesSwitchSelection(t *testing.T) {
	req, snapshots, cacheFn := registryTestFixture(t)

	prefixRef, _ := newPrefixAffinityScorer(registryBlockSize)
	preciseRef, _ := newPrecisePrefixCacheScorer(cacheFn)
	noHitRef, _ := newNoHitLRUScorer(cacheFn)

	cases := []struct {
		name         string
		ref          scorerFunc
		wantObserver bool
	}{
		{"prefix-affinity", prefixRef, true},
		{"precise-prefix-cache", preciseRef, false}, // stateless ground-truth: (scorer, nil)
		{"no-hit-lru", noHitRef, true},
		{"queue-depth", scoreQueueDepth, false},
		{"kv-utilization", scoreKVUtilization, false},
		{"load-balance", scoreLoadBalance, false},
		{"active-requests", scoreActiveRequests, false},
		{"running-requests", scoreRunningRequests, false},
		{"load-aware", scoreLoadAware, false},
		{"vllm-dp", scoreVLLMDP, false},
		{"lora-affinity", scoreLoRAAffinity, false},
	}
	require.Len(t, cases, 11, "all 11 built-in scorers must be covered (BC-1)")

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotScorer, gotObserver := newScorerWithObserver(tc.name, registryBlockSize, cacheFn)
			require.NotNil(t, gotScorer)
			// exact assert.Equal, not InDelta: byte-identity guard.
			assert.Equal(t, tc.ref(req, snapshots), gotScorer(req, snapshots))
			if tc.wantObserver {
				assert.NotNil(t, gotObserver, "expected non-nil observer for %q", tc.name)
			} else {
				assert.Nil(t, gotObserver, "expected nil observer for %q", tc.name)
			}
		})
	}
}

// TestScorerRegistry_UnknownPanics (BC-2): unknown name panics (R1, no silent
// nil-return — same failure mode as the old default: arm).
func TestScorerRegistry_UnknownPanics(t *testing.T) {
	_, _, cacheFn := registryTestFixture(t)
	assert.Panics(t, func() {
		newScorerWithObserver("definitely-not-a-scorer", registryBlockSize, cacheFn)
	})
}

// TestScorerRegistry_ValidityDerivedFromRegistry (BC-3): validity ≡ the live
// registry keys, and every valid name is constructible. Bidirectional
// set-equality closes the validity/constructibility drift gap in one assertion.
func TestScorerRegistry_ValidityDerivedFromRegistry(t *testing.T) {
	_, _, cacheFn := registryTestFixture(t)

	want := make([]string, 0, len(scorerRegistry))
	for n := range scorerRegistry {
		want = append(want, n)
	}
	sort.Strings(want)
	assert.Equal(t, want, ValidScorerNames(),
		"ValidScorerNames must return exactly the registry keys, sorted")

	for _, name := range ValidScorerNames() {
		assert.True(t, IsValidScorer(name), "ValidScorerNames returned non-valid %q", name)
		assert.NotPanics(t, func() {
			newScorerWithObserver(name, registryBlockSize, cacheFn)
		}, "valid scorer %q must be constructible", name)
	}
	assert.False(t, IsValidScorer("definitely-not-a-scorer"), "unregistered name must be rejected")
}

// TestScorerRegistry_DefaultsRegistered (BC-6 hardening): every default scorer
// name validates and constructs, closing the gap where a default could drift out
// of the registry and only fail at runtime policy-build.
func TestScorerRegistry_DefaultsRegistered(t *testing.T) {
	_, _, cacheFn := registryTestFixture(t)
	for _, cfg := range DefaultScorerConfigs() {
		assert.True(t, IsValidScorer(cfg.Name), "default scorer %q must be valid", cfg.Name)
		assert.NotPanics(t, func() {
			newScorerWithObserver(cfg.Name, registryBlockSize, cacheFn)
		}, "default scorer %q must construct", cfg.Name)
	}
}

// TestScorerRegistry_RegisterAddsName (BC-4): adding a scorer is one localized
// registerScorer call — the name validates and constructs with no other edits.
func TestScorerRegistry_RegisterAddsName(t *testing.T) {
	// no t.Parallel: mutates package-global scorerRegistry
	old := scorerRegistry
	t.Cleanup(func() { scorerRegistry = old })
	scorerRegistry = cloneRegistry(old)

	const name = "test-dummy-scorer"
	require.False(t, IsValidScorer(name), "dummy must not pre-exist")
	registerScorer(name, stubScorerCtor)
	assert.True(t, IsValidScorer(name))

	_, _, cacheFn := registryTestFixture(t)
	assert.NotPanics(t, func() {
		newScorerWithObserver(name, registryBlockSize, cacheFn)
	})
}

// TestRegisterScorer_DuplicateNamePanics (BC-4 guard): duplicate registration
// panics with the duplicate-scorer message (R4). Message-asserting makes this a
// genuine fail-first test against the "unimplemented" skeleton.
func TestRegisterScorer_DuplicateNamePanics(t *testing.T) {
	// no t.Parallel: mutates package-global scorerRegistry
	old := scorerRegistry
	t.Cleanup(func() { scorerRegistry = old })
	scorerRegistry = cloneRegistry(old)

	registerScorer("dup", stubScorerCtor)
	defer func() {
		r := recover()
		require.NotNil(t, r, "second registration must panic")
		assert.Contains(t, fmt.Sprint(r), "duplicate scorer")
	}()
	registerScorer("dup", stubScorerCtor)
}

// TestRegisterScorer_EmptyNamePanics (BC-4 guard): empty-name registration
// panics (an empty name would make IsValidScorer("") true, breaking parity).
func TestRegisterScorer_EmptyNamePanics(t *testing.T) {
	// no t.Parallel: mutates package-global scorerRegistry
	old := scorerRegistry
	t.Cleanup(func() { scorerRegistry = old })
	scorerRegistry = cloneRegistry(old)

	defer func() {
		r := recover()
		require.NotNil(t, r, "empty-name registration must panic")
		assert.Contains(t, fmt.Sprint(r), "empty scorer name")
	}()
	registerScorer("", stubScorerCtor)
}

// TestParseScorerConfigs_InvalidNameErrorIsDeterministic (INV-6): the "valid: …"
// list in the error comes from sorted ValidScorerNames(), so the message is
// stable run-to-run.
func TestParseScorerConfigs_InvalidNameErrorIsDeterministic(t *testing.T) {
	_, err1 := ParseScorerConfigs("bad-name:1")
	require.Error(t, err1)
	_, err2 := ParseScorerConfigs("bad-name:1")
	require.Error(t, err2)
	assert.Equal(t, err1.Error(), err2.Error())
}
