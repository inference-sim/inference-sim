package sim

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestExampleConfigs_EPPEstimatePrefix verifies that epp-estimate-prefix.yaml
// loads correctly and configures the expected scorers with correct weights.
func TestExampleConfigs_EPPEstimatePrefix(t *testing.T) {
	// GIVEN the epp-estimate-prefix.yaml example config
	path := filepath.Join("..", "examples", "epp-estimate-prefix.yaml")
	bundle, err := LoadPolicyBundle(path)
	require.NoError(t, err, "failed to load epp-estimate-prefix.yaml")

	// THEN validation passes
	require.NoError(t, bundle.Validate(), "validation failed")

	// THEN routing policy is weighted
	assert.Equal(t, "weighted", bundle.Routing.Policy)

	// THEN scorers match llm-d's epp-estimate-prefix-cache-config.yaml mapping
	require.Len(t, bundle.Routing.Scorers, 2, "expected 2 scorers")

	// Verify scorer names and weights (1:1 ratio)
	scorerMap := make(map[string]float64)
	for _, s := range bundle.Routing.Scorers {
		scorerMap[s.Name] = s.Weight
	}

	assert.Equal(t, 1.0, scorerMap["prefix-affinity"], "prefix-affinity weight")
	assert.Equal(t, 1.0, scorerMap["load-balance"], "load-balance weight")

	// THEN admission policy is always-admit
	assert.Equal(t, "always-admit", bundle.Admission.Policy)

	// THEN scheduler is fcfs
	assert.Equal(t, "fcfs", bundle.Scheduler)
}

// TestExampleConfigs_EPPPrecisePrefix verifies that epp-precise-prefix.yaml
// loads correctly and configures the expected scorers with correct weights.
func TestExampleConfigs_EPPPrecisePrefix(t *testing.T) {
	// GIVEN the epp-precise-prefix.yaml example config
	path := filepath.Join("..", "examples", "epp-precise-prefix.yaml")
	bundle, err := LoadPolicyBundle(path)
	require.NoError(t, err, "failed to load epp-precise-prefix.yaml")

	// THEN validation passes
	require.NoError(t, bundle.Validate(), "validation failed")

	// THEN routing policy is weighted
	assert.Equal(t, "weighted", bundle.Routing.Policy)

	// THEN scorers match llm-d's epp-precise-prefix-cache-config.yaml mapping
	require.Len(t, bundle.Routing.Scorers, 3, "expected 3 scorers")

	// Verify scorer names and weights (2:1:1 ratio)
	scorerMap := make(map[string]float64)
	for _, s := range bundle.Routing.Scorers {
		scorerMap[s.Name] = s.Weight
	}

	assert.Equal(t, 2.0, scorerMap["prefix-affinity"], "prefix-affinity weight")
	assert.Equal(t, 1.0, scorerMap["kv-utilization"], "kv-utilization weight")
	assert.Equal(t, 1.0, scorerMap["queue-depth"], "queue-depth weight")

	// THEN admission policy is always-admit
	assert.Equal(t, "always-admit", bundle.Admission.Policy)

	// THEN scheduler is fcfs
	assert.Equal(t, "fcfs", bundle.Scheduler)
}

// TestExampleConfigs_EPPEstimatePrefix_RoutingBehavior verifies that the
// epp-estimate-prefix configuration produces expected routing behavior.
func TestExampleConfigs_EPPEstimatePrefix_RoutingBehavior(t *testing.T) {
	// GIVEN the epp-estimate-prefix.yaml config
	path := filepath.Join("..", "examples", "epp-estimate-prefix.yaml")
	bundle, err := LoadPolicyBundle(path)
	require.NoError(t, err)

	// WHEN creating a routing policy from the config
	policy := NewRoutingPolicy(bundle.Routing.Policy, bundle.Routing.Scorers)

	// GIVEN instances with different loads
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 10, BatchSize: 5, KVUtilization: 0.5},
		{ID: "instance_1", QueueDepth: 2, BatchSize: 1, KVUtilization: 0.5},
		{ID: "instance_2", QueueDepth: 5, BatchSize: 3, KVUtilization: 0.5},
	}

	// WHEN routing a request
	req := &Request{ID: "req1", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}
	decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN a valid decision is made
	assert.NotEmpty(t, decision.TargetInstance)
	assert.NotNil(t, decision.Scores)

	// THEN all instances have scores
	assert.Len(t, decision.Scores, 3)

	// THEN target has highest score (argmax invariant)
	targetScore := decision.Scores[decision.TargetInstance]
	for id, score := range decision.Scores {
		assert.LessOrEqual(t, score, targetScore, "instance %s has higher score than target", id)
	}
}

// TestExampleConfigs_EPPPrecisePrefix_RoutingBehavior verifies that the
// epp-precise-prefix configuration produces expected routing behavior.
func TestExampleConfigs_EPPPrecisePrefix_RoutingBehavior(t *testing.T) {
	// GIVEN the epp-precise-prefix.yaml config
	path := filepath.Join("..", "examples", "epp-precise-prefix.yaml")
	bundle, err := LoadPolicyBundle(path)
	require.NoError(t, err)

	// WHEN creating a routing policy from the config
	policy := NewRoutingPolicy(bundle.Routing.Policy, bundle.Routing.Scorers)

	// GIVEN instances with different loads and utilizations
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 10, BatchSize: 0, KVUtilization: 0.8},
		{ID: "instance_1", QueueDepth: 2, BatchSize: 0, KVUtilization: 0.2},
		{ID: "instance_2", QueueDepth: 5, BatchSize: 0, KVUtilization: 0.5},
	}

	// WHEN routing a request
	req := &Request{ID: "req1", InputTokens: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}}
	decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN a valid decision is made
	assert.NotEmpty(t, decision.TargetInstance)
	assert.NotNil(t, decision.Scores)

	// THEN all instances have scores
	assert.Len(t, decision.Scores, 3)

	// THEN target has highest score (argmax invariant)
	targetScore := decision.Scores[decision.TargetInstance]
	for id, score := range decision.Scores {
		assert.LessOrEqual(t, score, targetScore, "instance %s has higher score than target", id)
	}
}

// TestExampleConfigs_EPPPrecisePrefix_WeightRatioEffect verifies that the 2:1:1
// weight ratio in epp-precise-prefix prioritizes prefix-affinity over load/utilization.
func TestExampleConfigs_EPPPrecisePrefix_WeightRatioEffect(t *testing.T) {
	// GIVEN the epp-precise-prefix.yaml config
	path := filepath.Join("..", "examples", "epp-precise-prefix.yaml")
	bundle, err := LoadPolicyBundle(path)
	require.NoError(t, err)

	// WHEN creating a routing policy from the config
	policy := NewRoutingPolicy(bundle.Routing.Policy, bundle.Routing.Scorers)

	// GIVEN instances where prefix-affinity and load-balancing disagree:
	// - First, route a request to build prefix cache
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 1, BatchSize: 0, KVUtilization: 0.1}, // low load
		{ID: "instance_1", QueueDepth: 5, BatchSize: 0, KVUtilization: 0.5}, // higher load
	}

	// Route first request (no prefix cache yet)
	tokens := make([]int, 32) // 2 blocks at block_size=16
	for i := range tokens {
		tokens[i] = i + 1
	}
	req1 := &Request{ID: "req1", InputTokens: tokens}
	decision1 := policy.Route(req1, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN first request goes to instance_0 (lowest load, no prefix cache yet)
	// The prefix-affinity scorer won't help since no cache exists yet
	// But load-balance and queue-depth favor instance_0
	assert.Equal(t, "instance_0", decision1.TargetInstance, "first request should go to lowest load instance")

	// NOW flip the load: instance_0 becomes busy, instance_1 becomes idle
	// But instance_0 has the prefix cached
	snapshots2 := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 10, BatchSize: 5, KVUtilization: 0.9}, // busy, but has cache
		{ID: "instance_1", QueueDepth: 0, BatchSize: 0, KVUtilization: 0.1},  // idle, no cache
	}

	// Route second request with SAME prefix
	req2 := &Request{ID: "req2", InputTokens: tokens}
	decision2 := policy.Route(req2, &RouterState{Snapshots: snapshots2, Clock: 2000})

	// THEN with 2:1:1 weights, prefix-affinity (weight 2) should outweigh
	// queue-depth (weight 1) + kv-utilization (weight 1), so instance_0 wins
	// because it has the prefix cached
	assert.Equal(t, "instance_0", decision2.TargetInstance,
		"prefix-affinity (weight 2) should outweigh load-balancing (weight 1+1) for cached prefix")
}

// TestExampleConfigs_EPPEstimatePrefix_LoadBalanceFormula verifies that the
// load-balance scorer uses the inverse transform formula: 1/(1+load).
func TestExampleConfigs_EPPEstimatePrefix_LoadBalanceFormula(t *testing.T) {
	// GIVEN the epp-estimate-prefix.yaml config
	path := filepath.Join("..", "examples", "epp-estimate-prefix.yaml")
	bundle, err := LoadPolicyBundle(path)
	require.NoError(t, err)

	// Verify load-balance is present
	hasLoadBalance := false
	for _, s := range bundle.Routing.Scorers {
		if s.Name == "load-balance" {
			hasLoadBalance = true
			break
		}
	}
	assert.True(t, hasLoadBalance, "load-balance scorer should be present")

	// WHEN using load-balance scorer directly
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 0, BatchSize: 0},  // load=0 → score=1/(1+0)=1.0
		{ID: "b", QueueDepth: 1, BatchSize: 0},  // load=1 → score=1/(1+1)=0.5
		{ID: "c", QueueDepth: 9, BatchSize: 0},  // load=9 → score=1/(1+9)=0.1
	}

	scores := scoreLoadBalance(nil, snapshots)

	// THEN scores follow 1/(1+load) formula
	assert.InDelta(t, 1.0, scores["a"], 0.001, "load=0 → score=1.0")
	assert.InDelta(t, 0.5, scores["b"], 0.001, "load=1 → score=0.5")
	assert.InDelta(t, 0.1, scores["c"], 0.001, "load=9 → score=0.1")
}

// TestExampleConfigs_EPPPrecisePrefix_QueueDepthFormula verifies that the
// queue-depth scorer uses min-max normalization.
func TestExampleConfigs_EPPPrecisePrefix_QueueDepthFormula(t *testing.T) {
	// GIVEN the epp-precise-prefix.yaml config
	path := filepath.Join("..", "examples", "epp-precise-prefix.yaml")
	bundle, err := LoadPolicyBundle(path)
	require.NoError(t, err)

	// Verify queue-depth is present
	hasQueueDepth := false
	for _, s := range bundle.Routing.Scorers {
		if s.Name == "queue-depth" {
			hasQueueDepth = true
			break
		}
	}
	assert.True(t, hasQueueDepth, "queue-depth scorer should be present")

	// WHEN using queue-depth scorer directly
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 0, BatchSize: 0},   // load=0 (min)
		{ID: "b", QueueDepth: 5, BatchSize: 0},   // load=5 (mid)
		{ID: "c", QueueDepth: 10, BatchSize: 0},  // load=10 (max)
	}

	scores := scoreQueueDepth(nil, snapshots)

	// THEN scores follow min-max normalization: (max-val)/(max-min)
	// min=0, max=10
	// a: (10-0)/(10-0) = 1.0
	// b: (10-5)/(10-0) = 0.5
	// c: (10-10)/(10-0) = 0.0
	assert.InDelta(t, 1.0, scores["a"], 0.001, "min load → score=1.0")
	assert.InDelta(t, 0.5, scores["b"], 0.001, "mid load → score=0.5")
	assert.InDelta(t, 0.0, scores["c"], 0.001, "max load → score=0.0")
}

// TestExampleConfigs_EPPPrecisePrefix_KVUtilizationFormula verifies that the
// kv-utilization scorer uses: score = 1 - utilization.
func TestExampleConfigs_EPPPrecisePrefix_KVUtilizationFormula(t *testing.T) {
	// GIVEN the epp-precise-prefix.yaml config
	path := filepath.Join("..", "examples", "epp-precise-prefix.yaml")
	bundle, err := LoadPolicyBundle(path)
	require.NoError(t, err)

	// Verify kv-utilization is present
	hasKVUtil := false
	for _, s := range bundle.Routing.Scorers {
		if s.Name == "kv-utilization" {
			hasKVUtil = true
			break
		}
	}
	assert.True(t, hasKVUtil, "kv-utilization scorer should be present")

	// WHEN using kv-utilization scorer directly
	snapshots := []RoutingSnapshot{
		{ID: "a", KVUtilization: 0.0},  // score = 1 - 0 = 1.0
		{ID: "b", KVUtilization: 0.3},  // score = 1 - 0.3 = 0.7
		{ID: "c", KVUtilization: 1.0},  // score = 1 - 1 = 0.0
	}

	scores := scoreKVUtilization(nil, snapshots)

	// THEN scores follow 1-utilization formula
	assert.InDelta(t, 1.0, scores["a"], 0.001, "util=0 → score=1.0")
	assert.InDelta(t, 0.7, scores["b"], 0.001, "util=0.3 → score=0.7")
	assert.InDelta(t, 0.0, scores["c"], 0.001, "util=1.0 → score=0.0")
}
