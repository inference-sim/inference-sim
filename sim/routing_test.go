package sim

import (
	"fmt"
	"math"
	"testing"
)

// TestRoutingPolicy_Interface_Contract verifies the RoutingPolicy interface contract (BC-1).
func TestRoutingPolicy_Interface_Contract(t *testing.T) {
	// GIVEN a RoutingPolicy implementation (RoundRobin)
	policy := NewRoutingPolicy("round-robin", nil)

	// WHEN Route() is called with valid inputs
	req := &Request{ID: "req1", InputTokens: []int{1, 2, 3}}
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5},
		{ID: "instance_1", QueueDepth: 3},
	}
	decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN RoutingDecision must have TargetInstance set
	if decision.TargetInstance == "" {
		t.Errorf("Expected non-empty TargetInstance, got empty")
	}

	// THEN TargetInstance must be in snapshots
	found := false
	for _, snap := range snapshots {
		if snap.ID == decision.TargetInstance {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("TargetInstance %q not found in snapshots", decision.TargetInstance)
	}
}

// TestRoundRobin_DeterministicOrdering verifies BC-2.
func TestRoundRobin_DeterministicOrdering(t *testing.T) {
	// GIVEN RoundRobin policy
	policy := NewRoutingPolicy("round-robin", nil)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0"},
		{ID: "instance_1"},
		{ID: "instance_2"},
	}

	// WHEN 6 requests are routed
	var targets []string
	for i := 0; i < 6; i++ {
		req := &Request{ID: fmt.Sprintf("req%d", i)}
		decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: int64(i * 1000)})
		targets = append(targets, decision.TargetInstance)
	}

	// THEN requests distributed round-robin: 0, 1, 2, 0, 1, 2
	expected := []string{"instance_0", "instance_1", "instance_2", "instance_0", "instance_1", "instance_2"}
	for i, exp := range expected {
		if targets[i] != exp {
			t.Errorf("Request %d: expected %q, got %q", i, exp, targets[i])
		}
	}
}

// TestRoundRobin_EmptySnapshots_Panics verifies BC-10.
func TestRoundRobin_EmptySnapshots_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic on empty snapshots, got none")
		}
	}()

	policy := NewRoutingPolicy("round-robin", nil)
	req := &Request{ID: "req1"}
	policy.Route(req, &RouterState{Snapshots: []RoutingSnapshot{}, Clock: 1000})
}

// TestNewRoutingPolicy_UnknownName_Panics verifies BC-11.
func TestNewRoutingPolicy_UnknownName_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic on unknown policy name, got none")
		}
	}()

	NewRoutingPolicy("invalid-policy", nil)
}

// TestNewRoutingPolicy_DefaultName verifies empty string defaults to round-robin behavior.
func TestNewRoutingPolicy_DefaultName(t *testing.T) {
	policy := NewRoutingPolicy("", nil)
	if policy == nil {
		t.Fatal("Expected non-nil policy for empty string, got nil")
	}
	// Verify round-robin behavior: routes to first instance, then second
	snapshots := []RoutingSnapshot{{ID: "a"}, {ID: "b"}}
	state := &RouterState{Snapshots: snapshots, Clock: 0}
	d1 := policy.Route(&Request{ID: "r1"}, state)
	d2 := policy.Route(&Request{ID: "r2"}, state)
	if d1.TargetInstance != "a" || d2.TargetInstance != "b" {
		t.Errorf("Expected round-robin (a, b), got (%s, %s)", d1.TargetInstance, d2.TargetInstance)
	}
}

// TestLeastLoaded_LoadBasedSelection verifies BC-3.
func TestLeastLoaded_LoadBasedSelection(t *testing.T) {
	policy := NewRoutingPolicy("least-loaded", nil)

	tests := []struct {
		name      string
		snapshots []RoutingSnapshot
		expected  string
	}{
		{
			name: "instance 1 has lowest load",
			snapshots: []RoutingSnapshot{
				{ID: "instance_0", QueueDepth: 10, BatchSize: 5},
				{ID: "instance_1", QueueDepth: 3, BatchSize: 2},
				{ID: "instance_2", QueueDepth: 7, BatchSize: 8},
			},
			expected: "instance_1",
		},
		{
			name: "tie broken by first occurrence (lowest index)",
			snapshots: []RoutingSnapshot{
				{ID: "instance_0", QueueDepth: 5, BatchSize: 5},
				{ID: "instance_1", QueueDepth: 8, BatchSize: 2},
				{ID: "instance_2", QueueDepth: 3, BatchSize: 12},
			},
			expected: "instance_0",
		},
		{
			name: "all instances equal load",
			snapshots: []RoutingSnapshot{
				{ID: "instance_0", QueueDepth: 5, BatchSize: 5},
				{ID: "instance_1", QueueDepth: 5, BatchSize: 5},
			},
			expected: "instance_0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &Request{ID: "req1"}
			decision := policy.Route(req, &RouterState{Snapshots: tt.snapshots, Clock: 1000})
			if decision.TargetInstance != tt.expected {
				t.Errorf("Expected %q, got %q", tt.expected, decision.TargetInstance)
			}
		})
	}
}

// TestLeastLoaded_PendingRequests_BreaksTie verifies that PendingRequests is included
// in load calculation, preventing pile-on at high request rates (#175).
func TestLeastLoaded_PendingRequests_BreaksTie(t *testing.T) {
	policy := NewRoutingPolicy("least-loaded", nil)

	// GIVEN two instances with equal QueueDepth+BatchSize but different PendingRequests
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 3, PendingRequests: 4},
		{ID: "instance_1", QueueDepth: 5, BatchSize: 3, PendingRequests: 0},
	}

	// WHEN routing a request
	req := &Request{ID: "req1"}
	decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN instance_1 is chosen (load=8 vs load=12)
	if decision.TargetInstance != "instance_1" {
		t.Errorf("expected instance_1 (fewer pending), got %q", decision.TargetInstance)
	}
}

// TestLeastLoaded_EmptySnapshots_Panics verifies BC-10.
func TestLeastLoaded_EmptySnapshots_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic on empty snapshots, got none")
		}
	}()

	policy := NewRoutingPolicy("least-loaded", nil)
	req := &Request{ID: "req1"}
	policy.Route(req, &RouterState{Snapshots: []RoutingSnapshot{}, Clock: 1000})
}

// === WeightedScoring Tests (rewritten for scorer pipeline) ===

// TestWeightedScoring_DefaultScorers_RoutesToBestComposite verifies BC-17-6 (argmax).
func TestWeightedScoring_DefaultScorers_RoutesToBestComposite(t *testing.T) {
	// GIVEN weighted policy with default scorers and instances with varying load/utilization
	policy := NewRoutingPolicy("weighted", nil)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 10, BatchSize: 0, KVUtilization: 0.8},
		{ID: "instance_1", QueueDepth: 2, BatchSize: 0, KVUtilization: 0.2},
		{ID: "instance_2", QueueDepth: 5, BatchSize: 0, KVUtilization: 0.5},
	}

	// WHEN routing a request
	req := &Request{ID: "req1"}
	decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN instance_1 wins (lowest load + lowest utilization = highest composite score)
	if decision.TargetInstance != "instance_1" {
		t.Errorf("expected instance_1 (best composite), got %q", decision.TargetInstance)
	}
	// THEN Scores map has all instances
	if len(decision.Scores) != 3 {
		t.Errorf("expected 3 scores, got %d", len(decision.Scores))
	}
}

// TestWeightedScoring_HighestScoreWins verifies BC-17-6: target has the highest score.
func TestWeightedScoring_HighestScoreWins(t *testing.T) {
	policy := NewRoutingPolicy("weighted", nil)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, KVUtilization: 0.5},
		{ID: "instance_1", QueueDepth: 1, KVUtilization: 0.1},
		{ID: "instance_2", QueueDepth: 8, KVUtilization: 0.9},
	}

	req := &Request{ID: "req1"}
	decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	// Behavioral invariant: target has the highest score
	targetScore, ok := decision.Scores[decision.TargetInstance]
	if !ok {
		t.Fatalf("target %q not in Scores map", decision.TargetInstance)
	}
	for id, score := range decision.Scores {
		if score > targetScore {
			t.Errorf("instance %q has higher score (%f) than target %q (%f)",
				id, score, decision.TargetInstance, targetScore)
		}
	}
}

// TestWeightedScoring_WeightsNormalized verifies BC-17-2: proportional weights produce identical routing.
func TestWeightedScoring_WeightsNormalized(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 8, KVUtilization: 0.2},
		{ID: "instance_1", QueueDepth: 2, KVUtilization: 0.8},
	}

	// WHEN using unnormalized weights [3,2,2] and scaled equivalent [6,4,4] (exact same ratios)
	unnormalized := NewRoutingPolicy("weighted", []ScorerConfig{
		{Name: "queue-depth", Weight: 3.0},
		{Name: "kv-utilization", Weight: 2.0},
		{Name: "load-balance", Weight: 2.0},
	})
	scaled := NewRoutingPolicy("weighted", []ScorerConfig{
		{Name: "queue-depth", Weight: 6.0},
		{Name: "kv-utilization", Weight: 4.0},
		{Name: "load-balance", Weight: 4.0},
	})

	d1 := unnormalized.Route(&Request{ID: "r1"}, &RouterState{Snapshots: snapshots, Clock: 1000})
	d2 := scaled.Route(&Request{ID: "r2"}, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN both produce same target (same weight ratios)
	if d1.TargetInstance != d2.TargetInstance {
		t.Errorf("expected same decision for proportional weights, got %q vs %q",
			d1.TargetInstance, d2.TargetInstance)
	}
}

// TestWeightedScoring_SingleScorer_LoadBalance verifies single-scorer configuration.
func TestWeightedScoring_SingleScorer_LoadBalance(t *testing.T) {
	policy := NewRoutingPolicy("weighted", []ScorerConfig{{Name: "load-balance", Weight: 1.0}})
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 10},
		{ID: "instance_1", QueueDepth: 2},
	}

	decision := policy.Route(&Request{ID: "r1"}, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN instance_1 wins (lower load → higher load-balance score)
	if decision.TargetInstance != "instance_1" {
		t.Errorf("expected instance_1 (lower load), got %q", decision.TargetInstance)
	}
}

// TestWeightedScoring_DifferentScorerWeights_FlipDecision verifies weight sensitivity.
func TestWeightedScoring_DifferentScorerWeights_FlipDecision(t *testing.T) {
	// GIVEN instances where queue-depth and kv-utilization disagree:
	// instance_0: high load (queue-depth disfavors), low utilization (kv-util favors)
	// instance_1: low load (queue-depth favors), high utilization (kv-util disfavors)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 10, KVUtilization: 0.1},
		{ID: "instance_1", QueueDepth: 1, KVUtilization: 0.9},
	}

	// WHEN using queue-depth-dominant weights
	qdDominant := NewRoutingPolicy("weighted", []ScorerConfig{
		{Name: "queue-depth", Weight: 9.0},
		{Name: "kv-utilization", Weight: 1.0},
	})
	d1 := qdDominant.Route(&Request{ID: "r1"}, &RouterState{Snapshots: snapshots, Clock: 1000})

	// WHEN using kv-utilization-dominant weights
	kvDominant := NewRoutingPolicy("weighted", []ScorerConfig{
		{Name: "queue-depth", Weight: 1.0},
		{Name: "kv-utilization", Weight: 9.0},
	})
	d2 := kvDominant.Route(&Request{ID: "r2"}, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN different weights produce different decisions
	if d1.TargetInstance == d2.TargetInstance {
		t.Errorf("expected different decisions for different scorer weights, both chose %q", d1.TargetInstance)
	}
}

// TestWeightedScoring_AllIdle_NoDivisionByZero verifies BC-17-9 (no NaN/Inf).
func TestWeightedScoring_AllIdle_NoDivisionByZero(t *testing.T) {
	policy := NewRoutingPolicy("weighted", nil)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 0, BatchSize: 0, KVUtilization: 0.0},
		{ID: "instance_1", QueueDepth: 0, BatchSize: 0, KVUtilization: 0.0},
	}

	req := &Request{ID: "req1"}
	decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	// All idle: equal scores everywhere → first occurrence wins
	if decision.TargetInstance != "instance_0" {
		t.Errorf("Expected instance_0 (tie broken by first occurrence), got %q", decision.TargetInstance)
	}

	for id, score := range decision.Scores {
		if math.IsNaN(score) || math.IsInf(score, 0) {
			t.Errorf("Score for %s is not finite: %f", id, score)
		}
	}
}

// TestWeightedScoring_EmptySnapshots_Panics verifies BC-17-8.
func TestWeightedScoring_EmptySnapshots_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic on empty snapshots, got none")
		}
	}()

	policy := NewRoutingPolicy("weighted", nil)
	policy.Route(&Request{ID: "req1"}, &RouterState{Snapshots: []RoutingSnapshot{}, Clock: 1000})
}

// TestWeightedScoring_NilConfigs_UsesDefaults verifies default scorer configuration.
func TestWeightedScoring_NilConfigs_UsesDefaults(t *testing.T) {
	// GIVEN nil scorerConfigs
	policy := NewRoutingPolicy("weighted", nil)

	// WHEN routing a request with differentiated snapshots
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 5, KVUtilization: 0.5},
		{ID: "b", QueueDepth: 1, KVUtilization: 0.1},
	}
	decision := policy.Route(&Request{ID: "r1"}, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN a valid decision is made (defaults applied)
	if decision.TargetInstance != "a" && decision.TargetInstance != "b" {
		t.Errorf("invalid target %q", decision.TargetInstance)
	}
	if decision.Scores == nil || len(decision.Scores) != 2 {
		t.Errorf("expected 2 scores, got %v", decision.Scores)
	}
}

// TestWeightedScoring_PendingRequests_AffectsScorers verifies that PendingRequests
// affects queue-depth and load-balance scorers.
func TestWeightedScoring_PendingRequests_AffectsScorers(t *testing.T) {
	policy := NewRoutingPolicy("weighted", []ScorerConfig{{Name: "load-balance", Weight: 1.0}})

	// GIVEN two instances with equal QueueDepth but different PendingRequests
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 0, PendingRequests: 3},
		{ID: "instance_1", QueueDepth: 0, PendingRequests: 0},
	}

	// WHEN routing a request
	decision := policy.Route(&Request{ID: "r1"}, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN instance_1 wins (lower effective load)
	if decision.TargetInstance != "instance_1" {
		t.Errorf("expected instance_1 (no pending), got %q", decision.TargetInstance)
	}
}

// TestWeightedScoring_EmptyConfigs_UsesDefaults verifies that empty scorer slice falls back to defaults.
func TestWeightedScoring_EmptyConfigs_UsesDefaults(t *testing.T) {
	policy := NewRoutingPolicy("weighted", []ScorerConfig{})
	snapshots := []RoutingSnapshot{{ID: "a", QueueDepth: 1}}
	decision := policy.Route(&Request{ID: "r1"}, &RouterState{Snapshots: snapshots, Clock: 1000})
	if decision.TargetInstance != "a" {
		t.Errorf("expected 'a', got %q", decision.TargetInstance)
	}
}

// === PrefixAffinity Tests (unchanged from before — just signature update) ===

// TestPrefixAffinity_CacheHit verifies BC-5 (cache-aware routing).
func TestPrefixAffinity_CacheHit(t *testing.T) {
	policy := NewRoutingPolicy("prefix-affinity", nil)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 10, BatchSize: 5},
		{ID: "instance_1", QueueDepth: 2, BatchSize: 1},
	}

	// First request with prefix [1, 2, 3] routed (cache miss → LeastLoaded)
	req1 := &Request{ID: "req1", InputTokens: []int{1, 2, 3}}
	decision1 := policy.Route(req1, &RouterState{Snapshots: snapshots, Clock: 1000})
	firstTarget := decision1.TargetInstance

	// Second request with same prefix → cache hit
	req2 := &Request{ID: "req2", InputTokens: []int{1, 2, 3}}
	decision2 := policy.Route(req2, &RouterState{Snapshots: snapshots, Clock: 2000})

	if decision2.TargetInstance != firstTarget {
		t.Errorf("Expected cache hit routing to %q, got %q", firstTarget, decision2.TargetInstance)
	}
}

// TestPrefixAffinity_CacheMiss verifies fallback to LeastLoaded.
func TestPrefixAffinity_CacheMiss(t *testing.T) {
	policy := NewRoutingPolicy("prefix-affinity", nil)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 10, BatchSize: 5}, // load=15
		{ID: "instance_1", QueueDepth: 2, BatchSize: 1},  // load=3 (min)
	}

	req := &Request{ID: "req1", InputTokens: []int{7, 8, 9}}
	decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	if decision.TargetInstance != "instance_1" {
		t.Errorf("Expected fallback to least-loaded (instance_1), got %q", decision.TargetInstance)
	}
}

// TestPrefixAffinity_DifferentPrefixes verifies distinct hashing.
func TestPrefixAffinity_DifferentPrefixes(t *testing.T) {
	policy := NewRoutingPolicy("prefix-affinity", nil)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 5},
		{ID: "instance_1", QueueDepth: 5, BatchSize: 5},
	}

	req1 := &Request{ID: "req1", InputTokens: []int{1, 2, 3}}
	req2 := &Request{ID: "req2", InputTokens: []int{4, 5, 6}}

	decision1 := policy.Route(req1, &RouterState{Snapshots: snapshots, Clock: 1000})
	decision2 := policy.Route(req2, &RouterState{Snapshots: snapshots, Clock: 2000})

	validIDs := map[string]bool{"instance_0": true, "instance_1": true}
	if !validIDs[decision1.TargetInstance] || !validIDs[decision2.TargetInstance] {
		t.Errorf("Invalid routing decisions: %q, %q", decision1.TargetInstance, decision2.TargetInstance)
	}
}

// TestPrefixAffinity_HashMatchesKVCache verifies hash format consistency.
func TestPrefixAffinity_HashMatchesKVCache(t *testing.T) {
	hash1 := hashTokens([]int{1, 2, 3})
	hash2 := hashTokens([]int{3, 2, 1})
	if hash1 == hash2 {
		t.Errorf("Expected different hashes for [1,2,3] and [3,2,1], got same: %s", hash1)
	}

	// Deterministic
	hash3 := hashTokens([]int{1, 2, 3})
	if hash1 != hash3 {
		t.Errorf("Expected identical hash for same tokens, got %q and %q", hash1, hash3)
	}
}

// TestPrefixAffinity_NoStateLeak verifies BC-8 (no cross-simulation state leak).
func TestPrefixAffinity_NoStateLeak(t *testing.T) {
	policy1 := NewRoutingPolicy("prefix-affinity", nil)
	policy2 := NewRoutingPolicy("prefix-affinity", nil)

	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 5},
		{ID: "instance_1", QueueDepth: 5, BatchSize: 5},
	}

	// policy1 routes and builds cache
	req1 := &Request{ID: "req1", InputTokens: []int{1, 2, 3}}
	policy1.Route(req1, &RouterState{Snapshots: snapshots, Clock: 1000})

	// policy2 routes same prefix — cache miss (independent state)
	req2 := &Request{ID: "req2", InputTokens: []int{1, 2, 3}}
	decision2 := policy2.Route(req2, &RouterState{Snapshots: snapshots, Clock: 1000})

	// Falls back to LeastLoaded → first occurrence (instance_0)
	if decision2.TargetInstance != "instance_0" {
		t.Errorf("Expected independent state (fallback to least-loaded), got %q", decision2.TargetInstance)
	}
}

// TestPrefixAffinity_EmptySnapshots_Panics verifies BC-10.
func TestPrefixAffinity_EmptySnapshots_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic on empty snapshots, got none")
		}
	}()

	policy := NewRoutingPolicy("prefix-affinity", nil)
	policy.Route(&Request{ID: "req1", InputTokens: []int{1}}, &RouterState{Snapshots: []RoutingSnapshot{}, Clock: 1000})
}

// TestPrefixAffinity_StaleEntry_FallsBackToLeastLoaded verifies stale cache path.
func TestPrefixAffinity_StaleEntry_FallsBackToLeastLoaded(t *testing.T) {
	policy := NewRoutingPolicy("prefix-affinity", nil)

	// First call: maps prefix to instance_1 (least loaded)
	snapshots1 := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 10, BatchSize: 5},
		{ID: "instance_1", QueueDepth: 2, BatchSize: 1},
	}
	req := &Request{ID: "req1", InputTokens: []int{1, 2, 3}}
	decision1 := policy.Route(req, &RouterState{Snapshots: snapshots1, Clock: 1000})
	if decision1.TargetInstance != "instance_1" {
		t.Fatalf("setup: expected instance_1, got %q", decision1.TargetInstance)
	}

	// Second call: instance_1 is gone, only instance_0 and instance_2
	snapshots2 := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 10, BatchSize: 5},
		{ID: "instance_2", QueueDepth: 1, BatchSize: 0},
	}
	req2 := &Request{ID: "req2", InputTokens: []int{1, 2, 3}}
	decision2 := policy.Route(req2, &RouterState{Snapshots: snapshots2, Clock: 2000})

	// Should fallback to least-loaded (instance_2)
	if decision2.TargetInstance != "instance_2" {
		t.Errorf("Expected fallback to instance_2, got %q", decision2.TargetInstance)
	}
}

// TestRoutingDecision_PriorityHint_DefaultZero verifies BC-9: default Priority is zero.
func TestRoutingDecision_PriorityHint_DefaultZero(t *testing.T) {
	policyNames := []string{"round-robin", "least-loaded", "weighted", "prefix-affinity"}

	for _, name := range policyNames {
		t.Run(name, func(t *testing.T) {
			policy := NewRoutingPolicy(name, nil)
			state := &RouterState{
				Snapshots: []RoutingSnapshot{{ID: "instance_0", QueueDepth: 1}},
				Clock:     1000,
			}
			req := &Request{ID: "req1", InputTokens: []int{1, 2, 3}}
			decision := policy.Route(req, state)

			if decision.Priority != 0 {
				t.Errorf("expected default Priority 0, got %f", decision.Priority)
			}
		})
	}
}

// === AlwaysBusiest Tests ===

// TestAlwaysBusiest_RouteToHighestLoad verifies BC-6.
func TestAlwaysBusiest_RouteToHighestLoad(t *testing.T) {
	policy := NewRoutingPolicy("always-busiest", nil)
	req := &Request{ID: "r1", InputTokens: []int{1, 2}}
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 2, BatchSize: 1},
		{ID: "instance_1", QueueDepth: 10, BatchSize: 5},
		{ID: "instance_2", QueueDepth: 0, BatchSize: 0},
	}

	decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	if decision.TargetInstance != "instance_1" {
		t.Errorf("expected instance_1 (busiest), got %q", decision.TargetInstance)
	}
}

// TestAlwaysBusiest_PendingRequests_IncludedInLoad verifies that PendingRequests
// is included in AlwaysBusiest load calculation (#175).
func TestAlwaysBusiest_PendingRequests_IncludedInLoad(t *testing.T) {
	policy := NewRoutingPolicy("always-busiest", nil)

	// GIVEN two instances with equal QueueDepth+BatchSize but different PendingRequests
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 3, PendingRequests: 0},
		{ID: "instance_1", QueueDepth: 5, BatchSize: 3, PendingRequests: 4},
	}

	req := &Request{ID: "r1"}
	decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN instance_1 is chosen (load=12 vs load=8)
	if decision.TargetInstance != "instance_1" {
		t.Errorf("expected instance_1 (more pending = busier), got %q", decision.TargetInstance)
	}
}

// TestAlwaysBusiest_EmptySnapshots_Panics verifies defensive convention.
func TestAlwaysBusiest_EmptySnapshots_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on empty snapshots")
		}
	}()
	policy := NewRoutingPolicy("always-busiest", nil)
	policy.Route(&Request{ID: "r1"}, &RouterState{Snapshots: []RoutingSnapshot{}, Clock: 0})
}
