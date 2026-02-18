package sim

import (
	"fmt"
	"math"
	"testing"
)

// TestRoutingPolicy_Interface_Contract verifies the RoutingPolicy interface contract (BC-1).
func TestRoutingPolicy_Interface_Contract(t *testing.T) {
	// GIVEN a RoutingPolicy implementation (RoundRobin)
	policy := NewRoutingPolicy("round-robin", 0, 0)

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
	policy := NewRoutingPolicy("round-robin", 0, 0)
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

	policy := NewRoutingPolicy("round-robin", 0, 0)
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

	NewRoutingPolicy("invalid-policy", 0, 0)
}

// TestNewRoutingPolicy_DefaultName verifies empty string defaults to round-robin behavior.
func TestNewRoutingPolicy_DefaultName(t *testing.T) {
	policy := NewRoutingPolicy("", 0, 0)
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
	policy := NewRoutingPolicy("least-loaded", 0, 0)

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

// TestLeastLoaded_EmptySnapshots_Panics verifies BC-10.
func TestLeastLoaded_EmptySnapshots_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic on empty snapshots, got none")
		}
	}()

	policy := NewRoutingPolicy("least-loaded", 0, 0)
	req := &Request{ID: "req1"}
	policy.Route(req, &RouterState{Snapshots: []RoutingSnapshot{}, Clock: 1000})
}

// TestWeightedScoring_MultiFactor verifies BC-4.
func TestWeightedScoring_MultiFactor(t *testing.T) {
	policy := NewRoutingPolicy("weighted", 0.6, 0.4)

	tests := []struct {
		name      string
		snapshots []RoutingSnapshot
		expected  string
		reason    string
	}{
		{
			name: "instance 1 wins on more free KV blocks",
			snapshots: []RoutingSnapshot{
				{ID: "instance_0", QueueDepth: 5, BatchSize: 5, FreeKVBlocks: 200},
				{ID: "instance_1", QueueDepth: 5, BatchSize: 5, FreeKVBlocks: 800},
			},
			expected: "instance_1",
			reason:   "equal load; instance_1 wins on more FreeKVBlocks",
		},
		{
			name: "instance 0 wins on low queue depth",
			snapshots: []RoutingSnapshot{
				{ID: "instance_0", QueueDepth: 2, BatchSize: 2, FreeKVBlocks: 500},
				{ID: "instance_1", QueueDepth: 8, BatchSize: 2, FreeKVBlocks: 500},
			},
			expected: "instance_0",
			reason:   "equal FreeKVBlocks; instance_0 wins on lower load",
		},
		{
			name: "all equal scores, first occurrence wins",
			snapshots: []RoutingSnapshot{
				{ID: "instance_0", QueueDepth: 5, BatchSize: 5, FreeKVBlocks: 500},
				{ID: "instance_1", QueueDepth: 5, BatchSize: 5, FreeKVBlocks: 500},
			},
			expected: "instance_0",
			reason:   "tie broken by first occurrence in snapshot order",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &Request{ID: "req1"}
			decision := policy.Route(req, &RouterState{Snapshots: tt.snapshots, Clock: 1000})
			if decision.TargetInstance != tt.expected {
				t.Errorf("%s: expected %q, got %q", tt.reason, tt.expected, decision.TargetInstance)
			}
			if decision.Scores == nil || len(decision.Scores) != len(tt.snapshots) {
				t.Errorf("Expected Scores map with %d entries, got %v", len(tt.snapshots), decision.Scores)
			}
		})
	}
}

// TestWeightedScoring_UniformLoad verifies cache differentiation when load is equal.
func TestWeightedScoring_UniformLoad(t *testing.T) {
	policy := NewRoutingPolicy("weighted", 0.6, 0.4)

	// All instances have identical load; instance_0 has more FreeKVBlocks → wins.
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 5, FreeKVBlocks: 700},
		{ID: "instance_1", QueueDepth: 5, BatchSize: 5, FreeKVBlocks: 300},
	}

	req := &Request{ID: "req1"}
	decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	// instance_0 wins on more FreeKVBlocks (load scores cancel out)
	if decision.TargetInstance != "instance_0" {
		t.Errorf("Expected instance_0 to win on cache score alone, got %q", decision.TargetInstance)
	}
}

// TestWeightedScoring_NegativeWeights_Panics verifies negative weight sum is rejected.
func TestWeightedScoring_NegativeWeights_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for negative weight sum")
		}
	}()
	NewRoutingPolicy("weighted", -0.5, -0.5)
}

// TestPrefixAffinity_CacheHit verifies BC-5 (cache-aware routing).
func TestPrefixAffinity_CacheHit(t *testing.T) {
	policy := NewRoutingPolicy("prefix-affinity", 0, 0)
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
	policy := NewRoutingPolicy("prefix-affinity", 0, 0)
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
	policy := NewRoutingPolicy("prefix-affinity", 0, 0)
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
	policy1 := NewRoutingPolicy("prefix-affinity", 0, 0)
	policy2 := NewRoutingPolicy("prefix-affinity", 0, 0)

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

// TestWeightedScoring_EmptySnapshots_Panics verifies empty snapshots cause panic.
func TestWeightedScoring_EmptySnapshots_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic on empty snapshots, got none")
		}
	}()

	policy := NewRoutingPolicy("weighted", 0.6, 0.4)
	policy.Route(&Request{ID: "req1"}, &RouterState{Snapshots: []RoutingSnapshot{}, Clock: 1000})
}

// TestPrefixAffinity_EmptySnapshots_Panics verifies BC-10.
func TestPrefixAffinity_EmptySnapshots_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic on empty snapshots, got none")
		}
	}()

	policy := NewRoutingPolicy("prefix-affinity", 0, 0)
	policy.Route(&Request{ID: "req1", InputTokens: []int{1}}, &RouterState{Snapshots: []RoutingSnapshot{}, Clock: 1000})
}

// TestWeightedScoring_HighestScoreWins verifies the behavioral invariant:
// the selected target must have the highest score in the Scores map.
func TestWeightedScoring_HighestScoreWins(t *testing.T) {
	policy := NewRoutingPolicy("weighted", 0.6, 0.4)

	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 5, FreeKVBlocks: 200},
		{ID: "instance_1", QueueDepth: 5, BatchSize: 5, FreeKVBlocks: 800},
		{ID: "instance_2", QueueDepth: 5, BatchSize: 5, FreeKVBlocks: 500},
	}

	req := &Request{ID: "req1"}
	decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	// Behavioral invariant: target has the highest score (instance_1 has most FreeKVBlocks)
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

	// All instances should have scores
	if len(decision.Scores) != len(snapshots) {
		t.Errorf("expected %d scores, got %d", len(snapshots), len(decision.Scores))
	}
}

// TestPrefixAffinity_StaleEntry_FallsBackToLeastLoaded verifies stale cache path.
func TestPrefixAffinity_StaleEntry_FallsBackToLeastLoaded(t *testing.T) {
	policy := NewRoutingPolicy("prefix-affinity", 0, 0)

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

// TestWeightedScoring_AllIdle_NoDivisionByZero verifies zero-load edge case.
func TestWeightedScoring_AllIdle_NoDivisionByZero(t *testing.T) {
	policy := NewRoutingPolicy("weighted", 0.6, 0.4)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 0, BatchSize: 0, FreeKVBlocks: 700},
		{ID: "instance_1", QueueDepth: 0, BatchSize: 0, FreeKVBlocks: 300},
	}

	req := &Request{ID: "req1"}
	decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	// Both idle: equal loadScore. instance_0 has more FreeKVBlocks → wins.
	if decision.TargetInstance != "instance_0" {
		t.Errorf("Expected instance_0, got %q", decision.TargetInstance)
	}

	// Verify scores are finite (no NaN from division by zero)
	for id, score := range decision.Scores {
		if math.IsNaN(score) || math.IsInf(score, 0) {
			t.Errorf("Score for %s is not finite: %f", id, score)
		}
	}
}

// TestRoutingDecision_PriorityHint_DefaultZero verifies BC-9: default Priority is zero.
func TestRoutingDecision_PriorityHint_DefaultZero(t *testing.T) {
	policyNames := []string{"round-robin", "least-loaded", "weighted", "prefix-affinity"}

	for _, name := range policyNames {
		t.Run(name, func(t *testing.T) {
			policy := NewRoutingPolicy(name, 0.6, 0.4)
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

// TestWeightedScoring_PendingRequests_AffectsLoadDimension verifies that
// PendingRequests (routed but not yet queued) increases effective load,
// causing routing to spread requests across instances instead of piling
// them on the same one (issue #169, #170).
func TestWeightedScoring_PendingRequests_AffectsLoadDimension(t *testing.T) {
	policy := NewRoutingPolicy("weighted", 0.5, 0.5)

	// GIVEN two instances with identical QueueDepth and FreeKVBlocks,
	// but instance_0 has 3 pending requests (routed, not yet in queue)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 0, FreeKVBlocks: 500, PendingRequests: 3},
		{ID: "instance_1", QueueDepth: 0, FreeKVBlocks: 500, PendingRequests: 0},
	}

	// WHEN routing a request
	decision := policy.Route(&Request{ID: "r1"}, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN instance_1 wins (lower effective load: 0+0 vs 0+3)
	if decision.TargetInstance != "instance_1" {
		t.Errorf("expected instance_1 (no pending), got %q", decision.TargetInstance)
	}
}

// TestWeightedScoring_PendingRequests_WeightsFlipDecision verifies that
// with pending requests creating load asymmetry, changing weights
// produces DIFFERENT routing decisions (the #169 acceptance criteria).
func TestWeightedScoring_PendingRequests_WeightsFlipDecision(t *testing.T) {
	// GIVEN an instance with pending requests (high effective load) but lots of free KV,
	// and another with no pending (low load) but few free KV blocks.
	// PendingRequests creates the signal disagreement: load increases immediately
	// while FreeKVBlocks stays unchanged (no KV blocks allocated for pending).
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 0, FreeKVBlocks: 900, PendingRequests: 5}, // cache: best, load: worst
		{ID: "instance_1", QueueDepth: 0, FreeKVBlocks: 100, PendingRequests: 0}, // cache: worst, load: best
	}

	// WHEN using cache-dominant weights
	cacheDominant := NewRoutingPolicy("weighted", 0.9, 0.1)
	d1 := cacheDominant.Route(&Request{ID: "r1"}, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN instance_0 wins (most FreeKVBlocks)
	if d1.TargetInstance != "instance_0" {
		t.Errorf("cache-dominant: expected instance_0, got %q", d1.TargetInstance)
	}

	// WHEN using load-dominant weights
	loadDominant := NewRoutingPolicy("weighted", 0.1, 0.9)
	d2 := loadDominant.Route(&Request{ID: "r2"}, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN instance_1 wins (no pending requests)
	if d2.TargetInstance != "instance_1" {
		t.Errorf("load-dominant: expected instance_1, got %q", d2.TargetInstance)
	}

	// THEN different weights produce different decisions
	if d1.TargetInstance == d2.TargetInstance {
		t.Errorf("expected different decisions, both chose %q", d1.TargetInstance)
	}
}

// TestAlwaysBusiest_RouteToHighestLoad verifies BC-6.
func TestAlwaysBusiest_RouteToHighestLoad(t *testing.T) {
	policy := NewRoutingPolicy("always-busiest", 0, 0)
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

// TestWeightedScoring_DifferentWeights_UncorrelatedDimensions verifies that
// changing weights produces different routing decisions when cache utilization
// and load rank instances differently (issue #169).
func TestWeightedScoring_DifferentWeights_UncorrelatedDimensions(t *testing.T) {
	// GIVEN two instances where cache and load rankings disagree:
	// - instance_0: lots of free KV (cache favors), high load (load disfavors)
	// - instance_1: few free KV blocks (cache disfavors), low load (load favors)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 8, BatchSize: 2, FreeKVBlocks: 900},
		{ID: "instance_1", QueueDepth: 0, BatchSize: 0, FreeKVBlocks: 100},
	}

	// WHEN cache weight dominates (0.9 vs 0.1)
	cacheDominant := NewRoutingPolicy("weighted", 0.9, 0.1)
	d1 := cacheDominant.Route(&Request{ID: "r1"}, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN instance_0 wins (better cache score)
	if d1.TargetInstance != "instance_0" {
		t.Errorf("cache-dominant: expected instance_0, got %q", d1.TargetInstance)
	}

	// WHEN load weight dominates (0.1 vs 0.9)
	loadDominant := NewRoutingPolicy("weighted", 0.1, 0.9)
	d2 := loadDominant.Route(&Request{ID: "r2"}, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN instance_1 wins (better load score)
	if d2.TargetInstance != "instance_1" {
		t.Errorf("load-dominant: expected instance_1, got %q", d2.TargetInstance)
	}

	// THEN different weights produce different decisions
	if d1.TargetInstance == d2.TargetInstance {
		t.Errorf("expected different routing decisions for different weights, both chose %q", d1.TargetInstance)
	}
}

// TestWeightedScoring_HomogeneousCluster_WeightsAffectDecisions verifies
// issue #169: in a realistic 4-instance cluster where one instance has
// slightly more load, different weight ratios MUST produce different
// routing decisions. This is the user-facing behavioral contract.
func TestWeightedScoring_HomogeneousCluster_WeightsAffectDecisions(t *testing.T) {
	// GIVEN a 4-instance cluster where FreeKVBlocks and load disagree:
	// instance_0: many queued requests → high load, lots of free KV
	// instance_1: few requests → low load, few free KV blocks
	// instance_2 & 3: moderate (balanced)
	// Cache signal (FreeKVBlocks) ranks:  0 > 2 > 3 > 1
	// Load signal (1/(1+load)) ranks:     1 > 3 > 2 > 0
	// These rankings DISAGREE — so weight changes must flip the winner.
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 12, BatchSize: 2, FreeKVBlocks: 800},
		{ID: "instance_1", QueueDepth: 1, BatchSize: 3, FreeKVBlocks: 200},
		{ID: "instance_2", QueueDepth: 6, BatchSize: 2, FreeKVBlocks: 600},
		{ID: "instance_3", QueueDepth: 3, BatchSize: 2, FreeKVBlocks: 400},
	}

	// WHEN using extreme weight ratios
	cacheDominant := NewRoutingPolicy("weighted", 0.95, 0.05)
	loadDominant := NewRoutingPolicy("weighted", 0.05, 0.95)

	req := &Request{ID: "r1"}
	d1 := cacheDominant.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})
	d2 := loadDominant.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN different weight ratios produce different routing decisions
	if d1.TargetInstance == d2.TargetInstance {
		t.Errorf("issue #169: cache-dominant and load-dominant weights both chose %q; "+
			"weights must affect routing decisions", d1.TargetInstance)
	}
}

// TestWeightedScoring_WeightsNormalized verifies that weights are auto-normalized
// so that (0.6, 0.2) behaves as (0.75, 0.25) — the ratio matters, not the sum.
func TestWeightedScoring_WeightsNormalized(t *testing.T) {
	// GIVEN uncorrelated instances where the crossover depends on exact weight ratio
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 8, BatchSize: 2, FreeKVBlocks: 900}, // lots of free KV, high load
		{ID: "instance_1", QueueDepth: 0, BatchSize: 0, FreeKVBlocks: 100}, // few free KV, low load
	}

	// WHEN using unnormalized weights (0.6, 0.2) which sum to 0.8
	unnormalized := NewRoutingPolicy("weighted", 0.6, 0.2)
	// AND using equivalent normalized weights (0.75, 0.25) which sum to 1.0
	normalized := NewRoutingPolicy("weighted", 0.75, 0.25)

	d1 := unnormalized.Route(&Request{ID: "r1"}, &RouterState{Snapshots: snapshots, Clock: 1000})
	d2 := normalized.Route(&Request{ID: "r2"}, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN both produce the same decision (same ratio = same behavior)
	if d1.TargetInstance != d2.TargetInstance {
		t.Errorf("expected same decision for proportional weights, got %q vs %q",
			d1.TargetInstance, d2.TargetInstance)
	}
}

// TestWeightedScoring_ZeroWeightSum_Panics verifies that zero weight sum is rejected.
func TestWeightedScoring_ZeroWeightSum_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for zero weight sum")
		}
	}()
	NewRoutingPolicy("weighted", 0, 0)
}

// TestAlwaysBusiest_EmptySnapshots_Panics verifies defensive convention.
func TestAlwaysBusiest_EmptySnapshots_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on empty snapshots")
		}
	}()
	policy := NewRoutingPolicy("always-busiest", 0, 0)
	policy.Route(&Request{ID: "r1"}, &RouterState{Snapshots: []RoutingSnapshot{}, Clock: 0})
}
