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
	decision := policy.Route(req, snapshots, 1000)

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
		decision := policy.Route(req, snapshots, int64(i*1000))
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
	policy.Route(req, []RoutingSnapshot{}, 1000)
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

// TestNewRoutingPolicy_DefaultName verifies empty string defaults to round-robin.
func TestNewRoutingPolicy_DefaultName(t *testing.T) {
	policy := NewRoutingPolicy("", 0, 0)
	if policy == nil {
		t.Errorf("Expected non-nil policy for empty string, got nil")
	}
	if _, ok := policy.(*RoundRobin); !ok {
		t.Errorf("Expected RoundRobin for empty string, got %T", policy)
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
			decision := policy.Route(req, tt.snapshots, 1000)
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
	policy.Route(req, []RoutingSnapshot{}, 1000)
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
			name: "instance 1 wins on low KV utilization",
			snapshots: []RoutingSnapshot{
				{ID: "instance_0", QueueDepth: 5, BatchSize: 5, KVUtilization: 0.8}, // load=10, norm=1.0, score = (1-0.8)*0.6 + (1-1.0)*0.4 = 0.12
				{ID: "instance_1", QueueDepth: 5, BatchSize: 5, KVUtilization: 0.2}, // load=10, norm=1.0, score = (1-0.2)*0.6 + (1-1.0)*0.4 = 0.48
			},
			expected: "instance_1",
			reason:   "equal load (both normalized to 1.0); instance_1 wins on lower KVUtilization",
		},
		{
			name: "instance 0 wins on low load",
			snapshots: []RoutingSnapshot{
				{ID: "instance_0", QueueDepth: 2, BatchSize: 2, KVUtilization: 0.5}, // load=4, norm=4/10=0.4, score = 0.5*0.6 + 0.6*0.4 = 0.54
				{ID: "instance_1", QueueDepth: 8, BatchSize: 2, KVUtilization: 0.5}, // load=10, norm=1.0, score = 0.5*0.6 + 0.0*0.4 = 0.3
			},
			expected: "instance_0",
			reason:   "equal KVUtilization; instance_0 wins on lower normalized load",
		},
		{
			name: "all equal scores, first occurrence wins",
			snapshots: []RoutingSnapshot{
				{ID: "instance_0", QueueDepth: 5, BatchSize: 5, KVUtilization: 0.5},
				{ID: "instance_1", QueueDepth: 5, BatchSize: 5, KVUtilization: 0.5},
			},
			expected: "instance_0",
			reason:   "tie broken by first occurrence in snapshot order",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &Request{ID: "req1"}
			decision := policy.Route(req, tt.snapshots, 1000)
			if decision.TargetInstance != tt.expected {
				t.Errorf("%s: expected %q, got %q", tt.reason, tt.expected, decision.TargetInstance)
			}
			if decision.Scores == nil || len(decision.Scores) != len(tt.snapshots) {
				t.Errorf("Expected Scores map with %d entries, got %v", len(tt.snapshots), decision.Scores)
			}
		})
	}
}

// TestWeightedScoring_UniformLoad verifies divide-by-zero safety.
func TestWeightedScoring_UniformLoad(t *testing.T) {
	policy := NewRoutingPolicy("weighted", 0.6, 0.4)

	// All instances have identical load → normalizedLoad = 1.0 for all
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 5, KVUtilization: 0.3}, // score = 0.7*0.6 + 0*0.4 = 0.42
		{ID: "instance_1", QueueDepth: 5, BatchSize: 5, KVUtilization: 0.7}, // score = 0.3*0.6 + 0*0.4 = 0.18
	}

	req := &Request{ID: "req1"}
	decision := policy.Route(req, snapshots, 1000)

	// instance_0 wins on lower KVUtilization (load component cancels out)
	if decision.TargetInstance != "instance_0" {
		t.Errorf("Expected instance_0 to win on cache score alone, got %q", decision.TargetInstance)
	}
}

// TestWeightedScoring_NegativeWeights verifies BC-12 (undefined but non-fatal).
func TestWeightedScoring_NegativeWeights(t *testing.T) {
	policy := NewRoutingPolicy("weighted", -0.5, -0.5)

	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 5, KVUtilization: 0.5},
	}

	req := &Request{ID: "req1"}
	decision := policy.Route(req, snapshots, 1000)
	if decision.TargetInstance == "" {
		t.Errorf("Expected non-empty TargetInstance even with negative weights")
	}
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
	decision1 := policy.Route(req1, snapshots, 1000)
	firstTarget := decision1.TargetInstance

	// Second request with same prefix → cache hit
	req2 := &Request{ID: "req2", InputTokens: []int{1, 2, 3}}
	decision2 := policy.Route(req2, snapshots, 2000)

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
	decision := policy.Route(req, snapshots, 1000)

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

	decision1 := policy.Route(req1, snapshots, 1000)
	decision2 := policy.Route(req2, snapshots, 2000)

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
	policy1.Route(req1, snapshots, 1000)

	// policy2 routes same prefix — cache miss (independent state)
	req2 := &Request{ID: "req2", InputTokens: []int{1, 2, 3}}
	decision2 := policy2.Route(req2, snapshots, 1000)

	// Falls back to LeastLoaded → first occurrence (instance_0)
	if decision2.TargetInstance != "instance_0" {
		t.Errorf("Expected independent state (fallback to least-loaded), got %q", decision2.TargetInstance)
	}
}

// === Fix #2: Missing empty-snapshot panic tests for WeightedScoring and PrefixAffinity (BC-10) ===

// TestWeightedScoring_EmptySnapshots_Panics verifies BC-10.
func TestWeightedScoring_EmptySnapshots_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic on empty snapshots, got none")
		}
	}()

	policy := NewRoutingPolicy("weighted", 0.6, 0.4)
	policy.Route(&Request{ID: "req1"}, []RoutingSnapshot{}, 1000)
}

// TestPrefixAffinity_EmptySnapshots_Panics verifies BC-10.
func TestPrefixAffinity_EmptySnapshots_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic on empty snapshots, got none")
		}
	}()

	policy := NewRoutingPolicy("prefix-affinity", 0, 0)
	policy.Route(&Request{ID: "req1", InputTokens: []int{1}}, []RoutingSnapshot{}, 1000)
}

// === Fix #3: WeightedScoring score value verification ===

// TestWeightedScoring_ScoreValues verifies actual computed scores match the formula.
func TestWeightedScoring_ScoreValues(t *testing.T) {
	policy := NewRoutingPolicy("weighted", 0.6, 0.4)

	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 5, KVUtilization: 0.8}, // load=10, norm=1.0, score = 0.2*0.6 + 0*0.4 = 0.12
		{ID: "instance_1", QueueDepth: 5, BatchSize: 5, KVUtilization: 0.2}, // load=10, norm=1.0, score = 0.8*0.6 + 0*0.4 = 0.48
	}

	req := &Request{ID: "req1"}
	decision := policy.Route(req, snapshots, 1000)

	const epsilon = 1e-9
	expectedScores := map[string]float64{"instance_0": 0.12, "instance_1": 0.48}
	for id, expected := range expectedScores {
		if got, ok := decision.Scores[id]; !ok {
			t.Errorf("Score for %s missing", id)
		} else if math.Abs(got-expected) > epsilon {
			t.Errorf("Score for %s: got %f, want %f", id, got, expected)
		}
	}
}

// === Fix #4: PrefixAffinity stale cache entry test ===

// TestPrefixAffinity_StaleEntry_FallsBackToLeastLoaded verifies stale cache path.
func TestPrefixAffinity_StaleEntry_FallsBackToLeastLoaded(t *testing.T) {
	policy := NewRoutingPolicy("prefix-affinity", 0, 0)

	// First call: maps prefix to instance_1 (least loaded)
	snapshots1 := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 10, BatchSize: 5},
		{ID: "instance_1", QueueDepth: 2, BatchSize: 1},
	}
	req := &Request{ID: "req1", InputTokens: []int{1, 2, 3}}
	decision1 := policy.Route(req, snapshots1, 1000)
	if decision1.TargetInstance != "instance_1" {
		t.Fatalf("setup: expected instance_1, got %q", decision1.TargetInstance)
	}

	// Second call: instance_1 is gone, only instance_0 and instance_2
	snapshots2 := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 10, BatchSize: 5},
		{ID: "instance_2", QueueDepth: 1, BatchSize: 0},
	}
	req2 := &Request{ID: "req2", InputTokens: []int{1, 2, 3}}
	decision2 := policy.Route(req2, snapshots2, 2000)

	// Should fallback to least-loaded (instance_2)
	if decision2.TargetInstance != "instance_2" {
		t.Errorf("Expected fallback to instance_2, got %q", decision2.TargetInstance)
	}
}

// === Fix #5: WeightedScoring all-idle (maxLoad==0) edge case ===

// TestWeightedScoring_AllIdle_NoDivisionByZero verifies zero-load edge case.
func TestWeightedScoring_AllIdle_NoDivisionByZero(t *testing.T) {
	policy := NewRoutingPolicy("weighted", 0.6, 0.4)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 0, BatchSize: 0, KVUtilization: 0.3},
		{ID: "instance_1", QueueDepth: 0, BatchSize: 0, KVUtilization: 0.7},
	}

	req := &Request{ID: "req1"}
	decision := policy.Route(req, snapshots, 1000)

	// With zero load, normalizedLoad=0, loadScore=1.0*0.4=0.4 for all.
	// instance_0 wins on lower KVUtilization: (0.7*0.6 + 0.4) = 0.82 vs (0.3*0.6 + 0.4) = 0.58
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
