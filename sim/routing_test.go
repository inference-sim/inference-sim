package sim

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestRoutingPolicy_Interface_Contract verifies the RoutingPolicy interface contract (BC-1).
func TestRoutingPolicy_Interface_Contract(t *testing.T) {
	// GIVEN a RoutingPolicy implementation (RoundRobin)
	policy := NewRoutingPolicy("round-robin", nil, 16, nil)

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
	policy := NewRoutingPolicy("round-robin", nil, 16, nil)
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

	policy := NewRoutingPolicy("round-robin", nil, 16, nil)
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

	NewRoutingPolicy("invalid-policy", nil, 16, nil)
}

// TestNewRoutingPolicy_DefaultName verifies empty string defaults to round-robin behavior.
func TestNewRoutingPolicy_DefaultName(t *testing.T) {
	policy := NewRoutingPolicy("", nil, 16, nil)
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
	policy := NewRoutingPolicy("least-loaded", nil, 16, nil)

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

// TestRoutingSnapshot_EffectiveLoad_IncludesInFlightRequests verifies BC-5:
// GIVEN a RoutingSnapshot with QueueDepth=2, BatchSize=1, InFlightRequests=3
// WHEN EffectiveLoad() is called
// THEN the result is 6
func TestRoutingSnapshot_EffectiveLoad_IncludesInFlightRequests(t *testing.T) {
	snap := RoutingSnapshot{
		ID:               "test",
		QueueDepth:       2,
		BatchSize:        1,
		InFlightRequests: 3,
	}
	if got := snap.EffectiveLoad(); got != 6 {
		t.Errorf("EffectiveLoad() = %d, want 6", got)
	}
}

// TestLeastLoaded_InFlightRequests_BreaksTie verifies that InFlightRequests is included
// in load calculation, preventing pile-on at high request rates (#175).
func TestLeastLoaded_InFlightRequests_BreaksTie(t *testing.T) {
	policy := NewRoutingPolicy("least-loaded", nil, 16, nil)

	// GIVEN two instances with equal QueueDepth+BatchSize but different InFlightRequests
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 3, InFlightRequests: 4},
		{ID: "instance_1", QueueDepth: 5, BatchSize: 3, InFlightRequests: 0},
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

	policy := NewRoutingPolicy("least-loaded", nil, 16, nil)
	req := &Request{ID: "req1"}
	policy.Route(req, &RouterState{Snapshots: []RoutingSnapshot{}, Clock: 1000})
}

// === WeightedScoring Tests (rewritten for scorer pipeline) ===

// TestWeightedScoring_DefaultScorers_RoutesToBestComposite verifies BC-17-6 (argmax).
func TestWeightedScoring_DefaultScorers_RoutesToBestComposite(t *testing.T) {
	// GIVEN weighted policy with default scorers and instances with varying load/utilization
	policy := NewRoutingPolicy("weighted", nil, 16, nil)
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
	policy := NewRoutingPolicy("weighted", nil, 16, nil)
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
	}, 16, nil)
	scaled := NewRoutingPolicy("weighted", []ScorerConfig{
		{Name: "queue-depth", Weight: 6.0},
		{Name: "kv-utilization", Weight: 4.0},
		{Name: "load-balance", Weight: 4.0},
	}, 16, nil)

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
	policy := NewRoutingPolicy("weighted", []ScorerConfig{{Name: "load-balance", Weight: 1.0}}, 16, nil)
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
	}, 16, nil)
	d1 := qdDominant.Route(&Request{ID: "r1"}, &RouterState{Snapshots: snapshots, Clock: 1000})

	// WHEN using kv-utilization-dominant weights
	kvDominant := NewRoutingPolicy("weighted", []ScorerConfig{
		{Name: "queue-depth", Weight: 1.0},
		{Name: "kv-utilization", Weight: 9.0},
	}, 16, nil)
	d2 := kvDominant.Route(&Request{ID: "r2"}, &RouterState{Snapshots: snapshots, Clock: 1000})

	// THEN different weights produce different decisions
	if d1.TargetInstance == d2.TargetInstance {
		t.Errorf("expected different decisions for different scorer weights, both chose %q", d1.TargetInstance)
	}
}

// TestWeightedScoring_AllIdle_NoDivisionByZero verifies BC-17-9 (no NaN/Inf).
func TestWeightedScoring_AllIdle_NoDivisionByZero(t *testing.T) {
	policy := NewRoutingPolicy("weighted", nil, 16, nil)
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

	policy := NewRoutingPolicy("weighted", nil, 16, nil)
	policy.Route(&Request{ID: "req1"}, &RouterState{Snapshots: []RoutingSnapshot{}, Clock: 1000})
}

// TestWeightedScoring_NilConfigs_UsesDefaults verifies default scorer configuration.
func TestWeightedScoring_NilConfigs_UsesDefaults(t *testing.T) {
	// GIVEN nil scorerConfigs
	policy := NewRoutingPolicy("weighted", nil, 16, nil)

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

// TestWeightedScoring_InFlightRequests_AffectsScorers verifies that InFlightRequests
// affects queue-depth and load-balance scorers.
func TestWeightedScoring_InFlightRequests_AffectsScorers(t *testing.T) {
	policy := NewRoutingPolicy("weighted", []ScorerConfig{{Name: "load-balance", Weight: 1.0}}, 16, nil)

	// GIVEN two instances with equal QueueDepth but different InFlightRequests
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 0, InFlightRequests: 3},
		{ID: "instance_1", QueueDepth: 0, InFlightRequests: 0},
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
	policy := NewRoutingPolicy("weighted", []ScorerConfig{}, 16, nil)
	snapshots := []RoutingSnapshot{{ID: "a", QueueDepth: 1}}
	decision := policy.Route(&Request{ID: "r1"}, &RouterState{Snapshots: snapshots, Clock: 1000})
	if decision.TargetInstance != "a" {
		t.Errorf("expected 'a', got %q", decision.TargetInstance)
	}
}

// TestRoutingDecision_PriorityHint_DefaultZero verifies BC-9: default Priority is zero.
func TestRoutingDecision_PriorityHint_DefaultZero(t *testing.T) {
	policyNames := []string{"round-robin", "least-loaded", "weighted"}

	for _, name := range policyNames {
		t.Run(name, func(t *testing.T) {
			policy := NewRoutingPolicy(name, nil, 16, nil)
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
	policy := NewRoutingPolicy("always-busiest", nil, 16, nil)
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

// TestAlwaysBusiest_InFlightRequests_IncludedInLoad verifies that InFlightRequests
// is included in AlwaysBusiest load calculation (#175).
func TestAlwaysBusiest_InFlightRequests_IncludedInLoad(t *testing.T) {
	policy := NewRoutingPolicy("always-busiest", nil, 16, nil)

	// GIVEN two instances with equal QueueDepth+BatchSize but different InFlightRequests
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 3, InFlightRequests: 0},
		{ID: "instance_1", QueueDepth: 5, BatchSize: 3, InFlightRequests: 4},
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
	policy := NewRoutingPolicy("always-busiest", nil, 16, nil)
	policy.Route(&Request{ID: "r1"}, &RouterState{Snapshots: []RoutingSnapshot{}, Clock: 0})
}

// === Random Tie-Breaking Tests (#565) ===

// TestLeastLoaded_TieBreaking_Random verifies BC-2: random uniform tie-breaking.
// GIVEN LeastLoaded with a non-nil RNG and 3 instances with equal EffectiveLoad
// WHEN Route() is called 300 times
// THEN each tied instance is selected with approximately equal frequency.
func TestLeastLoaded_TieBreaking_Random(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	policy := NewRoutingPolicy("least-loaded", nil, 16, rng)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 5},
		{ID: "instance_1", QueueDepth: 5, BatchSize: 5},
		{ID: "instance_2", QueueDepth: 5, BatchSize: 5},
	}

	counts := map[string]int{}
	N := 300
	for i := 0; i < N; i++ {
		req := &Request{ID: fmt.Sprintf("req%d", i)}
		decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})
		counts[decision.TargetInstance]++
	}

	// Each instance should get roughly N/3 = 100 requests.
	// Allow ±50% tolerance (50-150 range) for random variation — 6.1 sigma.
	for _, id := range []string{"instance_0", "instance_1", "instance_2"} {
		if counts[id] < 50 || counts[id] > 150 {
			t.Errorf("instance %s got %d/%d requests, expected ~100 (uniform)", id, counts[id], N)
		}
	}
}

// TestWeightedScoring_TieBreaking_Random verifies BC-1: random uniform tie-breaking.
// GIVEN WeightedScoring with a non-nil RNG and 3 instances with equal composite scores
// WHEN Route() is called 300 times
// THEN each tied instance is selected with approximately equal frequency.
func TestWeightedScoring_TieBreaking_Random(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	// Single queue-depth scorer: all idle → all score 1.0 → tie.
	policy := NewRoutingPolicy("weighted", []ScorerConfig{
		{Name: "queue-depth", Weight: 1.0},
	}, 16, rng)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 0},
		{ID: "instance_1", QueueDepth: 0},
		{ID: "instance_2", QueueDepth: 0},
	}

	counts := map[string]int{}
	N := 300
	for i := 0; i < N; i++ {
		req := &Request{ID: fmt.Sprintf("req%d", i)}
		decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})
		counts[decision.TargetInstance]++
	}

	for _, id := range []string{"instance_0", "instance_1", "instance_2"} {
		if counts[id] < 50 || counts[id] > 150 {
			t.Errorf("instance %s got %d/%d requests, expected ~100 (uniform)", id, counts[id], N)
		}
	}
}

// TestTieBreaking_Determinism verifies BC-3: same seed → same decisions.
// Tests both all-tied and partial-tie scenarios.
func TestTieBreaking_Determinism(t *testing.T) {
	cases := []struct {
		name      string
		snapshots []RoutingSnapshot
	}{
		{
			name: "all-tied",
			snapshots: []RoutingSnapshot{
				{ID: "a", QueueDepth: 0},
				{ID: "b", QueueDepth: 0},
				{ID: "c", QueueDepth: 0},
			},
		},
		{
			name: "partial-tie",
			snapshots: []RoutingSnapshot{
				{ID: "a", QueueDepth: 0},
				{ID: "b", QueueDepth: 0},
				{ID: "c", QueueDepth: 5},
			},
		},
	}

	for _, tc := range cases {
		for _, policyName := range []string{"least-loaded", "weighted"} {
			t.Run(tc.name+"/"+policyName, func(t *testing.T) {
				rng1 := rand.New(rand.NewSource(99))
				rng2 := rand.New(rand.NewSource(99))
				var scorers []ScorerConfig
				if policyName == "weighted" {
					scorers = []ScorerConfig{{Name: "queue-depth", Weight: 1.0}}
				}
				p1 := NewRoutingPolicy(policyName, scorers, 16, rng1)
				p2 := NewRoutingPolicy(policyName, scorers, 16, rng2)

				for i := 0; i < 50; i++ {
					req := &Request{ID: fmt.Sprintf("req%d", i)}
					d1 := p1.Route(req, &RouterState{Snapshots: tc.snapshots, Clock: 1000})
					d2 := p2.Route(req, &RouterState{Snapshots: tc.snapshots, Clock: 1000})
					if d1.TargetInstance != d2.TargetInstance {
						t.Errorf("request %d: different decisions with same seed: %s vs %s",
							i, d1.TargetInstance, d2.TargetInstance)
					}
				}
			})
		}
	}
}

// TestTieBreaking_NoTie_PreservesRNGState verifies BC-4: distinct scores → unique winner,
// RNG state not advanced (non-tie calls must not shift the RNG stream).
// Tests both LeastLoaded and WeightedScoring to ensure neither consumes RNG on non-ties.
func TestTieBreaking_NoTie_PreservesRNGState(t *testing.T) {
	nonTieSnaps := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 10, KVUtilization: 0.8},
		{ID: "instance_1", QueueDepth: 1, KVUtilization: 0.1},
		{ID: "instance_2", QueueDepth: 5, KVUtilization: 0.5},
	}

	tests := []struct {
		name    string
		policy  string
		scorers []ScorerConfig
		winner  string
	}{
		{"least-loaded", "least-loaded", nil, "instance_1"},
		{"weighted/queue-depth", "weighted", []ScorerConfig{{Name: "queue-depth", Weight: 1.0}}, "instance_1"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rng1 := rand.New(rand.NewSource(42))
			rng2 := rand.New(rand.NewSource(42))

			policy := NewRoutingPolicy(tc.policy, tc.scorers, 16, rng1)

			// Make 50 non-tie routing calls — should NOT consume RNG
			for i := 0; i < 50; i++ {
				req := &Request{ID: fmt.Sprintf("req%d", i)}
				decision := policy.Route(req, &RouterState{Snapshots: nonTieSnaps, Clock: 1000})
				if decision.TargetInstance != tc.winner {
					t.Fatalf("request %d: expected %s (unique winner), got %q", i, tc.winner, decision.TargetInstance)
				}
			}

			// Verify rng1 and rng2 are in the same state
			val1 := rng1.Intn(1000)
			val2 := rng2.Intn(1000)
			if val1 != val2 {
				t.Errorf("RNG state diverged after non-tie calls: rng1=%d, rng2=%d (RNG consumed on non-tie)", val1, val2)
			}
		})
	}
}

// TestTieBreaking_NilRNG_Positional verifies BC-5: nil RNG → positional tie-breaking.
// Tests both LeastLoaded and WeightedScoring to ensure parity (R23).
func TestTieBreaking_NilRNG_Positional(t *testing.T) {
	tests := []struct {
		name    string
		policy  string
		scorers []ScorerConfig
	}{
		{"least-loaded", "least-loaded", nil},
		{"weighted/queue-depth", "weighted", []ScorerConfig{{Name: "queue-depth", Weight: 1.0}}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			policy := NewRoutingPolicy(tc.policy, tc.scorers, 16, nil)
			snapshots := []RoutingSnapshot{
				{ID: "instance_0", QueueDepth: 5},
				{ID: "instance_1", QueueDepth: 5},
			}

			for i := 0; i < 10; i++ {
				req := &Request{ID: fmt.Sprintf("req%d", i)}
				decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})
				if decision.TargetInstance != "instance_0" {
					t.Errorf("nil RNG should use positional (first), got %q", decision.TargetInstance)
				}
			}
		})
	}
}
