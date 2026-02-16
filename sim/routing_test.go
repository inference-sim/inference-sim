package sim

import (
	"fmt"
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
