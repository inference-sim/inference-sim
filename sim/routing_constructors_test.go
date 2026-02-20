package sim

import "testing"

// TestNewRoutingDecision_Fields verifies BC-1: constructor sets Target and Reason,
// leaves Scores nil and Priority zero.
func TestNewRoutingDecision_Fields(t *testing.T) {
	// GIVEN target and reason
	target := "instance_0"
	reason := "test-reason"

	// WHEN NewRoutingDecision is called
	decision := NewRoutingDecision(target, reason)

	// THEN fields match expected values
	if decision.TargetInstance != target {
		t.Errorf("TargetInstance = %q, want %q", decision.TargetInstance, target)
	}
	if decision.Reason != reason {
		t.Errorf("Reason = %q, want %q", decision.Reason, reason)
	}
	if decision.Scores != nil {
		t.Errorf("Scores = %v, want nil", decision.Scores)
	}
	if decision.Priority != 0.0 {
		t.Errorf("Priority = %f, want 0.0", decision.Priority)
	}
}

// TestNewRoutingDecisionWithScores_Fields verifies BC-2: constructor sets Target, Reason,
// and Scores, leaves Priority zero.
func TestNewRoutingDecisionWithScores_Fields(t *testing.T) {
	// GIVEN target, reason, and scores
	target := "instance_1"
	reason := "weighted-scoring (score=0.850)"
	scores := map[string]float64{"instance_0": 0.3, "instance_1": 0.85}

	// WHEN NewRoutingDecisionWithScores is called
	decision := NewRoutingDecisionWithScores(target, reason, scores)

	// THEN fields match expected values
	if decision.TargetInstance != target {
		t.Errorf("TargetInstance = %q, want %q", decision.TargetInstance, target)
	}
	if decision.Reason != reason {
		t.Errorf("Reason = %q, want %q", decision.Reason, reason)
	}
	if len(decision.Scores) != 2 {
		t.Fatalf("Scores length = %d, want 2", len(decision.Scores))
	}
	if decision.Scores["instance_1"] != 0.85 {
		t.Errorf("Scores[instance_1] = %f, want 0.85", decision.Scores["instance_1"])
	}
	if decision.Priority != 0.0 {
		t.Errorf("Priority = %f, want 0.0", decision.Priority)
	}
}
