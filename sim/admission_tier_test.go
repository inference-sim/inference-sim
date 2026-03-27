package sim

import (
	"strings"
	"testing"
)

// makeOverloadedState returns a RouterState whose single snapshot has
// EffectiveLoad() == load (via QueueDepth). Useful for admission unit tests.
func makeOverloadedState(load int) *RouterState {
	return &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "inst_0", QueueDepth: load},
		},
		Clock: 0,
	}
}

// T005 — Table-driven unit tests for SLOTierPriority.
// Tests all 5 canonical classes + empty string + unknown string.
func TestSLOTierPriority_AllClasses(t *testing.T) {
	tests := []struct {
		class    string
		expected int
	}{
		{"critical", 4},
		{"standard", 3},
		{"sheddable", 2},
		{"batch", 1},
		{"background", 0},
		{"", 3},          // empty → Standard (backward compat)
		{"unknown", 3},   // unknown → Standard (defensive)
		{"CRITICAL", 3},  // case-sensitive → Standard (not critical)
		{"Standard", 3},  // mixed case → Standard
	}
	for _, tt := range tests {
		t.Run("class="+tt.class, func(t *testing.T) {
			got := SLOTierPriority(tt.class)
			if got != tt.expected {
				t.Errorf("SLOTierPriority(%q) = %d, want %d", tt.class, got, tt.expected)
			}
		})
	}
}

// T006 — Critical and Standard are admitted under overload.
func TestTierShedAdmission_CriticalAndStandardAdmittedUnderOverload(t *testing.T) {
	policy := &TierShedAdmission{OverloadThreshold: 0, MinAdmitPriority: 3}
	state := makeOverloadedState(10) // heavily overloaded

	for _, class := range []string{"critical", "standard"} {
		req := &Request{ID: "r", SLOClass: class}
		admitted, reason := policy.Admit(req, state)
		if !admitted {
			t.Errorf("class=%q: expected admitted=true under overload, got false (reason=%q)", class, reason)
		}
		if reason != "" {
			t.Errorf("class=%q: expected empty reason on admission, got %q", class, reason)
		}
	}
}

// T007 — Sheddable rejected when maxLoad > OverloadThreshold.
func TestTierShedAdmission_SheddableRejectedWhenOverloaded(t *testing.T) {
	policy := &TierShedAdmission{OverloadThreshold: 5, MinAdmitPriority: 3}
	state := makeOverloadedState(6) // load=6 > threshold=5

	req := &Request{ID: "r", SLOClass: "sheddable"}
	admitted, reason := policy.Admit(req, state)
	if admitted {
		t.Error("sheddable should be rejected when overloaded")
	}
	if reason == "" {
		t.Error("expected non-empty reason on rejection")
	}
	if !strings.Contains(reason, "tier-shed") {
		t.Errorf("reason should mention tier-shed, got %q", reason)
	}
}

// T008 — All tiers admitted when maxLoad <= OverloadThreshold.
func TestTierShedAdmission_AllAdmittedUnderThreshold(t *testing.T) {
	policy := &TierShedAdmission{OverloadThreshold: 10, MinAdmitPriority: 3}
	state := makeOverloadedState(5) // load=5 <= threshold=10

	for _, class := range []string{"critical", "standard", "sheddable", "batch", "background"} {
		req := &Request{ID: "r", SLOClass: class}
		admitted, reason := policy.Admit(req, state)
		if !admitted {
			t.Errorf("class=%q: expected admitted under threshold, got rejected (reason=%q)", class, reason)
		}
	}
}

// T009 — Empty snapshots: no panic, all requests admitted.
func TestTierShedAdmission_EmptySnapshotsNoPanic(t *testing.T) {
	policy := &TierShedAdmission{OverloadThreshold: 0, MinAdmitPriority: 3}
	state := &RouterState{Snapshots: []RoutingSnapshot{}} // no instances

	for _, class := range []string{"critical", "standard", "sheddable"} {
		req := &Request{ID: "r", SLOClass: class}
		admitted, _ := policy.Admit(req, state)
		if !admitted {
			t.Errorf("class=%q: expected admission when no instances, got rejected", class)
		}
	}
}

// T010 — Empty SLOClass treated as Standard (priority 3), never shed below Standard.
func TestTierShedAdmission_EmptySLOClassTreatedAsStandard(t *testing.T) {
	policy := &TierShedAdmission{OverloadThreshold: 0, MinAdmitPriority: 3}
	state := makeOverloadedState(10)

	req := &Request{ID: "r", SLOClass: ""}
	admitted, reason := policy.Admit(req, state)
	if !admitted {
		t.Errorf("empty SLOClass should be treated as Standard and admitted, got rejected (reason=%q)", reason)
	}
}

// T011 — Batch and Background always admitted regardless of load.
func TestTierShedAdmission_BatchAndBackgroundAlwaysAdmitted(t *testing.T) {
	policy := &TierShedAdmission{OverloadThreshold: 0, MinAdmitPriority: 3}
	// Extremely overloaded state
	state := makeOverloadedState(9999)

	for _, class := range []string{"batch", "background"} {
		req := &Request{ID: "r", SLOClass: class}
		admitted, reason := policy.Admit(req, state)
		if !admitted {
			t.Errorf("class=%q should always be admitted by tier-shed policy, got rejected (reason=%q)", class, reason)
		}
		if reason != "" {
			t.Errorf("class=%q: expected empty reason on admission, got %q", class, reason)
		}
	}
}

// Additional: verify MinAdmitPriority=0 admits all tiers regardless of load (I-2 footgun doc).
// A TierShedAdmission with MinAdmitPriority=0 is functionally identical to AlwaysAdmit.
func TestTierShedAdmission_ZeroMinPriorityAdmitsAll(t *testing.T) {
	policy := &TierShedAdmission{OverloadThreshold: 0, MinAdmitPriority: 0}
	state := makeOverloadedState(9999) // extremely overloaded
	for _, class := range []string{"critical", "standard", "sheddable", "batch", "background"} {
		req := &Request{ID: "r", SLOClass: class}
		admitted, reason := policy.Admit(req, state)
		if !admitted {
			t.Errorf("class=%q: MinAdmitPriority=0 should admit all tiers under any load, got rejected (reason=%q)", class, reason)
		}
	}
}

// Additional: verify Sheddable NOT shed when exactly at threshold (strict >).
func TestTierShedAdmission_ExactThresholdAdmits(t *testing.T) {
	policy := &TierShedAdmission{OverloadThreshold: 5, MinAdmitPriority: 3}
	state := makeOverloadedState(5) // load=5 == threshold → not overloaded

	req := &Request{ID: "r", SLOClass: "sheddable"}
	admitted, _ := policy.Admit(req, state)
	if !admitted {
		t.Error("sheddable should be admitted when load == threshold (strict > required)")
	}
}
