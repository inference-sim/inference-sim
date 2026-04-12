package sim

import (
	"math"
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

// T005 — Table-driven unit tests for SLOTierPriority (GAIE-compatible defaults).
// Tests all 5 canonical classes + empty string + unknown string.
func TestSLOTierPriority_AllClasses(t *testing.T) {
	tests := []struct {
		class    string
		expected int
	}{
		{"critical", 4},
		{"standard", 3},
		{"batch", -1},
		{"sheddable", -2},
		{"background", -3},
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

// BC-1: Default SLOPriorityMap returns GAIE-compatible values.
func TestSLOPriorityMap_Defaults(t *testing.T) {
	m := DefaultSLOPriorityMap()
	tests := []struct {
		class    string
		expected int
	}{
		{"critical", 4},
		{"standard", 3},
		{"batch", -1},
		{"sheddable", -2},
		{"background", -3},
		{"", 3},
		{"unknown", 3},
	}
	for _, tt := range tests {
		t.Run("class="+tt.class, func(t *testing.T) {
			got := m.Priority(tt.class)
			if got != tt.expected {
				t.Errorf("Priority(%q) = %d, want %d", tt.class, got, tt.expected)
			}
		})
	}
}

// BC-2: IsSheddable returns true iff Priority(class) < 0.
func TestSLOPriorityMap_IsSheddable(t *testing.T) {
	m := DefaultSLOPriorityMap()
	tests := []struct {
		class    string
		expected bool
	}{
		{"critical", false},
		{"standard", false},
		{"batch", true},
		{"sheddable", true},
		{"background", true},
		{"", false},       // empty → standard(3) → not sheddable
		{"unknown", false}, // unknown → standard(3) → not sheddable
	}
	for _, tt := range tests {
		t.Run("class="+tt.class, func(t *testing.T) {
			got := m.IsSheddable(tt.class)
			if got != tt.expected {
				t.Errorf("IsSheddable(%q) = %v, want %v", tt.class, got, tt.expected)
			}
		})
	}
}

// BC-3: Custom overrides replace specific defaults, others retained.
func TestSLOPriorityMap_CustomOverrides(t *testing.T) {
	m := NewSLOPriorityMap(map[string]int{"critical": 10, "batch": 0})
	if got := m.Priority("critical"); got != 10 {
		t.Errorf("critical should be overridden to 10, got %d", got)
	}
	if got := m.Priority("batch"); got != 0 {
		t.Errorf("batch should be overridden to 0, got %d", got)
	}
	// Non-overridden classes retain defaults
	if got := m.Priority("standard"); got != 3 {
		t.Errorf("standard should retain default 3, got %d", got)
	}
	if got := m.Priority("background"); got != -3 {
		t.Errorf("background should retain default -3, got %d", got)
	}
	// batch=0 is no longer sheddable
	if m.IsSheddable("batch") {
		t.Error("batch with priority 0 should NOT be sheddable")
	}
}

// T006 — Critical and Standard are admitted under overload.
func TestTierShedAdmission_CriticalAndStandardAdmittedUnderOverload(t *testing.T) {
	policy := NewTierShedAdmission(0, 3, nil)
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
	policy := NewTierShedAdmission(5, 3, nil)
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
	policy := NewTierShedAdmission(10, 3, nil)
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
	policy := NewTierShedAdmission(0, 3, nil)
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
	policy := NewTierShedAdmission(0, 3, nil)
	state := makeOverloadedState(10)

	req := &Request{ID: "r", SLOClass: ""}
	admitted, reason := policy.Admit(req, state)
	if !admitted {
		t.Errorf("empty SLOClass should be treated as Standard and admitted, got rejected (reason=%q)", reason)
	}
}

// T011 — Batch and Background are rejected under overload when below MinAdmitPriority.
func TestTierShedAdmission_BatchAndBackgroundRejectedUnderOverload(t *testing.T) {
	policy := NewTierShedAdmission(0, 3, nil)
	// Extremely overloaded state
	state := makeOverloadedState(9999)

	for _, class := range []string{"batch", "background"} {
		req := &Request{ID: "r", SLOClass: class}
		admitted, _ := policy.Admit(req, state)
		if admitted {
			t.Errorf("class=%q (priority < MinAdmitPriority=3) should be rejected under overload, got admitted", class)
		}
	}
}

// MinAdmitPriority=0 admits non-sheddable (priority >= 0) and rejects sheddable (priority < 0).
// With GAIE defaults: critical(4), standard(3) admitted; batch(-1), sheddable(-2), background(-3) rejected.
func TestTierShedAdmission_ZeroMinPriorityRejectsSheddable(t *testing.T) {
	policy := NewTierShedAdmission(0, 0, nil)
	state := makeOverloadedState(9999)

	// Non-sheddable classes admitted
	for _, class := range []string{"critical", "standard"} {
		req := &Request{ID: "r", SLOClass: class}
		admitted, reason := policy.Admit(req, state)
		if !admitted {
			t.Errorf("class=%q: MinAdmitPriority=0 should admit non-sheddable, got rejected (reason=%q)", class, reason)
		}
	}
	// Sheddable classes rejected (priority < 0)
	for _, class := range []string{"batch", "sheddable", "background"} {
		req := &Request{ID: "r", SLOClass: class}
		admitted, _ := policy.Admit(req, state)
		if admitted {
			t.Errorf("class=%q: MinAdmitPriority=0 should reject sheddable (priority < 0), got admitted", class)
		}
	}
}

// MinAdmitPriority=-3 admits all tiers (lowest default priority is background=-3, -3 < -3 is false).
func TestTierShedAdmission_NegativeMinPriorityAdmitsAll(t *testing.T) {
	policy := NewTierShedAdmission(0, -3, nil)
	state := makeOverloadedState(9999)
	for _, class := range []string{"critical", "standard", "sheddable", "batch", "background"} {
		req := &Request{ID: "r", SLOClass: class}
		admitted, reason := policy.Admit(req, state)
		if !admitted {
			t.Errorf("class=%q: MinAdmitPriority=-3 should admit all tiers, got rejected (reason=%q)", class, reason)
		}
	}
}

// Additional: verify Sheddable NOT shed when exactly at threshold (strict >).
func TestTierShedAdmission_ExactThresholdAdmits(t *testing.T) {
	policy := NewTierShedAdmission(5, 3, nil)
	state := makeOverloadedState(5) // load=5 == threshold → not overloaded

	req := &Request{ID: "r", SLOClass: "sheddable"}
	admitted, _ := policy.Admit(req, state)
	if !admitted {
		t.Error("sheddable should be admitted when load == threshold (strict > required)")
	}
}

// --- GAIELegacyAdmission tests ---

// BC-1: Non-sheddable requests always admitted, even at extreme saturation.
func TestGAIELegacy_NonSheddableAlwaysAdmitted(t *testing.T) {
	policy := NewGAIELegacyAdmission(5, 0.8, nil)
	// Saturated: QueueDepth=100 -> qRatio=20, KVUtil=1.0 -> kvRatio=1.25 -> sat=20 >> 1.0
	state := &RouterState{
		Snapshots: []RoutingSnapshot{{ID: "i0", QueueDepth: 100, KVUtilization: 1.0}},
	}
	for _, class := range []string{"critical", "standard", "", "unknown"} {
		req := &Request{ID: "r", SLOClass: class}
		admitted, _ := policy.Admit(req, state)
		if !admitted {
			t.Errorf("class=%q: non-sheddable must always be admitted, got rejected", class)
		}
	}
}

// BC-2: Sheddable requests rejected when saturation >= 1.0.
func TestGAIELegacy_SheddableRejectedWhenSaturated(t *testing.T) {
	policy := NewGAIELegacyAdmission(5, 0.8, nil)
	// qRatio=5/5=1.0, kvRatio=0.8/0.8=1.0 -> sat=1.0 (exactly at boundary)
	state := &RouterState{
		Snapshots: []RoutingSnapshot{{ID: "i0", QueueDepth: 5, KVUtilization: 0.8}},
	}
	for _, class := range []string{"batch", "sheddable", "background"} {
		req := &Request{ID: "r", SLOClass: class}
		admitted, reason := policy.Admit(req, state)
		if admitted {
			t.Errorf("class=%q: sheddable must be rejected at saturation=1.0", class)
		}
		if !strings.Contains(reason, "gaie-saturated") {
			t.Errorf("class=%q: reason should contain 'gaie-saturated', got %q", class, reason)
		}
	}
}

// BC-3: Sheddable requests admitted when saturation < 1.0.
func TestGAIELegacy_SheddableAdmittedWhenNotSaturated(t *testing.T) {
	policy := NewGAIELegacyAdmission(5, 0.8, nil)
	// qRatio=2/5=0.4, kvRatio=0.3/0.8=0.375 -> sat=0.4 < 1.0
	state := &RouterState{
		Snapshots: []RoutingSnapshot{{ID: "i0", QueueDepth: 2, KVUtilization: 0.3}},
	}
	for _, class := range []string{"batch", "sheddable", "background"} {
		req := &Request{ID: "r", SLOClass: class}
		admitted, _ := policy.Admit(req, state)
		if !admitted {
			t.Errorf("class=%q: sheddable should be admitted at saturation < 1.0", class)
		}
	}
}

// BC-3b: Sheddable admitted just below boundary (saturation=0.9875 < 1.0).
// Guards against off-by-epsilon error in the >= 1.0 comparison.
func TestGAIELegacy_SheddableAdmittedJustBelowBoundary(t *testing.T) {
	policy := NewGAIELegacyAdmission(5, 0.8, nil)
	// qRatio=4/5=0.8, kvRatio=0.79/0.8=0.9875 -> sat=max(0.8, 0.9875)=0.9875 < 1.0
	state := &RouterState{
		Snapshots: []RoutingSnapshot{{ID: "i0", QueueDepth: 4, KVUtilization: 0.79}},
	}
	req := &Request{ID: "r", SLOClass: "sheddable"}
	admitted, _ := policy.Admit(req, state)
	if !admitted {
		t.Error("sheddable should be admitted at saturation=0.9875 (just below 1.0)")
	}
}

// BC-4: Saturation formula matches GAIE: avg(max(qd/qdT, kv/kvT)).
func TestGAIELegacy_FormulaExact(t *testing.T) {
	policy := NewGAIELegacyAdmission(5, 0.8, nil)
	state := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "i0", QueueDepth: 10, KVUtilization: 0.4}, // max(10/5, 0.4/0.8) = max(2.0, 0.5) = 2.0
			{ID: "i1", QueueDepth: 1, KVUtilization: 0.9},  // max(1/5, 0.9/0.8) = max(0.2, 1.125) = 1.125
		},
	}
	// Expected: avg(2.0, 1.125) = 1.5625 -> sheddable rejected
	req := &Request{ID: "r", SLOClass: "sheddable"}
	admitted, _ := policy.Admit(req, state)
	if admitted {
		t.Error("sheddable should be rejected at saturation 1.5625")
	}

	// Both instances lightly loaded -> saturation < 1.0
	state2 := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "i0", QueueDepth: 2, KVUtilization: 0.3}, // max(0.4, 0.375) = 0.4
			{ID: "i1", QueueDepth: 1, KVUtilization: 0.2}, // max(0.2, 0.25) = 0.25
		},
	}
	// Expected: avg(0.4, 0.25) = 0.325 < 1.0 -> sheddable admitted
	admitted2, _ := policy.Admit(req, state2)
	if !admitted2 {
		t.Error("sheddable should be admitted at saturation 0.325")
	}
}

// BC-5: Empty snapshots -> saturation=1.0 -> sheddable rejected, non-sheddable admitted.
func TestGAIELegacy_EmptySnapshotsSaturated(t *testing.T) {
	policy := NewGAIELegacyAdmission(5, 0.8, nil)
	state := &RouterState{Snapshots: []RoutingSnapshot{}}

	req := &Request{ID: "r", SLOClass: "sheddable"}
	admitted, _ := policy.Admit(req, state)
	if admitted {
		t.Error("sheddable should be rejected with empty snapshots (saturation=1.0)")
	}

	reqCrit := &Request{ID: "r2", SLOClass: "critical"}
	admitted2, _ := policy.Admit(reqCrit, state)
	if !admitted2 {
		t.Error("non-sheddable should still be admitted with empty snapshots")
	}
}

// BC-6: Custom SLOPriorityMap changes sheddability for GAIELegacyAdmission.
func TestGAIELegacy_CustomPriorityMapMakesBatchNonSheddable(t *testing.T) {
	// Make batch non-sheddable (priority=0 instead of default -1).
	customMap := NewSLOPriorityMap(map[string]int{"batch": 0})
	policy := NewGAIELegacyAdmission(5, 0.8, customMap)
	// Saturated state: sat=1.0
	state := &RouterState{
		Snapshots: []RoutingSnapshot{{ID: "i0", QueueDepth: 5, KVUtilization: 0.8}},
	}
	// batch is now non-sheddable (priority=0 >= 0), so it should be admitted even at sat=1.0.
	req := &Request{ID: "r", SLOClass: "batch"}
	admitted, _ := policy.Admit(req, state)
	if !admitted {
		t.Error("batch with custom priority=0 should be non-sheddable and admitted at saturation=1.0")
	}
	// sheddable still has priority=-2, so it should be rejected.
	reqShed := &Request{ID: "r2", SLOClass: "sheddable"}
	admitted2, _ := policy.Admit(reqShed, state)
	if admitted2 {
		t.Error("sheddable should still be rejected at saturation=1.0 with custom map")
	}
}

// Constructor validation: NewGAIELegacyAdmission panics on invalid parameters (R3).
func TestGAIELegacy_ConstructorPanics(t *testing.T) {
	cases := []struct {
		name string
		qd   float64
		kv   float64
	}{
		{"zero qd", 0, 0.8},
		{"negative qd", -1, 0.8},
		{"NaN qd", math.NaN(), 0.8},
		{"Inf qd", math.Inf(1), 0.8},
		{"zero kv", 5, 0},
		{"negative kv", 5, -0.5},
		{"kv above 1.0", 5, 1.1},
		{"NaN kv", 5, math.NaN()},
		{"Inf kv", 5, math.Inf(1)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("expected panic for qd=%v kv=%v, got none", tc.qd, tc.kv)
				}
			}()
			NewGAIELegacyAdmission(tc.qd, tc.kv, nil)
		})
	}
}
