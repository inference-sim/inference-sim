package cluster

import (
	"testing"
)

// Unit tests for TenantTracker (sim/cluster/tenant.go).
// BDD scenarios from specs/002-tier-tenant-fairness/plan.md PR-3 test table.

// T_Tenant_001 — IsOverBudget returns false for empty tenantID (zero-value safe).
func TestTenantTracker_EmptyTenantIDNeverOverBudget(t *testing.T) {
	tracker := NewTenantTracker(map[string]float64{"alice": 0.5}, 10)
	if tracker.IsOverBudget("") {
		t.Error("expected IsOverBudget(\"\") to be false, got true")
	}
}

// T_Tenant_002 — IsOverBudget returns false when tenant has no configured budget.
func TestTenantTracker_NoBudgetNeverOverBudget(t *testing.T) {
	tracker := NewTenantTracker(map[string]float64{"alice": 0.5}, 10)
	if tracker.IsOverBudget("bob") {
		t.Error("expected IsOverBudget(\"bob\") to be false (no budget configured), got true")
	}
}

// T_Tenant_003 — nil budgets map never triggers over-budget (zero-value safe).
func TestTenantTracker_NilBudgetsNeverOverBudget(t *testing.T) {
	tracker := NewTenantTracker(nil, 10)
	tracker.OnStart("alice")
	tracker.OnStart("alice")
	if tracker.IsOverBudget("alice") {
		t.Error("expected IsOverBudget to be false with nil budgets, got true")
	}
}

// T_Tenant_004 — IsOverBudget returns true when inFlight exceeds budget fraction.
// Budget 0.3 * capacity 10 = 3 allowed. With 4 in-flight → over budget.
func TestTenantTracker_OverBudgetWhenExceedsFraction(t *testing.T) {
	tracker := NewTenantTracker(map[string]float64{"alice": 0.3}, 10)
	// 3 in-flight = exactly at budget → not over
	tracker.OnStart("alice")
	tracker.OnStart("alice")
	tracker.OnStart("alice")
	if tracker.IsOverBudget("alice") {
		t.Error("expected IsOverBudget to be false at exactly budget (3 in-flight, budget=3), got true")
	}
	// 4 in-flight = over budget
	tracker.OnStart("alice")
	if !tracker.IsOverBudget("alice") {
		t.Error("expected IsOverBudget to be true with 4 in-flight, budget=3, got false")
	}
}

// T_Tenant_005 — OnComplete decrements; once below threshold, no longer over budget.
func TestTenantTracker_OnCompleteDecrementsAndUnblocks(t *testing.T) {
	tracker := NewTenantTracker(map[string]float64{"alice": 0.3}, 10)
	tracker.OnStart("alice")
	tracker.OnStart("alice")
	tracker.OnStart("alice")
	tracker.OnStart("alice") // over budget (4 > 3)

	if !tracker.IsOverBudget("alice") {
		t.Error("pre-condition: should be over budget with 4 in-flight")
	}
	tracker.OnComplete("alice") // back to 3 → at limit, not over
	if tracker.IsOverBudget("alice") {
		t.Error("expected IsOverBudget to be false after completing to 3 in-flight, got true")
	}
}

// T_Tenant_006 — OnComplete never goes below zero (floor 0).
func TestTenantTracker_OnCompleteFloorZero(t *testing.T) {
	tracker := NewTenantTracker(map[string]float64{"alice": 0.5}, 10)
	// Complete without any start — should not panic and inFlight stays 0
	tracker.OnComplete("alice")
	tracker.OnComplete("alice")
	if tracker.IsOverBudget("alice") {
		t.Error("expected IsOverBudget to be false with inFlight=0, got true")
	}
}

// T_Tenant_007 — Two tenants tracked independently.
func TestTenantTracker_TwoTenantsIndependent(t *testing.T) {
	tracker := NewTenantTracker(map[string]float64{
		"alice": 0.3, // 3 allowed out of 10
		"bob":   0.5, // 5 allowed out of 10
	}, 10)

	// Put alice at 4 (over budget), bob at 2 (under budget)
	for i := 0; i < 4; i++ {
		tracker.OnStart("alice")
	}
	for i := 0; i < 2; i++ {
		tracker.OnStart("bob")
	}

	if !tracker.IsOverBudget("alice") {
		t.Error("expected alice to be over budget, got false")
	}
	if tracker.IsOverBudget("bob") {
		t.Error("expected bob to be under budget, got true")
	}
}

// T_Tenant_009 — budget=0.0 edge case: first request at any tick is not blocked.
// DES ordering: admission events fire at priority 1, routing events at priority 2.
// At admission time inFlight=0, so 0 > 0.0 is false → the first request passes through.
// This is the extreme edge of the same-tick burst tradeoff documented in IsOverBudget.
// The second request (after the first is routed and OnStart fires) will be blocked.
func TestTenantTracker_ZeroBudget_FirstRequestNotBlocked(t *testing.T) {
	tracker := NewTenantTracker(map[string]float64{"alice": 0.0}, 10)
	// Before any routing: inFlight=0. 0 > (0.0 * 10 = 0.0) is false → not over budget.
	if tracker.IsOverBudget("alice") {
		t.Error("zero budget: IsOverBudget should be false when inFlight=0 (DES ordering; first request slips through)")
	}
	// After routing fires (OnStart called): inFlight=1. 1 > 0.0 is true → over budget.
	tracker.OnStart("alice")
	if !tracker.IsOverBudget("alice") {
		t.Error("zero budget: IsOverBudget should be true when inFlight=1 (subsequent requests blocked)")
	}
}

// T_Tenant_008 — OnStart/OnComplete are no-ops for empty tenantID.
func TestTenantTracker_EmptyTenantIDNoop(t *testing.T) {
	tracker := NewTenantTracker(map[string]float64{"": 0.1}, 10)
	// Even if "" has a budget entry, OnStart/OnComplete should be no-ops for ""
	tracker.OnStart("")
	tracker.OnStart("")
	if tracker.IsOverBudget("") {
		t.Error("expected IsOverBudget(\"\") to always be false, even if budget entry exists for \"\"")
	}
}
