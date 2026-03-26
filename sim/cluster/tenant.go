package cluster

import (
	"fmt"
	"math"
)

// TenantTracker tracks in-flight request counts per tenant and enforces fair-share budgets.
// Zero-value is safe: when no budgets are configured, IsOverBudget always returns false.
//
// Behavioral contract: specs/002-tier-tenant-fairness/contracts/tenant-tracker.md
type TenantTracker struct {
	budgets       map[string]float64 // tenantID → fraction of totalCapacity (absent key = unlimited; 0.0 = zero slots allowed)
	inFlight      map[string]int     // tenantID → current in-flight count
	totalCapacity int                // max in-flight slots across cluster
}

// NewTenantTracker creates a TenantTracker from a budget map and cluster capacity.
// budgets may be nil (unlimited for all tenants).
// totalCapacity must be >= 1.
// Panics if any budget value is outside [0, 1] or is NaN/Inf (R3).
func NewTenantTracker(budgets map[string]float64, totalCapacity int) *TenantTracker {
	if totalCapacity < 1 {
		totalCapacity = 1
	}
	for tenantID, v := range budgets {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || v > 1 {
			panic(fmt.Sprintf("NewTenantTracker: budget for tenant %q is %v; must be in [0, 1]", tenantID, v))
		}
	}
	return &TenantTracker{
		budgets:       budgets,
		inFlight:      make(map[string]int),
		totalCapacity: totalCapacity,
	}
}

// IsOverBudget returns true when tenantID has a configured budget and currently exceeds it.
// Always returns false when tenantID is empty or has no configured budget.
// Pure query — no state mutation.
func (t *TenantTracker) IsOverBudget(tenantID string) bool {
	if tenantID == "" || t.budgets == nil {
		return false
	}
	budget, ok := t.budgets[tenantID]
	if !ok {
		return false
	}
	limit := budget * float64(t.totalCapacity)
	return float64(t.inFlight[tenantID]) > limit
}

// OnStart increments the in-flight count for tenantID.
// No-op when tenantID is empty.
func (t *TenantTracker) OnStart(tenantID string) {
	if tenantID == "" {
		return
	}
	t.inFlight[tenantID]++
}

// OnComplete decrements the in-flight count for tenantID, floor 0.
// No-op when tenantID is empty.
func (t *TenantTracker) OnComplete(tenantID string) {
	if tenantID == "" {
		return
	}
	if t.inFlight[tenantID] > 0 {
		t.inFlight[tenantID]--
	}
}
