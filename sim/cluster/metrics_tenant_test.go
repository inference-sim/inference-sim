package cluster

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// buildMetricsWithTenants constructs a *sim.Metrics with completed requests distributed
// across tenants. Each entry in tenantTokens maps tenantID → output tokens for one
// completed request. Multiple entries with the same key add separate requests.
func buildMetricsWithTenants(entries []struct {
	tenantID string
	tokens   int
}) *sim.Metrics {
	m := sim.NewMetrics()
	for i, e := range entries {
		id := fmt.Sprintf("req-%d", i) // deterministic unique IDs
		rm := sim.RequestMetrics{
			ID:              id,
			TenantID:        e.tenantID,
			NumDecodeTokens: e.tokens,
		}
		m.Requests[id] = rm
		m.RequestE2Es[id] = float64(i+1) * 1000 // non-zero E2E (completed)
	}
	return m
}

// BC-T1: ComputePerTenantMetrics returns nil when no requests have a TenantID.
//
// GIVEN aggregated metrics where all RequestMetrics have empty TenantID
// WHEN ComputePerTenantMetrics is called
// THEN it returns nil
func TestComputePerTenantMetrics_NoTenantIDs_ReturnsNil(t *testing.T) {
	entries := []struct {
		tenantID string
		tokens   int
	}{
		{"", 100},
		{"", 200},
	}
	m := buildMetricsWithTenants(entries)

	result := ComputePerTenantMetrics(m)

	if result != nil {
		t.Errorf("expected nil when no TenantIDs set, got map with %d entries", len(result))
	}
}

// BC-T2: Balanced two-tenant workload produces Jain index >= 0.99.
//
// GIVEN two tenants each with equal NumDecodeTokens across completed requests
// WHEN ComputePerTenantMetrics is called
// THEN JainFairnessIndex over the token map is >= 0.99
func TestComputePerTenantMetrics_Balanced_JainNearOne(t *testing.T) {
	entries := []struct {
		tenantID string
		tokens   int
	}{
		{"alice", 500},
		{"alice", 500},
		{"bob", 500},
		{"bob", 500},
	}
	m := buildMetricsWithTenants(entries)

	result := ComputePerTenantMetrics(m)

	if result == nil {
		t.Fatal("expected non-nil result for two-tenant workload")
	}
	if len(result) != 2 {
		t.Fatalf("expected 2 tenant entries, got %d", len(result))
	}

	tokenMap := make(map[string]float64, len(result))
	for id, tm := range result {
		tokenMap[id] = float64(tm.TotalTokensServed)
	}
	jain := JainFairnessIndex(tokenMap)
	if jain < 0.99 {
		t.Errorf("expected Jain index >= 0.99 for balanced workload, got %.4f", jain)
	}
}

// BC-T3: Skewed 10x workload produces Jain index < 0.70.
//
// GIVEN tenant "alice" with 10x the output tokens of tenant "bob"
// WHEN ComputePerTenantMetrics is called
// THEN JainFairnessIndex over the token map is < 0.70
func TestComputePerTenantMetrics_Skewed10x_JainBelow70(t *testing.T) {
	entries := []struct {
		tenantID string
		tokens   int
	}{
		{"alice", 10000},
		{"bob", 1000},
	}
	m := buildMetricsWithTenants(entries)

	result := ComputePerTenantMetrics(m)

	if result == nil {
		t.Fatal("expected non-nil result for two-tenant workload")
	}

	tokenMap := make(map[string]float64, len(result))
	for id, tm := range result {
		tokenMap[id] = float64(tm.TotalTokensServed)
	}
	jain := JainFairnessIndex(tokenMap)
	if jain >= 0.70 {
		t.Errorf("expected Jain index < 0.70 for 10x skewed workload, got %.4f", jain)
	}
}

// BC-T4: Single tenant produces a map with one entry and Jain index of 1.0.
//
// GIVEN exactly one tenant with completed requests
// WHEN ComputePerTenantMetrics is called
// THEN the returned map has one entry AND JainFairnessIndex returns 1.0
func TestComputePerTenantMetrics_SingleTenant_JainOne(t *testing.T) {
	entries := []struct {
		tenantID string
		tokens   int
	}{
		{"alice", 300},
		{"alice", 400},
	}
	m := buildMetricsWithTenants(entries)

	result := ComputePerTenantMetrics(m)

	if result == nil {
		t.Fatal("expected non-nil result for single-tenant workload")
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 tenant entry, got %d", len(result))
	}

	tm, ok := result["alice"]
	if !ok {
		t.Fatal("expected entry for tenant 'alice'")
	}
	if tm.CompletedRequests != 2 {
		t.Errorf("expected CompletedRequests=2, got %d", tm.CompletedRequests)
	}
	if tm.TotalTokensServed != 700 {
		t.Errorf("expected TotalTokensServed=700, got %d", tm.TotalTokensServed)
	}

	tokenMap := map[string]float64{"alice": float64(tm.TotalTokensServed)}
	jain := JainFairnessIndex(tokenMap)
	if jain != 1.0 {
		t.Errorf("expected Jain index = 1.0 for single tenant, got %.4f", jain)
	}
}

// BC-T5: Requests without TenantID are excluded from per-tenant counts.
//
// GIVEN a mix of tenanted requests (TenantID="alice") and untenanted (TenantID="")
// WHEN ComputePerTenantMetrics is called
// THEN only "alice" appears in the result map (no "" entry)
func TestComputePerTenantMetrics_UntenantedRequestsExcluded(t *testing.T) {
	entries := []struct {
		tenantID string
		tokens   int
	}{
		{"alice", 500},
		{"", 300}, // untenanted — must be excluded
		{"alice", 200},
		{"", 100}, // untenanted — must be excluded
	}
	m := buildMetricsWithTenants(entries)

	result := ComputePerTenantMetrics(m)

	if result == nil {
		t.Fatal("expected non-nil result (alice has tenanted requests)")
	}
	if _, ok := result[""]; ok {
		t.Error("expected no entry for empty TenantID, but found one")
	}
	if len(result) != 1 {
		t.Errorf("expected exactly 1 tenant entry (alice), got %d", len(result))
	}
	if result["alice"].CompletedRequests != 2 {
		t.Errorf("expected alice CompletedRequests=2, got %d", result["alice"].CompletedRequests)
	}
	if result["alice"].TotalTokensServed != 700 {
		t.Errorf("expected alice TotalTokensServed=700, got %d", result["alice"].TotalTokensServed)
	}
}

// BC-T6: Orphaned RequestE2Es entries (no matching Requests entry) are silently skipped.
//
// GIVEN aggregated metrics with a reqID in RequestE2Es but not in Requests
// WHEN ComputePerTenantMetrics is called
// THEN no panic occurs AND the orphaned ID does not appear in any tenant entry
func TestComputePerTenantMetrics_OrphanedE2E_Skipped(t *testing.T) {
	m := buildMetricsWithTenants([]struct {
		tenantID string
		tokens   int
	}{
		{"alice", 500},
	})
	// Inject an orphaned E2E entry with no corresponding Requests entry.
	m.RequestE2Es["ghost"] = 9999.0

	result := ComputePerTenantMetrics(m)

	if result == nil {
		t.Fatal("expected non-nil result for workload with one tenanted request")
	}
	for _, tm := range result {
		if tm.TenantID == "ghost" {
			t.Error("orphaned entry 'ghost' must not appear in any tenant")
		}
	}
	if len(result) != 1 {
		t.Errorf("expected exactly 1 tenant entry (alice), got %d", len(result))
	}
}
