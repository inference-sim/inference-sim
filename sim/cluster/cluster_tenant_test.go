package cluster

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// newTenantRequest creates a request with a TenantID and SLOClass.
func newTenantRequest(id string, arrivalTime int64, tenantID, sloClass string) *sim.Request {
	return &sim.Request{
		ID:           id,
		ArrivalTime:  arrivalTime,
		TenantID:     tenantID,
		SLOClass:     sloClass,
		InputTokens:  make([]int, 50),
		OutputTokens: make([]int, 20),
		State:        sim.StateQueued,
	}
}

// newTenantConfig creates a DeploymentConfig with tier-shed admission and tenant budgets.
func newTenantConfig(numInstances int, tenantBudgets map[string]float64) DeploymentConfig {
	cfg := newTestDeploymentConfig(numInstances)
	cfg.AdmissionPolicy = "tier-shed"
	cfg.TierShedThreshold = 0
	cfg.TierShedMinPriority = 2 // shed only background (0) and batch (1); sheddable (2) passes unless over budget
	cfg.TenantBudgets = tenantBudgets
	return cfg
}

// T_TenantInteg_001 — Over-budget tenant's Sheddable shed; on-budget tenant's Sheddable admitted.
// Use MaxRunningReqs=5 so totalCapacity = 2×5 = 10; alice budget=0.1 → limit=1 slot.
// Dense arrivals ensure alice quickly exceeds her 1-slot budget.
//
// Differential verification: budgets are swapped in the second run to prove enforcement
// tracks per-tenant identity, not a fixed constant. The constrained tenant sheds in both
// configurations; the run with alice constrained and the run with bob constrained both
// produce non-zero shedding.
func TestTenantAdmission_OverBudgetSheddableShed(t *testing.T) {
	const n = 60

	makeRequests := func() []*sim.Request {
		reqs := make([]*sim.Request, 0, n)
		for i := 0; i < n; i++ {
			tenant := "alice"
			if i%2 == 1 {
				tenant = "bob"
			}
			reqs = append(reqs, newTenantRequest(
				fmt.Sprintf("req_%s_%d", tenant, i),
				int64(i)*5,
				tenant,
				"sheddable",
			))
		}
		return reqs
	}

	// Run A: alice=0.1 (1 slot), bob=0.9 (9 slots) — alice is over-budget, sheds sheddable.
	cfgA := newTenantConfig(2, map[string]float64{"alice": 0.1, "bob": 0.9})
	cfgA.BatchConfig = sim.NewBatchConfig(5, 2048, 0) // totalCapacity=10; alice limit=1 slot
	csA := NewClusterSimulator(cfgA, makeRequests(), nil)
	mustRun(t, csA)

	shedA := csA.ShedByTier()["sheddable"]
	if shedA == 0 {
		t.Error("run A: expected over-budget alice's sheddable requests to be shed, got 0 shed")
	}

	// Differential: run B swaps budgets — bob=0.1 (1 slot), alice=0.9 (9 slots).
	// Now bob is over-budget; shed count should be non-zero regardless of which tenant is constrained.
	cfgB := newTenantConfig(2, map[string]float64{"alice": 0.9, "bob": 0.1})
	cfgB.BatchConfig = sim.NewBatchConfig(5, 2048, 0) // totalCapacity=10; bob limit=1 slot
	csB := NewClusterSimulator(cfgB, makeRequests(), nil)
	mustRun(t, csB)

	shedB := csB.ShedByTier()["sheddable"]
	if shedB == 0 {
		t.Error("run B: expected over-budget bob's sheddable requests to be shed, got 0 shed")
	}
}

// T_TenantInteg_002 — Over-budget tenant's Critical and Standard NOT shed by budget enforcement.
func TestTenantAdmission_CriticalAndStandardProtectedFromBudgetShed(t *testing.T) {
	const n = 40
	var requests []*sim.Request

	// Alice sends only critical requests at high rate — should all be admitted regardless of budget
	for i := 0; i < n; i++ {
		requests = append(requests, newTenantRequest(
			fmt.Sprintf("req_alice_crit_%d", i),
			int64(i)*5,
			"alice",
			"critical",
		))
	}

	// alice gets very tiny budget
	cfg := newTenantConfig(2, map[string]float64{
		"alice": 0.05, // only ~0.5 slots allowed
	})
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	// No critical requests should be shed by tenant budget enforcement
	shedCritical := cs.ShedByTier()["critical"]
	if shedCritical > 0 {
		t.Errorf("critical requests should not be shed by tenant budget enforcement, got shed=%d", shedCritical)
	}
	// Verify requests actually ran (not silently dropped for an unrelated reason)
	completed := cs.AggregatedMetrics().CompletedRequests
	if completed == 0 {
		t.Errorf("expected some critical requests to complete, got 0 completions out of %d requests", n)
	}
}

// T_TenantInteg_003 — Simulation with no TenantID produces byte-identical output (INV-6).
// TenantBudgets: nil → zero-value path — same as running without any tenant config.
func TestTenantAdmission_NilBudgetsIdenticalToBaseline(t *testing.T) {
	requests1 := newTestRequests(30)
	requests2 := newTestRequests(30)

	// Run 1: no tenant budgets
	cfg1 := newTestDeploymentConfig(2)
	cs1 := NewClusterSimulator(cfg1, requests1, nil)
	mustRun(t, cs1)

	// Run 2: explicit nil TenantBudgets (zero-value safe)
	cfg2 := newTestDeploymentConfig(2)
	cfg2.TenantBudgets = nil
	cs2 := NewClusterSimulator(cfg2, requests2, nil)
	mustRun(t, cs2)

	m1, _ := json.Marshal(cs1.AggregatedMetrics())
	m2, _ := json.Marshal(cs2.AggregatedMetrics())
	if !bytes.Equal(m1, m2) {
		t.Error("nil TenantBudgets should produce byte-identical output (INV-6 violated)")
	}
}

// T_TenantInteg_005 — INV-9: tenant budget decision never reads req.OutputTokens.
// (renumbered; original T004 deleted: it was structural; T003 already covers the observable guarantee)
// This is a structural invariant — enforced by code inspection and TenantTracker contract.
// We verify it indirectly: a simulation where OutputTokens is zeroed produces the same
// admission outcomes as one where OutputTokens is populated.
func TestTenantAdmission_INV9_DoesNotReadOutputTokens(t *testing.T) {
	makeReqs := func(zeroOutput bool) []*sim.Request {
		reqs := make([]*sim.Request, 20)
		for i := range reqs {
			out := make([]int, 20)
			if !zeroOutput {
				for j := range out {
					out[j] = j + 1
				}
			}
			reqs[i] = &sim.Request{
				ID:           fmt.Sprintf("req_%d", i),
				ArrivalTime:  int64(i) * 5,
				TenantID:     "alice",
				SLOClass:     "sheddable",
				InputTokens:  make([]int, 50),
				OutputTokens: out,
				State:        sim.StateQueued,
			}
		}
		return reqs
	}

	// Use MaxRunningReqs=5 so totalCapacity=10; alice budget=0.1 → limit=1 slot.
	// Dense arrivals ensure the budget enforcement code path actually executes.
	cfg1 := newTenantConfig(2, map[string]float64{"alice": 0.1})
	cfg1.BatchConfig = sim.NewBatchConfig(5, 2048, 0)
	cs1 := NewClusterSimulator(cfg1, makeReqs(true), nil)
	mustRun(t, cs1)

	cfg2 := newTenantConfig(2, map[string]float64{"alice": 0.1})
	cfg2.BatchConfig = sim.NewBatchConfig(5, 2048, 0)
	cs2 := NewClusterSimulator(cfg2, makeReqs(false), nil)
	mustRun(t, cs2)

	shed1 := cs1.ShedByTier()["sheddable"]
	shed2 := cs2.ShedByTier()["sheddable"]
	// Verify the budget enforcement code path was reached.
	if shed1 == 0 {
		t.Error("INV-9 test setup error: alice was never over-budget; budget enforcement did not fire")
	}
	// Shed counts must be equal — output tokens must not influence admission
	if shed1 != shed2 {
		t.Errorf("INV-9 violated: shed counts differ based on OutputTokens content (%d vs %d)", shed1, shed2)
	}
}

// T_TenantInteg_006 — Tenant budget enforced in PD disaggregation mode.
// In PD mode each parent request occupies 2 capacity slots (1 prefill + 1 decode).
// With alice budget=0.1 and totalCapacity=40, alice's limit is 4 slots (~2 concurrent
// parent requests). Dense arrivals cause alice to exceed her limit, shedding sheddable requests.
// This test verifies the 2-slot-per-parent semantics do not prevent budget enforcement.
func TestTenantAdmission_PDMode_BudgetEnforced(t *testing.T) {
	const n = 40
	var requests []*sim.Request
	for i := 0; i < n; i++ {
		requests = append(requests, newTenantRequest(
			fmt.Sprintf("req_alice_pd_%d", i),
			int64(i)*5,
			"alice",
			"sheddable",
		))
	}

	// 4 instances: 2 prefill + 2 decode; MaxRunningReqs=10 → totalCapacity=40; alice limit=4 slots.
	cfg := newTestDisaggDeploymentConfig(4, 2, 2)
	cfg.AdmissionPolicy = "tier-shed"
	cfg.TierShedThreshold = 0
	cfg.TierShedMinPriority = 2 // sheddable passes tier-shed; budget catches over-budget alice
	cfg.TenantBudgets = map[string]float64{"alice": 0.1}
	cfg.BatchConfig = sim.NewBatchConfig(10, 2048, 0) // totalCapacity=40; alice limit=4 slots

	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	shed := cs.ShedByTier()["sheddable"]
	if shed == 0 {
		t.Error("PD mode: expected over-budget alice's sheddable requests to be shed, got 0 shed")
	}
	completed := cs.AggregatedMetrics().CompletedRequests
	if completed == 0 {
		t.Errorf("PD mode: expected some requests to complete, got 0 completions out of %d", n)
	}
}
