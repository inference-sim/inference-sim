package cmd

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

// makeTestSpec returns a minimal WorkloadSpec for seed tests.
func makeTestSpec(seed int64) *workload.WorkloadSpec {
	return &workload.WorkloadSpec{
		Version: "1", Seed: seed, Category: "language", AggregateRate: 10.0,
		Clients: []workload.ClientSpec{{
			ID: "c1", TenantID: "t1", RateFraction: 1.0, SLOClass: "interactive",
			Arrival:    workload.ArrivalSpec{Process: "poisson"},
			InputDist:  workload.DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
			OutputDist: workload.DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
}

// TestSeedOverride_DifferentSeeds_DifferentWorkloads verifies BC-1/BC-2:
// when the CLI seed overrides the YAML seed, different seeds produce
// different workloads (arrival times and token counts differ).
func TestSeedOverride_DifferentSeeds_DifferentWorkloads(t *testing.T) {
	// GIVEN a workload spec with YAML seed 42
	spec1 := makeTestSpec(42)
	spec2 := makeTestSpec(42)

	// WHEN CLI --seed overrides to different values
	spec1.Seed = 100 // simulates Changed("seed") → spec.Seed = 100
	spec2.Seed = 200 // simulates Changed("seed") → spec.Seed = 200

	horizon := int64(1e6)
	r1, err := workload.GenerateRequests(spec1, horizon, 50)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := workload.GenerateRequests(spec2, horizon, 50)
	if err != nil {
		t.Fatal(err)
	}

	// THEN the workloads differ (at least one request has different arrival time)
	if len(r1) == 0 || len(r2) == 0 {
		t.Fatal("expected non-empty request sets")
	}
	anyDifferent := false
	minLen := min(len(r1), len(r2))
	for i := 0; i < minLen; i++ {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			anyDifferent = true
			break
		}
	}
	if len(r1) != len(r2) {
		anyDifferent = true
	}
	if !anyDifferent {
		t.Error("different seeds produced identical workloads — seed override is not working")
	}
}

// TestSeedOverride_SameSeed_IdenticalWorkload verifies BC-4:
// same seed produces byte-identical workload (determinism preserved).
func TestSeedOverride_SameSeed_IdenticalWorkload(t *testing.T) {
	// GIVEN two specs with the same seed (simulating CLI override to same value)
	spec1 := makeTestSpec(42)
	spec2 := makeTestSpec(42)
	spec1.Seed = 123
	spec2.Seed = 123

	horizon := int64(1e6)
	r1, err := workload.GenerateRequests(spec1, horizon, 50)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := workload.GenerateRequests(spec2, horizon, 50)
	if err != nil {
		t.Fatal(err)
	}

	// THEN output is identical
	if len(r1) != len(r2) {
		t.Fatalf("different counts: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			t.Errorf("request %d: arrival %d vs %d", i, r1[i].ArrivalTime, r2[i].ArrivalTime)
			break
		}
	}
}

// TestSeedOverride_YAMLSeedPreserved_WhenCLINotSpecified verifies BC-3/BC-5:
// when --seed is not explicitly passed, the YAML seed governs workload
// generation (backward compatibility).
func TestSeedOverride_YAMLSeedPreserved_WhenCLINotSpecified(t *testing.T) {
	// GIVEN a spec with YAML seed 42 (no CLI override)
	specA := makeTestSpec(42)
	specB := makeTestSpec(42)

	horizon := int64(1e6)
	r1, err := workload.GenerateRequests(specA, horizon, 50)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := workload.GenerateRequests(specB, horizon, 50)
	if err != nil {
		t.Fatal(err)
	}

	// THEN same YAML seed produces identical workload (YAML seed is the default)
	if len(r1) != len(r2) {
		t.Fatalf("different counts: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			t.Errorf("request %d: arrival %d vs %d — YAML seed not preserved", i, r1[i].ArrivalTime, r2[i].ArrivalTime)
			break
		}
	}
}
