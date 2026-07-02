package cmd

import (
	"testing"

	"blis/sim/workload"
)

// makeTestSpec returns a minimal WorkloadSpec for seed tests.
func makeTestSpec(seed int64) *workload.WorkloadSpec {
	return &workload.WorkloadSpec{
		Version: "2", Seed: seed, Category: "language", AggregateRate: 10.0,
		Clients: []workload.ClientSpec{{
			ID: "c1", TenantID: "t1", RateFraction: 1.0, SLOClass: "standard",
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

	// THEN the workloads differ (arrival times or token counts)
	if len(r1) == 0 || len(r2) == 0 {
		t.Fatal("expected non-empty request sets")
	}
	anyDifferent := false
	minLen := min(len(r1), len(r2))
	for i := 0; i < minLen; i++ {
		if r1[i].ArrivalTime != r2[i].ArrivalTime ||
			len(r1[i].InputTokens) != len(r2[i].InputTokens) {
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
	spec1 := makeTestSpec(123)
	spec2 := makeTestSpec(123)

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
//
// Note: This test validates the library-level determinism property (same spec.Seed
// → same workload). The CLI wiring (cmd.Flags().Changed("seed") gating the override)
// is verified by code inspection — it follows the exact Changed() pattern used by
// --horizon and --num-requests at cmd/root.go:222-229.
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

// makeMultiTurnAccumulateSpec returns a WorkloadSpec exercising the multi-turn
// accumulate path that PR #1445 changed (SessionTokenBuffer storage). Drives
// the open-loop reasoning generator.
func makeMultiTurnAccumulateSpec(seed int64) *workload.WorkloadSpec {
	rr := 10.0
	return &workload.WorkloadSpec{
		Version: "2", Seed: seed, Category: "language", AggregateRate: rr,
		Clients: []workload.ClientSpec{{
			ID: "c1", TenantID: "t1", RateFraction: 1.0, SLOClass: "standard",
			Arrival:    workload.ArrivalSpec{Process: "poisson"},
			InputDist:  workload.DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 80, "std_dev": 10, "min": 50, "max": 150}},
			OutputDist: workload.DistSpec{Type: "exponential", Params: map[string]float64{"mean": 40}},
			Reasoning: &workload.ReasoningSpec{
				MultiTurn: &workload.MultiTurnSpec{
					MaxRounds:     5,
					ThinkTimeUs:   1000,
					ContextGrowth: "accumulate",
				},
			},
		}},
	}
}

// TestSeedDeterminism_MultiTurnAccumulate verifies INV-6 under the delta-encoded
// multi-turn path introduced in PR #1445: same seed must produce byte-identical
// per-request arrival times, input lengths, AND input token VALUES across runs.
// The token-value check is the load-bearing one — it catches representation
// changes (e.g. SessionTokenBuffer slicing) that fail to preserve content.
func TestSeedDeterminism_MultiTurnAccumulate(t *testing.T) {
	const seed = 7777
	spec1 := makeMultiTurnAccumulateSpec(seed)
	spec2 := makeMultiTurnAccumulateSpec(seed)

	horizon := int64(1e7)
	r1, err := workload.GenerateRequests(spec1, horizon, 100)
	if err != nil {
		t.Fatal(err)
	}
	r2, err := workload.GenerateRequests(spec2, horizon, 100)
	if err != nil {
		t.Fatal(err)
	}

	if len(r1) != len(r2) {
		t.Fatalf("INV-6: different request counts: %d vs %d", len(r1), len(r2))
	}
	if len(r1) == 0 {
		t.Fatal("expected non-empty request set for multi-turn accumulate workload")
	}
	for i := range r1 {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			t.Errorf("INV-6: request %d arrival diverged: %d vs %d", i, r1[i].ArrivalTime, r2[i].ArrivalTime)
		}
		if r1[i].InputLen() != r2[i].InputLen() {
			t.Errorf("INV-6: request %d input length diverged: %d vs %d", i, r1[i].InputLen(), r2[i].InputLen())
		}
		// Spot-check token VALUES at three positions in each request — first,
		// middle, last. A representation change that breaks determinism would
		// surface as a value mismatch at one of these.
		a := r1[i].FullInputTokens()
		b := r2[i].FullInputTokens()
		if len(a) != len(b) {
			t.Errorf("INV-6: request %d FullInputTokens length diverged", i)
			continue
		}
		positions := []int{0, len(a) / 2, len(a) - 1}
		for _, p := range positions {
			if p < 0 || p >= len(a) {
				continue
			}
			if a[p] != b[p] {
				t.Errorf("INV-6: request %d token[%d] diverged: %d vs %d", i, p, a[p], b[p])
				break
			}
		}
	}
}
