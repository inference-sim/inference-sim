package workload

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/lora"
)

// adapterTestSpec builds a minimal single-client spec with the given adapter id.
func adapterTestSpec(model, adapter string) *WorkloadSpec {
	return &WorkloadSpec{
		Version: "2", Seed: 42, Category: "language", AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "c1", TenantID: "t1", RateFraction: 1.0, SLOClass: "standard",
			Model: model, Adapter: adapter,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
}

// TestGenerateRequests_ThreadsAdapter verifies ClientSpec.Adapter is threaded onto
// every generated Request.Adapter (US1). This is the identity plumbing the per-adapter
// metrics depend on.
func TestGenerateRequests_ThreadsAdapter(t *testing.T) {
	spec := adapterTestSpec("llama-3.1-8b", "adapter_0")
	reqs, err := GenerateRequests(spec, int64(1e6), 50)
	if err != nil {
		t.Fatalf("GenerateRequests: %v", err)
	}
	if len(reqs) == 0 {
		t.Fatal("expected non-empty requests")
	}
	for _, r := range reqs {
		if r.Adapter != "adapter_0" {
			t.Fatalf("request %s: Adapter = %q, want adapter_0", r.ID, r.Adapter)
		}
	}
}

// TestGenerateRequests_OmittedAdapterIsEmpty verifies the no-op default: a client with
// no adapter yields base-model-only requests (Request.Adapter == "").
func TestGenerateRequests_OmittedAdapterIsEmpty(t *testing.T) {
	spec := adapterTestSpec("llama-3.1-8b", "")
	reqs, err := GenerateRequests(spec, int64(1e6), 50)
	if err != nil {
		t.Fatalf("GenerateRequests: %v", err)
	}
	if len(reqs) == 0 {
		t.Fatal("expected non-empty requests")
	}
	for _, r := range reqs {
		if r.Adapter != "" {
			t.Fatalf("request %s: Adapter = %q, want empty (base-model-only)", r.ID, r.Adapter)
		}
	}
}

// TestGenerateRequests_ThreadsCohortAdapter verifies CohortSpec.Adapter flows through
// cohort expansion into every generated request.
func TestGenerateRequests_ThreadsCohortAdapter(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, Category: "language", AggregateRate: 10.0,
		Cohorts: []CohortSpec{{
			ID: "h0", Population: 2, SLOClass: "standard", Model: "llama-3.1-8b",
			Adapter: "adapter_1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	if err := ExpandClientsAndCohorts(spec); err != nil {
		t.Fatalf("ExpandClientsAndCohorts: %v", err)
	}
	reqs, err := GenerateRequests(spec, int64(1e6), 50)
	if err != nil {
		t.Fatalf("GenerateRequests: %v", err)
	}
	if len(reqs) == 0 {
		t.Fatal("expected non-empty requests")
	}
	for _, r := range reqs {
		if r.Adapter != "adapter_1" {
			t.Fatalf("request %s: Adapter = %q, want adapter_1", r.ID, r.Adapter)
		}
	}
}

// mustRegistry builds a lora.Registry (a sim.AdapterRegistry) for validation tests.
func mustRegistry(t *testing.T, specs ...sim.AdapterSpec) sim.AdapterRegistry {
	t.Helper()
	reg, err := lora.NewRegistry(specs)
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}
	return reg
}

// TestValidateAdapterReferences verifies workload-vs-registry cross-validation (US1):
//   - a referenced adapter id must be a registry key (completeness)
//   - a declared base model must match the client/cohort model
//   - an omitted adapter is always valid (base-model-only, no-op)
func TestValidateAdapterReferences(t *testing.T) {
	reg := mustRegistry(t,
		sim.AdapterSpec{ID: "adapter_0", Rank: 8, BaseModel: "llama-3.1-8b"},
		sim.AdapterSpec{ID: "adapter_1", Rank: 16}, // no base model declared => any model
	)

	t.Run("known id, matching base model", func(t *testing.T) {
		spec := adapterTestSpec("llama-3.1-8b", "adapter_0")
		if err := ValidateAdapterReferences(spec, reg); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})
	t.Run("no base model declared accepts any client model", func(t *testing.T) {
		spec := adapterTestSpec("some-other-model", "adapter_1")
		if err := ValidateAdapterReferences(spec, reg); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})
	t.Run("omitted adapter is valid", func(t *testing.T) {
		spec := adapterTestSpec("llama-3.1-8b", "")
		if err := ValidateAdapterReferences(spec, reg); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})
	t.Run("unknown adapter id rejected", func(t *testing.T) {
		spec := adapterTestSpec("llama-3.1-8b", "ghost")
		if err := ValidateAdapterReferences(spec, reg); err == nil {
			t.Error("expected error for unknown adapter id")
		}
	})
	t.Run("base model mismatch rejected", func(t *testing.T) {
		spec := adapterTestSpec("qwen-2.5-7b", "adapter_0") // adapter_0 is for llama-3.1-8b
		if err := ValidateAdapterReferences(spec, reg); err == nil {
			t.Error("expected error for base-model mismatch")
		}
	})
	t.Run("cohort adapter validated too", func(t *testing.T) {
		spec := &WorkloadSpec{
			Version: "2", Seed: 42, Category: "language", AggregateRate: 10.0,
			Cohorts: []CohortSpec{{
				ID: "h0", Population: 1, SLOClass: "standard", Model: "llama-3.1-8b",
				Adapter: "ghost", RateFraction: 1.0,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
			}},
		}
		if err := ValidateAdapterReferences(spec, reg); err == nil {
			t.Error("expected error for unknown cohort adapter id")
		}
	})
}
