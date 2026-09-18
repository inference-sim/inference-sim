package lora

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestRegistry_RankLookup verifies id -> rank resolution for declared adapters and a
// miss for an unknown id (contracts/config-schema.md, data-model.md "Adapter Registry").
func TestRegistry_RankLookup(t *testing.T) {
	reg, err := NewRegistry([]sim.AdapterSpec{
		{ID: "adapter_0", Rank: 8, BaseModel: "llama-3.1-8b"},
		{ID: "adapter_1", Rank: 16},
	})
	if err != nil {
		t.Fatalf("NewRegistry: unexpected error %v", err)
	}

	if r, ok := reg.RankOf("adapter_0"); !ok || r != 8 {
		t.Errorf("RankOf(adapter_0) = (%d, %v), want (8, true)", r, ok)
	}
	if r, ok := reg.RankOf("adapter_1"); !ok || r != 16 {
		t.Errorf("RankOf(adapter_1) = (%d, %v), want (16, true)", r, ok)
	}
	if _, ok := reg.RankOf("nope"); ok {
		t.Errorf("RankOf(nope) should miss")
	}
	if bm, ok := reg.BaseModelOf("adapter_0"); !ok || bm != "llama-3.1-8b" {
		t.Errorf("BaseModelOf(adapter_0) = (%q, %v), want (llama-3.1-8b, true)", bm, ok)
	}
	if !reg.Has("adapter_1") {
		t.Errorf("Has(adapter_1) should be true")
	}
	if reg.Len() != 2 {
		t.Errorf("Len() = %d, want 2", reg.Len())
	}
}

// TestRegistry_IDsSorted verifies IDs() returns keys in deterministic sorted order (R2)
// regardless of declaration order — a determinism prerequisite for any output derived
// from registry iteration.
func TestRegistry_IDsSorted(t *testing.T) {
	reg, err := NewRegistry([]sim.AdapterSpec{
		{ID: "zeta", Rank: 8},
		{ID: "alpha", Rank: 8},
		{ID: "mu", Rank: 8},
	})
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}
	got := reg.IDs()
	want := []string{"alpha", "mu", "zeta"}
	if len(got) != len(want) {
		t.Fatalf("IDs() len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("IDs()[%d] = %q, want %q (must be key-sorted, R2)", i, got[i], want[i])
		}
	}
}

// TestRegistry_CheckReferences verifies the completeness check flags unknown ids and
// accepts references that all resolve (empty id is ignored — base-model-only request).
func TestRegistry_CheckReferences(t *testing.T) {
	reg, err := NewRegistry([]sim.AdapterSpec{
		{ID: "adapter_0", Rank: 8},
		{ID: "adapter_1", Rank: 16},
	})
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}

	if err := reg.CheckReferences("adapter_0", "", "adapter_1"); err != nil {
		t.Errorf("CheckReferences of known ids (+ empty) should pass, got %v", err)
	}
	if err := reg.CheckReferences("adapter_0", "ghost"); err == nil {
		t.Errorf("CheckReferences must flag unknown id 'ghost'")
	}
}

// TestNewRegistry_Rejections verifies the constructor rejects malformed registries:
// duplicate id, empty id, and non-positive rank (R3). These mirror LoRAConfig.Validate
// so a registry built directly is as safe as one built from a validated config.
func TestNewRegistry_Rejections(t *testing.T) {
	tests := []struct {
		name  string
		specs []sim.AdapterSpec
	}{
		{"duplicate id", []sim.AdapterSpec{{ID: "a", Rank: 8}, {ID: "a", Rank: 16}}},
		{"empty id", []sim.AdapterSpec{{ID: "", Rank: 8}}},
		{"zero rank", []sim.AdapterSpec{{ID: "a", Rank: 0}}},
		{"negative rank", []sim.AdapterSpec{{ID: "a", Rank: -1}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := NewRegistry(tt.specs); err == nil {
				t.Errorf("NewRegistry(%s) should error", tt.name)
			}
		})
	}
}

// TestNewAdapterRegistryFunc_Registered verifies sim/lora's init() wired the registry
// factory into the sim package (breaking the import cycle, mirroring
// sim.NewLatencyModelFunc). Production code reaches the registry through this seam.
func TestNewAdapterRegistryFunc_Registered(t *testing.T) {
	if sim.NewAdapterRegistryFunc == nil {
		t.Fatalf("sim.NewAdapterRegistryFunc must be registered by sim/lora init()")
	}
	reg, err := sim.NewAdapterRegistryFunc([]sim.AdapterSpec{{ID: "a", Rank: 8}})
	if err != nil {
		t.Fatalf("factory error: %v", err)
	}
	if r, ok := reg.RankOf("a"); !ok || r != 8 {
		t.Errorf("factory-built registry RankOf(a) = (%d,%v), want (8,true)", r, ok)
	}
}

// TestNewResidentAdapterSetFunc_Registered verifies sim/lora's init() wired the
// resident-set factory into the sim package (mirrors NewAdapterRegistryFunc). A
// refactor that breaks this seam would otherwise silently leave the Simulator's
// resident set nil and zero every adapter metric with no test failure.
func TestNewResidentAdapterSetFunc_Registered(t *testing.T) {
	if sim.NewResidentAdapterSetFunc == nil {
		t.Fatalf("sim.NewResidentAdapterSetFunc must be registered by sim/lora init()")
	}
	set := sim.NewResidentAdapterSetFunc(2)
	if set == nil {
		t.Fatalf("factory returned nil resident set")
	}
	if set.Len() != 0 {
		t.Errorf("fresh resident set Len() = %d, want 0", set.Len())
	}
}
