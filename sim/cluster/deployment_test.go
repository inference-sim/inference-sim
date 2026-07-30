package cluster

import (
	"bytes"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"gopkg.in/yaml.v3"
)

// These are the B-5 (#1493) INV-PS2 (pre-placement conservation) contract tests
// for LoRA adapter placement. They exercise the exported ValidateLoRAPlacement
// checker directly (one row per DD-B5-f check), the strict YAML schema (R10), and
// the construction-time panic surface (Principle V, library layer). LoRA
// construction funcs are registered via the blank import in lora_import_test.go.

// loraPlacementConfig builds a LoRA-enabled DeploymentConfig with the given
// per-instance resident capacity, adapter registry contents, and placement map,
// plus a matching AdapterRegistry. Cost coefficients mirror the other cluster
// LoRA tests so full construction can reach the placement check.
func loraPlacementConfig(t *testing.T, numInstances, capacity int, placement map[int][]string, adapterIDs ...string) (DeploymentConfig, sim.AdapterRegistry) {
	t.Helper()
	dc := newTestDeploymentConfig(numInstances)
	c := capacity
	base, bw, fp := 1000.0, 2.0e6, 2.0e6
	specs := make([]sim.AdapterSpec, len(adapterIDs))
	for i, id := range adapterIDs {
		specs[i] = sim.AdapterSpec{ID: id, Rank: 8}
	}
	dc.LoRAConfig = sim.LoRAConfig{
		AdapterCapacity:       &c,
		LoadBaseLatencyUs:     &base,
		LoadBandwidthBytesUs:  &bw,
		FootprintBytesPerRank: &fp,
		Adapters:              specs,
	}
	dc.LoRAAdapterPlacement = placement
	reg, err := sim.BuildAdapterRegistry(dc.ToSimConfig())
	if err != nil {
		t.Fatalf("BuildAdapterRegistry: %v", err)
	}
	return dc, reg
}

// TestValidateLoRAPlacement covers the DD-B5-f ordered checks (0–4). Each error
// row asserts the message names the offending index / id / bound; the nil rows
// pin that valid and empty-list placements pass (C-6).
func TestValidateLoRAPlacement(t *testing.T) {
	tests := []struct {
		name    string
		setup   func(t *testing.T) (DeploymentConfig, sim.AdapterRegistry)
		wantErr []string // substrings the error must contain; empty ⇒ expect nil
	}{
		{
			name: "check0_placement_with_lora_disabled",
			setup: func(t *testing.T) (DeploymentConfig, sim.AdapterRegistry) {
				dc := newTestDeploymentConfig(2) // no LoRAConfig ⇒ subsystem off
				dc.LoRAAdapterPlacement = map[int][]string{0: {"A"}}
				return dc, nil
			},
			wantErr: []string{"LoRA disabled"},
		},
		{
			name: "check1_negative_index",
			setup: func(t *testing.T) (DeploymentConfig, sim.AdapterRegistry) {
				return loraPlacementConfig(t, 2, 4, map[int][]string{-1: {"A"}}, "A")
			},
			wantErr: []string{"-1", "[0,"},
		},
		{
			name: "check1_out_of_range_index",
			setup: func(t *testing.T) (DeploymentConfig, sim.AdapterRegistry) {
				return loraPlacementConfig(t, 2, 4, map[int][]string{5: {"A"}}, "A")
			},
			wantErr: []string{"5", "[0,"},
		},
		{
			name: "check2_empty_string_id",
			setup: func(t *testing.T) (DeploymentConfig, sim.AdapterRegistry) {
				return loraPlacementConfig(t, 2, 4, map[int][]string{0: {""}}, "A")
			},
			wantErr: []string{"0"},
		},
		{
			name: "check2_unregistered_id",
			setup: func(t *testing.T) (DeploymentConfig, sim.AdapterRegistry) {
				return loraPlacementConfig(t, 2, 4, map[int][]string{0: {"Z"}}, "A")
			},
			wantErr: []string{"Z", "0"},
		},
		{
			name: "check3_intra_index_duplicate",
			setup: func(t *testing.T) (DeploymentConfig, sim.AdapterRegistry) {
				return loraPlacementConfig(t, 2, 4, map[int][]string{0: {"A", "A"}}, "A")
			},
			wantErr: []string{"A", "0"},
		},
		{
			name: "check4_capacity_exceeded",
			setup: func(t *testing.T) (DeploymentConfig, sim.AdapterRegistry) {
				return loraPlacementConfig(t, 2, 1, map[int][]string{0: {"A", "B"}}, "A", "B")
			},
			wantErr: []string{"0", "2", "1"}, // index, requested count, capacity
		},
		{
			name: "valid_placement",
			setup: func(t *testing.T) (DeploymentConfig, sim.AdapterRegistry) {
				return loraPlacementConfig(t, 2, 4, map[int][]string{0: {"A"}, 1: {"B"}}, "A", "B")
			},
			wantErr: nil,
		},
		{
			name: "empty_list_value_is_noop",
			setup: func(t *testing.T) (DeploymentConfig, sim.AdapterRegistry) {
				return loraPlacementConfig(t, 2, 4, map[int][]string{0: {}}, "A")
			},
			wantErr: nil,
		},
		{
			name: "absent_placement_is_noop",
			setup: func(t *testing.T) (DeploymentConfig, sim.AdapterRegistry) {
				return loraPlacementConfig(t, 2, 4, nil, "A")
			},
			wantErr: nil,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dc, reg := tc.setup(t)
			err := ValidateLoRAPlacement(dc, reg)
			if len(tc.wantErr) == 0 {
				if err != nil {
					t.Fatalf("ValidateLoRAPlacement = %v, want nil", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("ValidateLoRAPlacement = nil, want error containing %v", tc.wantErr)
			}
			for _, sub := range tc.wantErr {
				if !strings.Contains(err.Error(), sub) {
					t.Errorf("error %q does not contain %q", err.Error(), sub)
				}
			}
		})
	}
}

// TestValidateLoRAPlacement_DeterministicFirstError pins INV-6: with violations at
// both index -1 (range) and index 0 (duplicate), sorted-ascending iteration reports
// the index -1 range error first, independent of Go map iteration order.
func TestValidateLoRAPlacement_DeterministicFirstError(t *testing.T) {
	dc, reg := loraPlacementConfig(t, 2, 4, map[int][]string{-1: {"A"}, 0: {"A", "A"}}, "A")
	err := ValidateLoRAPlacement(dc, reg)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "-1") {
		t.Errorf("first error %q is not the index -1 range violation (non-deterministic ordering?)", err.Error())
	}
}

// TestNewClusterSimulator_InvalidPlacementPanics pins C-6 / Principle V: an invalid
// placement fails at construction via panic (library layer), mirroring the existing
// ValidatePoolTopology panic surface.
func TestNewClusterSimulator_InvalidPlacementPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic from NewClusterSimulator with out-of-range placement index")
		}
	}()
	dc, _ := loraPlacementConfig(t, 1, 4, map[int][]string{3: {"A"}}, "A") // index 3 >= NumInstances 1
	_ = NewClusterSimulator(dc, NewSliceRequestSource(newTestRequests(1)), nil)
}

// TestDeploymentConfig_PlacementStrictParse pins DD-B5-e / R10: lora_adapter_placement
// parses under the strict decoder, and an unknown sibling key is rejected.
func TestDeploymentConfig_PlacementStrictParse(t *testing.T) {
	const good = "lora_adapter_placement:\n  0: [A, B]\n  1: [C]\n"
	var dc DeploymentConfig
	dec := yaml.NewDecoder(bytes.NewReader([]byte(good)))
	dec.KnownFields(true)
	if err := dec.Decode(&dc); err != nil {
		t.Fatalf("strict decode of valid placement failed: %v", err)
	}
	if got := dc.LoRAAdapterPlacement[0]; len(got) != 2 || got[0] != "A" || got[1] != "B" {
		t.Errorf("placement[0] = %v, want [A B]", got)
	}
	if got := dc.LoRAAdapterPlacement[1]; len(got) != 1 || got[0] != "C" {
		t.Errorf("placement[1] = %v, want [C]", got)
	}

	const bad = "lora_adapter_placement_typo:\n  0: [A]\n"
	dec2 := yaml.NewDecoder(bytes.NewReader([]byte(bad)))
	dec2.KnownFields(true)
	var dc2 DeploymentConfig
	if err := dec2.Decode(&dc2); err == nil {
		t.Error("strict decode accepted an unknown key; R10 strict parsing not in effect")
	}
}
