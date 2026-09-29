package cluster

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// Config-time rank coupling: per-instance max_lora_rank and capacity. LoRA construction
// funcs are registered via the blank import in lora_import_test.go.

const (
	instCfgFootprint = 2.0e6 // footprint_bytes_per_rank in every fixture below
	instCfgGPUMemGiB = 80.0
	instCfgMemUtil   = 0.9
	instCfgBlockSize = 16
)

// instanceConfigFixture builds a valid two-instance deployment with per-instance slots:
// adapters a8, b16, c64 (ranks 8, 16, 64), cluster-wide capacity 3, pre-placement, and
// KV auto-calc enabled on a model CalculateKVBlocks accepts. ranks/caps/placement are
// the per-test variables; pass nil lists for the cluster-wide configuration.
func instanceConfigFixture(t *testing.T, ranks, caps []int, placement map[int][]string) DeploymentConfig {
	t.Helper()
	dc := newTestDeploymentConfig(2)
	hw := testRooflineHWCalib()
	hw.MemoryGiB = instCfgGPUMemGiB
	dc.ModelHardwareConfig = sim.NewModelHardwareConfig(kvAutoCalcTestModel(), hw,
		"test-model", "H100", 1, 1, false, "", "roofline", 0)
	dc.KVCacheConfig = sim.NewKVCacheConfig(10000, instCfgBlockSize, 0, 0, 0, 0)
	capacity := 3
	base, bw, fp := 1000.0, 2.0e6, instCfgFootprint
	dc.LoRAConfig = sim.LoRAConfig{
		AdapterCapacity:       &capacity,
		LoadBaseLatencyUs:     &base,
		LoadBandwidthBytesUs:  &bw,
		FootprintBytesPerRank: &fp,
		Adapters: []sim.AdapterSpec{
			{ID: "a8", Rank: 8}, {ID: "b16", Rank: 16}, {ID: "c64", Rank: 64},
		},
		CreationPolicy: "pre-placement",
	}
	dc.RoutingPolicy = "route-to-holder"
	dc.LoRAAdapterPlacement = placement
	dc.LoRAInstanceMaxRank = ranks
	dc.LoRAInstanceCapacity = caps
	dc.KVAutoCalc = KVAutoCalcConfig{
		Enabled:              true,
		GPUMemoryUtilization: instCfgMemUtil,
		Params:               kvAutoCalcTestParams(),
	}
	return dc
}

func requirePanicContaining(t *testing.T, want string, f func()) {
	t.Helper()
	defer func() {
		t.Helper()
		r := recover()
		if r == nil {
			t.Fatalf("expected a panic containing %q, got none", want)
		}
		if msg := r.(string); !strings.Contains(msg, want) {
			t.Fatalf("panic %q does not contain %q", msg, want)
		}
	}()
	f()
}

var instCfgPlacement = map[int][]string{0: {"a8", "b16"}, 1: {"c64"}}

// Each guard in ValidateLoRAInstanceConfig, one mutation of a valid fixture per case. The
// premise — that the unmutated fixture passes — is asserted first, so a case can only
// fail on its own mutation.
func TestValidateLoRAInstanceConfig_Guards(t *testing.T) {
	valid := func() DeploymentConfig {
		return instanceConfigFixture(t, []int{16, 64}, []int{2, 1}, instCfgPlacement)
	}
	if err := ValidateLoRADeployment(valid()); err != nil {
		t.Fatalf("premise: valid fixture rejected: %v", err)
	}
	cases := []struct {
		name   string
		mutate func(*DeploymentConfig)
		want   string
	}{
		{"ranks without capacities", func(d *DeploymentConfig) { d.LoRAInstanceCapacity = nil }, "must be set together"},
		{"capacities without ranks", func(d *DeploymentConfig) { d.LoRAInstanceMaxRank = nil }, "must be set together"},
		{"LoRA disabled", func(d *DeploymentConfig) {
			d.Adapters = nil
			d.LoRAAdapterPlacement = nil
		}, "LoRA disabled"},
		{"rank list too short", func(d *DeploymentConfig) { d.LoRAInstanceMaxRank = []int{64} }, "lora_instance_max_rank has 1 entries"},
		{"capacity list too long", func(d *DeploymentConfig) { d.LoRAInstanceCapacity = []int{2, 1, 1} }, "lora_instance_capacity has 3 entries"},
		{"node pools", func(d *DeploymentConfig) { d.NodePools = []NodePoolConfig{{Name: "p"}} }, "not supported with node_pools"},
		{"PD disaggregation", func(d *DeploymentConfig) { d.PrefillInstances = 1 }, "not supported with PD disaggregation"},
		{"explicit KV blocks", func(d *DeploymentConfig) { d.KVAutoCalc.Enabled = false }, "requires auto-calculated KV capacity"},
		{"rank outside vLLM's set", func(d *DeploymentConfig) { d.LoRAInstanceMaxRank = []int{12, 64} }, "lora_instance_max_rank[0] = 12"},
		{"zero capacity", func(d *DeploymentConfig) { d.LoRAInstanceCapacity = []int{2, 0} }, "lora_instance_capacity[1] = 0"},
		{"seeded rank above cap", func(d *DeploymentConfig) { d.LoRAInstanceMaxRank = []int{8, 64} }, `adapter "b16" of rank 16, above its max_lora_rank 8`},
		{"seeded count above capacity", func(d *DeploymentConfig) { d.LoRAInstanceCapacity = []int{1, 1} }, "instance 0 assigned 2 adapters, exceeds capacity 1"},
		{"scheduled rank above cap", func(d *DeploymentConfig) {
			d.PlacementSchedule = []sim.PlacementScheduleEntry{{AtUs: 10, Placement: map[int][]string{0: {"c64"}}}}
		}, `adapter "c64" of rank 64, above its max_lora_rank 16`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := valid()
			tc.mutate(&d)
			err := ValidateLoRADeployment(d)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got error %v, want one containing %q", err, tc.want)
			}
		})
	}
}

// The per-instance capacity, not the cluster-wide one, bounds a placement: the same
// placement passes without the lists (cluster-wide capacity 3) and fails with them.
func TestValidateLoRAPlacement_UsesPerInstanceCapacity(t *testing.T) {
	placement := map[int][]string{0: {"a8", "b16"}}
	if err := ValidateLoRADeployment(instanceConfigFixture(t, nil, nil, placement)); err != nil {
		t.Fatalf("premise: placement within the cluster-wide capacity rejected: %v", err)
	}
	err := ValidateLoRADeployment(instanceConfigFixture(t, []int{16, 64}, []int{1, 3}, placement))
	if err == nil || !strings.Contains(err.Error(), "exceeds capacity 1") {
		t.Fatalf("got %v, want the per-instance capacity 1 to reject 2 seeded adapters", err)
	}
}

// Absent lists are valid whatever else is configured, including with LoRA off (R20).
func TestValidateLoRAInstanceConfig_AbsentIsNil(t *testing.T) {
	dc := newTestDeploymentConfig(2)
	if err := ValidateLoRAInstanceConfig(dc, nil); err != nil {
		t.Fatalf("absent lists with LoRA off: got %v, want nil", err)
	}
}

// NewClusterSimulator enforces the same checks, by panic (library layer).
func TestNewClusterSimulator_PanicsOnInvalidInstanceConfig(t *testing.T) {
	dc := instanceConfigFixture(t, []int{8, 64}, []int{2, 1}, instCfgPlacement)
	requirePanicContaining(t, "above its max_lora_rank 8", func() {
		NewClusterSimulator(dc, NewSliceRequestSource(nil), nil)
	})
}
