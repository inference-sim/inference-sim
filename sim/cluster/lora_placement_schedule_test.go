package cluster

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// scheduleFixture builds a LoRA-enabled DeploymentConfig with NumInstances=2,
// AdapterCapacity=2, four registered adapters (adapter_0..adapter_3), and the given
// PlacementSchedule, plus a matching AdapterRegistry. Mirrors loraPlacementConfig in
// deployment_test.go, adapted for the schedule field instead of LoRAAdapterPlacement.
func scheduleFixture(t *testing.T, entries []sim.PlacementScheduleEntry) (DeploymentConfig, sim.AdapterRegistry) {
	t.Helper()
	dc := newTestDeploymentConfig(2)
	capacity := 2
	base, bw, fp := 1000.0, 2.0e6, 2.0e6
	specs := []sim.AdapterSpec{
		{ID: "adapter_0", Rank: 8},
		{ID: "adapter_1", Rank: 8},
		{ID: "adapter_2", Rank: 8},
		{ID: "adapter_3", Rank: 8},
	}
	dc.LoRAConfig = sim.LoRAConfig{
		AdapterCapacity:       &capacity,
		LoadBaseLatencyUs:     &base,
		LoadBandwidthBytesUs:  &bw,
		FootprintBytesPerRank: &fp,
		Adapters:              specs,
	}
	dc.PlacementSchedule = entries
	reg, err := sim.BuildAdapterRegistry(dc.ToSimConfig())
	if err != nil {
		t.Fatalf("BuildAdapterRegistry: %v", err)
	}
	return dc, reg
}

// Each case is a structural defect that must be caught before any instance is built. The
// capacity case is the one that matters beyond hygiene: it is BLIS's own enforcement of the
// per-instance adapter bound the SOLVER does not enforce (lora-control issue #60).
func TestValidateLoRAPlacementScheduleRejectsStructuralDefects(t *testing.T) {
	for _, tc := range []struct {
		name    string
		entries []sim.PlacementScheduleEntry
		want    string
	}{
		{"index out of range",
			[]sim.PlacementScheduleEntry{{AtUs: 0, Placement: map[int][]string{99: {"adapter_0"}}}},
			"out of range"},
		{"unregistered adapter",
			[]sim.PlacementScheduleEntry{{AtUs: 0, Placement: map[int][]string{0: {"nope"}}}},
			"unregistered adapter"},
		{"duplicate within an instance",
			[]sim.PlacementScheduleEntry{{AtUs: 0, Placement: map[int][]string{0: {"adapter_0", "adapter_0"}}}},
			"duplicate adapter"},
		{"over capacity",
			[]sim.PlacementScheduleEntry{{AtUs: 0, Placement: map[int][]string{
				0: {"adapter_0", "adapter_1", "adapter_2"}}}},
			"capacity"},
		{"a LATER entry is over capacity",
			[]sim.PlacementScheduleEntry{
				{AtUs: 0, Placement: map[int][]string{0: {"adapter_0"}}},
				{AtUs: 10, Placement: map[int][]string{0: {"adapter_0", "adapter_1", "adapter_2"}}}},
			"capacity"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dc, registry := scheduleFixture(t, tc.entries)
			err := ValidateLoRAPlacementSchedule(dc, registry)
			if err == nil {
				t.Fatalf("want an error mentioning %q, got nil", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not mention %q", err.Error(), tc.want)
			}
			if !strings.Contains(err.Error(), "lora_placement_schedule") {
				t.Errorf("error %q does not name the schedule, so it reads as a "+
					"--lora-adapter-placement failure", err.Error())
			}
		})
	}
}

func TestValidateLoRAPlacementScheduleAcceptsAValidScheduleAndAnEmptyOne(t *testing.T) {
	dc, registry := scheduleFixture(t, []sim.PlacementScheduleEntry{
		{AtUs: 0, Placement: map[int][]string{0: {"adapter_0", "adapter_1"}, 1: {"adapter_2"}}},
		{AtUs: 10, Placement: map[int][]string{1: {"adapter_0"}}},
	})
	if err := ValidateLoRAPlacementSchedule(dc, registry); err != nil {
		t.Errorf("valid schedule rejected: %v", err)
	}
	empty, emptyRegistry := scheduleFixture(t, nil)
	if err := ValidateLoRAPlacementSchedule(empty, emptyRegistry); err != nil {
		t.Errorf("an absent schedule must be a no-op, got %v", err)
	}
}
