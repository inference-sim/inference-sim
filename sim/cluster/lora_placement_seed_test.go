package cluster

import (
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// B-5 (#1493) cluster-level seeding tests for the CreationPolicy.Initial seam
// wired into NewClusterSimulator. Two distinct guarantees are covered:
//
//   - Wiring: each instance receives EXACTLY its cluster-assigned subset
//     (config.LoRAAdapterPlacement[idx]) via CreationContext.Assigned, keyed by
//     the live construction-loop counter (DD-B5-g). Verified by swapping in a
//     seeding stub policy (the shipped on-demand default deliberately ignores
//     Assigned; a consuming policy arrives in B-6).
//   - Inertness: with the shipped on-demand default, a present placement seeds
//     nothing — byte-identical to no placement (INV-L1).
//   - C-5: the seam is invoked only for construction / node-ready placement,
//     never for autoscaler scale-up, which shares addLiveInstance.
//
// LoRA construction funcs are registered via lora_import_test.go's blank import.

// seedAllStub is a CreationPolicy whose Initial seeds the full assigned set
// (unlike on-demand, which ignores it) and whose OnResidentMiss admits every
// load. Used to observe that the cluster forwards the correct per-instance
// subset through the seam.
type seedAllStub struct{}

func (seedAllStub) Initial(ctx sim.CreationContext) []string { return ctx.Assigned }
func (seedAllStub) OnResidentMiss(sim.CreationContext) bool  { return true }

// withSeedingCreationPolicy temporarily replaces the creation-policy factory so
// every instance built during the test uses seedAllStub. Restored on cleanup.
func withSeedingCreationPolicy(t *testing.T) {
	t.Helper()
	orig := sim.NewCreationPolicyFunc
	sim.NewCreationPolicyFunc = func(string) (sim.CreationPolicy, error) { return seedAllStub{}, nil }
	t.Cleanup(func() { sim.NewCreationPolicyFunc = orig })
}

// TestStartupPlacement_WiringSeedsPerIndex pins DD-B5-g: with a seeding policy,
// each instance is seeded with exactly its construction-index subset.
func TestStartupPlacement_WiringSeedsPerIndex(t *testing.T) {
	withSeedingCreationPolicy(t)
	dc, _ := loraPlacementConfig(t, 2, 4, map[int][]string{0: {"A"}, 1: {"B"}}, "A", "B")
	cs := NewClusterSimulator(dc, NewSliceRequestSource(nil), nil)

	want := map[InstanceID]string{"instance_0": "A", "instance_1": "B"}
	for id, wantID := range want {
		inst := cs.instanceByID(id)
		if inst == nil {
			t.Fatalf("instance %q not constructed", id)
		}
		got := inst.ResidentAdapterIDs()
		if len(got) != 1 || got[0] != wantID {
			t.Errorf("instance %q resident = %v, want [%s]", id, got, wantID)
		}
	}
}

// TestStartupPlacement_UnplacedInstanceEmpty pins that an instance with no
// placement entry receives an empty assigned set (nil map lookup) even when a
// sibling is seeded.
func TestStartupPlacement_UnplacedInstanceEmpty(t *testing.T) {
	withSeedingCreationPolicy(t)
	dc, _ := loraPlacementConfig(t, 2, 4, map[int][]string{0: {"A"}}, "A")
	cs := NewClusterSimulator(dc, NewSliceRequestSource(nil), nil)

	inst := cs.instanceByID("instance_1")
	if inst == nil {
		t.Fatal("instance_1 not constructed")
	}
	if got := inst.ResidentAdapterIDs(); len(got) != 0 {
		t.Errorf("unplaced instance_1 resident = %v, want empty", got)
	}
}

// TestStartupPlacement_OnDemandDefaultInert pins INV-L1: the shipped on-demand
// default seeds nothing even when a placement is configured — the seam is inert
// until a consuming policy (B-6) is selected.
func TestStartupPlacement_OnDemandDefaultInert(t *testing.T) {
	dc, _ := loraPlacementConfig(t, 2, 4, map[int][]string{0: {"A"}, 1: {"B"}}, "A", "B")
	cs := NewClusterSimulator(dc, NewSliceRequestSource(nil), nil)

	for _, id := range []InstanceID{"instance_0", "instance_1"} {
		inst := cs.instanceByID(id)
		if inst == nil {
			t.Fatalf("instance %q not constructed", id)
		}
		if got := inst.ResidentAdapterIDs(); len(got) != 0 {
			t.Errorf("on-demand default: instance %q resident = %v, want empty (INV-L1)", id, got)
		}
	}
}

// TestAutoscaleInstance_GetsEmptyResidentSet pins C-5: even under a seeding
// policy, the scale-up path never invokes the Initial seam, so a scaled-up
// instance comes up with an empty resident set — seeding lives outside
// addLiveInstance, which scaleUp shares.
func TestAutoscaleInstance_GetsEmptyResidentSet(t *testing.T) {
	withSeedingCreationPolicy(t)
	dc, _ := loraPlacementConfig(t, 1, 4, map[int][]string{0: {"A"}}, "A")
	dc.Model = "test-model"
	dc.NodePools = []NodePoolConfig{
		{Name: "h100-pool", GPUType: "H100", GPUsPerNode: 4, InitialNodes: 1, MaxNodes: 2, GPUMemoryGiB: 80},
	}
	cs := NewClusterSimulator(dc, NewSliceRequestSource(nil), nil)

	actuator := NewDirectActuator(cs)
	if err := actuator.Apply([]ScaleDecision{
		{ModelID: "test-model", Variant: NewVariantSpec("H100", 1), Delta: 1},
	}); err != nil {
		t.Fatalf("scale-up Apply: %v", err)
	}

	var scaled *InstanceSimulator
	for _, inst := range cs.instances {
		if strings.HasPrefix(string(inst.ID()), "autoscale-") {
			scaled = inst
			break
		}
	}
	if scaled == nil {
		t.Fatal("no autoscale-* instance created by scale-up")
	}
	if got := scaled.ResidentAdapterIDs(); len(got) != 0 {
		sort.Strings(got)
		t.Errorf("autoscaled instance %q resident = %v, want empty (placement must not leak, C-5)", scaled.ID(), got)
	}
}
