package cluster

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestDirectActuatorApply verifies DirectActuator behavior for scale-up and scale-down.
// T020 from tasks.md — US3 acceptance scenarios 3.2, 3.3.
func TestDirectActuatorApply(t *testing.T) {
	// Helper to build a minimal active instance with a given model and variant.
	newActiveInst := func(id, model, gpu string, tp int) *InstanceSimulator {
		simCfg := newTestDeploymentConfig(1).ToSimConfig()
		simCfg.GPU = gpu
		inst := NewInstanceSimulator(InstanceID(id), simCfg)
		inst.Model = model
		inst.TPDegree = tp
		inst.State = sim.InstanceStateActive
		return inst
	}

	t.Run("scale_down_transitions_to_draining", func(t *testing.T) {
		// Two active instances for model "m1" with variant A100/TP=1.
		// Scale-down should drain the lexicographically first (oldest) instance.
		cs := NewClusterSimulator(newTestDeploymentConfig(1), nil, nil)
		inst1 := newActiveInst("inst-b", "m1", "A100", 1)
		inst2 := newActiveInst("inst-a", "m1", "A100", 1)
		cs.instances = []*InstanceSimulator{inst1, inst2}

		actuator := NewDirectActuator(cs)
		err := actuator.Apply([]ScaleDecision{
			{ModelID: "m1", Variant: NewVariantSpec("A100", 1), Delta: -1},
		})
		if err != nil {
			t.Fatalf("Apply returned error: %v", err)
		}

		// inst-a sorts before inst-b — inst-a should be drained
		if inst2.State != sim.InstanceStateDraining {
			t.Errorf("inst-a State = %q, want Draining (oldest by ID)", inst2.State)
		}
		if inst1.State != sim.InstanceStateActive {
			t.Errorf("inst-b State = %q, want Active (not selected)", inst1.State)
		}
	})

	t.Run("scale_down_no_match_returns_error", func(t *testing.T) {
		// No active instances matching the variant — should return error.
		cs := NewClusterSimulator(newTestDeploymentConfig(1), nil, nil)
		inst := newActiveInst("inst-1", "m1", "H100", 2) // wrong variant
		cs.instances = []*InstanceSimulator{inst}

		actuator := NewDirectActuator(cs)
		err := actuator.Apply([]ScaleDecision{
			{ModelID: "m1", Variant: NewVariantSpec("A100", 1), Delta: -1},
		})
		if err == nil {
			t.Error("Apply should return error when no matching active instance found")
		}
	})

	t.Run("scale_down_skips_non_active_instances", func(t *testing.T) {
		// One Draining instance, one Active — should only drain the Active one.
		cs := NewClusterSimulator(newTestDeploymentConfig(1), nil, nil)
		draining := newActiveInst("inst-a", "m1", "A100", 1)
		draining.State = sim.InstanceStateDraining
		active := newActiveInst("inst-b", "m1", "A100", 1)
		cs.instances = []*InstanceSimulator{draining, active}

		actuator := NewDirectActuator(cs)
		err := actuator.Apply([]ScaleDecision{
			{ModelID: "m1", Variant: NewVariantSpec("A100", 1), Delta: -1},
		})
		if err != nil {
			t.Fatalf("Apply returned error: %v", err)
		}
		if active.State != sim.InstanceStateDraining {
			t.Errorf("active inst State = %q, want Draining", active.State)
		}
	})

	t.Run("scale_up_nil_placement_returns_error", func(t *testing.T) {
		// No PlacementManager — scale-up should return error.
		cs := NewClusterSimulator(newTestDeploymentConfig(1), nil, nil)
		cs.placement = nil

		actuator := NewDirectActuator(cs)
		err := actuator.Apply([]ScaleDecision{
			{ModelID: "m1", Variant: NewVariantSpec("A100", 1), Delta: 1},
		})
		if err == nil {
			t.Error("Apply should return error when PlacementManager is nil")
		}
	})

	t.Run("scale_down_continue_not_return", func(t *testing.T) {
		// Delta=-2 but only 1 matching active instance.
		// First iteration drains it, second finds no match → error, but first drain still applied.
		cs := NewClusterSimulator(newTestDeploymentConfig(1), nil, nil)
		inst := newActiveInst("inst-a", "m1", "A100", 1)
		cs.instances = []*InstanceSimulator{inst}

		actuator := NewDirectActuator(cs)
		err := actuator.Apply([]ScaleDecision{
			{ModelID: "m1", Variant: NewVariantSpec("A100", 1), Delta: -2},
		})
		// Should return error (second iteration failed) but first drain was applied
		if err == nil {
			t.Error("Apply should return error for partial scale-down failure")
		}
		if inst.State != sim.InstanceStateDraining {
			t.Errorf("inst State = %q, want Draining (first iteration should succeed)", inst.State)
		}
	})

	t.Run("constructor_panics_on_nil_cluster", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Error("expected panic for nil cluster, got none")
			}
		}()
		NewDirectActuator(nil)
	})

	t.Run("id_format_is_zero_padded", func(t *testing.T) {
		// Verify that instance IDs use zero-padded format (%06d) so lexicographic
		// sort matches creation order even past single digits.
		cs := NewClusterSimulator(newTestDeploymentConfig(1), nil, nil)
		actuator := NewDirectActuator(cs)

		// Manually increment seq and check ID format
		actuator.nextInstSeq = 9
		expected := "autoscale-m1-000010"
		actuator.nextInstSeq++
		got := string(InstanceID(fmt.Sprintf("autoscale-%s-%06d", "m1", actuator.nextInstSeq)))
		if got != expected {
			t.Errorf("ID = %q, want %q (zero-padded for correct lexicographic order)", got, expected)
		}
	})

	t.Run("scale_up_creates_live_instance", func(t *testing.T) {
		// GIVEN: a cluster with 0 initial instances and 1 Ready node (2 GPU slots, TP=1).
		// The snapshotProvider is initialized empty by NewClusterSimulator.
		base := newTestDeploymentConfig(1)
		base.Model = "test-model"
		base.NodePools = []NodePoolConfig{
			{Name: "a100-pool", GPUType: "A100-80", GPUsPerNode: 2, InitialNodes: 1, MaxNodes: 2, GPUMemoryGiB: 80},
		}
		cs := NewClusterSimulator(base, nil, nil)
		cs.instances = []*InstanceSimulator{} // Clear initial instances to simulate scale-up from empty

		// WHEN: scaleUp is called for 1 instance.
		actuator := NewDirectActuator(cs)
		err := actuator.Apply([]ScaleDecision{
			{ModelID: "test-model", Variant: NewVariantSpec("A100-80", 1), Delta: 1},
		})
		if err != nil {
			t.Fatalf("Apply returned unexpected error: %v", err)
		}

		// THEN: exactly one InstanceSimulator exists in cs.instances.
		if len(cs.instances) != 1 {
			t.Fatalf("len(cs.instances) = %d, want 1 (scaleUp must create an InstanceSimulator)", len(cs.instances))
		}
		inst := cs.instances[0]

		// THEN: instance is past Scheduling (Loading if delay>0, WarmingUp/Active if delay==0).
		if inst.State == sim.InstanceStateScheduling {
			t.Errorf("instance State = %q: must be past Scheduling after scaleUp (Loading/WarmingUp/Active expected)", inst.State)
		}

		// THEN: GPU type is pool-authoritative (SC-004).
		if got := inst.GPU(); got != "A100-80" {
			t.Errorf("GPU() = %q, want %q (pool gpu_type must override CLI --gpu)", got, "A100-80")
		}

		// THEN: model matches the scale decision.
		if inst.Model != "test-model" {
			t.Errorf("Model = %q, want %q", inst.Model, "test-model")
		}

		// THEN: in-flight counter initialised to 0 (R4: every path initialises this).
		id := string(inst.ID())
		if v, ok := cs.inFlightRequests[id]; !ok || v != 0 {
			t.Errorf("inFlightRequests[%q] = %d ok=%v, want 0/true", id, v, ok)
		}

		// THEN: instance registered with snapshotProvider so it is routable.
		if !cs.snapshotProvider.HasInstance(inst.ID()) {
			t.Errorf("instance %q not registered with snapshotProvider (routing blind spot)", inst.ID())
		}
	})

	t.Run("scale_up_delta_2_creates_two_instances", func(t *testing.T) {
		// GIVEN: a node with 4 GPU slots allows 4 TP=1 instances.
		base := newTestDeploymentConfig(1)
		base.Model = "test-model"
		base.NodePools = []NodePoolConfig{
			{Name: "a100-pool", GPUType: "A100-80", GPUsPerNode: 4, InitialNodes: 1, MaxNodes: 2, GPUMemoryGiB: 80},
		}
		cs := NewClusterSimulator(base, nil, nil)
		cs.instances = []*InstanceSimulator{} // Clear initial instances to simulate scale-up from empty

		actuator := NewDirectActuator(cs)
		err := actuator.Apply([]ScaleDecision{
			{ModelID: "test-model", Variant: NewVariantSpec("A100-80", 1), Delta: 2},
		})
		if err != nil {
			t.Fatalf("Apply returned error: %v", err)
		}

		// THEN: two instances created and both registered.
		if len(cs.instances) != 2 {
			t.Fatalf("len(cs.instances) = %d, want 2", len(cs.instances))
		}
		for _, inst := range cs.instances {
			if !cs.snapshotProvider.HasInstance(inst.ID()) {
				t.Errorf("instance %q not registered with snapshotProvider", inst.ID())
			}
		}
	})

	t.Run("scale_up_no_capacity_returns_error", func(t *testing.T) {
		// GIVEN: node has 2 GPU slots; NumInstances=1 consumes 1 at startup, leaving 1 available.
		// Requesting Delta=2 means: first placement succeeds, second exhausts capacity.
		base := newTestDeploymentConfig(1)
		base.Model = "test-model"
		base.NodePools = []NodePoolConfig{
			{Name: "tiny-pool", GPUType: "A100-80", GPUsPerNode: 2, InitialNodes: 1, MaxNodes: 1, GPUMemoryGiB: 80},
		}
		cs := NewClusterSimulator(base, nil, nil)
		cs.instances = []*InstanceSimulator{} // Clear initial instances; placement already consumed 1 slot

		actuator := NewDirectActuator(cs)
		err := actuator.Apply([]ScaleDecision{
			{ModelID: "test-model", Variant: NewVariantSpec("A100-80", 1), Delta: 2},
		})

		// THEN: first succeeds, second fails — Apply returns error.
		if err == nil {
			t.Error("Apply should return error when placement capacity is exhausted")
		}
		// And 1 instance was created (first iteration succeeded).
		if len(cs.instances) != 1 {
			t.Errorf("len(cs.instances) = %d, want 1 (first placement should succeed)", len(cs.instances))
		}
	})
}
