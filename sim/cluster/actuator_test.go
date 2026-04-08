package cluster

import (
	"fmt"
	"testing"
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
		inst.State = InstanceStateActive
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
		if inst2.State != InstanceStateDraining {
			t.Errorf("inst-a State = %q, want Draining (oldest by ID)", inst2.State)
		}
		if inst1.State != InstanceStateActive {
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
		draining.State = InstanceStateDraining
		active := newActiveInst("inst-b", "m1", "A100", 1)
		cs.instances = []*InstanceSimulator{draining, active}

		actuator := NewDirectActuator(cs)
		err := actuator.Apply([]ScaleDecision{
			{ModelID: "m1", Variant: NewVariantSpec("A100", 1), Delta: -1},
		})
		if err != nil {
			t.Fatalf("Apply returned error: %v", err)
		}
		if active.State != InstanceStateDraining {
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
		if inst.State != InstanceStateDraining {
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
}
