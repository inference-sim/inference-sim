package sim

import "testing"

func TestInstanceState_Constants(t *testing.T) {
	states := []InstanceState{
		InstanceStateScheduling,
		InstanceStateLoading,
		InstanceStateWarmingUp,
		InstanceStateActive,
		InstanceStateDraining,
		InstanceStateTerminated,
	}
	for _, s := range states {
		if s == "" {
			t.Errorf("InstanceState constant is empty")
		}
		if !IsValidInstanceState(string(s)) {
			t.Errorf("IsValidInstanceState(%q) = false, want true", s)
		}
	}
	if IsValidInstanceState("bogus") {
		t.Error("IsValidInstanceState(bogus) = true, want false")
	}
}
