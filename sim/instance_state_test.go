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
		if !IsValidInstanceState(string(s)) {
			t.Errorf("IsValidInstanceState(%q) = false, want true", s)
		}
	}
	if IsValidInstanceState("bogus") {
		t.Error("IsValidInstanceState(bogus) = true, want false")
	}
}

func TestInstanceState_StringValues(t *testing.T) {
	cases := []struct {
		constant InstanceState
		want     string
	}{
		{InstanceStateScheduling, "Scheduling"},
		{InstanceStateLoading, "Loading"},
		{InstanceStateWarmingUp, "WarmingUp"},
		{InstanceStateActive, "Active"},
		{InstanceStateDraining, "Draining"},
		{InstanceStateTerminated, "Terminated"},
	}
	for _, tc := range cases {
		if got := string(tc.constant); got != tc.want {
			t.Errorf("InstanceState constant: got %q, want %q", got, tc.want)
		}
	}
}
