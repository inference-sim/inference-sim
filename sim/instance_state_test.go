package sim

import "testing"

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
