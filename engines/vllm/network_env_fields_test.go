// Original to blis-vllm.

package blisvllm

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
)

// AgentX recipes state three variables vLLM does not read: the CUDA runtime reads CUDA_LAUNCH_BLOCKING, Gloo
// reads GLOO_SOCKET_IFNAME and NCCL reads NCCL_SOCKET_IFNAME. The field list prices each as a stand-in so the point
// is scored and says so: the two interface names, stated as the corpus states them, are left unset and the
// stand-in is exact; CUDA_LAUNCH_BLOCKING=1 slows a real step and is named on the point as an execution mode
// substituted, never exact. A value outside what the evidence covers stays refused.
func TestTheNetworkInterfaceAndLaunchVariablesAreStoodIn(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	find := func(name string) *fields.Field {
		for i := range l.Fields {
			if l.Fields[i].Name == name && l.Fields[i].Source == fields.SourceEnv {
				return &l.Fields[i]
			}
		}
		t.Fatalf("%s is not in the field list, so a recipe that states it is refused as a field the release does not have", name)
		return nil
	}
	for _, c := range []struct {
		field, value string
		exact        bool
		approximated bool
	}{
		{"GLOO_SOCKET_IFNAME", "eth0", true, false},
		{"GLOO_SOCKET_IFNAME", "ib0", false, false},
		{"NCCL_SOCKET_IFNAME", "eth0", true, false},
		{"NCCL_SOCKET_IFNAME", "ib0", false, false},
		{"CUDA_LAUNCH_BLOCKING", "1", false, true},
		{"CUDA_LAUNCH_BLOCKING", "2", false, false},
	} {
		f := find(c.field)
		facts := fields.Facts{Value: c.value, ComputeCapability: "9.0"}
		if got := fields.Exact(f.Estimation, facts); got != c.exact {
			t.Errorf("%s=%s: exact %v, want %v", c.field, c.value, got, c.exact)
		}
		if _, got := fields.Approximate(f.Estimation, facts); got != c.approximated {
			t.Errorf("%s=%s: approximated %v, want %v", c.field, c.value, got, c.approximated)
		}
	}
}
