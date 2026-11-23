// Original to blis-vllm.

package blisvllm

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"
)

// NCCL reads NCCL_MNNVL_ENABLE (src/init.cc:1074 at NVIDIA/nccl v2.32.6-1 ad2d192c; vLLM does not). At 2, the default,
// a communicator runs NCCL's multi-node NVLink check only when it spans more than one node (init.cc:1377:
// nNodes > 1 && enable != 0 && p2pLevel != 0); at 1 it runs the check for every communicator, a communicator inside one
// node included. So the unset stand-in is exact for 2 on any communicator and for 1 only where the tensor-parallel
// group spans nodes; a stated 1 on one node runs a path the sweep behind the collectives did not, and 0 turns the path off.
func TestNcclMnnvlEnableIsTheDefaultOnlyWhereNcclChoosesTheSamePath(t *testing.T) {
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	var est fields.Estimation
	found := false
	for _, f := range l.Fields {
		if f.Name == "NCCL_MNNVL_ENABLE" && f.Source == fields.SourceEnv {
			est, found = f.Estimation, true
		}
	}
	if !found {
		t.Fatal("NCCL_MNNVL_ENABLE is not in the field list")
	}
	one, two, four := 1, 2, 4
	for _, c := range []struct {
		value string
		nodes *int
		exact bool
	}{
		{"2", &one, true},
		{"2", &two, true},
		{"2", nil, true},
		{"1", &two, true},
		{"1", &four, true},
		{"1", &one, false},
		{"1", nil, false},
		{"0", &one, false},
		{"0", &two, false},
		{"", &two, false},
	} {
		got := fields.Exact(est, fields.Facts{Value: c.value, TPGroupNodes: c.nodes, ComputeCapability: "10.3"})
		if got != c.exact {
			n := -1
			if c.nodes != nil {
				n = *c.nodes
			}
			t.Errorf("NCCL_MNNVL_ENABLE=%q on %d node(s): exact %v, want %v", c.value, n, got, c.exact)
		}
	}
}
