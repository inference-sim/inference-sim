// Original to blis-vllm.

package kernelmodel

import (
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

// A pool's EnginePlacement is the chip and fabric its scenario names, the layout its deployment states and
// the nodes it fills, counted from the pool's own first node, with the rack of each node as the kernel
// places it: the decode pool of a 2 + 2 node deployment is on nodes 0 and 1 of its own pool, in the racks of
// the cluster's nodes 2 and 3, and its plan accepts a pull from the prefill pool's first node.
func TestEnginePlacementIsThePoolsOwnNodes(t *testing.T) {
	r := repos()
	r.Scenarios, _ = filepath.Abs(filepath.Join("..", "..", "vllmengine", "testdata"))
	for i, wantRacks := range [][]int{{0, 1}, {2, 3}} {
		m, err := OpenPool("muse-h100-roce-1p1d.yaml", r, i)
		if err != nil {
			t.Fatal(err)
		}
		p := m.EnginePlacement()
		if err := p.Validate(); err != nil {
			t.Fatalf("pool %d: %v", i, err)
		}
		if p.Hardware != "h100" || p.Fabric != "roce-200g" || p.Tensor != 16 || p.Pipeline != 1 || p.Nodes != 2 || p.RanksPerNode != 8 || p.Node != 0 {
			t.Errorf("pool %d: placement %+v, want h100 on roce-200g, tensor 16, two nodes of 8, engine on node 0", i, p)
		}
		for n, rack := range wantRacks {
			if got := p.EndpointOf(n); got != (plan.Endpoint{Node: n, Rack: rack}) {
				t.Errorf("pool %d: node %d is %+v, want rack %d", i, n, got, rack)
			}
		}
		move := plan.Transfer{Tokens: 100, From: plan.Endpoint{Node: 0, Rack: 0}, To: p.Endpoint(), Kind: plan.TransferPeer}
		if err := p.Plan(nil).CheckTransfer(move); err != nil {
			t.Errorf("pool %d: the plan refuses a pull: %v", i, err)
		}
	}
}
