// Original to blis-vllm.

package kernelmodel

import (
	"path/filepath"
	"reflect"
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

// Each replica of a data-parallel pool sits on the node its ranks fill: the Muse GB200 pool of eight single-GPU
// ranks on two nodes of four has ranks 0 to 3 on node 0 and ranks 4 to 7 on node 1, and everything else of the
// placement is the pool's.
func TestEnginePlacementOfAReplicaIsItsOwnNode(t *testing.T) {
	r := repos()
	r.Scenarios, _ = filepath.Abs(filepath.Join("..", "..", "vllmengine", "testdata"))
	m, err := OpenPool("muse-gb200-nvl-dep8-dspark.yaml", r, 0)
	if err != nil {
		t.Fatal(err)
	}
	base := m.EnginePlacement()
	for rank, node := range []int{0, 0, 0, 0, 1, 1, 1, 1} {
		p := m.EnginePlacementOf(rank)
		if p.Node != node {
			t.Errorf("rank %d is on node %d, want %d", rank, p.Node, node)
		}
		p.Node = base.Node
		if !reflect.DeepEqual(p, base) {
			t.Errorf("rank %d: the placement differs from rank 0's in more than the node:\n%+v\n%+v", rank, p, base)
		}
	}
	if err := m.EnginePlacementOf(5).Validate(); err != nil {
		t.Error(err)
	}
}
