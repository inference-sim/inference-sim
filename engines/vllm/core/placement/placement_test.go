// Original to blis-vllm.

package placement

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/units"
)

func TestValidateRefusesWhatThePlanOrThePoolRefuses(t *testing.T) {
	ok := Placement{Hardware: "h100", Fabric: "ib", Nodes: 2, RanksPerNode: 8, Tensor: 8, Node: 1, NodeRacks: []int{0, 1}}
	if err := ok.Validate(); err != nil {
		t.Fatal(err)
	}
	for name, p := range map[string]Placement{
		"no chip":                 {Nodes: 1},
		"a node the pool lacks":   {Hardware: "h100", Nodes: 2, Node: 2},
		"racks for another size":  {Hardware: "h100", Nodes: 2, NodeRacks: []int{0}},
		"a negative rack":         {Hardware: "h100", Nodes: 1, NodeRacks: []int{-1}},
		"two nodes, no network":   {Hardware: "h100", Nodes: 2},
		"fewer places than ranks": {Hardware: "h100", Tensor: 16, Nodes: 1, RanksPerNode: 8},
	} {
		if p.Validate() == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestForRankFillsNodesInOrder(t *testing.T) {
	p := Placement{Hardware: "h100", Fabric: "ib", Nodes: 4, RanksPerNode: 8, NodeRacks: []int{0, 0, 1, 1}}
	for rank, want := range map[int]plan.Endpoint{0: {Node: 0, Rack: 0}, 1: {Node: 1, Rack: 0}, 2: {Node: 2, Rack: 1}, 3: {Node: 3, Rack: 1}, 9: {Node: 3, Rack: 1}} {
		if got := p.ForRank(rank, 8).Endpoint(); got != want {
			t.Errorf("rank %d: %+v, want %+v", rank, got, want)
		}
	}
	if got := p.ForRank(1, 4).Endpoint(); got.Node != 0 {
		t.Errorf("a replica of 4 workers, rank 1: node %d, want 0 (workers 4 to 7 share node 0)", got.Node)
	}
}

func TestPlanSharesNoMemoryWithThePlacement(t *testing.T) {
	p := Placement{Hardware: "h100", Nodes: 1}
	tiers := []plan.KVTier{{Device: CPUDRAM, Bytes: units.Rank(units.Bytes(10))}}
	got := p.Plan(tiers)
	tiers[0].Device = "mutated"
	if got.Tiers[0].Device != CPUDRAM {
		t.Error("the plan's tiers alias the caller's slice")
	}
}

func TestAtStatesTheEnginesNodeAndItsRack(t *testing.T) {
	p := Placement{Hardware: "h100", Fabric: "ib", Nodes: 3, RanksPerNode: 8, NodeRacks: []int{0, 0, 1}}
	q := p.At(1, 5)
	if got := q.Endpoint(); got != (plan.Endpoint{Node: 1, Rack: 5}) {
		t.Errorf("endpoint %+v, want node 1 rack 5", got)
	}
	if p.NodeRacks[1] != 0 || p.Node != 0 {
		t.Error("At changed the placement it was called on")
	}
	if err := q.Validate(); err != nil {
		t.Error(err)
	}
	if err := p.At(3, 0).Validate(); err == nil {
		t.Error("a node the pool lacks was accepted")
	}
	if got := (Placement{Hardware: "h100", Nodes: 1}).At(0, 4).Endpoint(); got != (plan.Endpoint{Node: 0, Rack: 4}) {
		t.Errorf("a one-node pool, endpoint %+v", got)
	}
}
