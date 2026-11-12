// Original to blis-vllm.

package vllm

import (
	"errors"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/conformance/enginetest"
	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

// pool4 is a decode pool of four nodes of eight ranks, the shape a host places one instance per node of.
const pool4 = "placement: {hardware: h100, fabric: ib-400g, nodes: 4, ranks_per_node: 8}\n"

func pullArrival() enginetest.Arrival {
	r := req("pulls", 80, 6)
	r.RemotePrefill, r.Peer = true, &plan.Endpoint{Node: 1, Rack: 1}
	return enginetest.Arrival{At: 0, Req: r}
}

// The engine passes the contract's placement conformance check: placed at the first node and at the last of a
// four-node pool, the pull of a request ends where its instance was put, and the two ends differ. Without
// PlaceAt both would end at the pool's first node (the check refuses such an engine).
func TestTheEngineIsPlaceableAndItsPullsEndWhereItWasPut(t *testing.T) {
	if err := enginetest.CheckPlacement(build(t, consumer+pool4), pullArrival(), 4); err != nil {
		t.Fatal(err)
	}
}

// The host's refusals reach the host through engine.Place: a place outside the pool is rejected, a pool
// that states no placement cannot be placed anywhere but at the first node, and an engine that has
// started a transfer cannot be moved.
func TestThePlacementRefusalsReachTheHost(t *testing.T) {
	e := mustEngine(t, consumer+pool4)
	if err := engine.Place(e, plan.Endpoint{Node: 4}, 4); !errors.Is(err, engine.ErrPlacementRejected) {
		t.Errorf("a node the pool lacks: %v, want ErrPlacementRejected", err)
	}
	if err := engine.Place(mustEngine(t, ""), plan.Endpoint{Node: 1}, 4); !errors.Is(err, engine.ErrPlacementRejected) {
		t.Errorf("an engine whose pool states no placement, placed at node 1: %v, want ErrPlacementRejected", err)
	}
	if err := engine.Place(mustEngine(t, ""), plan.Endpoint{}, 1); err != nil {
		t.Errorf("an engine with no placement, at the first node of a one-node pool: %v", err)
	}
	// After a pull has started the engine's end is part of what it stated.
	r := pullArrival().Req
	if err := e.Add(r, 0); err != nil {
		t.Fatal(err)
	}
	runOne(t, e, 500)
	if err := engine.Place(e, plan.Endpoint{Node: 2, Rack: 2}, 4); !errors.Is(err, engine.ErrPlacementRejected) {
		t.Errorf("an engine that has started a transfer: %v, want ErrPlacementRejected", err)
	}
}

// A data-parallel group is placed as the pool: its ranks sit where the layout puts them, so it starts at the
// first node and refuses another.
func TestADataParallelGroupIsPlacedAtTheFirstNode(t *testing.T) {
	g, err := build(t, "data_parallel_size: 2\nplacement: {hardware: h100, fabric: ib-400g, data: 2, nodes: 2, ranks_per_node: 1, node_racks: [0, 1]}\n")()
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := g.(engine.Placeable); !ok {
		t.Fatalf("%T is not engine.Placeable", g)
	}
	if err := engine.Place(g, plan.Endpoint{Node: 1}, 2); !errors.Is(err, engine.ErrPlacementRejected) {
		t.Errorf("a group placed at node 1: %v, want ErrPlacementRejected", err)
	}
	if err := engine.Place(g, plan.Endpoint{Node: 0, Rack: 3}, 2); err != nil {
		t.Errorf("a group placed at the first node: %v", err)
	}
}

// Placing a group at its first node changes the first node's rack only: the racks the pool gave its
// other nodes stay, so a later rank is still where the pool put it (the endpoint of a transfer and the
// link it crosses follow from it). Placed again with another rack for node 0, rank 1 keeps rack 1.
func TestPlacingAGroupKeepsTheRacksOfItsOtherNodes(t *testing.T) {
	g, err := build(t, "data_parallel_size: 2\nplacement: {hardware: h100, fabric: ib-400g, data: 2, nodes: 2, ranks_per_node: 1, node_racks: [0, 1]}\n")()
	if err != nil {
		t.Fatal(err)
	}
	grp, ok := g.(*DPGroup)
	if !ok {
		t.Fatalf("%T is not a DPGroup", g)
	}
	for _, rack := range []int{0, 3} {
		if err := engine.Place(g, plan.Endpoint{Node: 0, Rack: rack}, 2); err != nil {
			t.Fatal(err)
		}
		if got := grp.ranks[0].e.at; got != (plan.Endpoint{Node: 0, Rack: rack}) {
			t.Errorf("rank 0 at %+v, want node 0 in rack %d", got, rack)
		}
		if got := grp.ranks[1].e.at; got != (plan.Endpoint{Node: 1, Rack: 1}) {
			t.Errorf("after placing node 0 in rack %d, rank 1 at %+v, want node 1 in rack 1 as the pool stated", rack, got)
		}
	}
}
