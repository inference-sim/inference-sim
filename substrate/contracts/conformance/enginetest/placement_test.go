package enginetest

import (
	"errors"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

func pullArrival() Arrival {
	peer := plan.Endpoint{Node: 5, Rack: 2}
	r := req("pulls", 4, 1)
	r.RemotePrefill, r.Peer = true, &peer
	return at(0, r)
}

// An engine of a disaggregated pool with Plan, Move and Peer and no placement support is refused:
// left alone, every instance of the pool would report the pool's first node as its end.
func TestAnEngineWithoutPlacementSupportIsRefusedForAPoolOfSeveralNodes(t *testing.T) {
	build := func() (engine.Engine, error) { return NewPuller(fifoCfg(), 4) }
	err := CheckPlacement(build, pullArrival(), 4)
	if !errors.Is(err, engine.ErrPlacementUnsupported) {
		t.Fatalf("CheckPlacement = %v, want a refusal wrapping ErrPlacementUnsupported", err)
	}
	// The failure the refusal prevents, shown: without it both instances end at the first node.
	var ends []plan.Endpoint
	for range 2 {
		_, derr := Drive(build, []Arrival{pullArrival()}, Options{Boundary: func(_ plan.PoolPlan, x engine.Transfer) error {
			ends = append(ends, x.Move.To)
			return nil
		}})
		if derr != nil {
			t.Fatal(derr)
		}
	}
	if len(ends) != 2 || ends[0] != (plan.Endpoint{}) || ends[1] != (plan.Endpoint{}) {
		t.Errorf("an unplaced engine's pull ends at %v, want the first node for both", ends)
	}
}

// An engine with placement support reports the end it was placed at: instances on different nodes
// report different Move.To, and each pull ends where its instance was put.
func TestAPlacedEngineReportsTheEndItWasPlacedAt(t *testing.T) {
	build := func() (engine.Engine, error) { return NewPlacedPuller(fifoCfg(), 4) }
	if err := CheckPlacement(build, pullArrival(), 4); err != nil {
		t.Fatal(err)
	}
}

// An engine that has the capability and rejects a place of its pool is refused with the rejection.
func TestAPlacementTheEngineRejectsIsRefused(t *testing.T) {
	// The engine is built for a pool of two nodes and the host says the pool has four: node 3 is
	// outside what the engine models.
	build := func() (engine.Engine, error) { return NewPlacedPuller(fifoCfg(), 2) }
	if err := CheckPlacement(build, pullArrival(), 4); !errors.Is(err, engine.ErrPlacementRejected) {
		t.Fatalf("CheckPlacement = %v, want a refusal wrapping ErrPlacementRejected", err)
	}
}

func TestCheckPlacementNeedsAPoolOfSeveralNodes(t *testing.T) {
	build := func() (engine.Engine, error) { return NewPlacedPuller(fifoCfg(), 1) }
	if err := CheckPlacement(build, pullArrival(), 1); err == nil {
		t.Error("a pool of one node was accepted")
	}
}
