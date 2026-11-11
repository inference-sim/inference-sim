package engine

import (
	"errors"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

// bare is an engine with no placement support; placed records where it was put.
type bare struct{ Engine }

type placed struct {
	Engine
	at  plan.Endpoint
	err error
}

func (p *placed) PlaceAt(at plan.Endpoint) error { p.at = at; return p.err }

// Place decides when the absence of the capability is a refusal: a pool of several nodes, or an
// instance away from the first node and rack, must be placeable; a one-node pool at the origin
// needs nothing.
func TestPlaceRefusesAnEngineThatCannotBeTold(t *testing.T) {
	for name, c := range map[string]struct {
		at    plan.Endpoint
		nodes int
		want  error
	}{
		"several nodes, origin":         {plan.Endpoint{}, 2, ErrPlacementUnsupported},
		"several nodes, elsewhere":      {plan.Endpoint{Node: 1}, 4, ErrPlacementUnsupported},
		"one node, another rack":        {plan.Endpoint{Rack: 1}, 1, ErrPlacementUnsupported},
		"a node past the pool":          {plan.Endpoint{Node: 4}, 4, ErrPlacementRejected},
		"a negative rack":               {plan.Endpoint{Rack: -1}, 4, ErrPlacementRejected},
		"one node at the origin (fine)": {plan.Endpoint{}, 1, nil},
	} {
		err := Place(bare{}, c.at, c.nodes)
		if !errors.Is(err, c.want) || (c.want == nil) != (err == nil) {
			t.Errorf("%s: Place = %v, want %v", name, err, c.want)
		}
		var pe *PlacementError
		if c.want != nil && (!errors.As(err, &pe) || pe.At != c.at || pe.PoolNodes != c.nodes) {
			t.Errorf("%s: the refusal does not carry the placement: %v", name, err)
		}
	}
}

func TestPlaceTellsAPlaceableEngineAndPassesItsRejection(t *testing.T) {
	p := &placed{}
	if err := Place(p, plan.Endpoint{Node: 2, Rack: 1}, 4); err != nil || p.at != (plan.Endpoint{Node: 2, Rack: 1}) {
		t.Fatalf("Place = %v, engine placed at %+v", err, p.at)
	}
	p.err = errors.New("no such rank")
	if err := Place(p, plan.Endpoint{Node: 1}, 4); !errors.Is(err, ErrPlacementRejected) {
		t.Errorf("a rejected placement: %v", err)
	}
}
