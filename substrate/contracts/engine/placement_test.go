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
		"several nodes, origin":                         {plan.Endpoint{}, 2, ErrPlacementUnsupported},
		"several nodes, elsewhere":                      {plan.Endpoint{Node: 1}, 4, ErrPlacementUnsupported},
		"one node, another rack":                        {plan.Endpoint{Rack: 1}, 1, ErrPlacementUnsupported},
		"a node past the pool":                          {plan.Endpoint{Node: 4}, 4, ErrPlacementRejected},
		"a negative rack":                               {plan.Endpoint{Rack: -1}, 4, ErrPlacementRejected},
		"one node at the origin (fine)":                 {plan.Endpoint{}, 1, nil},
		"no node count means one: node 1 is outside it": {plan.Endpoint{Node: 1}, 0, ErrPlacementRejected},
		"no node count: the origin needs nothing":       {plan.Endpoint{}, 0, nil},
		"no node count, another rack":                   {plan.Endpoint{Rack: 1}, 0, ErrPlacementUnsupported},
	} {
		err := Place(bare{}, c.at, c.nodes)
		if !errors.Is(err, c.want) || (c.want == nil) != (err == nil) {
			t.Errorf("%s: Place = %v, want %v", name, err, c.want)
		}
		var pe *PlacementError
		if c.want != nil && (!errors.As(err, &pe) || pe.At != c.at || pe.PoolNodes != max(1, c.nodes)) {
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

// A plan that states no node count is valid and means one node: a Placeable engine is not told
// node 1 of it (the engine's own rejection must not be what keeps it out), and an engine that is
// not Placeable is refused as rejected, not as unsupported, for the endpoint outside the pool.
func TestAZeroNodeCountIsOneNode(t *testing.T) {
	p := &placed{}
	err := Place(p, plan.Endpoint{Node: 1}, 0)
	if !errors.Is(err, ErrPlacementRejected) || p.at != (plan.Endpoint{}) {
		t.Fatalf("a Placeable engine at node 1 of a pool with no node count: err %v, engine told %+v", err, p.at)
	}
	err = Place(bare{}, plan.Endpoint{Node: 1}, 0)
	if !errors.Is(err, ErrPlacementRejected) || errors.Is(err, ErrPlacementUnsupported) {
		t.Fatalf("an engine without placement at node 1 of such a pool: %v, want ErrPlacementRejected", err)
	}
	var pe *PlacementError
	if !errors.As(err, &pe) || pe.PoolNodes != 1 {
		t.Errorf("the refusal states the effective node count: %+v", pe)
	}
	if err := Place(bare{}, plan.Endpoint{}, -1); !errors.Is(err, ErrPlacementRejected) {
		t.Errorf("a negative node count: %v", err)
	}
}

// Two pools may each have a local rack 0 that is a different physical rack. The origin turns a
// pool-relative endpoint into the cluster's and back; an offset of the node alone puts both
// pools' local rack 0 on one rack.
func TestPoolsWithTheSameLocalRackAreOnDifferentPhysicalRacks(t *testing.T) {
	prefill := PoolOrigin{Node: 0, Rack: 0}
	decode := PoolOrigin{Node: 8, Rack: 3}
	local := plan.Endpoint{Node: 1, Rack: 0}
	gp, gd := prefill.Global(local), decode.Global(local)
	if gp == gd || gp.Rack == gd.Rack {
		t.Fatalf("the same local endpoint %+v is %+v in one pool and %+v in the other: want different nodes and racks", local, gp, gd)
	}
	if gd != (plan.Endpoint{Node: 9, Rack: 3}) {
		t.Errorf("decode pool global = %+v, want node 9 rack 3", gd)
	}
	for _, o := range []PoolOrigin{prefill, decode} {
		back, err := o.Local(o.Global(local))
		if err != nil || back != local {
			t.Errorf("origin %+v: Local(Global(%+v)) = %+v, %v", o, local, back, err)
		}
	}
	if _, err := decode.Local(plan.Endpoint{Node: 8, Rack: 2}); err == nil {
		t.Error("a rack below the pool's origin was accepted as local")
	}
}
