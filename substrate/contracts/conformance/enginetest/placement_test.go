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

func pulls() []Arrival { return []Arrival{pullArrival()} }

// twoPulls are two requests that pull from different peers.
func twoPulls() []Arrival {
	p1, p2 := plan.Endpoint{Node: 5, Rack: 2}, plan.Endpoint{Node: 6, Rack: 3}
	r1, r2 := req("pull-a", 4, 1), req("pull-b", 4, 1)
	r1.RemotePrefill, r1.Peer = true, &p1
	r2.RemotePrefill, r2.Peer = true, &p2
	return []Arrival{at(0, r1), at(1, r2)}
}

// counting counts the requests an engine is given.
type counting struct {
	engine.Engine
	adds *int
}

func (c counting) Add(r engine.Request, now engine.Time) error {
	*c.adds++
	return c.Engine.Add(r, now)
}

// rejecting is a placed engine that refuses every placement.
type rejecting struct{ PlacedPuller }

func (rejecting) PlaceAt(plan.Endpoint) error { return errors.New("no such rank") }

// countingPlacing is counting that passes a placement on to the engine it wraps.
type countingPlacing struct {
	counting
	p engine.Placeable
}

func (c countingPlacing) PlaceAt(at plan.Endpoint) error { return c.p.PlaceAt(at) }

// An engine of a disaggregated pool with Plan, Move and Peer and no placement support is refused:
// left alone, every instance of the pool would report the pool's first node as its end.
func TestAnEngineWithoutPlacementSupportIsRefusedForAPoolOfSeveralNodes(t *testing.T) {
	build := func() (engine.Engine, error) { return NewPuller(fifoCfg(), 4) }
	err := CheckPlacement(build, pulls(), 4)
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
	if err := CheckPlacement(build, pulls(), 4); err != nil {
		t.Fatal(err)
	}
}

// An engine that has the capability and rejects a place of its pool is refused with the rejection.
func TestAPlacementTheEngineRejectsIsRefused(t *testing.T) {
	// The engine is built for a pool of two nodes and the host says the pool has four: node 3 is
	// outside what the engine models.
	build := func() (engine.Engine, error) { return NewPlacedPuller(fifoCfg(), 2) }
	if err := CheckPlacement(build, pulls(), 4); !errors.Is(err, engine.ErrPlacementRejected) {
		t.Fatalf("CheckPlacement = %v, want a refusal wrapping ErrPlacementRejected", err)
	}
}

func TestCheckPlacementNeedsAPoolOfSeveralNodes(t *testing.T) {
	build := func() (engine.Engine, error) { return NewPlacedPuller(fifoCfg(), 1) }
	if err := CheckPlacement(build, pulls(), 1); err == nil {
		t.Error("a pool of one node was accepted")
	}
}

// A refused placement ends the check before the engine runs: an engine that is not Placeable is
// returned the refusal and is given no request, so the check does not drive it and report a
// placement error afterwards.
func TestARefusedPlacementDoesNotRunTheEngine(t *testing.T) {
	for name, c := range map[string]struct {
		build func() (engine.Engine, error)
		want  error
	}{
		"not placeable": {func() (engine.Engine, error) { return NewPuller(fifoCfg(), 4) }, engine.ErrPlacementUnsupported},
		"rejected":      {func() (engine.Engine, error) { p, err := NewPlacedPuller(fifoCfg(), 4); return rejecting{p}, err }, engine.ErrPlacementRejected},
	} {
		adds := 0
		build := func() (engine.Engine, error) {
			e, err := c.build()
			if p, ok := e.(engine.Placeable); ok {
				return countingPlacing{counting{e, &adds}, p}, err
			}
			return counting{e, &adds}, err
		}
		err := CheckPlacement(build, pulls(), 4)
		if !errors.Is(err, c.want) {
			t.Errorf("%s: CheckPlacement = %v, want %v", name, err, c.want)
		}
		if adds != 0 {
			t.Errorf("%s: the engine was given %d request(s) after the placement was refused", name, adds)
		}
	}
}

// earlyWrong states its own end at the origin for its first pull and correctly for every later one:
// only a check of every pull sees it, since the last pull is right.
type earlyWrong struct {
	PlacedPuller
	seen int
}

func (l *earlyWrong) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := l.PlacedPuller.Schedule(now)
	for i, x := range b.TransfersStarted {
		if x.Move.Kind != plan.TransferPeer {
			continue
		}
		l.seen++
		if l.seen == 1 {
			b.TransfersStarted[i].Move.To = plan.Endpoint{}
		}
	}
	return b, err
}

// Every pull is validated, not the last: an engine wrong for its first pull and right for its last
// is refused.
func TestEveryPullMustEndWhereTheInstanceWasPlaced(t *testing.T) {
	good := func() (engine.Engine, error) { return NewPlacedPuller(fifoCfg(), 4) }
	if err := CheckPlacement(good, twoPulls(), 4); err != nil {
		t.Fatalf("a correct engine with two pulls: %v", err)
	}
	early := func() (engine.Engine, error) {
		p, err := NewPlacedPuller(fifoCfg(), 4)
		return &earlyWrong{PlacedPuller: p}, err
	}
	err := CheckPlacement(early, twoPulls(), 4)
	if err == nil || errors.Is(err, engine.ErrPlacementRejected) || errors.Is(err, engine.ErrPlacementUnsupported) {
		t.Fatalf("an engine wrong for its first pull and right for its last: %v, want a report of the pull that ends elsewhere", err)
	}
}

// Two pools may each have a local rack 0 that is a different physical rack. The prefill engine is
// placed at local node 0 rack 0 of its pool, the decode engine at local node 0 rack 0 of its own;
// the decode engine's pull reads from the prefill engine's local endpoint and lands on its own, both
// pool-relative, and the cluster's endpoints, found with each pool's origin, are on different racks.
func TestTwoPoolsWithTheSameLocalRackPullAcrossPhysicalRacks(t *testing.T) {
	prefillOrigin, decodeOrigin := engine.PoolOrigin{Node: 0, Rack: 0}, engine.PoolOrigin{Node: 2, Rack: 3}
	local := plan.Endpoint{} // node 0, rack 0 of each pool
	prefill, err := NewPlacedPuller(fifoCfg(), 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := engine.Place(prefill, local, 2); err != nil {
		t.Fatal(err)
	}
	peer := local // the prefill engine's end, in the prefill pool's numbering
	r := req("p", 4, 1)
	r.RemotePrefill, r.Peer = true, &peer
	var move plan.Transfer
	_, err = Drive(func() (engine.Engine, error) {
		d, err := NewPlacedPuller(fifoCfg(), 2)
		if err != nil {
			return nil, err
		}
		return d, engine.Place(d, local, 2)
	}, []Arrival{at(0, r)}, Options{Boundary: func(_ plan.PoolPlan, x engine.Transfer) error {
		if x.Move.Kind == plan.TransferPeer {
			move = x.Move
		}
		return nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	if move.From != local || move.To != local {
		t.Fatalf("both ends are local node 0 rack 0: From %+v To %+v", move.From, move.To)
	}
	from, to := prefillOrigin.Global(move.From), decodeOrigin.Global(move.To)
	if from.Rack == to.Rack || from.Node == to.Node {
		t.Errorf("the pull leaves %+v and lands on %+v: different pools' local rack 0 must be different physical racks", from, to)
	}
}
