package enginetest

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

// PullerFabric is the catalog network a Puller's plan names when its pool spans nodes.
const PullerFabric = "fifo-fabric"

// Puller is a FIFO that also disaggregates: a request marked RemotePrefill is loaded from the peer
// it names, as a Transfer whose Move is a pull with From the peer and To this engine's own end.
// It has Plan, Move and Peer and does NOT implement engine.Placeable, so its own end is always the
// pool's first node: the known-bad engine for placement (an engine of a disaggregated pool that
// cannot be told where it runs). PlacedPuller is the same engine with the capability.
type Puller struct {
	*FIFO
	nodes int
	at    plan.Endpoint
	peers map[string]plan.Endpoint
}

// NewPuller builds a Puller for a pool of the given number of nodes.
func NewPuller(cfg FIFOConfig, nodes int) (*Puller, error) {
	f, err := NewFIFO(cfg)
	if err != nil {
		return nil, err
	}
	return &Puller{FIFO: f, nodes: max(1, nodes), peers: map[string]plan.Endpoint{}}, nil
}

// Plan implements engine.Planned: FIFO's plan over the pool's nodes.
func (p *Puller) Plan() plan.PoolPlan {
	pl := p.FIFO.Plan()
	pl.Layout.Nodes = p.nodes
	if p.nodes > 1 {
		pl.Fabric = PullerFabric
	}
	return pl
}

// Add implements engine.Engine: a pull is loaded in full before the request runs.
func (p *Puller) Add(r engine.Request, now engine.Time) error {
	if r.RemotePrefill {
		if r.Peer == nil {
			return fmt.Errorf("puller: request %s pulls and names no peer", r.ID)
		}
		p.peers[r.ID] = *r.Peer
		tags := map[string]string{}
		for k, v := range r.Tags {
			tags[k] = v
		}
		tags["fifo.load_tokens"] = fmt.Sprint(len(r.Prompt))
		r.Tags = tags
	}
	return p.FIFO.Add(r, now)
}

// Schedule implements engine.Engine, restating FIFO's tier loads of pulled requests as pulls
// from their peers into this engine's own end.
func (p *Puller) Schedule(now engine.Time) (engine.Batch, error) {
	b, err := p.FIFO.Schedule(now)
	for i, x := range b.TransfersStarted {
		if peer, ok := p.peers[x.Request]; ok {
			b.TransfersStarted[i].Tier = ""
			b.TransfersStarted[i].Move = plan.Transfer{Tokens: x.Move.Tokens, Kind: plan.TransferPeer, From: peer, To: p.at}
		}
	}
	return b, err
}

// PlacedPuller is a Puller that states its own end where the host placed it.
type PlacedPuller struct{ *Puller }

var _ engine.Placeable = PlacedPuller{}

// NewPlacedPuller builds a PlacedPuller for a pool of the given number of nodes.
func NewPlacedPuller(cfg FIFOConfig, nodes int) (PlacedPuller, error) {
	p, err := NewPuller(cfg, nodes)
	return PlacedPuller{p}, err
}

// PlaceAt implements engine.Placeable.
func (p PlacedPuller) PlaceAt(at plan.Endpoint) error {
	if at.Node < 0 || at.Rack < 0 || at.Node >= p.nodes {
		return fmt.Errorf("puller: node %d rack %d is outside a pool of %d node(s)", at.Node, at.Rack, p.nodes)
	}
	p.at = at
	return nil
}

// CheckPlacement is the conformance check of the placement capability for an engine of a
// disaggregated pool of poolNodes (more than one) nodes that starts peer pulls. build makes a fresh
// engine; pull is a RemotePrefill arrival that names its peer. The check places one instance at
// the first node and one at the last, runs the pull on each, and requires: the host's call to
// succeed (an engine that is not engine.Placeable, or that rejects a place of the pool, is
// refused, and the refusal is returned unchanged so a caller can match it with errors.Is on
// engine.ErrPlacementUnsupported or engine.ErrPlacementRejected); every pull to end at the
// endpoint its instance was placed at; and the two instances to report different ends.
func CheckPlacement(build func() (engine.Engine, error), pull Arrival, poolNodes int) error {
	if poolNodes < 2 {
		return fmt.Errorf("enginetest: CheckPlacement needs a pool of more than one node, got %d", poolNodes)
	}
	var ends []plan.Endpoint
	for _, node := range []int{0, poolNodes - 1} {
		want := plan.Endpoint{Node: node, Rack: node}
		var placeErr error
		var got *plan.Endpoint
		_, err := Drive(func() (engine.Engine, error) {
			e, err := build()
			if err != nil {
				return nil, err
			}
			placeErr = engine.Place(e, want, poolNodes)
			return e, nil
		}, []Arrival{pull}, Options{Boundary: func(_ plan.PoolPlan, x engine.Transfer) error {
			if x.Move.Kind == plan.TransferPeer {
				to := x.Move.To
				got = &to
			}
			return nil
		}})
		if placeErr != nil {
			return placeErr
		}
		if err != nil {
			return err
		}
		if got == nil {
			return fmt.Errorf("enginetest: the engine placed at %+v started no peer pull for the request", want)
		}
		if *got != want {
			return fmt.Errorf("enginetest: the engine was placed at %+v and its pull ends at %+v", want, *got)
		}
		ends = append(ends, *got)
	}
	if ends[0] == ends[1] {
		return fmt.Errorf("enginetest: instances placed on different nodes report the same end %+v", ends[0])
	}
	return nil
}
