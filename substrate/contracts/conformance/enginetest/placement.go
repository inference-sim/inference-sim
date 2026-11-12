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
// engine; pulls are RemotePrefill arrivals that name their peers. The check places one instance at
// the first node and one at the last (both coordinates in the pool's own numbering), runs the
// pulls on each, and requires: the host's call to succeed (an engine that is not engine.Placeable,
// or that rejects a place of the pool, is refused; the refusal is returned unchanged so a caller
// can match it with errors.Is on engine.ErrPlacementUnsupported or engine.ErrPlacementRejected, and
// the engine is not run); EVERY peer pull of EVERY request to end at the endpoint its instance
// was placed at and to read from the peer its request named; every pull arrival to start a peer
// pull; and the two instances to report different ends.
func CheckPlacement(build func() (engine.Engine, error), pulls []Arrival, poolNodes int) error {
	if poolNodes < 2 {
		return fmt.Errorf("enginetest: CheckPlacement needs a pool of more than one node, got %d", poolNodes)
	}
	if len(pulls) == 0 {
		return fmt.Errorf("enginetest: CheckPlacement needs at least one pull arrival")
	}
	peers := map[string]plan.Endpoint{}
	for _, a := range pulls {
		if !a.Req.RemotePrefill || a.Req.Peer == nil {
			return fmt.Errorf("enginetest: request %s is not a pull that names its peer", a.Req.ID)
		}
		peers[a.Req.ID] = *a.Req.Peer
	}
	var ends []plan.Endpoint
	for _, node := range []int{0, poolNodes - 1} {
		want := plan.Endpoint{Node: node, Rack: node}
		e, err := build()
		if err != nil {
			return err
		}
		// A refused placement ends the check: the engine is not run, and the refusal is the result.
		if err := engine.Place(e, want, poolNodes); err != nil {
			return err
		}
		pulled := map[string]int{}
		var bad error
		_, err = Drive(func() (engine.Engine, error) { return e, nil }, pulls, Options{Boundary: func(_ plan.PoolPlan, x engine.Transfer) error {
			if x.Move.Kind != plan.TransferPeer {
				return nil
			}
			pulled[x.Request]++
			switch {
			case x.Move.To != want:
				bad = fmt.Errorf("enginetest: the engine was placed at %+v and the pull of request %s ends at %+v", want, x.Request, x.Move.To)
			case x.Move.From != peers[x.Request]:
				bad = fmt.Errorf("enginetest: the pull of request %s reads from %+v, the peer its request named is %+v", x.Request, x.Move.From, peers[x.Request])
			}
			return bad
		}})
		if bad != nil {
			return bad
		}
		if err != nil {
			return err
		}
		for id := range peers {
			if pulled[id] == 0 {
				return fmt.Errorf("enginetest: the engine placed at %+v started no peer pull for request %s", want, id)
			}
		}
		ends = append(ends, want)
	}
	if ends[0] == ends[1] {
		return fmt.Errorf("enginetest: instances placed on different nodes report the same end %+v", ends[0])
	}
	return nil
}
