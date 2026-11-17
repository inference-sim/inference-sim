package sim

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
)

// Peer messages and peer WRITEs (engine.Peered, engine.PeerPlaced). An engine that talks to other
// engines of the run, as vLLM's p2p secondary tier does over ZMQ and NIXL, queues messages in its
// Outbox and reports a WRITE as a store transfer whose Move is a peer move. The driver drains the
// Outbox after every call it makes to the engine and hands each message to the host's PeerRouter,
// which picks the delivery time. A WRITE joins the host's link arbiter like any other transfer, so
// it shares the links it crosses; nothing in it fails.

// PeerRouter carries an engine's peer messages to the engines they name. SendPeer is called at now
// with each message the sender queued, in the order it queued them.
type PeerRouter interface {
	SendPeer(m engine.PeerMessage, now int64)
}

// PeerAddr is the engine's peer address, or "" when it is not Peered.
func (d *EngineDriver) PeerAddr() string {
	if p, ok := d.eng.(engine.Peered); ok {
		return p.PeerAddr()
	}
	return ""
}

// SetPeerAddr gives the engine the address its peers reach it at, when the engine takes one
// (interface{ SetPeerAddr(string) error }, an engine-specific method of blis-vllm's v0.33.0
// engine); false when it does not. It is called before the engine's first Schedule.
func (d *EngineDriver) SetPeerAddr(addr string) (bool, error) {
	s, ok := d.eng.(interface{ SetPeerAddr(string) error })
	if !ok {
		return false, nil
	}
	return true, s.SetPeerAddr(addr)
}

// SetPeerRouter installs the host's router for the engine's peer messages.
func (d *EngineDriver) SetPeerRouter(r PeerRouter) { d.peerRouter = r }

// PlacePeer tells the engine where the peer at addr sits, in the engine's own pool numbering. An
// engine that is not engine.PeerPlaced has nothing to place and the call is a no-op.
func (d *EngineDriver) PlacePeer(addr string, at plan.Endpoint) error {
	if p, ok := d.eng.(engine.PeerPlaced); ok {
		return p.PlacePeer(addr, at)
	}
	return nil
}

// drainPeer sends every message the engine queued to the router, at now. A Peered engine with no
// router has no peers to reach: its messages are dropped without a count, which cannot happen in a
// cluster run (the cluster installs a router for every Peered engine).
func (d *EngineDriver) drainPeer(now int64) {
	p, ok := d.eng.(engine.Peered)
	if !ok {
		return
	}
	for _, m := range p.Outbox() {
		if err := m.Validate(); err != nil {
			panic(fmt.Sprintf("engine driver: the engine queued an invalid peer message: %v", err))
		}
		if d.peerRouter != nil {
			d.peerRouter.SendPeer(m, now)
		}
	}
}

// DeliverPeer hands the engine a peer message at now, then drains what it queued in answer and
// steps it, since a message can start work.
func (d *EngineDriver) DeliverPeer(m engine.PeerMessage, now int64) {
	p, ok := d.eng.(engine.Peered)
	if !ok {
		panic(fmt.Sprintf("engine driver: a peer message was delivered to an engine that is not Peered (%T)", d.eng))
	}
	if err := p.Deliver(m, engine.Time(now)); err != nil {
		panic(fmt.Sprintf("engine driver: Deliver(%d from %q) at %d: %v", m.Seq, m.From, now, err))
	}
	d.drainPeer(now)
	d.kick(now)
	d.drainPeer(now)
}

// PeerWritePricer is the part of an Environment that prices a peer WRITE (design 6.1). A WRITE is
// not a pull: it leaves the engine's own pool, carries the bytes the engine states (its blocks as
// the p2p tier stores them) and may stay on one node. The Environment is asked through this
// optional interface, because env.Environment's TransferDemand cannot take the bytes.
type PeerWritePricer interface {
	PeerWriteDemand(p plan.PoolPlan, t engine.Transfer, bytes int64) (kernel.TransferDemand, error)
}

// peerWriteBytes is the optional engine method that states a WRITE's bytes.
type peerWriteBytes interface {
	PeerWriteBytes(engine.Transfer) (int64, bool)
}

// priceWrite is what the peer WRITE t demands of the fabric: the bytes the engine states
// (PeerWriteBytes) priced by the Environment's PeerWritePricer. An engine that is not Planned or
// states no bytes for the WRITE, a host with no Environment or an Environment that cannot price a
// WRITE, and a WRITE the cost model refuses are each a failure of the run, never a zero price.
func (d *EngineDriver) priceWrite(t engine.Transfer) kernel.TransferDemand {
	p, ok := d.eng.(engine.Planned)
	if !ok {
		panic(fmt.Sprintf("engine driver: the engine started a peer WRITE %d for %q and states no plan "+
			"(engine.Planned), so the cost model cannot price it", t.ID, t.Request))
	}
	b, ok := d.eng.(peerWriteBytes)
	if !ok {
		panic(fmt.Sprintf("engine driver: the engine started a peer WRITE %d for %q and cannot state its bytes "+
			"(PeerWriteBytes), so the cost model cannot price it", t.ID, t.Request))
	}
	bytes, ok := b.PeerWriteBytes(t)
	if !ok {
		panic(fmt.Sprintf("engine driver: the engine states no bytes for its peer WRITE %d of %q", t.ID, t.Request))
	}
	if d.cfg.Environment == nil {
		panic(fmt.Sprintf("engine driver: the engine started peer WRITE %d for %q, and the driver has no "+
			"Environment to price it", t.ID, t.Request))
	}
	w, ok := d.cfg.Environment.(PeerWritePricer)
	if !ok {
		panic(fmt.Sprintf("engine driver: peer WRITE %d of %q: the driver's Environment (%T) cannot price a "+
			"peer WRITE (PeerWriteDemand)", t.ID, t.Request, d.cfg.Environment))
	}
	if err := t.Validate(); err != nil {
		panic(fmt.Sprintf("engine driver: peer WRITE %d of %q: %v", t.ID, t.Request, err))
	}
	if err := p.Plan().CheckTransfer(t.Move); err != nil {
		panic(fmt.Sprintf("engine driver: peer WRITE %d of %q: %v", t.ID, t.Request, err))
	}
	demand, err := w.PeerWriteDemand(p.Plan(), t, bytes)
	if err != nil {
		panic(fmt.Sprintf("engine driver: peer WRITE %d of %q: %v", t.ID, t.Request, err))
	}
	return demand
}

// startPeerWrite runs an engine's WRITE to a peer. With a host that shares link capacity
// (OnPeerWritePriced) the WRITE joins the arbiter and lands when it completes (PeerWriteLanded);
// without one it takes its demand alone (TierTicks). Writes do not queue behind one another: each
// is its own NIXL request.
func (d *EngineDriver) startPeerWrite(t engine.Transfer, demand kernel.TransferDemand, now int64) {
	if d.OnPeerWritePriced != nil {
		if d.peerWrites == nil {
			d.peerWrites = map[engine.TransferID]bool{}
		}
		d.peerWrites[t.ID] = true
		d.OnPeerWritePriced(t.ID, demand, now)
		return
	}
	d.schedule(&driverTierTransferDone{at: now + TierTicks(demand), id: t.ID})
}

// PeerWriteLanded reports at now that the WRITE id (OnPeerWritePriced) has landed.
func (d *EngineDriver) PeerWriteLanded(id engine.TransferID, now int64) {
	if !d.peerWrites[id] {
		panic(fmt.Sprintf("engine driver: PeerWriteLanded(%d): no peer WRITE by that id is in flight", id))
	}
	delete(d.peerWrites, id)
	if err := d.eng.TransferDone(id, true, now); err != nil {
		panic(fmt.Sprintf("engine driver: TransferDone(%d) of a peer WRITE: %v", id, err))
	}
	d.kick(now)
	d.drainPeer(now)
}

// isPeerWrite reports whether t is a WRITE to a peer: a store whose move is a peer move.
func isPeerWrite(t engine.Transfer) bool {
	return t.Tier == "" && t.Kind == engine.TransferStore && t.Move.Kind == plan.TransferPeer
}
