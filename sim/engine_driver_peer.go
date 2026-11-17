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
