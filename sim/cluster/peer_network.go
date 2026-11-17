package cluster

import (
	"fmt"
	"sort"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/sim"
)

// The peer network carries the messages engines of one run send each other (engine.Peered): vLLM's
// p2p secondary tier talks over ZMQ between instances. The simulator picks the delivery time: a
// message sent at t arrives at t + PeerMessageLatency, a named substitution
// (peer_message_delivery_latency) until something measures the ZMQ round trip. Messages of one
// sender-receiver pair arrive in the order sent, because one latency serves every pair, and the
// messages that arrive at one instant are delivered in (From, Seq) order. A message to a stopped
// engine or an unknown address is dropped, never delivered, and counted (peer_messages_dropped).

// peerNetwork is the cluster's state for peer messages.
type peerNetwork struct {
	latency int64
	byAddr  map[string]*InstanceSimulator
	// due are the messages not yet delivered, by arrival tick; wake is set for the ticks that have
	// an event queued.
	due  map[int64][]engine.PeerMessage
	wake map[int64]bool

	sent, delivered, dropped int
}

// SendPeer implements sim.PeerRouter for the engines of the cluster.
func (cs *ClusterSimulator) sendPeer(m engine.PeerMessage, now int64) {
	pn := cs.peers
	pn.sent++
	at := now + pn.latency
	pn.due[at] = append(pn.due[at], m)
	if !pn.wake[at] {
		pn.wake[at] = true
		cs.schedule(&PeerDeliverEvent{time: at}, &cs.kvConnectorSender)
	}
}

type peerRouterFunc func(m engine.PeerMessage, now int64)

func (f peerRouterFunc) SendPeer(m engine.PeerMessage, now int64) { f(m, now) }

// registerPeer makes inst's engine reachable by its peer address, tells it where every earlier
// peer sits and every earlier peer where it sits (engine.PeerPlaced), and installs the router.
// An instance whose engine is not Peered is ignored.
func (cs *ClusterSimulator) registerPeer(inst *InstanceSimulator) {
	if inst.drv == nil || inst.drv.PeerAddr() == "" {
		return
	}
	if cs.peers == nil {
		cs.peers = &peerNetwork{latency: cs.config.PeerMessageLatency, byAddr: map[string]*InstanceSimulator{},
			due: map[int64][]engine.PeerMessage{}, wake: map[int64]bool{}}
	}
	addr := inst.drv.PeerAddr()
	if prev, dup := cs.peers.byAddr[addr]; dup {
		panic(fmt.Sprintf("ClusterSimulator: instances %s and %s both have the peer address %q", prev.id, inst.id, addr))
	}
	inst.drv.SetPeerRouter(peerRouterFunc(func(m engine.PeerMessage, now int64) { cs.sendPeer(m, now) }))
	if cs.config.PeerEndpoint != nil {
		for _, other := range cs.peerInstances() {
			cs.placePeer(other, inst)
			cs.placePeer(inst, other)
		}
	}
	cs.peers.byAddr[addr] = inst
}

// placePeer tells at's engine where peer sits, in the numbering of at's own pool.
func (cs *ClusterSimulator) placePeer(at, peer *InstanceSimulator) {
	var ep plan.Endpoint = cs.config.PeerEndpoint(peer.id)
	if err := at.drv.PlacePeer(peer.drv.PeerAddr(), ep); err != nil {
		panic(fmt.Sprintf("ClusterSimulator: placing peer %s for %s: %v", peer.id, at.id, err))
	}
}

// peerInstances are the registered peers, in address order.
func (cs *ClusterSimulator) peerInstances() []*InstanceSimulator {
	addrs := make([]string, 0, len(cs.peers.byAddr))
	for a := range cs.peers.byAddr {
		addrs = append(addrs, a)
	}
	sort.Strings(addrs)
	out := make([]*InstanceSimulator, 0, len(addrs))
	for _, a := range addrs {
		out = append(out, cs.peers.byAddr[a])
	}
	return out
}

// PeerDeliverEvent delivers the peer messages that arrive at its tick, in (From, Seq) order.
// Class 6, the KV connector's: it runs before every instance event of the tick, so an engine
// steps with the message already delivered.
type PeerDeliverEvent struct{ time int64 }

func (e *PeerDeliverEvent) Timestamp() int64 { return e.time }
func (e *PeerDeliverEvent) Priority() int    { return 6 }

// Execute delivers the batch.
func (e *PeerDeliverEvent) Execute(cs *ClusterSimulator) {
	pn := cs.peers
	batch := pn.due[e.time]
	delete(pn.due, e.time)
	delete(pn.wake, e.time)
	sort.SliceStable(batch, func(i, j int) bool {
		if batch[i].From != batch[j].From {
			return batch[i].From < batch[j].From
		}
		return batch[i].Seq < batch[j].Seq
	})
	for _, m := range batch {
		to, ok := pn.byAddr[m.To]
		if !ok || to.State == sim.InstanceStateTerminated {
			pn.dropped++
			continue
		}
		pn.delivered++
		to.drv.DeliverPeer(m, e.time)
	}
}

// PeerStats are the peer network's counts and its stated latency: messages sent, delivered and
// dropped (peer_messages_dropped), and the delivery latency in ticks (peer_message_delivery_latency).
// All zero when no engine of the run is Peered.
type PeerStats struct {
	Sent, Delivered, Dropped int
	LatencyTicks             int64
}

// PeerStats reports the peer network's counts.
func (cs *ClusterSimulator) PeerStats() PeerStats {
	if cs.peers == nil {
		return PeerStats{}
	}
	return PeerStats{Sent: cs.peers.sent, Delivered: cs.peers.delivered, Dropped: cs.peers.dropped, LatencyTicks: cs.peers.latency}
}
