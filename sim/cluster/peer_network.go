package cluster

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

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

	// onNode counts the Peered instances registered on each node's hostname, for the ports.
	onNode map[string]int

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
			due: map[int64][]engine.PeerMessage{}, wake: map[int64]bool{}, onNode: map[string]int{}}
	}
	cs.assignPeerAddr(inst)
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

// assignPeerAddr gives inst's engine the address "blis-node-<node>:<port>": the hostname the router
// is told for the instance (EndpointTopology.Hostname), and the port the engine's own address states
// plus the instance's index among the Peered instances of that node (0 for the first). It runs
// before the instance is placed with its peers, and so before any engine places a peer by address,
// and before the engine's first Schedule. A engine that takes no address (no SetPeerAddr) keeps the
// one it states; an instance with no single hostname (none stated, or one spanning nodes) keeps it
// too, and registerPeer refuses two instances that then share one. A refused address stops the run.
//
// Named substitution (blis-vllm#204): this stands for the per-node VLLM_P2P_SIDE_CHANNEL_HOST a
// deployment sets, and for the distinct side-channel port each engine of one node is given.
func (cs *ClusterSimulator) assignPeerAddr(inst *InstanceSimulator) {
	host := inst.endpointTopology.Hostname
	if host == "" {
		return
	}
	stated := inst.drv.PeerAddr()
	i := strings.LastIndex(stated, ":")
	port, err := strconv.Atoi(stated[i+1:])
	if i < 0 || err != nil {
		panic(fmt.Sprintf("ClusterSimulator: instance %s states the peer address %q, which is not host:port", inst.id, stated))
	}
	addr := fmt.Sprintf("%s:%d", host, port+cs.peers.onNode[host])
	taken, err := inst.drv.SetPeerAddr(addr)
	if err != nil {
		panic(fmt.Sprintf("ClusterSimulator: instance %s refused the peer address %q: %v", inst.id, addr, err))
	}
	if taken {
		cs.peers.onNode[host]++
	}
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

// p2pHandoff reports whether parent's P/D handoff goes through vLLM's p2p KV tier: both its prefill
// and its decode instance are peered (an engine with a p2p tier and an address). Their addresses are
// returned. A handoff whose instances are not both peered goes through the P/D connector's pull.
//
// The gateway keeps its order, prefill first and the decode at the prefill's end, as it does for every
// connector. vLLM leaves when to send each sub-request to the orchestrator (docs/features/
// kv_offloading_usage.md, "Orchestration-Layer Protocol", vllm-project/vllm v0.33.0): the prefill
// side keeps its stored blocks for the decoder's fetch, so the decode may start at its end.
func (cs *ClusterSimulator) p2pHandoff(parent *ParentRequest) (prefill, decode string, ok bool) {
	p := cs.instanceByID(parent.PrefillInstanceID)
	d := cs.instanceByID(parent.DecodeInstanceID)
	if p == nil || d == nil || p.drv == nil || d.drv == nil {
		return "", "", false
	}
	prefill, decode = p.drv.PeerAddr(), d.drv.PeerAddr()
	return prefill, decode, prefill != "" && decode != ""
}

// p2pTransferID is the kv_request_id the orchestrator allocates for parent's handoff: the parent
// request's id, unique in a run.
func p2pTransferID(parent *ParentRequest) string { return parent.ID }

// p2pPrefillParams are the prefill sub-request's kv_transfer_params for a p2p handoff: keep the
// computed KV for the remote decoder (remote_decoder{kv_request_id}).
func (cs *ClusterSimulator) p2pPrefillParams(parent *ParentRequest) (string, bool) {
	if _, _, ok := cs.p2pHandoff(parent); !ok {
		return "", false
	}
	b, _ := json.Marshal(map[string]any{"remote_decoder": map[string]any{"kv_request_id": p2pTransferID(parent)}})
	return string(b), true
}

// p2pDecodeParams are the decode sub-request's kv_transfer_params for a p2p handoff: pull the KV from
// the prefill instance at its address (remote_prefiller{kv_request_id, remote_host, remote_port}).
func (cs *ClusterSimulator) p2pDecodeParams(parent *ParentRequest) (string, bool) {
	prefill, _, ok := cs.p2pHandoff(parent)
	if !ok {
		return "", false
	}
	host, portText, found := strings.Cut(prefill, ":")
	port, err := strconv.Atoi(portText)
	if !found || err != nil {
		panic(fmt.Sprintf("ClusterSimulator: prefill instance %s has the peer address %q, which is not host:port",
			parent.PrefillInstanceID, prefill))
	}
	b, _ := json.Marshal(map[string]any{"remote_prefiller": map[string]any{"kv_request_id": p2pTransferID(parent),
		"remote_host": host, "remote_port": port}})
	return string(b), true
}
