// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/tiering/p2p/control/zmq.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).
// Also p2p/data/nixl.py, over the in-memory sockets and NIXL agent of blis-vllm's oracle
// (oracle/fake_zmq, oracle/fake_nixl).

package tiering

import (
	"errors"
	"fmt"
)

// P2PNetwork is the host's p2p transport for the instances of one process: the sockets the control
// transports talk over and the NIXL agents their WRITEs run on, held in memory. It has the
// oracle's semantics (oracle/fake_zmq, oracle/fake_nixl), and the host decides when things move:
//
//   - a control message sent on a DEALER is on the wire, in send order, until Deliver moves it to
//     the ROUTER bound at its address (a message to an address nobody has bound stays on it);
//   - a WRITE is in flight until Complete ends it: DONE, or ERR when its target's agent is down;
//   - a peer registration a session started is pending until ResolveRegistrations;
//   - Down is an instance that died: its ROUTER is unbound, what is on the wire to it is dropped,
//     every DEALER connected to it reports a disconnect, and WRITEs to its agent fail.
//
// A WRITE moves no bytes: it is the record of which local blocks go to which of the peer's.
type P2PNetwork struct {
	routers map[string]*memControl
	wire    []wired
	dealers []*memDealer
	held    bool
	xfers   []*memXfer
	// downAgents are the agents whose WRITEs fail; heldAgents are those whose WRITEs stay in flight.
	downAgents map[string]bool
	heldAgents map[string]bool
	regs       []*memReg
	draining   int
}

type wired struct {
	addr, ident string
	msg         P2PMsg
}

// NewP2PNetwork is an empty network.
func NewP2PNetwork() *P2PNetwork {
	return &P2PNetwork{routers: map[string]*memControl{}, downAgents: map[string]bool{}, heldAgents: map[string]bool{}}
}

// Hold keeps control messages on the wire across RunAll (hold_net), until it is released.
func (n *P2PNetwork) Hold(on bool) { n.held = on }

// HoldWrites keeps the WRITEs instance local submits in flight across RunAll, until released; a
// transfer state check inside drain_jobs still ends them (fake_nixl.hold).
func (n *P2PNetwork) HoldWrites(local string, on bool) { n.heldAgents[agentName(local)] = on }

// DataDown makes WRITEs to instance local's agent fail, its control socket untouched (nixl_down).
func (n *P2PNetwork) DataDown(local string) { n.downAgents[agentName(local)] = true }

// Down is instance local dying (fake_zmq.down and fake_nixl.agent_down).
func (n *P2PNetwork) Down(local string) {
	delete(n.routers, local)
	keep := n.wire[:0:0]
	for _, w := range n.wire {
		if w.addr != local {
			keep = append(keep, w)
		}
	}
	n.wire = keep
	for _, d := range n.dealers {
		if d.target == local && !d.closed {
			d.events++
		}
	}
	n.downAgents[agentName(local)] = true
}

// Deliver moves every message on the wire to its bound ROUTER, in send order.
func (n *P2PNetwork) Deliver() {
	var keep []wired
	for _, w := range n.wire {
		if r := n.routers[w.addr]; r != nil {
			r.inbox = append(r.inbox, w)
		} else {
			keep = append(keep, w)
		}
	}
	n.wire = keep
}

// Complete ends every WRITE in flight, in submission order, except a held agent's (complete_all).
func (n *P2PNetwork) Complete() { n.complete(false) }

func (n *P2PNetwork) complete(draining bool) {
	var keep []*memXfer
	for _, x := range n.xfers {
		if !draining && n.heldAgents[x.agent] {
			keep = append(keep, x)
			continue
		}
		x.state = "DONE"
		if n.downAgents[x.peer] {
			x.state = "ERR"
		}
	}
	n.xfers = keep
}

// ResolveRegistrations ends every pending peer registration (NixlTransport's worker thread).
func (n *P2PNetwork) ResolveRegistrations() {
	for _, r := range n.regs {
		r.run()
	}
	n.regs = nil
}

// RunAll is the oracle's transport rule after each call (run_tiering.py wait_p2p): every WRITE in
// flight ends, then every control message is delivered unless the wire is held, then every
// pending registration resolves.
func (n *P2PNetwork) RunAll() {
	n.Complete()
	if !n.held {
		n.Deliver()
	}
	n.ResolveRegistrations()
}

// Control is the control transport of the instance bound at local ("host:port"): ZmqTransport
// (control/zmq.py:156-399@4ba787ec) over the network's sockets. It binds the instance's ROUTER.
func (n *P2PNetwork) Control(local string) P2PControl {
	if n.routers[local] != nil {
		panic(fmt.Sprintf("tiering: address %s already in use", local))
	}
	c := &memControl{net: n, local: local}
	n.routers[local] = c
	return c
}

// memDealer is a DEALER socket and its monitor: events counts the EVENT_DISCONNECTED it holds.
type memDealer struct {
	target string
	closed bool
	events int
}

// memConn is ZmqConnection (zmq.py:85-153).
type memConn struct {
	net          *P2PNetwork
	local, peer  string
	dealer       *memDealer
	dead, closed bool
	inbox        []P2PMsg
}

func (c *memConn) PeerID() string { return c.peer }
func (c *memConn) Alive() bool    { return !c.dead && !c.closed }
func (c *memConn) MarkDead()      { c.dead = true }

func (c *memConn) Send(m P2PMsg) error {
	if !c.Alive() {
		return errors.New("send on closed connection to " + c.peer)
	}
	c.net.wire = append(c.net.wire, wired{addr: c.dealer.target, ident: c.local, msg: m})
	return nil
}

func (c *memConn) Recv() []P2PMsg {
	out := c.inbox
	c.inbox = nil
	return out
}

func (c *memConn) Close() {
	if c.closed {
		return
	}
	c.closed, c.dead = true, true
	c.dealer.closed = true
}

// memControl is ZmqTransport (zmq.py:156-399).
type memControl struct {
	net         *P2PNetwork
	local       string
	conns       omap[string, *memConn]
	pending     []wired
	inbox       []wired // the ROUTER's received messages
	closedState bool
}

// Connect is connect (zmq.py:184-207): a dead connection to the peer is retired first.
func (c *memControl) Connect(peer string) (P2PConn, error) {
	if old, ok := c.conns.get(peer); ok {
		if old.Alive() {
			panic(fmt.Sprintf("tiering: connection to %s already exists", peer))
		}
		c.conns.pop(peer)
		old.Close()
	}
	return c.open(peer), nil
}

func (c *memControl) open(peer string) *memConn {
	d := &memDealer{target: peer}
	c.net.dealers = append(c.net.dealers, d)
	conn := &memConn{net: c.net, local: c.local, peer: peer, dealer: d}
	c.conns.set(peer, conn)
	return conn
}

// Poll is poll (zmq.py:209-249).
func (c *memControl) Poll() []P2PConn {
	c.sweep()
	// _recv_router (:322-367).
	for _, w := range c.inbox {
		if conn, ok := c.conns.get(w.ident); ok {
			conn.inbox = append(conn.inbox, w.msg)
		} else {
			c.pending = append(c.pending, w)
		}
	}
	c.inbox = nil
	// _check_monitors (:369-399).
	for _, conn := range c.conns.values() {
		if conn.Alive() && conn.dealer.events > 0 {
			conn.dealer.events--
			conn.MarkDead()
		}
	}
	c.sweep()
	var out []P2PConn
	for _, w := range c.pending {
		conn, ok := c.conns.get(w.ident)
		if !ok {
			conn = c.open(w.ident)
			out = append(out, conn)
		}
		conn.inbox = append(conn.inbox, w.msg)
	}
	c.pending = nil
	return out
}

// sweep is _sweep_dead_connections (zmq.py:317-320).
func (c *memControl) sweep() {
	for _, p := range c.conns.keys() {
		if conn, _ := c.conns.get(p); !conn.Alive() {
			c.conns.pop(p)
			conn.Close()
		}
	}
}

func (c *memControl) Close() {
	if c.closedState {
		return
	}
	c.closedState = true
	for _, conn := range c.conns.values() {
		conn.Close()
	}
	c.conns.clear()
	if c.net.routers[c.local] == c {
		delete(c.net.routers, c.local)
	}
}

// agentName is the NIXL agent name of the instance at local. vLLM names it with a uuid4 that only
// travels inside the agent metadata (p2p/manager.py:349); any unique name serves.
func agentName(local string) string { return "nixl:" + local }

// memXfer is one WRITE on the agent: its local and remote block indexes, and its state (PROC until
// the host ends it, then DONE or ERR).
type memXfer struct {
	agent, peer   string
	local, remote []int
	state         string
}

// memReg is a registration future (data/nixl.py:235-255): it publishes the peer's handles unless
// the peer was removed or re-registered meanwhile.
type memReg struct {
	d          *memData
	peer, name string
	gen        int
	done       bool
	err        error
	cancelled  bool
}

func (r *memReg) Done() bool { return r.done }
func (r *memReg) Err() error { return r.err }
func (r *memReg) Cancel()    { r.cancelled = true }
func (r *memReg) run() {
	if r.done {
		return
	}
	r.done = true
	if r.d.peerGen[r.peer] != r.gen {
		r.err = fmt.Errorf("registration for peer %s was superseded (peer removed or re-registered while registering)", r.peer)
		return
	}
	r.d.remote[r.peer] = r.name
}

// memInflight is _Inflight (data/nixl.py:101-112). Its struck flag marks a transfer whose state
// check raised; the in-memory agent never raises, so no transfer is struck.
type memInflight struct {
	peer string
	x    *memXfer
}

// memData is NixlTransport (data/nixl.py:115-665) over the network's in-memory agent: the
// transport's own bookkeeping (transfer ids, peer generations, the three-strike peer fault) is
// vLLM's, and the agent's state checks are the oracle's (fake_nixl). The agent never raises, so
// the state-check faults of :619-633 do not arise.
type memData struct {
	net         *P2PNetwork
	local       string
	numBlocks   int
	blockLen    int64
	fingerprint string
	remote      map[string]string // peer → its agent's name (the remote dlist)
	inflight    omap[int, *memInflight]
	nextID      int
	strikes     map[string]int
	faults      map[string]string
	peerGen     map[string]int
}

// p2pMaxTransferFailures is _MAX_CONSECUTIVE_TRANSFER_FAILURES (data/nixl.py:64).
const p2pMaxTransferFailures = 3

// Data is the data transport of the instance at local, over a primary region of numBlocks blocks
// of blockLen bytes. fingerprint is the config fingerprint its handshake advertises.
func (n *P2PNetwork) Data(local string, numBlocks int, blockLen int64, fingerprint string) P2PData {
	return &memData{net: n, local: local, numBlocks: numBlocks, blockLen: blockLen, fingerprint: fingerprint,
		remote: map[string]string{}, strikes: map[string]int{}, faults: map[string]string{}, peerGen: map[string]int{}}
}

func (d *memData) AgentMetadata() []byte { return []byte(agentName(d.local)) }
func (d *memData) BaseAddr() int64       { return 0 }
func (d *memData) NumBlocks() int        { return d.numBlocks }
func (d *memData) BlockLen() int64       { return d.blockLen }
func (d *memData) Fingerprint() string   { return d.fingerprint }
func (d *memData) Close()                {}

func (d *memData) SetDraining(on bool) {
	if on {
		d.net.draining++
	} else {
		d.net.draining--
	}
}

// AddRemotePeerAsync is add_remote_peer_async (data/nixl.py:235-255); it resolves when the host
// resolves registrations.
func (d *memData) AddRemotePeerAsync(peer string, metadata []byte, _ int64, _ int, _ int64) P2PRegistration {
	d.peerGen[peer]++
	r := &memReg{d: d, peer: peer, name: string(metadata), gen: d.peerGen[peer]}
	// The registration runs on the transport's worker thread; the oracle's agent resolves it at the
	// driver's transport rule after the call, never in the poll that started it.
	d.net.regs = append(d.net.regs, r)
	return r
}

// RemoveRemotePeer is remove_remote_peer (data/nixl.py:257-267).
func (d *memData) RemoveRemotePeer(peer string) {
	d.peerGen[peer]++
	delete(d.remote, peer)
	delete(d.strikes, peer)
	delete(d.faults, peer)
}

// WriteBlocks is write_blocks (data/nixl.py:372-423).
func (d *memData) WriteBlocks(peer string, local, remote []int) (int, bool) {
	name, ok := d.remote[peer]
	if !ok {
		return 0, false
	}
	x := &memXfer{agent: agentName(d.local), peer: name, local: append([]int(nil), local...),
		remote: append([]int(nil), remote...), state: "PROC"}
	d.net.xfers = append(d.net.xfers, x)
	id := d.nextID
	d.nextID++
	d.inflight.set(id, &memInflight{peer: peer, x: x})
	return id, true
}

// state is check_xfer_state: inside drain_jobs a WRITE still in flight is ended first.
func (d *memData) state(x *memXfer) string {
	if x.state == "PROC" && d.net.draining > 0 {
		d.net.complete(true)
	}
	return x.state
}

// release is release_xfer_handle: a released WRITE no longer runs.
func (d *memData) release(x *memXfer) {
	for i, y := range d.net.xfers {
		if y == x {
			d.net.xfers = append(d.net.xfers[:i:i], d.net.xfers[i+1:]...)
			return
		}
	}
}

// Poll is poll (data/nixl.py:425-501), scoped to one peer.
func (d *memData) Poll(peer string) P2PPollResult {
	if d.inflight.len() == 0 {
		return P2PPollResult{PeerFault: d.faults[peer]}
	}
	var done, failed []int
	succeeded := map[string]bool{}
	type failure struct{ peer, cause string }
	var failures []failure
	for _, tid := range d.inflight.keys() {
		e, _ := d.inflight.get(tid)
		if e.peer != peer {
			continue
		}
		switch st := d.state(e.x); st {
		case "DONE":
			done = append(done, tid)
			succeeded[e.peer] = true
		case "PROC", "PEND":
		default:
			failed = append(failed, tid)
			failures = append(failures, failure{e.peer, "state " + st})
		}
	}
	for p := range succeeded {
		delete(d.strikes, p)
	}
	for _, f := range failures {
		d.strikes[f.peer]++
		if d.strikes[f.peer] >= p2pMaxTransferFailures {
			d.setFault(f.peer, fmt.Sprintf("%d consecutive transfer failures (last: %s)", d.strikes[f.peer], f.cause))
		}
	}
	for _, tid := range append(append([]int(nil), done...), failed...) {
		e, _ := d.inflight.pop(tid)
		d.release(e.x)
	}
	return P2PPollResult{Done: done, Failed: failed, PeerFault: d.faults[peer]}
}

func (d *memData) setFault(peer, cause string) {
	if _, ok := d.faults[peer]; !ok {
		d.faults[peer] = cause
	}
}

// Cancel is cancel (data/nixl.py:518-583).
func (d *memData) Cancel(ids []int, wait bool) []int {
	if !wait {
		for _, tid := range ids {
			if e, ok := d.inflight.pop(tid); ok {
				d.release(e.x)
			}
		}
		return nil
	}
	var still []int
	for _, tid := range ids {
		e, ok := d.inflight.get(tid)
		if !ok {
			continue
		}
		if st := d.state(e.x); st == "PROC" || st == "PEND" {
			still = append(still, tid)
			continue
		}
		d.release(e.x)
		d.inflight.pop(tid)
	}
	return still
}
