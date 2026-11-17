// Original to blis-vllm.

package vllm

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/engine"
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"

	kvtier "github.com/inference-sim/blis-vllm/core/connector/tiering"
)

// The p2p secondary tier through engine.Engine (blis-vllm#204).
//
// kvtier.P2PTier decides: which peers to talk to, which blocks a peer fetches, which WRITE moves
// them. It moves nothing and reads no clock. Here the engine gives it the two transports it
// needs, wired to the simulator contract instead of to the oracle's in-memory network:
//
//   - the control transport turns each message the tier sends into an engine.PeerMessage in the
//     engine's outbox (the simulator delivers it to the engine at the address it names, later),
//     and turns each message Deliver hands in into one the tier receives at its next poll;
//   - the data transport reports each WRITE the tier submits as an engine.Transfer of Kind store
//     and Move.Kind plan.TransferPeer, from this engine's endpoint to the peer's (PlacePeer), and
//     ends it when the simulator reports the transfer done.
//
// Both have the semantics of the oracle's host (kvtier.P2PNetwork); only the wiring differs.
//
// Time. The tier's deadlines read a clock the engine sets from the instant of every call that
// carries one (Add, Schedule, Complete, TransferDone, Finish, Deliver); it never goes back, and no
// call reads the time package or starts a goroutine. vLLM's tier-poll thread (a round of
// Manager.poll_tiers every tier_poll_interval_s, 1 ms unstated; tiering/manager.py:65-70,
// 312-326@4ba787ec) is the engine's poll: Schedule runs a round when the interval has passed since
// the last, and NextWake asks the simulator for the next one.
//
// Substitutions, named (blis-vllm#204):
//
//   - A message's delivery latency is the simulator's, as is the WRITE's duration (priced from its
//     Move); the engine learns that a peer died only through the session's own timeouts.
//   - vLLM's p2p has_pending_work is always true, which would make a simulator tick forever. The
//     engine asks for poll wakes only while the tier has work (kvtier.P2PTier.Busy) or the
//     transports hold a message or a WRITE. A session that is only idle is not polled until the
//     next message or call, so its idle reap and the like are observed at that call rather than on
//     the poll thread's schedule. No wake is asked for while the batch queue is full: the
//     simulator steps the engine at the Complete that frees it, and the round due runs there.
//   - A reset_cache's drain (DrainJobs) ends every WRITE at once, as the oracle's agent does.
//   - The side channel's host, port and the VLLM_P2P_* limits come from the engine's environment
//     (the engine's env: block) and otherwise vLLM's defaults.
//
// A scenario with more than one instance states each one's port (VLLM_P2P_SIDE_CHANNEL_PORT or the
// tier's port), as vLLM's operator does.

// p2pIssue is the p2p tier's engine side.
const p2pIssue = issueBase + "204"

// p2pKeys are the p2p tier's keys in secondary_tiers (P2PSecondaryTierManager.__init__,
// tiering/p2p/manager.py:232-246@4ba787ec). type is the factory's (tiering/factory.py:59-63).
// backpressure is refused by name, not read. locality is not a p2p key.
var p2pKeys = map[string]bool{"type": true, "host": true, "port": true, "backends": true, "num_threads": true,
	"unbound_store_timeout_s": true}

// p2pEnvNames are the engine environment variables the p2p tier reads (vllm/envs.py:234-238@4ba787ec).
var p2pEnvNames = []string{"VLLM_P2P_SIDE_CHANNEL_HOST", "VLLM_P2P_SIDE_CHANNEL_PORT", "VLLM_P2P_MAX_PEERS",
	"VLLM_P2P_HANDSHAKE_TIMEOUT_S", "VLLM_P2P_IDLE_TIMEOUT_S"}

// defaultTierPollUs is DEFAULT_TIER_POLL_INTERVAL_S, 1 ms (tiering/manager.py:70@4ba787ec).
const defaultTierPollUs = engine.Duration(1000)

// p2pPeer is the engine's end of a p2p tier: its address, its clock, the two transports, and
// the poll schedule.
type p2pPeer struct {
	addr string
	tier *kvtier.P2PTier
	// mgr is the manager that polls the tier; set when the connector builds it.
	mgr  func() *kvtier.Manager
	ctl  *p2pControl
	data *p2pData
	// placed are the peers' endpoints (PlacePeer), by address.
	placed map[string]plan.Endpoint

	// clock is the latest instant any call carried, in microseconds.
	clock engine.Time
	// every is tier_poll_interval_s as a duration; 0 is no poll thread. nextPoll is the instant of
	// the next round.
	every    engine.Duration
	nextPoll engine.Time
	seq      uint64
	out      []engine.PeerMessage
}

// nowS is the tier's clock, in seconds.
func (p *p2pPeer) nowS() float64 { return float64(p.clock) / 1e6 }

// advance sets the clock to a call's instant; it never goes back.
func (p *p2pPeer) advance(now engine.Time) { p.clock = max(p.clock, now) }

// emit queues a control message for the simulator.
func (p *p2pPeer) emit(to string, payload []byte) {
	p.seq++
	p.out = append(p.out, engine.PeerMessage{From: p.addr, To: to, Seq: p.seq, Bytes: int64(len(payload)), Payload: payload})
}

// outbox returns the messages sent since the last call, and forgets them.
func (p *p2pPeer) outbox() []engine.PeerMessage {
	out := p.out
	p.out = nil
	return out
}

// round is one round of the tier-poll thread, if one is due at the clock: the registrations a
// previous call started resolve first (the oracle resolves them between calls), then
// Manager.poll_tiers.
func (p *p2pPeer) round() {
	p.data.resolve()
	if p.every <= 0 || p.clock < p.nextPoll {
		return
	}
	p.mgr().PollTiers()
	p.nextPoll = p.clock + p.every
}

// arm schedules the next round when the tier has work and no round is pending: the thread, asleep,
// looks within one interval.
func (p *p2pPeer) arm() {
	if p.every > 0 && p.nextPoll <= p.clock && p.busy() {
		p.nextPoll = p.clock + p.every
	}
}

// busy reports whether a round would advance anything: the tier's work, a message waiting for a
// poll, or a WRITE in flight or ended and unpolled.
func (p *p2pPeer) busy() bool {
	return p.tier.Busy() || len(p.out) > 0 || len(p.ctl.inbox) > 0 || len(p.ctl.pending) > 0 || p.data.busy()
}

// wake is the instant the engine asks to be stepped for the tier, if any.
func (p *p2pPeer) wake(now engine.Time) (engine.Time, bool) {
	if p.every <= 0 || !p.busy() {
		return 0, false
	}
	return max(p.nextPoll, now+1), true
}

// peer is the p2p tier of the set; nil with none.
func (ts *tierSet) peer() *p2pPeer {
	if ts == nil {
		return nil
	}
	for _, b := range ts.tiers {
		if b.p2p != nil {
			return b.p2p
		}
	}
	return nil
}

// peer is the engine's p2p tier; nil with none (or without an offload).
func (o *offload) peer() *p2pPeer {
	if o == nil {
		return nil
	}
	return o.tiers.peer()
}

// buildP2P builds the p2p tier of secondary_tiers[i]. chunks is the CPU tier's chunks, the
// blocks the data plane registers; env the engine's environment; the poll interval is the spec's.
func buildP2P(i int, raw map[string]any, topBackpressure map[string]any, chunkBytes int64, chunks int, ns kvtier.Namespace,
	c *offloadConfig) (*builtTier, error) {
	at := func(format string, a ...any) error {
		return fmt.Errorf("vllm: kv_offload secondary_tiers[%d] (p2p): "+format, append([]any{i}, a...)...)
	}
	for _, k := range sortedAnyKeys(raw) {
		if k == "backpressure" {
			continue
		}
		if !p2pKeys[k] {
			// vLLM passes every other key to P2PSecondaryTierManager, whose **kwargs are reserved
			// (tiering/p2p/manager.py:232-246); the engine refuses what it cannot say it models.
			return nil, at("%q is not a key of the p2p tier (p2p/manager.py:232-246@4ba787ec)", k)
		}
	}
	if raw["backpressure"] != nil || topBackpressure != nil {
		// The p2p tier itself refuses a detector (p2p/manager.py:291-304@4ba787ec).
		return nil, at("store backpressure is not supported for the p2p tier: vLLM refuses it (p2p/manager.py:291-304@4ba787ec; %s)",
			issueBase+"13")
	}
	env := c.env
	host := "localhost"
	if v, ok := env["VLLM_P2P_SIDE_CHANNEL_HOST"]; ok {
		host = v
	}
	if v, stated := raw["host"]; stated {
		s, ok := v.(string)
		if !ok || s == "" {
			return nil, at("host %v must be a non-empty string", v)
		}
		host = s
	}
	port := 5710
	if v, ok := env["VLLM_P2P_SIDE_CHANNEL_PORT"]; ok {
		n, err := strconv.Atoi(v)
		if err != nil {
			return nil, at("env VLLM_P2P_SIDE_CHANNEL_PORT %q must be a whole number", v)
		}
		port = n
	}
	if v, stated := raw["port"]; stated {
		n, ok := v.(int)
		if !ok {
			return nil, at("port %v must be a whole number", v)
		}
		port = n
	}
	// The bound port is the base plus the data-parallel index (p2p/manager.py:325-333@4ba787ec);
	// an engine with an offload is one rank (config.go), index 0.
	if port < 1 || port > 65535 {
		return nil, at("port %d must be in 1..65535", port)
	}
	if v, stated := raw["backends"]; stated && v != nil {
		list, ok := v.([]any)
		if !ok {
			return nil, at("backends %v must be a list of backend names", v)
		}
		for _, b := range list {
			if s, _ := b.(string); s != "UCX" {
				// The data plane is the simulator's: it prices the WRITE from the move, over the
				// fabric the scenario states, whatever NIXL backend would carry it. Only UCX, NIXL's
				// default, is the data plane the tier is diff-tested on.
				return nil, at("backend %v is not modelled: the data plane is the simulator's, and only the UCX backend (NIXL's "+
					"default) is the one the tier is diff-tested on (%s)", b, p2pIssue)
			}
		}
	}
	if v, stated := raw["num_threads"]; stated {
		// The transport's thread count changes how fast the bytes move, which the simulator's
		// fabric prices, not what the tier decides.
		if n, ok := v.(int); !ok || n < 1 {
			return nil, at("num_threads %v must be a positive whole number", v)
		}
	}
	cfg := kvtier.P2PConfig{LocalID: host + ":" + strconv.Itoa(port), BlockBytes: chunkBytes}
	if v, stated := raw["unbound_store_timeout_s"]; stated {
		var f float64
		switch x := v.(type) {
		case int:
			f = float64(x)
		case float64:
			f = x
		default:
			return nil, at("unbound_store_timeout_s %v must be a positive number", v)
		}
		if !(f > 0) || math.IsInf(f, 0) {
			return nil, at("unbound_store_timeout_s %v must be a positive number (p2p/manager.py:309-313@4ba787ec)", v)
		}
		cfg.UnboundStoreTimeoutS = f
	}
	for _, e := range []struct {
		name string
		into *int
		off  *bool
	}{{"VLLM_P2P_MAX_PEERS", &cfg.MaxPeers, nil}, {"VLLM_P2P_HANDSHAKE_TIMEOUT_S", &cfg.HandshakeTimeoutS, nil},
		{"VLLM_P2P_IDLE_TIMEOUT_S", &cfg.IdleTimeoutS, &cfg.IdleReapOff}} {
		v, ok := env[e.name]
		if !ok {
			continue
		}
		n, err := strconv.Atoi(v)
		if err != nil {
			return nil, at("env %s %q must be a whole number", e.name, v)
		}
		if e.off != nil && n <= 0 {
			*e.off = true
			continue
		}
		if n <= 0 {
			return nil, at("env %s %d must be positive", e.name, n)
		}
		*e.into = n
	}
	cfg.HashSeed = env["PYTHONHASHSEED"]
	every := defaultTierPollUs
	if s := c.TierPollIntervalS; s != nil {
		every = 0
		if *s > 0 {
			every = max(engine.Duration(math.Round(*s*1e6)), 1)
		}
	}

	p := &p2pPeer{addr: cfg.LocalID, placed: map[string]plan.Endpoint{}, every: every}
	cfg.Now = p.nowS
	p.ctl = &p2pControl{p: p, conns: map[string]*p2pConn{}}
	// The fingerprint is the config the handshake compares (FileMapper's run config): the layout
	// the namespace states.
	p.data = &p2pData{p: p, numBlocks: chunks, blockLen: chunkBytes, fingerprint: fmt.Sprintf("%+v", ns),
		remote: map[string]bool{}, strikes: map[string]int{}, faults: map[string]string{}, peerGen: map[string]int{}}
	p.tier = kvtier.NewP2PTier(cfg, p.ctl, p.data)
	p.data.tier = p.tier
	return &builtTier{label: fmt.Sprintf("%d:p2p", i+1), tier: p.tier, host: p.data, p2p: p}, nil
}

// PlacePeer states where the engine at addr runs: the endpoint the WRITEs to it end at. It is the
// contract's PeerPlaced (inference-sim/blis-substrate#266, d213eca6), which the pinned substrate does
// not yet declare; Go interfaces are structural, so a simulator that looks for PeerPlaced finds it.
// The simulator calls it for every other Peered engine before the first Schedule; it may repeat for
// an address, and the last wins. A negative node or rack is an error. An engine with no p2p tier
// takes the endpoint and uses none.
func (e *Engine) PlacePeer(addr string, at plan.Endpoint) error {
	if addr == "" {
		return errors.New("vllm: PlacePeer: no address")
	}
	if at.Node < 0 || at.Rack < 0 {
		return fmt.Errorf("vllm: PlacePeer %s: endpoint node %d rack %d is negative", addr, at.Node, at.Rack)
	}
	if p := e.off.peer(); p != nil {
		p.placed[addr] = at
	}
	return nil
}

// PeerAddr implements engine.Peered: the address the p2p tier's peers dial; "" with no p2p tier.
func (e *Engine) PeerAddr() string {
	if p := e.off.peer(); p != nil {
		return p.addr
	}
	return ""
}

// Outbox implements engine.Peered: the control messages the tier has sent since the last call.
func (e *Engine) Outbox() []engine.PeerMessage {
	if p := e.off.peer(); p != nil {
		return p.outbox()
	}
	return nil
}

// Deliver implements engine.Peered: a peer's control message reaches the tier, which reads it at
// its next poll.
func (e *Engine) Deliver(m engine.PeerMessage, now engine.Time) error {
	p := e.off.peer()
	if p == nil {
		return fmt.Errorf("vllm: peer message %d from %q: the engine has no p2p tier", m.Seq, m.From)
	}
	p.advance(now)
	defer p.arm()
	if m.To != p.addr {
		return fmt.Errorf("vllm: peer message %d is addressed to %q, not this engine's %q", m.Seq, m.To, p.addr)
	}
	if err := m.Validate(); err != nil {
		return err
	}
	msg, err := decodeP2PMsg(m.Payload)
	if err != nil {
		return fmt.Errorf("vllm: peer message %d from %q: %w", m.Seq, m.From, err)
	}
	p.ctl.inbox = append(p.ctl.inbox, wiredMsg{ident: m.From, msg: msg})
	return nil
}

var _ engine.Peered = (*Engine)(nil)

// ---- the control transport ----

// wireMsg is a P2PMsg as it travels in a PeerMessage's payload: JSON in field order, keys in hex.
type wireMsg struct {
	Type          string   `json:"type"`
	PeerID        string   `json:"peer_id,omitempty"`
	AgentMetadata []byte   `json:"agent_metadata,omitempty"`
	BaseAddr      int64    `json:"base_addr,omitempty"`
	NumBlocks     int      `json:"num_blocks,omitempty"`
	BlockLen      int64    `json:"block_len,omitempty"`
	Fingerprint   string   `json:"fingerprint,omitempty"`
	HashSeed      string   `json:"hash_seed,omitempty"`
	KVRequestID   string   `json:"kv_request_id,omitempty"`
	Keys          []string `json:"keys,omitempty"`
	BlockIndexes  []int    `json:"block_indexes,omitempty"`
	RoundSeq      int      `json:"round_seq,omitempty"`
	Hits          []bool   `json:"hits,omitempty"`
	Success       bool     `json:"success,omitempty"`
}

var p2pMsgTypes = map[string]bool{"connect": true, "connect_ack": true, "disconnect": true, "fetch": true, "lookup": true,
	"lookup_resp": true, "transfer_done": true, "abort_fetch": true, "abort_ack": true}

// encodeP2PMsg is the payload of a control message.
func encodeP2PMsg(m kvtier.P2PMsg) []byte {
	w := wireMsg{Type: m.Type, PeerID: m.PeerID, AgentMetadata: m.AgentMetadata, BaseAddr: m.BaseAddr, NumBlocks: m.NumBlocks,
		BlockLen: m.BlockLen, Fingerprint: m.Fingerprint, HashSeed: m.HashSeed, KVRequestID: m.KVRequestID,
		BlockIndexes: m.BlockIndexes, RoundSeq: m.RoundSeq, Hits: m.Hits, Success: m.Success}
	for _, k := range m.Keys {
		w.Keys = append(w.Keys, hex.EncodeToString([]byte(k)))
	}
	b, err := json.Marshal(w)
	if err != nil {
		panic(fmt.Sprintf("vllm: encoding a p2p control message: %v", err)) // the fields are plain values
	}
	return b
}

// decodeP2PMsg reads a payload encodeP2PMsg wrote; anything else is an error.
func decodeP2PMsg(b []byte) (kvtier.P2PMsg, error) {
	var w wireMsg
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&w); err != nil {
		return kvtier.P2PMsg{}, fmt.Errorf("malformed p2p control message: %w", err)
	}
	if dec.More() {
		return kvtier.P2PMsg{}, errors.New("malformed p2p control message: trailing data")
	}
	if !p2pMsgTypes[w.Type] {
		return kvtier.P2PMsg{}, fmt.Errorf("malformed p2p control message: type %q", w.Type)
	}
	m := kvtier.P2PMsg{Type: w.Type, PeerID: w.PeerID, AgentMetadata: w.AgentMetadata, BaseAddr: w.BaseAddr, NumBlocks: w.NumBlocks,
		BlockLen: w.BlockLen, Fingerprint: w.Fingerprint, HashSeed: w.HashSeed, KVRequestID: w.KVRequestID,
		BlockIndexes: w.BlockIndexes, RoundSeq: w.RoundSeq, Hits: w.Hits, Success: w.Success}
	for _, k := range w.Keys {
		raw, err := hex.DecodeString(k)
		if err != nil || len(raw) < 4 {
			return kvtier.P2PMsg{}, fmt.Errorf("malformed p2p control message: key %q", k)
		}
		m.Keys = append(m.Keys, kvtier.Key(raw))
	}
	return m, nil
}

type wiredMsg struct {
	ident string
	msg   kvtier.P2PMsg
}

// p2pControl is the tier's control transport (kvtier.P2PControl) over the engine's outbox and
// Deliver. It behaves as the oracle's: a message from a peer with no connection opens an inbound
// connection at the next Poll, and a dead connection is swept.
type p2pControl struct {
	p       *p2pPeer
	conns   map[string]*p2pConn
	inbox   []wiredMsg
	pending []wiredMsg
	closed  bool
}

// p2pConn is the connection to one peer.
type p2pConn struct {
	c            *p2pControl
	peer         string
	dead, closed bool
	in           []kvtier.P2PMsg
}

func (c *p2pConn) PeerID() string { return c.peer }
func (c *p2pConn) Alive() bool    { return !c.dead && !c.closed }
func (c *p2pConn) MarkDead()      { c.dead = true }
func (c *p2pConn) Close()         { c.closed, c.dead = true, true }

func (c *p2pConn) Send(m kvtier.P2PMsg) error {
	if !c.Alive() {
		return errors.New("send on closed connection to " + c.peer)
	}
	c.c.p.emit(c.peer, encodeP2PMsg(m))
	return nil
}

func (c *p2pConn) Recv() []kvtier.P2PMsg {
	out := c.in
	c.in = nil
	return out
}

// Connect opens an outbound connection to peer; a dead connection to it is retired first.
func (c *p2pControl) Connect(peer string) (kvtier.P2PConn, error) {
	if peer == c.p.addr {
		return nil, fmt.Errorf("p2p connection to %s, this engine's own address", peer)
	}
	if old, ok := c.conns[peer]; ok {
		if old.Alive() {
			return nil, fmt.Errorf("p2p connection to %s already exists", peer)
		}
		delete(c.conns, peer)
		old.Close()
	}
	return c.open(peer), nil
}

func (c *p2pControl) open(peer string) *p2pConn {
	conn := &p2pConn{c: c, peer: peer}
	c.conns[peer] = conn
	return conn
}

// peers are the connected peers in address order.
func (c *p2pControl) peers() []string {
	out := make([]string, 0, len(c.conns))
	for p := range c.conns {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

func (c *p2pControl) sweep() {
	for _, p := range c.peers() {
		if conn := c.conns[p]; !conn.Alive() {
			delete(c.conns, p)
			conn.Close()
		}
	}
}

// Poll routes the messages that arrived to their connections and returns the connections they
// opened, each with its first messages buffered.
func (c *p2pControl) Poll() []kvtier.P2PConn {
	c.sweep()
	for _, w := range c.inbox {
		if conn, ok := c.conns[w.ident]; ok {
			conn.in = append(conn.in, w.msg)
		} else {
			c.pending = append(c.pending, w)
		}
	}
	c.inbox = nil
	c.sweep()
	var out []kvtier.P2PConn
	for _, w := range c.pending {
		conn, ok := c.conns[w.ident]
		if !ok {
			conn = c.open(w.ident)
			out = append(out, conn)
		}
		conn.in = append(conn.in, w.msg)
	}
	c.pending = nil
	return out
}

func (c *p2pControl) Close() {
	if c.closed {
		return
	}
	c.closed = true
	for _, conn := range c.conns {
		conn.Close()
	}
	c.conns = map[string]*p2pConn{}
}

// ---- the data transport ----

// p2pWrite is one WRITE: the local chunks and the peer's they go to. It is in flight (PROC) until
// the simulator reports its transfer done (DONE) or failed (ERR), and leaves when polled.
type p2pWrite struct {
	id            int
	peer          string
	local, remote []int
	state         string
}

// p2pReg is a peer registration: it publishes the peer's handles at the next round unless the peer
// was removed or registered again meanwhile.
type p2pReg struct {
	d    *p2pData
	peer string
	gen  int
	done bool
	err  error
}

func (r *p2pReg) Done() bool { return r.done }
func (r *p2pReg) Err() error { return r.err }
func (r *p2pReg) Cancel()    { r.gen = -1 }
func (r *p2pReg) run() {
	r.done = true
	switch {
	case r.gen < 0:
		r.err = fmt.Errorf("registration for peer %s was cancelled", r.peer)
	case r.d.peerGen[r.peer] != r.gen:
		r.err = fmt.Errorf("registration for peer %s was superseded (peer removed or re-registered while registering)", r.peer)
	default:
		r.d.remote[r.peer] = true
	}
}

// p2pData is the tier's data transport (kvtier.P2PData), and the tier's host (hostTier): its
// in-flight WRITEs are the jobs the engine reports as transfers. The transport's own bookkeeping
// (transfer ids, peer generations, the three-strike peer fault) is vLLM's (data/nixl.py), as in the
// oracle's host.
type p2pData struct {
	p           *p2pPeer
	tier        *kvtier.P2PTier
	numBlocks   int
	blockLen    int64
	fingerprint string
	remote      map[string]bool
	regs        []*p2pReg
	writes      []*p2pWrite
	nextID      int
	strikes     map[string]int
	faults      map[string]string
	peerGen     map[string]int
	draining    int
}

func (d *p2pData) AgentMetadata() []byte { return []byte("agent:" + d.p.addr) }
func (d *p2pData) BaseAddr() int64       { return 0 }
func (d *p2pData) NumBlocks() int        { return d.numBlocks }
func (d *p2pData) BlockLen() int64       { return d.blockLen }
func (d *p2pData) Fingerprint() string   { return d.fingerprint }
func (d *p2pData) Close()                {}

func (d *p2pData) SetDraining(on bool) {
	if on {
		d.draining++
	} else {
		d.draining--
	}
}

// busy reports a registration pending, or a WRITE in flight or ended and not yet polled.
func (d *p2pData) busy() bool { return len(d.regs) > 0 || len(d.writes) > 0 }

// resolve ends the registrations started before this call.
func (d *p2pData) resolve() {
	regs := d.regs
	d.regs = nil
	for _, r := range regs {
		r.run()
	}
}

// AddRemotePeerAsync starts a registration, which resolves at the next round.
func (d *p2pData) AddRemotePeerAsync(peer string, _ []byte, _ int64, _ int, _ int64) kvtier.P2PRegistration {
	d.peerGen[peer]++
	r := &p2pReg{d: d, peer: peer, gen: d.peerGen[peer]}
	d.regs = append(d.regs, r)
	return r
}

func (d *p2pData) RemoveRemotePeer(peer string) {
	d.peerGen[peer]++
	delete(d.remote, peer)
	delete(d.strikes, peer)
	delete(d.faults, peer)
}

// WriteBlocks submits one WRITE; false, with no WRITE, for a peer that is not registered or was
// never placed (PlacePeer): the engine could not say where the bytes go.
func (d *p2pData) WriteBlocks(peer string, local, remote []int) (int, bool) {
	if _, placed := d.p.placed[peer]; !placed || !d.remote[peer] {
		return 0, false
	}
	w := &p2pWrite{id: d.nextID, peer: peer, local: append([]int(nil), local...), remote: append([]int(nil), remote...), state: "PROC"}
	d.nextID++
	d.writes = append(d.writes, w)
	return w.id, true
}

// stateOf is check_xfer_state: inside drain_jobs a WRITE in flight ends at once (the oracle's
// agent does the same).
func (d *p2pData) stateOf(w *p2pWrite) string {
	if w.state == "PROC" && d.draining > 0 {
		for _, x := range d.writes {
			if x.state == "PROC" {
				x.state = "DONE"
			}
		}
	}
	return w.state
}

func (d *p2pData) drop(w *p2pWrite) {
	for i, x := range d.writes {
		if x == w {
			d.writes = append(d.writes[:i:i], d.writes[i+1:]...)
			return
		}
	}
}

func (d *p2pData) find(id int) *p2pWrite {
	for _, w := range d.writes {
		if w.id == id {
			return w
		}
	}
	return nil
}

// Poll reports the peer's WRITEs that ended since the last poll.
func (d *p2pData) Poll(peer string) kvtier.P2PPollResult {
	var done, failed []int
	var ended []*p2pWrite
	for _, w := range append([]*p2pWrite(nil), d.writes...) {
		if w.peer != peer {
			continue
		}
		switch d.stateOf(w) {
		case "DONE":
			done = append(done, w.id)
			ended = append(ended, w)
		case "PROC", "PEND":
		default:
			failed = append(failed, w.id)
			ended = append(ended, w)
		}
	}
	if len(done) > 0 {
		delete(d.strikes, peer)
	}
	for range failed {
		d.strikes[peer]++
		if d.strikes[peer] >= 3 { // _MAX_CONSECUTIVE_TRANSFER_FAILURES (data/nixl.py:64)
			if _, ok := d.faults[peer]; !ok {
				d.faults[peer] = fmt.Sprintf("%d consecutive transfer failures (last: state ERR)", d.strikes[peer])
			}
		}
	}
	for _, w := range ended {
		d.drop(w)
	}
	return kvtier.P2PPollResult{Done: done, Failed: failed, PeerFault: d.faults[peer]}
}

// Cancel is cancel(ids, mode).
func (d *p2pData) Cancel(ids []int, wait bool) []int {
	if !wait {
		for _, id := range ids {
			if w := d.find(id); w != nil {
				d.drop(w)
			}
		}
		return nil
	}
	var still []int
	for _, id := range ids {
		w := d.find(id)
		if w == nil {
			continue
		}
		if st := d.stateOf(w); st == "PROC" || st == "PEND" {
			still = append(still, id)
			continue
		}
		d.drop(w)
	}
	return still
}

// Jobs lists the WRITEs in flight, in submission order, as the engine reports them (hostTier): a
// store of the WRITE's chunks, for the request whose fetch they serve.
func (d *p2pData) Jobs() []kvtier.FSJob {
	var out []kvtier.FSJob
	for _, w := range d.writes {
		if w.state != "PROC" {
			continue
		}
		req, ok := d.tier.WriteRequest(w.peer, w.id)
		if !ok {
			req = w.peer
		}
		out = append(out, kvtier.FSJob{ID: w.id, Store: true, Chunks: len(w.local), Req: req})
	}
	return out
}

// Run ends a WRITE whose transfer the simulator reports done; a WRITE already settled, cancelled or
// drained is none.
func (d *p2pData) Run(id int) {
	if w := d.find(id); w != nil && w.state == "PROC" {
		w.state = "DONE"
	}
}

// Fail ends a WRITE whose transfer the simulator reports failed; the tier's three-strike rule
// applies at the next poll.
func (d *p2pData) Fail(id int) {
	if w := d.find(id); w != nil && w.state == "PROC" {
		w.state = "ERR"
	}
}

// PeerEnd is the endpoint the WRITE's bytes end at.
func (d *p2pData) PeerEnd(id int) plan.Endpoint {
	if w := d.find(id); w != nil {
		return d.p.placed[w.peer]
	}
	return plan.Endpoint{}
}

// ResolveLookups is the fs and obj tiers' lookup worker, which the p2p tier does not have.
func (d *p2pData) ResolveLookups() {}

// PeerWriteBytes is the bytes a WRITE to a peer moves, for the simulator to price it: its chunks of
// the CPU tier, each the primary view's stride over every worker's copy, which is the region
// NixlTransport registers and writes from (tiering/p2p/data/nixl.py:171-213@4ba787ec9959e566d3b587a796dff07673324da5). False for a transfer that is
// not a WRITE to a peer. A simulator reads it by type-asserting
// interface{ PeerWriteBytes(engine.Transfer) (int64, bool) } on the engine.Engine.
func (e *Engine) PeerWriteBytes(t engine.Transfer) (int64, bool) {
	if !e.off.isPeerWrite(t.ID) {
		return 0, false
	}
	return int64(t.Blocks/e.off.tiers.blocksPerChunk) * e.off.tiers.chunkBytes, true
}

// SetPeerAddr gives the engine the address its peers reach it at, replacing the one its p2p tier
// block states, before it has a session. Every instance of a pool is built from one engine block, so
// each would state the block's host and port; a host that runs several assigns each its own, as a
// deployment sets VLLM_P2P_SIDE_CHANNEL_HOST to each node's address. The address is host:port with a
// port in 1-65535. A simulator reads it by type-asserting interface{ SetPeerAddr(string) error } on
// the engine.Engine, and calls it before the engine's first Schedule.
func (e *Engine) SetPeerAddr(addr string) error {
	p := e.off.peer()
	if p == nil {
		return fmt.Errorf("vllm: the engine has no p2p tier to give an address")
	}
	host, port, ok := strings.Cut(addr, ":")
	n, err := strconv.Atoi(port)
	if !ok || host == "" || err != nil || n < 1 || n > 65535 {
		return fmt.Errorf("vllm: peer address %q is not host:port with a port in 1-65535", addr)
	}
	if err := p.tier.SetLocalID(addr); err != nil {
		return err
	}
	p.addr = addr
	return nil
}
