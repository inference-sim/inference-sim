// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/tiering/p2p/manager.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).

package tiering

import (
	"fmt"
	"strconv"
)

// The tier's own deadlines (p2p/manager.py:49-85@4ba787ec), in seconds.
const (
	p2pUnboundStoreTimeoutS = 60.0
	p2pReapedIDRetentionS   = 60.0
	p2pIdleReapHeadroom     = 8
	p2pReconnectMinBackoffS = 1.0
	p2pReconnectMaxBackoffS = 60.0
	// p2pDrainRounds bounds DrainJobs. vLLM's drain_jobs polls until no session has work, which
	// waits forever on work only a peer can end (its messages are delivered between calls).
	p2pDrainRounds = 10000
)

// P2PConfig is a secondary tier of type "p2p": its keys in secondary_tiers
// (P2PSecondaryTierManager.__init__, p2p/manager.py:232-402), what the spec hands it, and the
// VLLM_P2P_* environment it reads.
type P2PConfig struct {
	// Now is time.monotonic: the virtual time, which every p2p deadline runs on.
	Now func() float64
	// LocalID is "host:port", the control identity peers dial back (:348).
	LocalID string
	// BlockBytes is the primary tier's bytes per chunk (primary_kv_view.strides[0]).
	BlockBytes int64
	// UnboundStoreTimeoutS is unbound_store_timeout_s (default 60, :241); it must be positive.
	UnboundStoreTimeoutS float64
	// HashSeed is the prefix-cache hash seed the handshake compares (get_none_hash_seed).
	HashSeed string
	// The environment (vllm/envs.py:236-238): VLLM_P2P_MAX_PEERS (128),
	// VLLM_P2P_HANDSHAKE_TIMEOUT_S (30) and VLLM_P2P_IDLE_TIMEOUT_S (300). Zero takes vLLM's default.
	MaxPeers, HandshakeTimeoutS, IdleTimeoutS int
	// IdleReapOff is VLLM_P2P_IDLE_TIMEOUT_S <= 0, which turns the idle reap off.
	IdleReapOff bool
}

// srcInfo is P2PSourceInfo (:130-136): this request fetches from a peer.
type srcInfo struct {
	kvID, peer string
	probe      bool // remote_kv_source (symmetric); false for remote_prefiller (P/D)
}

// dstInfo is P2PDestInfo (:139-148): a peer fetches this request's blocks. kvID "" is a
// remote_decoder block without an id, which still marks the request remote-decode.
type dstInfo struct{ kvID string }

func roleParams(params map[string]any, name string) (map[string]any, bool) {
	if len(params) == 0 {
		return nil, false
	}
	v, ok := params[name]
	if !ok || v == nil {
		return nil, false
	}
	m, _ := v.(map[string]any)
	return m, true
}

func truthyStr(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case float64:
		if x == 0 {
			return ""
		}
		return strconv.FormatFloat(x, 'f', -1, 64)
	case int:
		if x == 0 {
			return ""
		}
		return strconv.Itoa(x)
	}
	return ""
}

// parseSource is _parse_source (:151-169). vLLM parses kv_transfer_params once in on_new_request
// and caches the result on the request context (_annotate_req_context); the parse depends only on
// the params, so the port parses them where it reads them.
func parseSource(params map[string]any) *srcInfo {
	role, ok := roleParams(params, "remote_prefiller")
	probe := false
	if !ok {
		role, ok = roleParams(params, "remote_kv_source")
		probe = true
	}
	if !ok || len(role) == 0 {
		return nil
	}
	host, port := truthyStr(role["remote_host"]), truthyStr(role["remote_port"])
	id := truthyStr(role["kv_request_id"])
	if host == "" || port == "" || id == "" {
		return nil
	}
	return &srcInfo{kvID: id, peer: host + ":" + port, probe: probe}
}

// parseDest is _parse_dest (:172-178).
func parseDest(params map[string]any) *dstInfo {
	role, ok := roleParams(params, "remote_decoder")
	if !ok {
		return nil
	}
	return &dstInfo{kvID: truthyStr(role["kv_request_id"])}
}

// unboundBatch is _UnboundStoreBatch (:196-209): a store parked until a fetch binds its id.
type unboundBatch struct {
	job         int
	keys        []Key
	blockIDs    []int
	submittedAt float64
}

type backoff struct{ wait, retryAt float64 }

// P2PTier is P2PSecondaryTierManager (p2p/manager.py:212-1127): KV shared between instances. Each
// remote peer has one session carrying both directions. A P/D decoder's lookups HIT at once and
// its promotion fetches from the prefiller, whose stores are parked until that fetch binds them; a
// symmetric consumer probes the peer, which answers from its own tiers through the parent.
//
// The tier decides and moves no bytes: its control messages go over a P2PControl and its WRITEs
// over a P2PData, both the host's, and every deadline reads cfg.Now.
//
// vLLM iterates several sets in PYTHONHASHSEED order (the client's active loads and flush list,
// the server's serve list, a lookup's pending keys); the port iterates them in insertion order,
// and its sets of small ids in ascending order (see intSet). No sequence depends on these orders.
// Shutdown (shutdown, :1074-1127) is not ported.
type P2PTier struct {
	cfg     P2PConfig
	now     func() float64
	data    P2PData
	control P2PControl

	sessions         omap[string, *p2pSession]
	reconnectBackoff omap[string, backoff]
	dataPlaneFaults  map[string]int
	kvToSession      omap[string, *p2pSession]
	unboundStores    omap[string, []unboundBatch]
	reapedStores     omap[string, float64]
	finished         []JobResult
	failedReqIDs     map[string]bool
	failedServeCtxs  []*ReqContext
}

// NewP2PTier builds a p2p tier over the host's control and data transports.
func NewP2PTier(cfg P2PConfig, control P2PControl, data P2PData) *P2PTier {
	if !(cfg.UnboundStoreTimeoutS > 0) {
		if cfg.UnboundStoreTimeoutS != 0 {
			panic(fmt.Sprintf("tiering: unbound_store_timeout_s must be a positive number, got %v", cfg.UnboundStoreTimeoutS))
		}
		cfg.UnboundStoreTimeoutS = p2pUnboundStoreTimeoutS
	}
	if cfg.MaxPeers == 0 {
		cfg.MaxPeers = 128
	}
	if cfg.HandshakeTimeoutS == 0 {
		cfg.HandshakeTimeoutS = 30
	}
	if cfg.IdleTimeoutS == 0 && !cfg.IdleReapOff {
		cfg.IdleTimeoutS = 300
	}
	if cfg.Now == nil {
		panic("tiering: a p2p tier needs the virtual clock (P2PConfig.Now)")
	}
	return &P2PTier{cfg: cfg, now: cfg.Now, data: data, control: control, dataPlaneFaults: map[string]int{},
		failedReqIDs: map[string]bool{}}
}

func (t *P2PTier) Type() string           { return "p2p" }
func (t *P2PTier) Medium() Medium         { return MediumNone }
func (t *P2PTier) Locality() Locality     { return LocalityNone }
func (t *P2PTier) HitSource() string      { return "P2P" } // CacheHitSource.P2P (:229)
func (t *P2PTier) BlockBytes() int64      { return t.cfg.BlockBytes }
func (t *P2PTier) Backpressure() Detector { return nil }  // refused for p2p (:291-304)
func (t *P2PTier) OffThreadPolling() bool { return true } // :230
func (t *P2PTier) TakeEvents() []Event    { return nil }
func (t *P2PTier) Touch([]Key, *ReqContext) {
}

// HasPendingWork is has_pending_work (:647-653): always, so the engine keeps polling the peers.
func (t *P2PTier) HasPendingWork() bool { return true }

// Lookup is lookup (:408-435): MISS for a request that is no consumer or has failed; a symmetric
// consumer's probe RETRYs until the peer answers; a P/D decoder HITs at once.
func (t *P2PTier) Lookup(k Key, c *ReqContext) Lookup {
	src := parseSource(c.Params)
	if src == nil || t.failedReqIDs[src.kvID] {
		return Miss
	}
	if src.probe {
		s, ok := t.sessions.get(src.peer)
		if !ok {
			return Miss
		}
		switch s.registerLookup(src.kvID, k) {
		case probeHit:
			return Hit
		case probeMiss:
			return Miss
		}
		return Retry
	}
	return Hit
}

// OnNewRequest is on_new_request (:437-463): a consumer opens its session to the producer, and a
// producer for a remote decoder with an id asks for REQUEST_LEVEL.
func (t *P2PTier) OnNewRequest(c *ReqContext) Policy {
	if src := parseSource(c.Params); src != nil {
		if t.getOrCreateSession(src.peer) == nil {
			t.failedReqIDs[src.kvID] = true
		}
	}
	if dst := parseDest(c.Params); dst != nil && dst.kvID != "" {
		return RequestLevel
	}
	return ChunkLevel
}

// OnRequestFinished is on_request_finished (:465-500).
func (t *P2PTier) OnRequestFinished(c *ReqContext) {
	src, dst := parseSource(c.Params), parseDest(c.Params)
	id := ""
	if src != nil {
		id = src.kvID
	} else if dst != nil {
		id = dst.kvID
	}
	if id == "" {
		return
	}
	delete(t.failedReqIDs, id)
	if src != nil {
		if s, ok := t.sessions.get(src.peer); ok {
			s.finishRequest(id)
		}
		return
	}
	if s, ok := t.kvToSession.pop(id); ok {
		s.finishRequest(id)
	}
}

// SubmitStore is submit_store (:502-575): a store for no remote decoder succeeds at once; one for
// a reaped id fails; one whose id a fetch has bound goes to that session; the rest are parked.
func (t *P2PTier) SubmitStore(j *TransferJob) {
	if len(j.Keys) != len(j.ChunkIDs) {
		panic("tiering: a p2p store job's keys and chunk ids differ in length")
	}
	dst := parseDest(j.Req.Params)
	switch {
	case dst == nil:
		t.finished = append(t.finished, JobResult{ID: j.ID, Success: true})
		return
	case dst.kvID == "":
		t.finished = append(t.finished, JobResult{ID: j.ID})
		return
	case t.reapedStores.has(dst.kvID):
		t.finished = append(t.finished, JobResult{ID: j.ID})
		return
	}
	keys, ids := append([]Key(nil), j.Keys...), append([]int(nil), j.ChunkIDs...)
	if s, ok := t.kvToSession.get(dst.kvID); ok {
		s.addStoredBlocks(dst.kvID, keys, ids, j.ID)
		return
	}
	b, _ := t.unboundStores.get(dst.kvID)
	t.unboundStores.set(dst.kvID, append(b, unboundBatch{job: j.ID, keys: keys, blockIDs: ids, submittedAt: t.now()}))
}

// SubmitLoad is submit_load (:577-634).
func (t *P2PTier) SubmitLoad(j *TransferJob) {
	src := parseSource(j.Req.Params)
	if src == nil {
		t.finished = append(t.finished, JobResult{ID: j.ID})
		return
	}
	if len(j.Keys) == 0 {
		t.finished = append(t.finished, JobResult{ID: j.ID, Success: true})
		return
	}
	s, ok := t.sessions.get(src.peer)
	if !ok {
		t.finished = append(t.finished, JobResult{ID: j.ID})
		t.failedReqIDs[src.kvID] = true
		return
	}
	s.requestBlocks(j.ID, src.kvID, j.Keys, j.ChunkIDs)
}

// FinishedJobs is get_finished_jobs (:636-645): one polling sweep, then what has accumulated.
func (t *P2PTier) FinishedJobs() []JobResult {
	t.pollOnce()
	out := t.finished
	t.finished = nil
	return out
}

// DrainJobs is drain_jobs (:655-679): poll until no session has a load or WRITE outstanding.
// Inside it the host ends each WRITE whose state is checked (P2PData.SetDraining).
func (t *P2PTier) DrainJobs() {
	t.data.SetDraining(true)
	defer t.data.SetDraining(false)
	for i := 0; ; i++ {
		t.pollOnce()
		pending := false
		for _, s := range t.sessions.values() {
			pending = pending || s.hasPendingWork()
		}
		if !pending {
			return
		}
		if i >= p2pDrainRounds {
			panic("tiering: p2p drain_jobs waits on work only a peer can end; vLLM would block here")
		}
	}
}

// ServeExternal is serve_external_requests (:681-698): the synthetic requests a reaped session
// left are released first, then every session answers its peer's lookups.
func (t *P2PTier) ServeExternal(p Parent) {
	if len(t.failedServeCtxs) > 0 {
		for _, c := range t.failedServeCtxs {
			p.OnRequestFinished(c)
		}
		t.failedServeCtxs = nil
	}
	for _, s := range t.sessions.values() {
		s.serveExternal(p)
	}
}

// OnScheduleEnd is on_schedule_end (:700-706): each session sends the step's probes.
func (t *P2PTier) OnScheduleEnd(ScheduleEnd) {
	for _, s := range t.sessions.values() {
		s.flushPendingLookups()
	}
}

func (t *P2PTier) newSession(peer string, conn P2PConn) *p2pSession {
	return newP2PSession(peer, t.cfg.LocalID, t.data, t.cfg.HashSeed, float64(t.cfg.HandshakeTimeoutS), t.now, conn)
}

// getOrCreateSession is _get_or_create_session (:717-770): nil is no session (a reconnect backoff,
// the peer limit, or a failed connect).
func (t *P2PTier) getOrCreateSession(peer string) *p2pSession {
	if s, ok := t.sessions.get(peer); ok {
		return s
	}
	if t.reconnectWait(peer) > 0 {
		return nil
	}
	if t.sessions.len() >= t.cfg.MaxPeers {
		return nil
	}
	conn, err := t.control.Connect(peer)
	if err != nil {
		return nil
	}
	s := t.newSession(peer, conn)
	t.sessions.set(peer, s)
	return s
}

// acceptNewPeers is _accept_new_peers (:772-825).
func (t *P2PTier) acceptNewPeers(conns []P2PConn) {
	for _, conn := range conns {
		if t.sessions.len() >= t.cfg.MaxPeers {
			conn.Close()
			continue
		}
		if t.reconnectWait(conn.PeerID()) > 0 {
			// Inside a reconnect backoff the peer is told at once (:800-805).
			_ = conn.Send(P2PMsg{Type: p2pDisconnect})
			conn.Close()
			continue
		}
		if t.sessions.has(conn.PeerID()) {
			conn.Close() // a duplicate connection (:807-809)
			continue
		}
		t.sessions.set(conn.PeerID(), t.newSession(conn.PeerID(), conn))
	}
}

// teardown is the reap common to _reap_dead_sessions and _reap_idle_sessions (:840-863, :980-993).
func (t *P2PTier) teardown(peer string) *p2pSession {
	s, _ := t.sessions.pop(peer)
	for _, id := range t.kvToSession.keys() {
		if b, _ := t.kvToSession.get(id); b == s {
			t.kvToSession.pop(id)
		}
	}
	jobs, ids, stores, serves := s.close()
	for _, j := range jobs {
		t.finished = append(t.finished, JobResult{ID: j})
	}
	for _, j := range stores {
		t.finished = append(t.finished, JobResult{ID: j})
	}
	for _, id := range ids {
		t.failedReqIDs[id] = true
	}
	t.failedServeCtxs = append(t.failedServeCtxs, serves...)
	t.data.RemoveRemotePeer(peer)
	return s
}

// reapDeadSessions is _reap_dead_sessions (:827-878): a data-plane fault also schedules a
// reconnect backoff.
func (t *P2PTier) reapDeadSessions() {
	var dead []string
	for _, p := range t.sessions.keys() {
		if s, _ := t.sessions.get(p); s.connected && !s.alive() {
			dead = append(dead, p)
		}
	}
	for _, p := range dead {
		fault := ""
		if s, _ := t.sessions.get(p); s != nil {
			fault = s.server.peerFault
		}
		t.teardown(p)
		if fault != "" {
			t.dataPlaneFaults[p]++
			t.scheduleReconnectBackoff(p)
		}
	}
}

// scheduleReconnectBackoff is _schedule_reconnect_backoff (:880-898): 0 for a first fault, then
// doubling from 1 s to 60 s while each fault follows the previous wait's end by at most 60 s.
func (t *P2PTier) scheduleReconnectBackoff(peer string) float64 {
	now := t.now()
	for _, p := range t.reconnectBackoff.keys() {
		if b, _ := t.reconnectBackoff.get(p); now-b.retryAt > p2pReconnectMaxBackoffS {
			t.reconnectBackoff.pop(p)
		}
	}
	wait := 0.0
	if prev, ok := t.reconnectBackoff.get(peer); ok {
		wait = min(max(2*prev.wait, p2pReconnectMinBackoffS), p2pReconnectMaxBackoffS)
	}
	t.reconnectBackoff.set(peer, backoff{wait: wait, retryAt: now + wait})
	return wait
}

// reconnectWait is _reconnect_wait (:900-902).
func (t *P2PTier) reconnectWait(peer string) float64 {
	b, ok := t.reconnectBackoff.get(peer)
	if !ok {
		return 0
	}
	return b.retryAt - t.now()
}

// reapUnboundStores is _reap_unbound_stores (:904-952): parked stores no fetch bound within the
// timeout fail, and their id is remembered as reaped for the retention.
func (t *P2PTier) reapUnboundStores() {
	now := t.now()
	if t.reapedStores.len() > 0 {
		keep := now - p2pReapedIDRetentionS
		for _, id := range t.reapedStores.keys() {
			if at, _ := t.reapedStores.get(id); at <= keep {
				t.reapedStores.pop(id)
			}
		}
	}
	if t.unboundStores.len() == 0 {
		return
	}
	deadline := now - t.cfg.UnboundStoreTimeoutS
	var expired []string
	for _, id := range t.unboundStores.keys() {
		if b, _ := t.unboundStores.get(id); len(b) > 0 && b[0].submittedAt <= deadline {
			expired = append(expired, id)
		}
	}
	for _, id := range expired {
		b, _ := t.unboundStores.pop(id)
		t.reapedStores.set(id, now)
		for _, x := range b {
			t.finished = append(t.finished, JobResult{ID: x.job})
		}
	}
}

// reapIdleSessions is _reap_idle_sessions (:954-999): idle sessions are evicted only once the
// free peer slots are down to the headroom.
func (t *P2PTier) reapIdleSessions() {
	if t.cfg.IdleReapOff || t.cfg.IdleTimeoutS <= 0 {
		return
	}
	headroom := min(p2pIdleReapHeadroom, max(0, t.cfg.MaxPeers-1))
	if t.sessions.len() < t.cfg.MaxPeers-headroom {
		return
	}
	deadline := t.now() - float64(t.cfg.IdleTimeoutS)
	var idle []string
	for _, p := range t.sessions.keys() {
		if s, _ := t.sessions.get(p); !s.hasPendingWork() && s.lastActivityAt <= deadline {
			idle = append(idle, p)
		}
	}
	for _, p := range idle {
		t.teardown(p)
	}
}

// pollOnce is _poll_once (:1005-1068): accept new peers, poll every session, bind each fetched id
// to its session (replaying its parked stores), then the reaps.
func (t *P2PTier) pollOnce() {
	t.acceptNewPeers(t.control.Poll())
	for _, s := range t.sessions.values() {
		r := s.poll()
		for _, l := range r.loads {
			t.finished = append(t.finished, JobResult{ID: l.jobID, Success: l.ok})
			if !l.ok {
				t.failedReqIDs[l.kvID] = true
			}
		}
		for _, x := range r.stores {
			t.finished = append(t.finished, JobResult{ID: x.jobID, Success: x.ok})
		}
		for _, id := range r.fetchIDs {
			if t.reapedStores.has(id) {
				// A fetch for a reaped id is answered at once with a failure, and the id stays
				// unbound (:1042-1059).
				s.finishRequest(id)
				b, _ := t.unboundStores.pop(id)
				for _, x := range b {
					t.finished = append(t.finished, JobResult{ID: x.job})
				}
				continue
			}
			t.kvToSession.set(id, s)
			b, _ := t.unboundStores.pop(id)
			for _, x := range b {
				s.addStoredBlocks(id, x.keys, x.blockIDs, x.job)
			}
		}
	}
	t.reapDeadSessions()
	t.reapUnboundStores()
	t.reapIdleSessions()
}
