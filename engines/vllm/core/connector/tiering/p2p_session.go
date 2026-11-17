// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/tiering/p2p/session/session.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).

package tiering

import "fmt"

// p2pPeerFaultDrainS is _PEER_FAULT_DRAIN_S (session.py:68@4ba787ec): how long a data-plane fault
// waits for in-flight WRITEs before it disconnects.
const p2pPeerFaultDrainS = 10.0

// sessionPoll is SessionPollResult (session.py:71-84).
type sessionPoll struct {
	loads    []loadResult
	stores   []storeResult
	fetchIDs []string
}

// p2pSession is P2PSession (session.py:103-722): one peer's connection, with the client role (this
// instance loads from the peer) and the server role (it serves the peer) on it.
type p2pSession struct {
	peer, local string
	data        P2PData
	blockLen    int64
	hashSeed    string
	handshakeS  float64
	now         func() float64

	conn      P2PConn
	sendReady bool
	queued    []P2PMsg
	// deathReason is the first cause _mark_dead recorded; "" is none.
	deathReason    string
	peerFaultAt    float64
	peerFaultSeen  bool
	connectedAt    float64
	connected      bool
	lastActivityAt float64
	newFetchIDs    []string
	pendingReg     P2PRegistration

	client *p2pClient
	server *p2pServer
}

// newP2PSession is P2PSession.__init__ (session.py:121-173) with a connection: attach_connection
// sends this side's ConnectMsg at once.
func newP2PSession(peer, local string, data P2PData, hashSeed string, handshakeS float64, now func() float64,
	conn P2PConn) *p2pSession {
	s := &p2pSession{peer: peer, local: local, data: data, blockLen: data.BlockLen(), hashSeed: hashSeed,
		handshakeS: handshakeS, now: now, lastActivityAt: now()}
	s.client = &p2pClient{peer: peer, send: s.send, now: now}
	s.server = &p2pServer{peer: peer, data: data, send: s.send, now: now}
	if conn != nil {
		s.attach(conn)
	}
	return s
}

// alive is alive (session.py:179-183): a session without a connection counts as alive.
func (s *p2pSession) alive() bool { return s.conn == nil || s.conn.Alive() }

// hasPendingWork is has_pending_work (session.py:194-201).
func (s *p2pSession) hasPendingWork() bool {
	return s.pendingReg != nil || s.client.hasActiveLoads() || s.server.hasInflight()
}

// attach is attach_connection (session.py:226-238).
func (s *p2pSession) attach(conn P2PConn) {
	if s.conn != nil {
		panic(fmt.Sprintf("tiering: p2p session %s already connected", s.peer))
	}
	s.conn = conn
	now := s.now()
	s.connectedAt, s.connected, s.lastActivityAt = now, true, now
	// _send_connect (:674-688). A fresh connection is alive, so the send does not fail.
	_ = s.conn.Send(P2PMsg{Type: p2pConnect, PeerID: s.local, AgentMetadata: s.data.AgentMetadata(),
		BaseAddr: s.data.BaseAddr(), NumBlocks: s.data.NumBlocks(), BlockLen: s.data.BlockLen(),
		Fingerprint: s.data.Fingerprint(), HashSeed: s.hashSeed})
}

func (s *p2pSession) requestBlocks(job int, id string, keys []Key, blockIDs []int) {
	s.client.requestBlocks(job, id, keys, blockIDs)
}

func (s *p2pSession) addStoredBlocks(id string, keys []Key, blockIDs []int, job int) {
	s.server.addStoredBlocks(id, keys, blockIDs, job, 0, false)
}

// finishRequest is finish_request (session.py:266-275).
func (s *p2pSession) finishRequest(id string) {
	s.client.finish(id)
	s.server.finish(id)
}

func (s *p2pSession) registerLookup(id string, k Key) probe { return s.client.registerLookup(id, k) }
func (s *p2pSession) flushPendingLookups()                  { s.client.flushPendingLookups(s.sendReady) }
func (s *p2pSession) serveExternal(p Parent)                { s.server.serveExternal(p) }

// poll is poll (session.py:302-372): dispatch what arrived, release the ConnectAck once the peer's
// registration lands, then run the handshake, load, store, peer-fault and lookup deadlines.
func (s *p2pSession) poll() sessionPoll {
	if s.conn == nil {
		return sessionPoll{stores: s.server.timeoutPendingStoreJobs()}
	}
	msgs := s.conn.Recv()
	if len(msgs) > 0 {
		s.lastActivityAt = s.now()
	}
	for _, m := range msgs {
		s.onMessage(m)
	}
	s.pollRegistration()
	if !s.sendReady && s.conn.Alive() && s.connected && s.now()-s.connectedAt > s.handshakeS {
		s.markDead("handshake timed out")
	}
	loads := s.client.collectResults()
	stores := s.server.collectResults()
	s.server.drainPendingAborts()
	s.escalatePeerFault()
	if _, ok := s.client.expiredLookup(); ok {
		s.disconnect("lookup response timed out")
	}
	ids := s.newFetchIDs
	s.newFetchIDs = nil
	return sessionPoll{loads: loads, stores: stores, fetchIDs: ids}
}

// close is close (session.py:374-410).
func (s *p2pSession) close() (failedJobs []int, failedIDs []string, failedStores []int, failedServes []*ReqContext) {
	if s.pendingReg != nil {
		s.pendingReg.Cancel()
		s.pendingReg = nil
	}
	failedJobs, failedIDs = s.client.close()
	failedStores, failedServes = s.server.close()
	if s.conn != nil {
		// A dead connection refuses the DisconnectMsg, as vLLM's suppressed send does.
		_ = s.conn.Send(P2PMsg{Type: p2pDisconnect})
		s.conn.Close()
		s.conn = nil
	}
	return failedJobs, failedIDs, failedStores, failedServes
}

// onMessage is _on_message (session.py:416-446): a protocol violation disconnects the peer. vLLM
// also counts consecutive unexpected exceptions (internal bugs) and disconnects at five; the port
// panics on such a bug instead.
func (s *p2pSession) onMessage(m P2PMsg) {
	if err := s.dispatch(m); err != nil {
		s.disconnect(fmt.Sprintf("protocol error: malformed %q: %v", m.Type, err))
	}
}

// escalatePeerFault is _escalate_peer_fault (session.py:462-485).
func (s *p2pSession) escalatePeerFault() {
	fault := s.server.peerFault
	if fault == "" || s.conn == nil || !s.conn.Alive() {
		return
	}
	now := s.now()
	if !s.peerFaultSeen {
		s.peerFaultSeen, s.peerFaultAt = true, now
	}
	if s.server.hasInflight() && now-s.peerFaultAt < p2pPeerFaultDrainS {
		return
	}
	s.disconnect("data plane: " + fault)
}

// disconnect is _disconnect (session.py:487-492): a best-effort DisconnectMsg, then dead.
func (s *p2pSession) disconnect(reason string) {
	if s.conn != nil {
		_ = s.conn.Send(P2PMsg{Type: p2pDisconnect})
	}
	s.markDead(reason)
}

// markDead is _mark_dead (session.py:494-500).
func (s *p2pSession) markDead(reason string) {
	if s.conn == nil {
		return
	}
	if s.deathReason == "" {
		s.deathReason = reason
	}
	s.conn.MarkDead()
}

// dispatch is _dispatch_message (session.py:502-572). A message buffered behind the connection's
// death is dropped.
func (s *p2pSession) dispatch(m P2PMsg) error {
	if s.conn != nil && !s.conn.Alive() {
		return nil
	}
	switch m.Type {
	case p2pConnect:
		s.onConnect(m)
	case p2pConnectAck:
		s.onConnectAck()
	case p2pFetch:
		if err := s.server.onFetch(m.KVRequestID, m.Keys, m.BlockIndexes, m.RoundSeq); err != nil {
			return err
		}
		s.newFetchIDs = append(s.newFetchIDs, m.KVRequestID)
	case p2pAbortFetch:
		s.server.onAbortFetch(m.KVRequestID, m.RoundSeq)
	case p2pTransferDone:
		s.client.onTransferDone(m.KVRequestID, m.Success, m.RoundSeq)
	case p2pAbortAck:
		s.client.onAbortAck(m.KVRequestID, m.RoundSeq)
	case p2pLookup:
		s.server.onLookup(m.KVRequestID, m.Keys, m.RoundSeq)
	case p2pLookupResp:
		s.client.onLookupResp(m.KVRequestID, m.Keys, m.Hits)
	case p2pDisconnect:
		s.markDead("peer disconnected")
	}
	return nil
}

// onConnect is _on_connect (session.py:578-626): a compatible peer's registration starts, and its
// ConnectAck waits for it.
func (s *p2pSession) onConnect(m P2PMsg) {
	if s.sendReady || s.pendingReg != nil {
		s.disconnect("protocol error: duplicate connect after handshake")
		return
	}
	switch {
	case m.BlockLen != s.blockLen:
		s.markDead(fmt.Sprintf("rejected peer connect: block_len mismatch from %s", s.peer))
		return
	case s.data.Fingerprint() != "" && m.Fingerprint != "" && m.Fingerprint != s.data.Fingerprint():
		s.markDead(fmt.Sprintf("rejected peer connect: config fingerprint mismatch from %s", s.peer))
		return
	case m.HashSeed != s.hashSeed:
		s.markDead(fmt.Sprintf("rejected peer connect: hash seed mismatch from %s", s.peer))
		return
	}
	s.pendingReg = s.data.AddRemotePeerAsync(s.peer, m.AgentMetadata, m.BaseAddr, m.NumBlocks, m.BlockLen)
}

// pollRegistration is _poll_registration (session.py:628-656).
func (s *p2pSession) pollRegistration() {
	reg := s.pendingReg
	if reg == nil || !reg.Done() {
		return
	}
	s.pendingReg = nil
	if err := reg.Err(); err != nil {
		s.markDead("peer registration failed: " + err.Error())
		return
	}
	if s.conn != nil && s.conn.Alive() {
		s.doSend(P2PMsg{Type: p2pConnectAck, PeerID: s.local})
	}
}

// onConnectAck is _on_connect_ack (session.py:658-668): the queued messages go out.
func (s *p2pSession) onConnectAck() {
	s.sendReady = true
	for _, m := range s.queued {
		s.doSend(m)
	}
	s.queued = nil
}

// send is _send (session.py:690-701): messages wait for the peer's ConnectAck.
func (s *p2pSession) send(m P2PMsg) {
	if s.conn == nil || !s.sendReady {
		s.queued = append(s.queued, m)
		return
	}
	s.doSend(m)
}

// doSend is _do_send (session.py:703-722): a failed send kills the connection.
func (s *p2pSession) doSend(m P2PMsg) {
	if s.conn == nil {
		return
	}
	if err := s.conn.Send(m); err != nil {
		s.markDead("send of " + m.Type + " failed")
		return
	}
	s.lastActivityAt = s.now()
}
