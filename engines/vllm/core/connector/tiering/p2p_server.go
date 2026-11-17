// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/tiering/p2p/session/server.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).

package tiering

import "fmt"

// storeResult is StoreResult (server.py:53-57).
type storeResult struct {
	jobID int
	ok    bool
}

// match is _MatchResult (server.py:60-66).
type match struct {
	local, remote []int
	jobs          intSet
}

type supply struct{ job, local int }

// outRound is _OutboundRequestState (server.py:69-153): one fetch round of a peer's request.
type outRound struct {
	demandReceived bool
	available      omap[Key, supply]
	demanded       omap[Key, int]
	remaining      int
	finishing      bool
	inflight       int
	cancelled      bool
	pendingJobs    intSet
}

func newOutRound() *outRound { return &outRound{pendingJobs: intSet{}} }

// addStored is _OutboundRequestState.add_stored_blocks (server.py:105-126).
func (r *outRound) addStored(keys []Key, blockIDs []int, job int) match {
	r.pendingJobs[job] = true
	var m match
	for i, k := range keys {
		if remote, ok := r.demanded.pop(k); ok {
			m.local = append(m.local, blockIDs[i])
			m.remote = append(m.remote, remote)
		} else {
			r.available.set(k, supply{job, blockIDs[i]})
		}
	}
	m.jobs = intSet{}
	if len(m.local) > 0 {
		m.jobs[job] = true
	}
	return m
}

// addDemand is add_fetch_demand (server.py:128-153).
func (r *outRound) addDemand(keys []Key, indexes []int) match {
	r.demandReceived = true
	r.remaining = len(keys)
	m := match{jobs: intSet{}}
	for i, k := range keys {
		if s, ok := r.available.pop(k); ok {
			m.local = append(m.local, s.local)
			m.remote = append(m.remote, indexes[i])
			m.jobs[s.job] = true
		} else {
			r.demanded.set(k, indexes[i])
		}
	}
	return m
}

// inflightXfer is _InflightXfer (server.py:156-168).
type inflightXfer struct {
	kvID   string
	blocks int
	jobs   intSet
	round  *outRound
	key    int
}

// serverReq is _ServerRequestState (server.py:216-241).
type serverReq struct {
	outbound     omap[int, *outRound]
	inflightTids intSet
}

type abortKey struct {
	kvID  string
	round int
}

// p2pServer is ServerRole (server.py:244-1292): the blocks this instance serves to one peer.
type p2pServer struct {
	peer string
	data P2PData
	send func(P2PMsg)
	now  func() float64

	requests            omap[string, *serverReq]
	inflight            omap[int, *inflightXfer]
	storeJobs           omap[int, float64]
	pendingStoreResults []storeResult
	pendingAborts       omap[abortKey, float64]
	// peerFault is PollResult.peer_fault once reported; "" is None.
	peerFault string
}

func (s *p2pServer) getOrCreate(id string) *serverReq {
	st, ok := s.requests.get(id)
	if !ok {
		st = &serverReq{inflightTids: intSet{}}
		s.requests.set(id, st)
	}
	return st
}

// maybePrune is _maybe_prune (server.py:304-315).
func (s *p2pServer) maybePrune(id string) {
	st, ok := s.requests.get(id)
	if !ok || st.outbound.len() > 0 || len(st.inflightTids) > 0 {
		return
	}
	for _, k := range s.pendingAborts.keys() {
		if k.kvID == id {
			return
		}
	}
	s.requests.pop(id)
}

// addStoredBlocks is add_stored_blocks (server.py:321-349).
func (s *p2pServer) addStoredBlocks(id string, keys []Key, blockIDs []int, job, round int) {
	s.storeJobs.set(job, s.now())
	st := s.getOrCreate(id)
	rnd, ok := st.outbound.get(round)
	if !ok {
		rnd = newOutRound()
		st.outbound.set(round, rnd)
	}
	if rnd.cancelled {
		s.storeJobs.pop(job)
		s.pendingStoreResults = append(s.pendingStoreResults, storeResult{job, false})
		return
	}
	m := rnd.addStored(keys, blockIDs, job)
	if len(m.local) > 0 && rnd.demandReceived {
		s.submitTransfer(id, m, rnd, round)
	}
}

// onFetch is on_fetch (server.py:351-418). A second fetch for a round that holds demand is a
// protocol violation (the ValueError of :376-379).
func (s *p2pServer) onFetch(id string, keys []Key, indexes []int, round int) error {
	if st, ok := s.requests.get(id); ok {
		if ex, ok := st.outbound.get(round); ok && ex.demandReceived {
			return fmt.Errorf("duplicate fetch for kv_request_id=%s round=%d", id, round)
		}
	}
	st := s.getOrCreate(id)
	req, ok := st.outbound.get(round)
	if !ok {
		req = newOutRound()
		st.outbound.set(round, req)
	}
	m := req.addDemand(keys, indexes)
	if len(keys) == 0 {
		// The terminal empty fetch closes the id: every round is drained without a TransferDone.
		for _, k := range st.outbound.keys() {
			s.finalizeOutbound(id, k, nil, false)
		}
		return nil
	}
	if len(m.local) > 0 {
		s.submitTransfer(id, m, req, round)
	} else if req.finishing && req.inflight == 0 {
		s.finalizeOutbound(id, round, nil, true)
	}
	return nil
}

// onAbortFetch is on_abort_fetch (server.py:420-439); vLLM's warning for an unknown id is a log
// line only.
func (s *p2pServer) onAbortFetch(id string, round int) {
	s.getOrCreate(id)
	s.pendingAborts.setDefault(abortKey{id, round}, s.now())
	s.drainAbort(id, round)
}

// finish is finish (server.py:723-755).
func (s *p2pServer) finish(id string) {
	st, ok := s.requests.get(id)
	if !ok {
		return
	}
	for _, k := range st.outbound.keys() {
		req, _ := st.outbound.get(k)
		req.finishing = true
		if !req.demandReceived || req.inflight > 0 {
			continue
		}
		s.finalizeOutbound(id, k, nil, true)
	}
}

// collectResults is collect_results (server.py:757-847).
func (s *p2pServer) collectResults() []storeResult {
	var results []storeResult
	pr := s.data.Poll(s.peer)
	if pr.PeerFault != "" {
		s.peerFault = pr.PeerFault
	}
	for _, tid := range pr.Done {
		x := s.inflightPop(tid)
		if x == nil {
			continue
		}
		results = append(results, s.settleXferJobs(x, !x.round.cancelled)...)
		rnd := x.round
		if st, ok := s.requests.get(x.kvID); ok {
			if cur, _ := st.outbound.get(x.key); cur == rnd {
				rnd.remaining -= x.blocks
				if rnd.remaining < 0 {
					panic(fmt.Sprintf("tiering: remaining went negative for kv_request_id=%s", x.kvID))
				}
				switch {
				case rnd.cancelled:
					if rnd.inflight == 0 {
						s.finalizeCancelled(x.kvID, x.key)
					}
				case rnd.remaining == 0:
					t := true
					s.finalizeOutbound(x.kvID, x.key, &t, true)
				case rnd.finishing && rnd.inflight == 0:
					f := false
					s.finalizeOutbound(x.kvID, x.key, &f, true)
				}
			}
		}
		s.maybePrune(x.kvID)
	}
	for _, tid := range pr.Failed {
		x := s.inflightPop(tid)
		if x == nil {
			continue
		}
		rnd := x.round
		inRound := false
		if st, ok := s.requests.get(x.kvID); ok {
			cur, _ := st.outbound.get(x.key)
			inRound = cur == rnd
		}
		if inRound && !rnd.cancelled {
			rnd.cancelled = true
			results = append(results, s.failUnwrittenRoundJobs(rnd)...)
		}
		results = append(results, s.settleXferJobs(x, false)...)
		if inRound && rnd.inflight == 0 {
			s.finalizeCancelled(x.kvID, x.key)
		}
		s.maybePrune(x.kvID)
	}
	if len(s.pendingStoreResults) > 0 {
		results = append(results, s.pendingStoreResults...)
		s.pendingStoreResults = nil
	}
	return results
}

// drainPendingAborts is drain_pending_aborts (server.py:858-861).
func (s *p2pServer) drainPendingAborts() {
	for _, k := range s.pendingAborts.keys() {
		s.drainAbort(k.kvID, k.round)
	}
}

// close is close (server.py:863-891): the store jobs to fail.
func (s *p2pServer) close() (failedStores []int) {
	failedStores = s.storeJobs.keys()
	s.storeJobs.clear()
	if s.inflight.len() > 0 {
		s.data.Cancel(s.inflight.keys(), false)
	}
	s.inflight.clear()
	s.pendingStoreResults = nil
	s.requests.clear()
	s.pendingAborts.clear()
	return failedStores
}

func (s *p2pServer) hasInflight() bool { return s.inflight.len() > 0 }

// inflightAdd and inflightPop are _inflight_add and _inflight_pop (server.py:906-925).
func (s *p2pServer) inflightAdd(tid int, x *inflightXfer) {
	s.inflight.set(tid, x)
	s.getOrCreate(x.kvID).inflightTids[tid] = true
}

func (s *p2pServer) inflightPop(tid int) *inflightXfer {
	x, ok := s.inflight.pop(tid)
	if !ok {
		return nil
	}
	x.round.inflight--
	if x.round.inflight < 0 {
		panic("tiering: a p2p round's inflight count went negative")
	}
	if st, ok := s.requests.get(x.kvID); ok {
		delete(st.inflightTids, tid)
	}
	return x
}

// settleXferJobs is _settle_xfer_jobs (server.py:927-946).
func (s *p2pServer) settleXferJobs(x *inflightXfer, ok bool) []storeResult {
	var pinned intSet
	if x.round.cancelled {
		pinned = s.readJobIDs(nil)
	}
	var out []storeResult
	for _, j := range x.jobs.sorted() {
		if pinned[j] {
			continue
		}
		if _, had := s.storeJobs.pop(j); !had {
			continue
		}
		out = append(out, storeResult{j, ok})
		delete(x.round.pendingJobs, j)
	}
	return out
}

// failRoundJobs is _fail_round_jobs (server.py:952-960).
func (s *p2pServer) failRoundJobs(r *outRound) []storeResult {
	var out []storeResult
	for _, j := range r.pendingJobs.sorted() {
		if _, had := s.storeJobs.pop(j); had {
			out = append(out, storeResult{j, false})
		}
	}
	r.pendingJobs = intSet{}
	return out
}

// finalizeCancelled is _finalize_cancelled (server.py:962-973): an aborted round's answer is the
// AbortAck, not a TransferDone.
func (s *p2pServer) finalizeCancelled(id string, round int) {
	f := false
	s.finalizeOutbound(id, round, &f, !s.pendingAborts.has(abortKey{id, round}))
}

// readJobIDs is _read_job_ids (server.py:975-983); r nil is every round.
func (s *p2pServer) readJobIDs(r *outRound) intSet {
	out := intSet{}
	for _, x := range s.inflight.values() {
		if r == nil || x.round == r {
			for j := range x.jobs {
				out[j] = true
			}
		}
	}
	return out
}

// failUnwrittenRoundJobs is _fail_unwritten_round_jobs (server.py:985-1002).
func (s *p2pServer) failUnwrittenRoundJobs(r *outRound) []storeResult {
	written := s.readJobIDs(r)
	var out []storeResult
	for _, j := range r.pendingJobs.sorted() {
		if written[j] {
			continue
		}
		delete(r.pendingJobs, j)
		if _, had := s.storeJobs.pop(j); had {
			out = append(out, storeResult{j, false})
		}
	}
	return out
}

// finalizeOutbound is _finalize_outbound (server.py:1004-1057); success nil derives it from the
// round's remaining demand.
func (s *p2pServer) finalizeOutbound(id string, round int, success *bool, sendDone bool) {
	st, _ := s.requests.get(id)
	req, _ := st.outbound.get(round)
	if req.cancelled && req.inflight > 0 {
		return
	}
	st.outbound.pop(round)
	var ok bool
	switch {
	case req.cancelled:
		ok = false
	case success != nil:
		ok = *success
	default:
		ok = req.demandReceived && req.remaining == 0
	}
	if !ok {
		s.pendingStoreResults = append(s.pendingStoreResults, s.failRoundJobs(req)...)
	} else {
		for _, j := range req.pendingJobs.sorted() {
			if _, had := s.storeJobs.pop(j); had {
				s.pendingStoreResults = append(s.pendingStoreResults, storeResult{j, true})
			}
		}
		req.pendingJobs = intSet{}
	}
	if sendDone && req.demandReceived {
		s.send(P2PMsg{Type: p2pTransferDone, KVRequestID: id, Success: ok, RoundSeq: round})
	}
	s.maybePrune(id)
}

// drainAbort is _drain_abort (server.py:1059-1126): the AbortAck goes out once none of the round's
// transfers can still write to the peer.
func (s *p2pServer) drainAbort(id string, round int) {
	st, _ := s.requests.get(id)
	rnd, _ := st.outbound.get(round)
	if rnd != nil && !rnd.cancelled {
		rnd.cancelled = true
		s.pendingStoreResults = append(s.pendingStoreResults, s.failUnwrittenRoundJobs(rnd)...)
	}
	var ids []int
	for _, tid := range s.inflight.keys() {
		if x, _ := s.inflight.get(tid); x.kvID == id && x.key == round {
			ids = append(ids, tid)
		}
	}
	finish := func() {
		if cur, _ := st.outbound.get(round); rnd != nil && cur == rnd {
			f := false
			s.finalizeOutbound(id, round, &f, false)
		}
		s.finalizeAbort(id, round)
	}
	if len(ids) == 0 {
		finish()
		return
	}
	still := map[int]bool{}
	for _, tid := range s.data.Cancel(ids, true) {
		still[tid] = true
	}
	for _, tid := range ids {
		if !still[tid] {
			s.inflightPop(tid)
		}
	}
	if len(still) == 0 {
		finish()
	}
}

// finalizeAbort is _finalize_abort (server.py:1143-1153).
func (s *p2pServer) finalizeAbort(id string, round int) {
	k := abortKey{id, round}
	s.pendingAborts.pop(k)
	s.send(P2PMsg{Type: p2pAbortAck, KVRequestID: id, RoundSeq: round})
	s.maybePrune(id)
}

// submitTransfer is _submit_transfer (server.py:1159-1225): no WRITE goes to a faulted peer.
func (s *p2pServer) submitTransfer(id string, m match, rnd *outRound, round int) {
	tid, ok := 0, false
	if s.peerFault == "" {
		tid, ok = s.data.WriteBlocks(s.peer, m.local, m.remote)
	}
	if ok {
		rnd.inflight++
		s.inflightAdd(tid, &inflightXfer{kvID: id, blocks: len(m.local), jobs: m.jobs, round: rnd, key: round})
		return
	}
	rnd.finishing = true
	if st, ok := s.requests.get(id); ok {
		if cur, _ := st.outbound.get(round); cur == rnd && rnd.inflight == 0 {
			f := false
			s.finalizeOutbound(id, round, &f, true)
		}
	}
}
