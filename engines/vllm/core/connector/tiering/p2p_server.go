// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/tiering/p2p/session/server.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).

package tiering

import "fmt"

// The server role's deadlines (server.py:40-50@4ba787ec), in seconds.
const (
	p2pStoreTimeoutS         = 30.0
	p2pCancelDrainTimeoutS   = 10.0
	p2pCancelRetryIntervalS  = 0.1
	p2pLookupPendingTimeoutS = 5.0
)

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
	lookupSupplied, demandReceived bool
	available                      omap[Key, supply]
	demanded                       omap[Key, int]
	remaining                      int
	finishing                      bool
	inflight                       int
	cancelled                      bool
	cancelTried                    bool
	lastCancelAt                   float64
	pendingJobs                    intSet
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

// activeLookup is _ActiveLookup (server.py:171-200). pending is a set of keys, iterated in
// insertion order where vLLM's follows PYTHONHASHSEED; no sequence parks two keys.
type activeLookup struct {
	id       int
	kvID     string
	ctx      *ReqContext
	round    int
	keys     []Key
	resolved map[Key]bool
	pending  omap[Key, struct{}]
	deadline float64
}

// pendingLookup is _PendingLookup (server.py:203-213).
type pendingLookup struct {
	keys       []Key
	enqueuedAt float64
	round      int
}

// serverReq is _ServerRequestState (server.py:216-241).
type serverReq struct {
	outbound       omap[int, *outRound]
	pendingLookups []pendingLookup
	lookups        omap[int, *activeLookup]
	inflightTids   intSet
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
	servePending        strSet
	inflight            omap[int, *inflightXfer]
	storeJobs           omap[int, float64]
	pendingStoreResults []storeResult
	finishedLookupCtxs  []*ReqContext
	lookupIDCounter     int
	pendingAborts       omap[abortKey, float64]
	slowAborts          map[abortKey]bool
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
	if !ok || st.outbound.len() > 0 || len(st.inflightTids) > 0 || st.lookups.len() > 0 || len(st.pendingLookups) > 0 {
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
func (s *p2pServer) addStoredBlocks(id string, keys []Key, blockIDs []int, job, round int, fromLookup bool) {
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
	if fromLookup {
		rnd.lookupSupplied = true
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
		s.finishInboundLookups(id)
		for _, k := range st.outbound.keys() {
			s.finalizeOutbound(id, k, nil, false)
		}
		return nil
	}
	if req.lookupSupplied && req.demanded.len() > 0 {
		// A symmetric round's supply precedes its fetch, so unmatched demand fails now (:392-408).
		f := false
		s.finalizeOutbound(id, round, &f, true)
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

// onLookup is on_lookup (server.py:441-470): the LookupMsg waits for the next serve.
func (s *p2pServer) onLookup(id string, keys []Key, round int) {
	st := s.getOrCreate(id)
	st.pendingLookups = append(st.pendingLookups, pendingLookup{keys: append([]Key(nil), keys...), enqueuedAt: s.now(), round: round})
	s.servePending.set(id, struct{}{})
}

// serveExternal is serve_external_requests (server.py:472-501).
func (s *p2pServer) serveExternal(p Parent) {
	for _, id := range s.servePending.keys() {
		st, ok := s.requests.get(id)
		if !ok {
			s.servePending.pop(id)
			continue
		}
		if len(st.pendingLookups) > 0 {
			pending := st.pendingLookups
			st.pendingLookups = nil
			for _, pl := range pending {
				s.processInboundLookup(id, pl, p)
			}
		}
		s.resolvePendingLookups(id, p)
		st, ok = s.requests.get(id)
		if !ok || (len(st.pendingLookups) == 0 && st.lookups.len() == 0) {
			s.servePending.pop(id)
			s.maybePrune(id)
		}
	}
	if len(s.finishedLookupCtxs) > 0 {
		for _, c := range s.finishedLookupCtxs {
			p.OnRequestFinished(c)
		}
		s.finishedLookupCtxs = nil
	}
}

// pollLookupKeys is _poll_lookup_keys (server.py:503-538): each unresolved key is looked up
// through the parent, and the new hits are pinned in one store job.
func (s *p2pServer) pollLookupKeys(lu *activeLookup, keys []Key, p Parent) {
	var hits []Key
	for _, k := range keys {
		if _, ok := lu.resolved[k]; ok {
			continue
		}
		switch p.Lookup(k, lu.ctx) {
		case Hit:
			hits = append(hits, k)
			lu.resolved[k] = true
			lu.pending.pop(k)
		case Miss:
			lu.resolved[k] = false
			lu.pending.pop(k)
		default:
			lu.pending.set(k, struct{}{})
		}
	}
	if len(hits) > 0 {
		// _pin_and_register_hits (server.py:601-626).
		j := p.CreateStoreJob(hits, lu.ctx)
		s.addStoredBlocks(lu.kvID, j.Keys, j.ChunkIDs, j.ID, lu.round, true)
	}
}

// processInboundLookup is _process_inbound_lookup (server.py:540-599). Its synthetic request is
// "p2p:<peer>:<kv_request_id>:lu<n>".
func (s *p2pServer) processInboundLookup(id string, pl pendingLookup, p Parent) {
	s.lookupIDCounter++
	n := s.lookupIDCounter
	lu := &activeLookup{id: n, kvID: id, ctx: NewReqContext(fmt.Sprintf("p2p:%s:%s:lu%d", s.peer, id, n)),
		round: pl.round, keys: pl.keys, resolved: map[Key]bool{}, deadline: pl.enqueuedAt + p2pLookupPendingTimeoutS}
	p.OnNewRequest(lu.ctx)
	// dict.fromkeys: each key once, in wire order.
	seen := map[Key]bool{}
	var uniq []Key
	for _, k := range lu.keys {
		if !seen[k] {
			seen[k] = true
			uniq = append(uniq, k)
		}
	}
	s.pollLookupKeys(lu, uniq, p)
	if lu.pending.len() > 0 {
		s.getOrCreate(id).lookups.set(n, lu)
	} else {
		s.finalizeLookup(lu, p)
	}
}

// resolvePendingLookups is _resolve_pending_lookups (server.py:628-660).
func (s *p2pServer) resolvePendingLookups(id string, p Parent) {
	st, ok := s.requests.get(id)
	if !ok || st.lookups.len() == 0 {
		return
	}
	now := s.now()
	var done []int
	for _, n := range st.lookups.keys() {
		lu, _ := st.lookups.get(n)
		s.pollLookupKeys(lu, lu.pending.keys(), p)
		if lu.pending.len() > 0 && now >= lu.deadline {
			for _, k := range lu.pending.keys() {
				lu.resolved[k] = false
			}
			lu.pending.clear()
		}
		if lu.pending.len() == 0 {
			done = append(done, n)
		}
	}
	for _, n := range done {
		lu, _ := st.lookups.pop(n)
		s.finalizeLookup(lu, p)
	}
}

// finalizeLookup is _finalize_lookup (server.py:662-690).
func (s *p2pServer) finalizeLookup(lu *activeLookup, p Parent) {
	if len(lu.keys) > 0 {
		hits := make([]bool, len(lu.keys))
		for i, k := range lu.keys {
			hits[i] = lu.resolved[k]
		}
		s.send(P2PMsg{Type: p2pLookupResp, KVRequestID: lu.kvID, Keys: append([]Key(nil), lu.keys...), Hits: hits})
	}
	p.OnRequestFinished(lu.ctx)
}

// finishInboundLookups is _finish_inbound_lookups (server.py:692-721).
func (s *p2pServer) finishInboundLookups(id string) {
	st, ok := s.requests.get(id)
	if !ok {
		return
	}
	st.pendingLookups = nil
	for _, lu := range st.lookups.values() {
		s.finishedLookupCtxs = append(s.finishedLookupCtxs, lu.ctx)
	}
	st.lookups.clear()
	s.servePending.pop(id)
	s.maybePrune(id)
}

// finish is finish (server.py:723-755).
func (s *p2pServer) finish(id string) {
	s.finishInboundLookups(id)
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
	results := s.timeoutPendingStoreJobs()
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

// close is close (server.py:863-891): the store jobs to fail and the synthetic lookup requests
// still owed an on_request_finished.
func (s *p2pServer) close() (failedStores []int, failedServes []*ReqContext) {
	failedStores = s.storeJobs.keys()
	s.storeJobs.clear()
	if s.inflight.len() > 0 {
		s.data.Cancel(s.inflight.keys(), false)
	}
	s.inflight.clear()
	s.pendingStoreResults = nil
	for _, st := range s.requests.values() {
		for _, lu := range st.lookups.values() {
			failedServes = append(failedServes, lu.ctx)
		}
	}
	failedServes = append(failedServes, s.finishedLookupCtxs...)
	s.requests.clear()
	s.servePending.clear()
	s.pendingAborts.clear()
	s.slowAborts = nil
	s.finishedLookupCtxs = nil
	return failedStores, failedServes
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
	now := s.now()
	key := abortKey{id, round}
	if at, _ := s.pendingAborts.get(key); now-at >= p2pCancelDrainTimeoutS && !s.slowAborts[key] {
		// Past the drain timeout vLLM only warns (:1093-1107); the pins and the ack still wait.
		if s.slowAborts == nil {
			s.slowAborts = map[abortKey]bool{}
		}
		s.slowAborts[key] = true
	}
	first, _ := s.inflight.get(ids[0])
	if !cancelDue(first.round, now) {
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

// cancelDue is _cancel_due (server.py:1128-1141).
func cancelDue(r *outRound, now float64) bool {
	if r.cancelTried && now-r.lastCancelAt < p2pCancelRetryIntervalS {
		return false
	}
	r.cancelTried, r.lastCancelAt = true, now
	return true
}

// finalizeAbort is _finalize_abort (server.py:1143-1153).
func (s *p2pServer) finalizeAbort(id string, round int) {
	k := abortKey{id, round}
	s.pendingAborts.pop(k)
	delete(s.slowAborts, k)
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

// timeoutPendingStoreJobs is _timeout_pending_store_jobs (server.py:1227-1292): a store job past
// the store timeout cancels its rounds in wait mode, and fails once no transfer reads its blocks.
func (s *p2pServer) timeoutPendingStoreJobs() []storeResult {
	if s.storeJobs.len() == 0 {
		return nil
	}
	now := s.now()
	deadline := now - p2pStoreTimeoutS
	var timedOut []int
	for _, j := range s.storeJobs.keys() {
		if at, _ := s.storeJobs.get(j); at <= deadline {
			timedOut = append(timedOut, j)
		}
	}
	if timedOut == nil {
		anyCancelled := false
		for _, x := range s.inflight.values() {
			anyCancelled = anyCancelled || x.round.cancelled
		}
		if !anyCancelled {
			return nil
		}
	}
	expired := intSet{}
	for _, j := range timedOut {
		expired[j] = true
	}
	var results []storeResult
	var rounds omap[abortKey, *outRound]
	for _, x := range s.inflight.values() {
		hit := x.round.cancelled
		for j := range x.jobs {
			hit = hit || expired[j]
		}
		if hit {
			rounds.set(abortKey{x.kvID, x.key}, x.round)
		}
	}
	for _, k := range rounds.keys() {
		rnd, _ := rounds.get(k)
		rnd.cancelled = true
		if !cancelDue(rnd, now) {
			continue
		}
		var tids []int
		for _, tid := range s.inflight.keys() {
			if x, _ := s.inflight.get(tid); x.round == rnd {
				tids = append(tids, tid)
			}
		}
		still := map[int]bool{}
		for _, tid := range s.data.Cancel(tids, true) {
			still[tid] = true
		}
		for _, tid := range tids {
			if still[tid] {
				continue
			}
			results = append(results, s.settleXferJobs(s.inflightPop(tid), false)...)
		}
		if st, ok := s.requests.get(k.kvID); ok && rnd.inflight == 0 {
			if cur, _ := st.outbound.get(k.round); cur == rnd {
				s.finalizeCancelled(k.kvID, k.round)
			}
		}
	}
	reading := s.readJobIDs(nil)
	for _, j := range timedOut {
		if reading[j] {
			continue
		}
		if _, had := s.storeJobs.pop(j); had {
			results = append(results, storeResult{j, false})
		}
	}
	for _, st := range s.requests.values() {
		for _, rnd := range st.outbound.values() {
			hit := false
			for j := range rnd.pendingJobs {
				hit = hit || expired[j]
			}
			if !hit {
				continue
			}
			for j := range expired {
				delete(rnd.pendingJobs, j)
			}
			for _, k := range rnd.available.keys() {
				if a, _ := rnd.available.get(k); expired[a.job] {
					rnd.available.pop(k)
				}
			}
		}
	}
	return results
}
