// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/tiering/p2p/session/client.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).

package tiering

import "fmt"

// The client role's deadlines (client.py:33-35@4ba787ec), in seconds.
const (
	p2pLoadTimeoutS     = 30.0
	p2pAbortAckTimeoutS = 10.0
	p2pLookupTimeoutS   = 30.0
)

// probe is a symmetric lookup's outcome for one key: pending (None), hit (True) or miss (False).
type probe int

const (
	probePending probe = iota
	probeHit
	probeMiss
)

// inboundLoad is _InboundLoadState (client.py:38-48).
type inboundLoad struct {
	jobID       int
	submittedAt float64
	aborted     bool
	abortedAt   float64
}

// clientReq is _ClientRequestState (client.py:51-85).
type clientReq struct {
	probes         omap[Key, probe]
	unsent         []Key
	roundSeq       int
	probed         bool
	peerLookupOpen bool
	loads          omap[int, *inboundLoad]
}

// loadResult is LoadResult (client.py:88-93).
type loadResult struct {
	jobID int
	kvID  string
	ok    bool
}

// p2pClient is ClientRole (client.py:111-548): the loads this instance fetches from one peer, and
// its symmetric probes.
type p2pClient struct {
	peer string
	send func(P2PMsg)
	now  func() float64

	requests omap[string, *clientReq]
	// flushPending and activeLoads are sets of ids (client.py:131-136), iterated in insertion order.
	flushPending    strSet
	activeLoads     strSet
	completedLoads  []loadResult
	lookupDeadlines omap[string, float64]
}

func (c *p2pClient) getOrCreate(id string) *clientReq {
	st, ok := c.requests.get(id)
	if !ok {
		st = &clientReq{}
		c.requests.set(id, st)
	}
	return st
}

// maybePrune is _maybe_prune (client.py:154-163).
func (c *p2pClient) maybePrune(id string) {
	if st, ok := c.requests.get(id); ok && st.loads.len() == 0 && st.probes.len() == 0 && len(st.unsent) == 0 {
		c.requests.pop(id)
	}
}

// onLoadTerminal is _on_load_terminal (client.py:165-170).
func (c *p2pClient) onLoadTerminal(id string, st *clientReq) {
	if st.loads.len() > 0 {
		return
	}
	c.activeLoads.pop(id)
	c.maybePrune(id)
}

func (c *p2pClient) hasActiveLoads() bool { return c.activeLoads.len() > 0 }

// expiredLookup is get_expired_lookup (client.py:177-183).
func (c *p2pClient) expiredLookup() (string, bool) {
	now := c.now()
	for _, id := range c.lookupDeadlines.keys() {
		if d, _ := c.lookupDeadlines.get(id); now >= d {
			return id, true
		}
	}
	return "", false
}

// requestBlocks is request_blocks (client.py:189-241): the FetchMsg that closes the current round.
func (c *p2pClient) requestBlocks(jobID int, id string, keys []Key, blockIDs []int) {
	st := c.getOrCreate(id)
	round := st.roundSeq
	st.roundSeq++
	st.loads.set(round, &inboundLoad{jobID: jobID, submittedAt: c.now()})
	c.activeLoads.set(id, struct{}{})
	st.peerLookupOpen = false
	c.send(P2PMsg{Type: p2pFetch, KVRequestID: id, Keys: append([]Key(nil), keys...),
		BlockIndexes: append([]int(nil), blockIDs...), RoundSeq: round})
	// The assertions of :235-239: a symmetric fetch carries only confirmed probes.
	if st.probed && len(keys) > 0 {
		if st.probes.len() == 0 {
			panic(fmt.Sprintf("tiering: symmetric fetch for %s has keys but no probes", id))
		}
		for _, k := range keys {
			if p, _ := st.probes.get(k); p != probeHit {
				panic(fmt.Sprintf("tiering: symmetric fetch for %s carries a key no probe confirmed", id))
			}
		}
	}
	st.probes.clear()
	c.lookupDeadlines.pop(id)
}

// finish is finish (client.py:243-288).
func (c *p2pClient) finish(id string) {
	st, ok := c.requests.get(id)
	if !ok {
		return
	}
	for _, round := range st.loads.keys() {
		ld, _ := st.loads.get(round)
		if ld.aborted {
			continue
		}
		c.send(P2PMsg{Type: p2pAbortFetch, KVRequestID: id, RoundSeq: round})
		ld.aborted, ld.abortedAt = true, c.now()
	}
	if st.peerLookupOpen {
		st.peerLookupOpen = false
		c.send(P2PMsg{Type: p2pFetch, KVRequestID: id, Keys: []Key{}, BlockIndexes: []int{}, RoundSeq: st.roundSeq})
	}
	st.probes.clear()
	st.unsent = nil
	c.lookupDeadlines.pop(id)
	c.flushPending.pop(id)
	c.maybePrune(id)
}

// onTransferDone is on_transfer_done (client.py:290-318); onAbortAck (:320-350) is the same with
// a failure. A message for no load in flight is dropped.
func (c *p2pClient) onTransferDone(id string, ok bool, round int) {
	st, found := c.requests.get(id)
	if !found {
		return
	}
	ld, found := st.loads.pop(round)
	if !found {
		return
	}
	c.completedLoads = append(c.completedLoads, loadResult{jobID: ld.jobID, kvID: id, ok: ok})
	c.onLoadTerminal(id, st)
}

func (c *p2pClient) onAbortAck(id string, round int) { c.onTransferDone(id, false, round) }

// registerLookup is register_lookup (client.py:356-390).
func (c *p2pClient) registerLookup(id string, k Key) probe {
	st := c.getOrCreate(id)
	st.probed = true
	if p, ok := st.probes.get(k); ok {
		return p
	}
	st.probes.set(k, probePending)
	st.unsent = append(st.unsent, k)
	c.flushPending.set(id, struct{}{})
	return probePending
}

// flushPendingLookups is flush_pending_lookups (client.py:392-435).
func (c *p2pClient) flushPendingLookups(ready bool) {
	if !ready {
		return
	}
	for _, id := range c.flushPending.keys() {
		st, ok := c.requests.get(id)
		if !ok || len(st.unsent) == 0 {
			continue
		}
		st.peerLookupOpen = true
		c.lookupDeadlines.setDefault(id, c.now()+p2pLookupTimeoutS)
		c.send(P2PMsg{Type: p2pLookup, KVRequestID: id, Keys: st.unsent, RoundSeq: st.roundSeq})
		st.unsent = nil
	}
	c.flushPending.clear()
}

// onLookupResp is on_lookup_resp (client.py:437-471).
func (c *p2pClient) onLookupResp(id string, keys []Key, hits []bool) {
	st, ok := c.requests.get(id)
	if !ok {
		return
	}
	progressed := false
	for i := 0; i < len(keys) && i < len(hits); i++ {
		if p, ok := st.probes.get(keys[i]); ok {
			progressed = progressed || p == probePending
			if hits[i] {
				st.probes.set(keys[i], probeHit)
			} else {
				st.probes.set(keys[i], probeMiss)
			}
		}
	}
	all := true
	for _, p := range st.probes.values() {
		if p == probePending {
			all = false
		}
	}
	switch {
	case all:
		c.lookupDeadlines.pop(id)
	case progressed:
		c.lookupDeadlines.set(id, c.now()+p2pLookupTimeoutS)
	}
}

// collectResults is collect_results (client.py:473-526): a load past the load timeout sends
// AbortFetch, and an abort unanswered past the abort-ack timeout fails its load.
func (c *p2pClient) collectResults() []loadResult {
	now := c.now()
	type rm struct {
		id    string
		round int
	}
	var remove []rm
	for _, id := range c.activeLoads.keys() {
		st, _ := c.requests.get(id)
		if st.loads.len() == 0 {
			panic("tiering: an active p2p load id holds no load")
		}
		for _, round := range st.loads.keys() {
			ld, _ := st.loads.get(round)
			if !ld.aborted {
				if now-ld.submittedAt >= p2pLoadTimeoutS {
					ld.aborted, ld.abortedAt = true, now
					c.send(P2PMsg{Type: p2pAbortFetch, KVRequestID: id, RoundSeq: round})
				}
			} else if now-ld.abortedAt >= p2pAbortAckTimeoutS {
				remove = append(remove, rm{id, round})
				c.completedLoads = append(c.completedLoads, loadResult{jobID: ld.jobID, kvID: id})
			}
		}
	}
	for _, r := range remove {
		st, _ := c.requests.get(r.id)
		st.loads.pop(r.round)
		c.onLoadTerminal(r.id, st)
	}
	out := c.completedLoads
	c.completedLoads = nil
	return out
}

// close is close (client.py:528-548): the loads in flight and the requests a dead peer leaves
// waiting (a load, or a probe never answered).
func (c *p2pClient) close() (failedJobs []int, failedIDs []string) {
	for _, st := range c.requests.values() {
		for _, ld := range st.loads.values() {
			failedJobs = append(failedJobs, ld.jobID)
		}
	}
	for _, id := range c.requests.keys() {
		st, _ := c.requests.get(id)
		open := st.loads.len() > 0
		for _, p := range st.probes.values() {
			open = open || p == probePending
		}
		if open {
			failedIDs = append(failedIDs, id)
		}
	}
	c.requests.clear()
	c.flushPending.clear()
	c.activeLoads.clear()
	c.completedLoads = nil
	c.lookupDeadlines.clear()
	return failedJobs, failedIDs
}
