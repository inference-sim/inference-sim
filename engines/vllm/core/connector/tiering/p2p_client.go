// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/tiering/p2p/session/client.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).

package tiering

// inboundLoad is _InboundLoadState (client.py:38-48).
type inboundLoad struct {
	jobID       int
	submittedAt float64
	aborted     bool
	abortedAt   float64
}

// clientReq is _ClientRequestState (client.py:51-85).
type clientReq struct {
	roundSeq int
	loads    omap[int, *inboundLoad]
}

// loadResult is LoadResult (client.py:88-93).
type loadResult struct {
	jobID int
	kvID  string
	ok    bool
}

// p2pClient is ClientRole (client.py:111-548): the loads this instance fetches from one peer.
type p2pClient struct {
	peer string
	send func(P2PMsg)
	now  func() float64

	requests omap[string, *clientReq]
	// activeLoads is a set of ids (client.py:136), iterated in insertion order.
	activeLoads    strSet
	completedLoads []loadResult
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
	if st, ok := c.requests.get(id); ok && st.loads.len() == 0 {
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

// requestBlocks is request_blocks (client.py:189-241): the FetchMsg that closes the current round.
func (c *p2pClient) requestBlocks(jobID int, id string, keys []Key, blockIDs []int) {
	st := c.getOrCreate(id)
	round := st.roundSeq
	st.roundSeq++
	st.loads.set(round, &inboundLoad{jobID: jobID, submittedAt: c.now()})
	c.activeLoads.set(id, struct{}{})
	c.send(P2PMsg{Type: p2pFetch, KVRequestID: id, Keys: append([]Key(nil), keys...),
		BlockIndexes: append([]int(nil), blockIDs...), RoundSeq: round})
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

// collectResults is collect_results (client.py:473-526): the loads that ended.
func (c *p2pClient) collectResults() []loadResult {
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
		if st.loads.len() > 0 {
			failedIDs = append(failedIDs, id)
		}
	}
	c.requests.clear()
	c.activeLoads.clear()
	c.completedLoads = nil
	return failedJobs, failedIDs
}
