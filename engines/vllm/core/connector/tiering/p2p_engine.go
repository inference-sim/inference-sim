// Original to blis-vllm.

package tiering

import "fmt"

// The p2p tier's view for an engine that drives it from outside (core/vllm). The tier decides and
// the engine's transports move: the engine needs to know which request a WRITE serves, and
// whether the tier has anything left to poll.

// WriteRequest is the kv_request_id the WRITE tid on the data transport to peer serves: the id of
// the fetch whose blocks it carries (inflightXfer.kvID). False when the tier holds no such WRITE,
// as for one it has already settled or cancelled.
func (t *P2PTier) WriteRequest(peer string, tid int) (string, bool) {
	s, ok := t.sessions.get(peer)
	if !ok {
		return "", false
	}
	x, ok := s.server.inflight.get(tid)
	if !ok {
		return "", false
	}
	return x.kvID, true
}

// Busy reports whether the tier has work that a poll would advance with no new input: a session
// still shaking hands, with a load, a probe, a lookup, a store or a WRITE outstanding, or a dead
// connection to reap; parked stores awaiting their fetch; results or released requests the next
// poll hands the manager. A session that is only idle is not busy: nothing it holds changes until
// a message or a call (vLLM's has_pending_work is always true, which on a virtual clock would
// tick forever; blis-vllm#204).
func (t *P2PTier) Busy() bool {
	if t.unboundStores.len() > 0 || len(t.finished) > 0 || len(t.failedServeCtxs) > 0 {
		return true
	}
	for _, s := range t.sessions.values() {
		if s.busy() {
			return true
		}
	}
	return false
}

// busy is the part of Busy a session holds.
func (s *p2pSession) busy() bool {
	if s.conn != nil && (!s.sendReady || !s.conn.Alive()) {
		return true
	}
	if s.hasPendingWork() || len(s.queued) > 0 || len(s.newFetchIDs) > 0 {
		return true
	}
	c, v := s.client, s.server
	if c.flushPending.len() > 0 || len(c.completedLoads) > 0 || c.lookupDeadlines.len() > 0 {
		return true
	}
	if v.servePending.len() > 0 || v.storeJobs.len() > 0 || v.pendingAborts.len() > 0 || len(v.pendingStoreResults) > 0 ||
		len(v.finishedLookupCtxs) > 0 {
		return true
	}
	for _, r := range v.requests.values() {
		if r.outbound.len() > 0 || r.lookups.len() > 0 || len(r.pendingLookups) > 0 || len(r.inflightTids) > 0 {
			return true
		}
	}
	return false
}

// SetLocalID gives the tier the control identity peers dial back, before it has a session: a host
// that runs several instances of one engine block assigns each its own (the side channel's host
// and port, P2PConfig.LocalID, :348). An error once any session exists, since a peer may already
// know the old identity.
func (t *P2PTier) SetLocalID(id string) error {
	if t.sessions.len() > 0 {
		return fmt.Errorf("tiering: the p2p tier's identity cannot change once it has a session (it is %q)", t.cfg.LocalID)
	}
	t.cfg.LocalID = id
	return nil
}
