// Moved from inference-sim/sim/cluster/pd_events.go at 63b211633e587a48a650922a2a2c2c2927199533.

// The engine's admission of remote-KV pulls (a P/D decode engine's handoffs): which of the
// waiting handoffs start their pull, and what each admitted pull reserves until its KV lands.
// It moved here from inference-sim's cluster (serveDecodeKVWaiters, #2021 B2 verification gap
// 5), which keeps the waiting queue, the transfer events and the drops.
package engine

// RemoteKVAdmission is one admitted remote-KV pull (AdmitRemoteKV): its request and the
// prompt tokens the pull must move.
type RemoteKVAdmission struct {
	Req          *Request
	PulledTokens int64
}

// AdmitRemoteKV admits waiting remote-KV pulls, in the engine's admission order
// (remoteKVOrder), for as long as the head's blocks fit, and returns the admitted ones in that
// order. The head that does not fit stops the scan: vLLM's waiting queue stops at the first
// request it cannot allocate (sched/scheduler.py:1195-1229@5df63a72). A pull fits when its
// reservation leaves free the blocks the engine's other admitted remote-KV requests still need
// to run: the first-step blocks of its pulls in flight and of the waiting requests whose pull
// landed (reserved_blocks, :1241-1250; _inflight_prefill_reserved_blocks, :3088-3091). An
// admitted pull's first-step blocks count as in flight until RemoteKVLanded.
//
// waiting is the handoffs in the order they reached this engine; a P/D host holds them and
// delivers them in that order. Which of them start, and in what order, is the engine's rule
// (#2021 B2, verification pass 4, gap 2).
func (sim *EngineCore) AdmitRemoteKV(waiting []*Request) []RemoteKVAdmission {
	var admitted []RemoteKVAdmission
	for _, req := range sim.remoteKVOrder(waiting) {
		reserved := sim.inflightPullBlocks + sim.WaitingHolderBlocks()
		pulled, ok := sim.ReserveRemoteKVBehind(req, reserved)
		if !ok {
			break
		}
		sim.notePullInFlight(req)
		admitted = append(admitted, RemoteKVAdmission{Req: req, PulledTokens: pulled})
	}
	return admitted
}

// remoteKVOrder is the order this engine admits its waiting remote-KV pulls in: the order they
// reached it, under every scheduling policy. That is how inference-sim's engine admitted them,
// and the compatibility engine keeps it (#2021 D7). vLLM's own waiting queue orders them by its
// scheduling_policy: under priority, by (priority, arrival, request id)
// (vllm/v1/core/sched/request_queue.py:202-205@5df63a72, Request.__lt__ in
// vllm/v1/request.py:357-367@5df63a72), so a priority decode pool admits pulls in another order.
// That parity is https://github.com/inference-sim/blis-vllm/issues/30.
func (sim *EngineCore) remoteKVOrder(waiting []*Request) []*Request { return waiting }

// notePullInFlight records an admitted pull's first-step blocks as in flight.
func (sim *EngineCore) notePullInFlight(req *Request) {
	n := sim.FirstDecodeStepBlocks(req)
	if sim.inflightPulls == nil {
		sim.inflightPulls = make(map[string]int64)
	}
	sim.inflightPulls[req.ID] = n
	sim.inflightPullBlocks += n
}

// RemoteKVLanded ends an admitted pull's in-flight reservation: its KV has landed (or the pull
// was dropped), so its first-step blocks no longer count against new pulls
// (_inflight_prefills.discard, sched/scheduler.py:1638-1642,2721@5df63a72). A request with no
// pull in flight is ignored.
func (sim *EngineCore) RemoteKVLanded(req *Request) {
	n, ok := sim.inflightPulls[req.ID]
	if !ok {
		return
	}
	delete(sim.inflightPulls, req.ID)
	sim.inflightPullBlocks -= n
}

// InflightPullBlocks is the first-step blocks of this engine's remote-KV pulls in flight.
func (sim *EngineCore) InflightPullBlocks() int64 { return sim.inflightPullBlocks }
