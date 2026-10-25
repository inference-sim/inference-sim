// sim/engine_events.go: the engine's side of the host's events and idle park: what an arrival,
// a queued request and a timeout do to the engine, and the engine state a parked instance's
// skipped polls record (#2021 B3; the events and the park are the host's, event.go and
// idle_park.go).
package sim

// queuedAt is when a request arriving at t enters the wait queue: after its admission overhead
// (render + tokenize), which runs on the endpoint's render stage when one is configured, so it
// queues behind earlier requests (INV-R1).
func (sim *EngineCore) QueuedAt(req *Request, t int64) int64 {
	render := sim.latencyModel.QueueingTime(req)
	if sim.renderer != nil {
		return sim.renderer.Admit(t, render)
	}
	return t + render
}

// hasWaiting reports whether the wait queue holds a request.
func (sim *EngineCore) HasWaiting() bool { return sim.WaitQ.Len() > 0 }

// hasRunning reports whether the running batch holds a request.
func (sim *EngineCore) HasRunning() bool {
	return sim.RunningBatch != nil && len(sim.RunningBatch.Requests) > 0
}

// timeout cancels a request whose client deadline passed, and reports whether it did: false
// when the request had already completed or timed out (BC-3). Three paths: (1) running
// request — new-slice removal from RunningBatch (R21), (2) queued request — WaitQ.Remove,
// (3) pre-QueuedEvent race — request not in any container yet (WaitQ.Remove returns false,
// safe no-op).
func (sim *EngineCore) CancelOnTimeout(req *Request) bool {
	// No-op guard: request already completed or timed out (BC-3)
	if req.State == StateCompleted || req.State == StateTimedOut {
		return false
	}
	wasRunning := req.State == StateRunning
	// INV-2 (request lifecycle): a terminal transition to timed-out, reachable from
	// either queued or running. The guard above makes it one-way — a request that
	// already reached a terminal state is never re-transitioned. INV-2's statement
	// names only queued -> running -> completed, so timed-out is a terminal state the
	// statement does not yet enumerate; this is not the only site that sets it.
	req.State = StateTimedOut
	sim.Rec.timedOut(req)

	// Release the adapter pin (cold-load gate, #1466) for a request cancelled while
	// running: it no longer uses its adapter. A queued/gated request never pinned,
	// so releaseAdapterPin is a no-op there (guarded by the per-request flag).
	sim.releaseAdapterPin(req)

	// Release KV blocks (safe for zero-block queued requests per BC-15)
	sim.KVCache.ReleaseKVBlocks(req)

	// Clean up computed-token tracking for ALL timed-out requests (prevents memory leak).
	// A preempted-then-queued request still has an entry from its prior running phase.
	delete(sim.reqNumComputedTokens, req.ID)

	if wasRunning {
		// New-slice construction (R21): build excluding timed-out request.
		// Allocates O(batch_size) per running-request timeout. Acceptable because:
		// (a) running-request timeouts are infrequent, (b) batch size is typically O(100s),
		// (c) R21 compliance requires avoiding in-place mutation of iterated slices.
		remaining := make([]*Request, 0, len(sim.RunningBatch.Requests)-1)
		for _, r := range sim.RunningBatch.Requests {
			if r != req {
				remaining = append(remaining, r)
			}
		}
		sim.RunningBatch.Requests = remaining
		// If batch is now empty, clear RunningBatch. Do NOT nil the host's pending step:
		// leaving it pointing to the already-scheduled StepEvent prevents the
		// INV-8 guard in TimeoutEvent.Execute from creating a duplicate StepEvent at the
		// current tick. At the same tick, StepEvents fire before TimeoutEvents (priority
		// 2 < 5), so a newly-created StepEvent would pull a queued request and
		// then its same-deadline TimeoutEvent would immediately time it out —
		// cascading once per queued seed and leaving N orphaned StepEvents.
		// The already-scheduled StepEvent fires at the correct future time and
		// the Step() orphan guard handles the empty-batch case.
		if len(remaining) == 0 {
			sim.RunningBatch = nil
		}
	} else {
		sim.WaitQ.Remove(req)
		sim.ClearDeferredKV(req.ID) // H3 (#1591): a timed-out queued request may be mid-deferral
	}
	return true
}

// quiescent reports whether the engine's next poll would schedule exactly what its last step
// did, nothing: the last step scheduled nothing and preempted nothing, no adapter load is in
// flight, and the KV store has no transfer pending or in flight (QuiescentKVStore).
func (sim *EngineCore) IsQuiescent() bool {
	if !sim.lastStepEmpty || sim.lastStepPreempted || sim.loadingAdapter != "" {
		return false
	}
	if sim.KVCache.PendingTransferLatency() != 0 {
		return false
	}
	q, ok := sim.KVCache.(QuiescentKVStore)
	return ok && q.Quiescent()
}

// parkedView is what every skipped poll of a parked engine would record: the waiting
// requests, the KV blocks in use, and the offload miss counter's advance per poll.
func (sim *EngineCore) ParkedView() (waiting int, used, missDelta int64) {
	q := sim.KVCache.(QuiescentKVStore)
	return sim.WaitQ.Len(), sim.KVCache.UsedBlocks(), q.PollMisses() - sim.pollMissesAtStep
}

// chargeSkippedPolls records the telemetry of k empty polls of a parked engine (p) as polling
// would have: the step count, the queue snapshots, the peak and the time-weighted KV usage (on
// the StepTime grid only), and the offload miss counter.
func (sim *EngineCore) ChargeSkippedPolls(k int64, p SkippedPolls) {
	sim.stepCount += int(k)
	for i := int64(0); i < k; i++ {
		sim.Rec.queueSnapshot(p.Waiting, 0)
	}
	// The peak once, then the time-weighted usage once per poll (on the StepTime grid only):
	// the float sum is accumulated poll by poll, in the same order as before.
	sim.Rec.kvUsage(p.Used, 0, false)
	if !p.Free {
		for i := int64(0); i < k; i++ {
			sim.Rec.kvUsage(p.Used, p.Quantum, true)
		}
	}
	sim.KVCache.(QuiescentKVStore).AddPollMisses(k * p.MissDelta)
}

// skippedPolls is what each skipped poll of a parked engine records: the waiting requests and
// KV blocks in use when it parked, one poll's length, whether polls cost GPU time (free: they
// do not, IdlePollEnv), and the offload miss counter's advance per poll.
type SkippedPolls struct {
	Waiting   int
	Used      int64
	Quantum   int64
	Free      bool
	MissDelta int64
}

// recordSkippedPollsEnd closes the server series' gauges at the last skipped poll's end.
func (sim *EngineCore) RecordSkippedPollsEnd(end int64) {
	if sim.serverSeries != nil {
		sim.serverSeries.recordGeneration(end, 0)
		sim.recordServerStepEnd(end, 0)
	}
}
