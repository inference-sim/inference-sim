// sim/engine_lockstep.go: the engine's part of a data-parallel lockstep step (#1909): a rank's
// schedule and finish, and how a group's step is priced. The group itself, which arms the
// step event and keeps the ranks' clocks, is the host's (dp_lockstep.go).
package sim

// DPSyncInterval is how many steps a data-parallel wave runs between the ranks'
// unfinished-request all-reduces: vLLM v0.31.0's dp_sync_interval default
// (vllm/config/parallel.py:150), which blis-schemas has no field to state.
const DPSyncInterval = 16

// DPStepPricer prices one lockstep step of a data-parallel group (#1909).
//
// batches[i] is rank i's scheduled requests; an empty one is a rank with nothing ready,
// which runs vLLM's dummy batch. The result is each rank's own step duration in ticks,
// each >= 1. A pricer applies whatever padding the engine applies across the ranks.
type DPStepPricer interface {
	StepTimesAcrossDP(batches [][]*Request) []int64
}

// hasWork reports whether a rank has unfinished requests, as the scheduler's
// has_unfinished_requests does: running or waiting.
func hasWork(r *engineCore) bool {
	return (r.RunningBatch != nil && len(r.RunningBatch.Requests) > 0) || r.WaitQ.Len() > 0
}

// lockstepSchedule is a rank's schedule in a lockstep step at now: it returns the requests
// the rank's forward pass runs.
func (sim *engineCore) LockstepSchedule(now int64) []*Request {
	sim.scheduleBatch(now)
	return sim.scheduledRequests()
}

// lockstepFinish applies a lockstep step of advance ticks that started at now to a rank that
// ran: progress, mirroring, completions and the server series' step end, then keeps or
// clears its running batch.
func (sim *engineCore) LockstepFinish(now, advance int64) {
	sim.advanceBatch(now, advance)
	sim.KVCache.MirrorToCPU(sim.RunningBatch.Requests)
	remaining := sim.processCompletions(now, advance)
	sim.recordServerStepEnd(now+advance, len(remaining))
	if len(remaining) > 0 {
		sim.RunningBatch.Requests = remaining
	} else {
		sim.RunningBatch = nil
	}
}

// dpStepPricer is the engine's latency model as a DPStepPricer, nil when it is not one.
func (sim *engineCore) DPStepPricerOf() DPStepPricer {
	if p, ok := sim.latencyModel.(DPStepPricer); ok {
		return p
	}
	return nil
}

// stepTimeAtLeastOne prices batch on the engine's own latency model, at least one tick.
func (sim *engineCore) StepTimeAtLeastOne(batch []*Request) int64 {
	return max(1, sim.latencyModel.StepTime(batch))
}

// consumeTransferLatency returns and clears the KV store's pending transfer latency.
func (sim *engineCore) ConsumeTransferLatency() int64 {
	return sim.KVCache.ConsumePendingTransferLatency()
}
