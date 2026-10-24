package sim

import (
	"math"
	"os"
	"strconv"
)

// Idle park (event-driven wake). An engine whose step scheduled nothing while requests wait
// would poll again one empty step later, and keep polling until some event frees KV or adds
// work. vLLM v0.31.0's EngineCore does the same busy loop, but an empty schedule runs no
// forward pass (v1/engine/core.py:630-659 step returns total_num_scheduled_tokens > 0 as
// model_executed; :1526-1542 _process_engine_step only sleeps 1 ms when nothing executed).
// Every poll of an unchanged engine forms the same empty batch, so the simulator instead
// parks the engine and, when an event wakes it, schedules the step at the first poll time
// at or after the wake: the timing polling gives, exactly, on a fixed poll grid. The skipped
// polls' telemetry (step count, queue snapshots, KV-usage integral, server gauges, the
// offload miss counter) is charged at the wake, so the run's metrics are unchanged.
//
// An engine parks only when nothing but an external event can change what it would schedule:
// no in-flight offload work (QuiescentKVStore), no adapter load, no data-parallel lockstep
// group, and a last step that preempted nothing. A step that preempts skips its WAITING phase
// (vllm/v1/core/sched/scheduler.py:900@5df63a72), so it can schedule nothing while the blocks
// it freed let the next poll admit a waiting request: parking there would strand that request
// with no event left to wake the engine. A P/D decode engine parks too: its remote-KV
// admission runs only with an empty wait queue (admitRemoteKV), so a poll of a parked engine
// never reaches it; a pull that lands enqueues its request, which wakes the engine.
type idlePark struct {
	at        int64 // the first skipped poll's time
	quantum   int64 // one empty step
	waiting   int   // WaitQ length while parked
	used      int64 // KV blocks in use while parked
	missDelta int64 // the offload miss counter's advance per poll
	free      bool  // polls cost no GPU time (IdlePollEnv): no KV-usage integral, no busy time
}

// canPark reports whether the engine, whose last step scheduled nothing, may park.
func (sim *Simulator) canPark() bool {
	if sim.idlePolling || !sim.lastStepEmpty || sim.lastStepPreempted || sim.lockstep != nil || sim.loadingAdapter != "" {
		return false
	}
	if sim.KVCache.PendingTransferLatency() != 0 {
		return false
	}
	q, ok := sim.KVCache.(QuiescentKVStore)
	return ok && q.Quiescent()
}

// IdlePollEnv names the opt-in vLLM idle-loop period, in us. Set (vLLM's is 1000:
// v1/engine/core.py:1538-1542 sleeps 1 ms when a step executed nothing), a parked engine wakes
// at the end of the poll period an event falls in, t + (x - 1 - (t-at) mod x) from the park,
// and its skipped polls cost no time (core.py:630-659 runs no forward pass for an empty
// schedule). Unset, an engine wakes on the StepTime(empty) grid, byte-identical to polling.
const IdlePollEnv = "BLIS_ENGINE_IDLE_POLL_US"

var idlePollUs = func() int64 {
	v, err := strconv.ParseInt(os.Getenv(IdlePollEnv), 10, 64)
	if err != nil || v < 0 {
		return -1
	}
	return v
}()

// parkIdle parks the engine; at is when polling would step next.
func (sim *Simulator) parkIdle(at, quantum int64) {
	q := sim.KVCache.(QuiescentKVStore)
	if idlePollUs >= 0 {
		at, quantum = at-quantum+max(idlePollUs, 1), max(idlePollUs, 1)
		sim.park = &idlePark{at: at, quantum: quantum, waiting: sim.WaitQ.Len(),
			used: sim.KVCache.UsedBlocks(), missDelta: q.PollMisses() - sim.pollMissesAtStep, free: true}
		sim.stepEvent = nil
		return
	}
	sim.park = &idlePark{at: at, quantum: quantum, waiting: sim.WaitQ.Len(),
		used: sim.KVCache.UsedBlocks(), missDelta: q.PollMisses() - sim.pollMissesAtStep}
	sim.stepEvent = nil
}

// unpark ends the park for a step wanted at t, charges the skipped polls and returns the
// first poll time at or after t.
func (sim *Simulator) unpark(t int64) int64 {
	p := sim.park
	k := sim.pollsBefore(t)
	sim.chargePolls(k)
	sim.park = nil
	return p.at + k*p.quantum
}

// chargePolls records k skipped empty polls as polling would have.
func (sim *Simulator) chargePolls(k int64) {
	if k <= 0 {
		return
	}
	p := sim.park
	sim.stepCount += int(k)
	for i := int64(0); i < k; i++ {
		sim.Metrics.NumWaitQRequests = append(sim.Metrics.NumWaitQRequests, p.waiting)
		sim.Metrics.NumRunningBatchRequests = append(sim.Metrics.NumRunningBatchRequests, 0)
	}
	if p.used > sim.Metrics.PeakKVBlocksUsed {
		sim.Metrics.PeakKVBlocksUsed = p.used
	}
	if !p.free {
		for i := int64(0); i < k; i++ {
			sim.Metrics.KVBlocksUsed += float64(p.used) * float64(p.quantum)
		}
	}
	sim.KVCache.(QuiescentKVStore).AddPollMisses(k * p.missDelta)
	end := p.at + k*p.quantum
	if p.free {
		end = p.at + (k-1)*p.quantum
	}
	sim.busyUntil = end
	if sim.serverSeries != nil {
		sim.serverSeries.recordGeneration(end, 0)
		sim.recordServerStepEnd(end, 0)
	}
}

// WakeIfParked schedules a parked engine's next step for an event at t that may let it
// admit work (KV freed, a pull landed). A no-op unless parked.
func (sim *Simulator) WakeIfParked(t int64) {
	switch {
	case sim.park == nil:
	case sim.WaitQ.Len() == 0:
		sim.dropPark(t)
	case sim.stepEvent == nil:
		sim.scheduleStepAt(t)
	}
}

// finalizePark charges the polls a still-parked engine would have run up to the horizon.
func (sim *Simulator) finalizePark() {
	p := sim.park
	if p == nil || sim.Horizon == math.MaxInt64 || sim.Horizon < p.at {
		return
	}
	k := (sim.Horizon-p.at)/p.quantum + 1
	sim.chargePolls(k)
	sim.Clock = max(sim.Clock, p.at+(k-1)*p.quantum)
	sim.park = nil
}

// dropPark ends a park whose wait queue emptied at t: the polls before t ran empty, and the
// next one finds nothing to do.
func (sim *Simulator) dropPark(t int64) {
	sim.chargePolls(sim.pollsBefore(t))
	sim.park = nil
}

// SetIdlePolling makes the engine poll empty steps instead of parking: the reference
// behaviour the park reproduces, kept for equivalence tests.
func (sim *Simulator) SetIdlePolling(on bool) { sim.idlePolling = on }

// EventsProcessed is the number of events this engine has executed.
func (sim *Simulator) EventsProcessed() int64 { return sim.eventsProcessed }

// pollsBefore counts the skipped polls that run before an event at t: those earlier than t,
// and one at t itself when the event sorts after a step (a timeout, PriorityTimeout).
func (sim *Simulator) pollsBefore(t int64) int64 {
	p := sim.park
	if t < p.at {
		return 0
	}
	if sim.curPriority > PriorityStep {
		return (t-p.at)/p.quantum + 1
	}
	return (t - p.at + p.quantum - 1) / p.quantum
}

// SetIdlePollPeriod sets the IdlePollEnv period process-wide (-1: the StepTime grid). For tests.
func SetIdlePollPeriod(us int64) { idlePollUs = us }
