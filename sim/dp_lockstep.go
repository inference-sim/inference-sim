package sim

import "fmt"

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

// LockstepGroup steps the data-parallel ranks of one MoE replica together, as vLLM's
// DPEngineCoreProc does (vllm/v1/engine/core.py:2251-2350 at v0.31.0).
//
// Every rank runs every step: a rank with requests ready runs them, one without runs a
// dummy batch (core.py:2280-2287, execute_dummy_batch :1005-1006). The MoE dispatch is a
// collective over the group, so no rank finishes a step before the slowest one: the step
// lasts the longest rank's duration, and every rank's clock advances by it. The wave ends
// when an unfinished-request all-reduce finds no rank with work; that all-reduce runs on
// the wave's first step and every DPSyncInterval-th step after (core.py:2333-2350), so a
// wave runs up to DPSyncInterval-1 dummy steps past its last request.
//
// The group's next step is one StepEvent in the leader's (rank 0's) queue; every rank's
// stepEvent points at it, so each rank's existing "a step is armed" guards see the group's
// state. A rank that asks for a step while one is armed is stepped by it.
type LockstepGroup struct {
	ranks   []*Simulator
	pricer  DPStepPricer
	pending *StepEvent
	// stepCounter counts the wave's steps since its start, as vLLM's step_counter.
	stepCounter int
	// idleSteps counts the wave's steps since one last ran a request. lastWave and
	// lastWaveIdle record both counts for the wave that ended last (for tests).
	idleSteps              int
	lastWave, lastWaveIdle int
}

// NewLockstepGroup binds ranks into one lockstep group. Rank 0 is the leader. It is an
// error to bind fewer than two ranks, a rank already in a group, or a rank with a step
// already armed.
//
// The group is priced by rank 0's latency model when it is a DPStepPricer. Otherwise each
// rank is priced at its own batch by its own model, an idle rank at an empty batch, with no
// padding across ranks.
func NewLockstepGroup(ranks []*Simulator) (*LockstepGroup, error) {
	if len(ranks) < 2 {
		return nil, fmt.Errorf("NewLockstepGroup: a data-parallel group needs >= 2 ranks, got %d", len(ranks))
	}
	g := &LockstepGroup{ranks: append([]*Simulator(nil), ranks...)}
	for i, r := range ranks {
		if r == nil {
			return nil, fmt.Errorf("NewLockstepGroup: rank %d is nil", i)
		}
		if r.lockstep != nil {
			return nil, fmt.Errorf("NewLockstepGroup: rank %d is already in a lockstep group", i)
		}
		if r.stepEvent != nil {
			return nil, fmt.Errorf("NewLockstepGroup: rank %d already has a step armed", i)
		}
	}
	if p, ok := ranks[0].latencyModel.(DPStepPricer); ok {
		g.pricer = p
	}
	for _, r := range ranks {
		r.lockstep = g
	}
	return g, nil
}

// Size is the group's rank count.
func (g *LockstepGroup) Size() int { return len(g.ranks) }

// wake arms the group's next step at t unless one is already armed.
func (g *LockstepGroup) wake(t int64) {
	if g.pending != nil {
		return
	}
	g.arm(t)
}

func (g *LockstepGroup) arm(t int64) {
	ev := &StepEvent{time: t}
	g.pending = ev
	for _, r := range g.ranks {
		r.stepEvent = ev
	}
	g.ranks[0].Schedule(ev)
}

func (g *LockstepGroup) disarm() {
	g.pending = nil
	g.lastWave, g.lastWaveIdle = g.stepCounter, g.idleSteps
	g.stepCounter, g.idleSteps = 0, 0
	for _, r := range g.ranks {
		r.stepEvent = nil
	}
}

// hasWork reports whether a rank has unfinished requests, as the scheduler's
// has_unfinished_requests does: running or waiting.
func hasWork(r *Simulator) bool {
	return (r.RunningBatch != nil && len(r.RunningBatch.Requests) > 0) || r.WaitQ.Len() > 0
}

// step runs one lockstep step for the whole group. e must be the armed event; any other
// (none is ever scheduled) is ignored.
func (g *LockstepGroup) step(e *StepEvent) {
	if e != g.pending {
		return
	}
	now := e.time
	g.drainSameTick(now)
	n := len(g.ranks)
	active := make([]bool, n)
	batches := make([][]*Request, n)
	for i, r := range g.ranks {
		if !hasWork(r) {
			continue
		}
		active[i] = true
		if r.Clock < now {
			r.Clock = now
		}
		r.scheduleBatch(now)
		batches[i] = r.scheduledRequests()
	}
	if !active[0] {
		// A dummy step is not work: the leader's clock stays at its last real event, so
		// SimEndedTime and the rates divided by it are not stretched by the wave's trailing
		// dummy steps (cf. ProcessNextEvent's lazy cancellation).
		g.ranks[0].Clock = g.ranks[0].preEventClock
	}

	durations := g.price(batches)
	var advance int64 = 1
	for i, r := range g.ranks {
		d := durations[i]
		if active[i] {
			d += r.KVCache.ConsumePendingTransferLatency()
		}
		advance = max(advance, d)
	}

	for i, r := range g.ranks {
		if !active[i] {
			continue
		}
		r.advanceBatch(now, advance)
		r.KVCache.MirrorToCPU(r.RunningBatch.Requests)
		remaining := r.processCompletions(now, advance)
		if len(remaining) > 0 {
			r.RunningBatch.Requests = remaining
		} else {
			r.RunningBatch = nil
		}
	}

	g.stepCounter++
	g.idleSteps++
	for _, a := range active {
		if a {
			g.idleSteps = 0
			break
		}
	}
	running := g.stepCounter != 1 && g.stepCounter%DPSyncInterval != 0
	for _, r := range g.ranks {
		running = running || hasWork(r)
	}
	g.pending = nil
	if running {
		g.arm(now + advance)
		return
	}
	g.disarm()
}

// drainSameTick runs, on every rank but the leader, the events due at now that order before
// a step (arrival, queueing, adapter-load completion). The group's StepEvent sits in the
// leader's queue and the cluster breaks a time tie by the lowest instance index, so without
// this a request queued on rank k > 0 at the step's own tick would wait a whole step that a
// request queued on rank 0 at the same tick joins.
func (g *LockstepGroup) drainSameTick(now int64) {
	for _, r := range g.ranks[1:] {
		for len(r.eventQueue) > 0 {
			head := r.eventQueue[0].event
			if head.Timestamp() != now || head.Priority() >= PriorityStep {
				break
			}
			r.ProcessNextEvent()
		}
	}
}

func (g *LockstepGroup) price(batches [][]*Request) []int64 {
	if g.pricer != nil {
		d := g.pricer.StepTimesAcrossDP(batches)
		if len(d) != len(batches) {
			panic(fmt.Sprintf("LockstepGroup: DPStepPricer returned %d durations for %d ranks", len(d), len(batches)))
		}
		return d
	}
	d := make([]int64, len(batches))
	for i, r := range g.ranks {
		d[i] = max(1, r.latencyModel.StepTime(batches[i]))
	}
	return d
}
