// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/engine/core.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/engine/core_client.py (DPLBAsyncMPClient, the internal load balancer) and
// vllm/v1/engine/coordinator.py (the stats publication and the wave start).

package vllm

import (
	"fmt"
	"slices"
	"sort"

	"github.com/inference-sim/blis-schemas/engine"

	core "github.com/inference-sim/blis-vllm/core/engine"
	"github.com/inference-sim/blis-vllm/core/sched"
)

// DPGroup is a data-parallel group of vLLM v0.32.0 engine ranks behind one endpoint, through
// blis-schemas' engine.Engine: what vLLM's internal load balancer exposes (one front end, every
// rank behind it, data_parallel_load_balancing internal). A request is routed to a rank by the
// balancer (DPLBAsyncMPClient.get_core_engine_for_request, core_client.py:1642-1697@5df63a72);
// every rank steps in lockstep, one pass of DPEngineCoreProc.run_busy_loop
// (core.py:2328-2405@5df63a72) per batch, its waves, dummy batches and prefill throttle being
// core/engine.DP's, which tier 3 diff-tests against vLLM (oracle/run_dp.py).
//
// One contract batch is one lockstep iteration: every rank's scheduled requests, IDs unique
// across ranks. Its step time is the slowest rank's forward pass, the group's all-reduce making
// every rank wait for it; a rank with nothing ready runs vLLM's dummy batch inside that step.
// The pricer prices the iteration's per-rank batches (DPPricer) when it can, the merged batch
// otherwise.
//
// With a batch queue (async scheduling, max_concurrent_batches 2), each rank's step is its own
// step_with_batch_queue (core.py:698-812@5df63a72), diff-tested inside the DP loop against vLLM
// (oracle/run_dp.py, core/engine/testdata/golden-dp-async): a rank's batch stays queued past the
// iteration that scheduled it, and the iteration's Complete applies the output each rank's step
// blocked on, which is that rank's oldest queued batch, scheduled this iteration or earlier.
// The iteration's own forward passes are what it prices; the group's all-reduce still ends
// every iteration, so the contract keeps one iteration in flight.
type DPGroup struct {
	dp     *core.DP
	ranks  []*rankState
	pricer Pricer
	rankOf map[string]int
	// abortedOn are the requests Finish aborted, or that finished, while an iteration in flight
	// scheduled them, by rank: their rankOf entries go once no iteration in flight does.
	abortedOn map[string]int
	// inflight are the lockstep iterations whose batches run tokens and have not completed,
	// oldest first: at most the ranks' batch queue size (InFlightLimit).
	inflight []*dpIter
	// open is the iteration whose step still waits on a rank's output: a rank blocked on its
	// oldest queued batch, which runs tokens and completes with an in-flight iteration. Its
	// wave all-reduce (core.DP.EndStep) runs once every such rank has its output.
	open      *dpOpen
	queueSize int
	nextID    engine.BatchID
	// statsIntervalUs is the coordinator's publication interval; epoch is the interval the
	// balancer's counts were last published in, -1 before the first.
	statsIntervalUs int64
	epoch           int64
	// busyUntil is when an iteration that ran only dummy batches ends: no batch of it is in
	// flight through the contract, so the group asks to be stepped then (NextWake).
	busyUntil engine.Time
	// iterations counts the lockstep iterations run; last is the last ended iteration's
	// per-rank record.
	iterations int
	last       []core.RankStep
}

// rankState is one rank: its scheduler, its engine core and the requests it serves.
type rankState struct {
	e *Engine
}

// dpIter is one lockstep iteration in flight: its contract batch and each rank's batch it
// scheduled that runs tokens.
type dpIter struct {
	id    engine.BatchID
	batch engine.Batch
	rank  map[int]core.Batch
}

// dpOpen is the iteration under way: every rank's step record, and which ranks' steps wait on
// their oldest queued batch's output.
type dpOpen struct {
	steps []core.RankStep
	waits []bool
}

// DPPricer prices one lockstep iteration of a data-parallel group from its per-rank batches,
// returning each rank's forward pass; the group's step is the slowest. A rank with an empty
// batch runs the dummy batch.
type DPPricer interface {
	StepTimesAcrossDP(ranks []engine.Batch) []engine.Duration
}

// newDPGroup builds a group of st.dp.ranks engines from one resolved configuration.
func newDPGroup(st settings, pricer Pricer) (*DPGroup, error) {
	if pricer == nil {
		return nil, fmt.Errorf("vllm: a pricer is required: the latency kernel's, or fixed durations")
	}
	n := st.dp.ranks
	g := &DPGroup{dp: &core.DP{LB: core.NewLoadBalancer(n)}, pricer: pricer, rankOf: map[string]int{}, abortedOn: map[string]int{},
		queueSize: max(st.queueSize, 1), statsIntervalUs: max(st.dp.statsIntervalUs, 1), epoch: -1}
	for i := 0; i < n; i++ {
		e, err := newEngine(st, pricer)
		if err != nil {
			return nil, err
		}
		rank := i
		e.queued = func(id string) bool { return g.inFlight(rank, id) }
		g.ranks = append(g.ranks, &rankState{e: e})
		g.dp.Ranks = append(g.dp.Ranks, e.core)
		g.dp.Waves = append(g.dp.Waves, &core.Wave{PrefillScheduleInterval: st.dp.prefillInterval,
			SyncInterval: st.dp.syncInterval})
	}
	return g, nil
}

// publish is the coordinator's stats update at now: the balancer's counts become the ranks'
// [waiting, running, kv_cache_usage] at the first routing of each publication interval
// (coordinator.py:154-162, 267-269; core_client.py:1562-1575@5df63a72).
func (g *DPGroup) publish(now engine.Time) {
	epoch := int64(now) / g.statsIntervalUs
	if epoch == g.epoch {
		return
	}
	g.epoch = epoch
	counts := make([][3]float64, len(g.ranks))
	for i, r := range g.ranks {
		nr, nw := r.e.s.Counts()
		counts[i] = [3]float64{float64(nw), float64(nr), r.e.s.KV.Pool.Usage()}
	}
	g.dp.LB.Publish(counts)
}

// Add routes the request with the internal load balancer and hands it to that rank; a request
// arriving at an idle wave starts the next wave on every rank (core.DP.Add).
func (g *DPGroup) Add(r engine.Request, now engine.Time) error {
	if _, dup := g.rankOf[r.ID]; dup {
		return fmt.Errorf("vllm: request %s added twice", r.ID)
	}
	g.publish(now)
	rank := g.dp.Add(r.ID)
	if err := g.ranks[rank].e.Add(r, now); err != nil {
		g.dp.LB.Finished(r.ID)
		return err
	}
	g.rankOf[r.ID] = rank
	return nil
}

// Schedule starts one lockstep iteration: every rank with work takes its engine step up to where
// it blocks on a GPU result (core.DP.StepBegin). The contract batch is the iteration's batches
// that run tokens, in flight until Complete. A rank whose step blocks on a schedule that ran no
// tokens applies it at once: it has no forward pass to wait for. A rank whose step blocks on a
// batch that runs tokens waits for it: with one batch in flight (step()) that is the batch it
// just scheduled, with a batch queue (step_with_batch_queue) the one it queued the iteration
// before (vllm/v1/engine/core.py:754-766@5df63a72). The iteration ends, with its dummy batches
// and the wave's all-reduce, when no rank waits any longer.
//
// No iteration starts while one is under way, while an iteration of dummy batches is still
// running, or when no rank has work. An iteration that runs no tokens on any rank is not in
// flight: its dummy batches are priced, and the group steps again when they end.
func (g *DPGroup) Schedule(now engine.Time) (engine.Batch, error) {
	if len(g.inflight) >= g.InFlightLimit() {
		return engine.Batch{}, engine.ErrNoCapacity
	}
	if g.open != nil || now < g.busyUntil || !g.stepping() {
		return engine.Batch{}, nil
	}
	steps := g.dp.StepBegin()
	g.iterations++
	it := &dpIter{rank: map[int]core.Batch{}}
	open := &dpOpen{steps: steps, waits: make([]bool, len(steps))}
	waiting := false
	for i, st := range steps {
		e := g.ranks[i].e
		if st.Step.Scheduled != nil {
			cb := *st.Step.Scheduled
			rb, err := e.batchOf(cb)
			if err != nil {
				return engine.Batch{}, err
			}
			e.recordSchedule(cb)
			if core.Executed(cb) {
				it.batch.Reqs = append(it.batch.Reqs, rb.Reqs...)
				it.rank[i] = cb
			}
			it.batch.Preempted = append(it.batch.Preempted, rb.Preempted...)
		}
		if !st.Block {
			continue
		}
		if oldest, _ := e.core.Oldest(); core.Executed(oldest) {
			open.waits[i], waiting = true, true
			continue
		}
		res := e.core.StepEnd(st.Step)
		steps[i].Step = res
		e.finishStep(*res.Processed, nil, res.Finished, now)
	}
	if !it.batch.Empty() {
		g.nextID++
		it.id, it.batch.ID = g.nextID, g.nextID
		g.inflight = append(g.inflight, it)
	}
	switch {
	case waiting:
		g.open = open
	case it.batch.Empty():
		// Every rank ran a dummy batch, or nothing: no forward pass of it is in flight.
		g.endStep(steps)
		g.busyUntil = now + g.dummyTime()
	default:
		g.endStep(steps)
	}
	return it.batch, nil
}

// endStep ends an iteration: the dummy batches and the wave's all-reduce (core.DP.EndStep).
func (g *DPGroup) endStep(steps []core.RankStep) {
	g.dp.EndStep(steps)
	g.last = steps
}

// dummyTime prices an iteration of dummy batches on every rank.
func (g *DPGroup) dummyTime() engine.Duration {
	if p, ok := g.pricer.(DPPricer); ok {
		var worst engine.Duration
		for _, d := range p.StepTimesAcrossDP(make([]engine.Batch, len(g.ranks))) {
			worst = max(worst, d)
		}
		return max(worst, 1)
	}
	return max(g.pricer.StepTime(engine.Batch{}), 1)
}

// stepping reports whether a lockstep iteration would do anything: a rank has requests or
// unfinished work, or a wave is still running (run_busy_loop's idle check, core.py:2353-2356).
func (g *DPGroup) stepping() bool {
	for i, r := range g.ranks {
		if r.e.core.HasWork() || r.e.s.HasUnfinished() || g.dp.Waves[i].Running {
			return true
		}
	}
	return false
}

// StepTime prices the iteration: the slowest rank's forward pass. A DPPricer prices each rank's
// batch with the group's padding; another pricer prices each rank's batch alone.
func (g *DPGroup) StepTime(b engine.Batch) engine.Duration {
	per := g.rankBatches(b)
	if p, ok := g.pricer.(DPPricer); ok {
		var worst engine.Duration
		for _, d := range p.StepTimesAcrossDP(per) {
			worst = max(worst, d)
		}
		return max(worst, 1)
	}
	var worst engine.Duration
	for _, rb := range per {
		if len(rb.Reqs) > 0 {
			worst = max(worst, g.pricer.StepTime(rb))
		}
	}
	return max(worst, 1)
}

// inFlight reports whether an iteration in flight scheduled the request on rank r.
func (g *DPGroup) inFlight(r int, id string) bool {
	for _, it := range g.inflight {
		if cb, ok := it.rank[r]; ok {
			if _, sched := cb.Out.Scheduled[id]; sched {
				return true
			}
		}
	}
	return false
}

// rankBatches splits an iteration's batch into the ranks' own batches.
func (g *DPGroup) rankBatches(b engine.Batch) []engine.Batch {
	per := make([]engine.Batch, len(g.ranks))
	for _, s := range b.Reqs {
		r := g.rankOf[s.ID]
		per[r].Reqs = append(per[r].Reqs, s)
	}
	return per
}

// Complete applies an in-flight iteration's batches: on each rank, its batch's sampled tokens
// and update_from_output, the batch being that rank's oldest queued one (Core.StepEnd). The
// iteration under way whose ranks waited on these outputs then ends (endStep).
func (g *DPGroup) Complete(id engine.BatchID, outcome engine.Outcome, end engine.Time) (engine.Result, error) {
	idx := -1
	for i, it := range g.inflight {
		if it.id == id {
			idx = i
			break
		}
	}
	switch {
	case idx < 0:
		return engine.Result{}, engine.ErrUnknownBatch
	case idx > 0:
		return engine.Result{}, engine.ErrOutOfOrder
	}
	it := g.inflight[0]
	g.inflight = g.inflight[1:]
	res := engine.Result{Emitted: map[string]int{}}
	for i, r := range g.ranks {
		cb, ok := it.rank[i]
		if !ok {
			continue
		}
		e := r.e
		if oldest, ok := e.core.Oldest(); !ok || oldest.ID != cb.ID {
			return engine.Result{}, fmt.Errorf("vllm: rank %d's batch %d is not its oldest queued batch", i, cb.ID)
		}
		waited := g.open != nil && g.open.waits[i]
		var base, step core.StepResult
		if waited {
			base = g.open.steps[i].Step
		}
		rr, _ := e.apply(cb, outcome, end, func() []sched.Finished {
			step = e.core.StepEnd(base)
			return step.Finished
		})
		if waited {
			g.open.steps[i].Step, g.open.waits[i] = step, false
		}
		for k, v := range rr.Emitted {
			res.Emitted[k] = v
		}
		res.FirstTokens = append(res.FirstTokens, rr.FirstTokens...)
		res.Finished = append(res.Finished, rr.Finished...)
		res.KVEvents = append(res.KVEvents, rr.KVEvents...)
	}
	if g.open != nil && !slices.Contains(g.open.waits, true) {
		steps := g.open.steps
		g.open = nil
		g.endStep(steps)
	}
	for _, f := range res.Finished {
		g.dp.LB.Finished(f.ID)
		if r := g.rankOf[f.ID]; g.inFlight(r, f.ID) {
			// A later iteration in flight still names it: its rank entry stays, and its ID is
			// refused, until that iteration completes, as an aborted request's does.
			g.abortedOn[f.ID] = r
		} else {
			delete(g.rankOf, f.ID)
		}
	}
	for id, r := range g.abortedOn {
		if !g.inFlight(r, id) {
			delete(g.rankOf, id)
			delete(g.abortedOn, id)
		}
	}
	sort.Strings(res.FirstTokens)
	sort.Slice(res.Finished, func(i, j int) bool { return res.Finished[i].ID < res.Finished[j].ID })
	return res, nil
}

// TransferDone implements engine.Engine: the group starts no transfers.
func (g *DPGroup) TransferDone(engine.TransferID, bool, engine.Time) error {
	return engine.ErrUnknownTransfer
}

// Finish ends a request on its rank.
func (g *DPGroup) Finish(id string, why engine.FinishReason, now engine.Time) error {
	r, ok := g.rankOf[id]
	if !ok {
		return nil
	}
	g.dp.LB.Finished(id)
	err := g.ranks[r].e.Finish(id, why, now)
	if !g.inFlight(r, id) {
		// No iteration in flight names it any more; one that does still maps it to its rank
		// (rankBatches) until it completes, when its rank engine forgets it too.
		delete(g.rankOf, id)
	} else {
		g.abortedOn[id] = r
	}
	return err
}

// NextWake implements engine.Engine: with no batch in flight, a group with work steps again when
// its dummy batches end, as run_busy_loop steps again after them.
func (g *DPGroup) NextWake(now engine.Time) (engine.Time, bool) {
	if len(g.inflight) > 0 || g.open != nil || !g.stepping() {
		return 0, false
	}
	if g.busyUntil > now {
		return g.busyUntil, true
	}
	return now + idlePollUs, true
}

// InFlightLimit implements engine.Engine: each rank's max_concurrent_batches, as many lockstep
// iterations' batches as a rank's batch queue holds.
func (g *DPGroup) InFlightLimit() int { return g.queueSize }

// Metrics implements engine.Engine: every rank's families, each sample labelled with its rank
// as vLLM labels a DP engine's series with its index (engine="i").
func (g *DPGroup) Metrics(now engine.Time) engine.Metrics {
	out := engine.Metrics{At: now}
	byName := map[string]int{}
	for i, r := range g.ranks {
		for _, f := range r.e.Metrics(now).Families {
			k, ok := byName[f.Name]
			if !ok {
				k = len(out.Families)
				byName[f.Name] = k
				out.Families = append(out.Families, engine.Family{Name: f.Name, Kind: f.Kind, Help: f.Help})
			}
			for _, s := range f.Samples {
				labels := map[string]string{"engine": fmt.Sprint(i)}
				for lk, lv := range s.Labels {
					labels[lk] = lv
				}
				out.Families[k].Samples = append(out.Families[k].Samples,
					engine.Sample{Name: s.Name, Labels: labels, Value: s.Value})
			}
		}
	}
	return out
}

// Snapshot implements engine.Engine: the group's totals.
func (g *DPGroup) Snapshot() engine.State {
	var st engine.State
	for _, r := range g.ranks {
		s := r.e.Snapshot()
		st.Waiting += s.Waiting
		st.Running += s.Running
		st.KVBlocksTotal += s.KVBlocksTotal
		st.KVBlocksUsed += s.KVBlocksUsed
		st.KVBlockSize = s.KVBlockSize
	}
	st.InFlight = len(g.inflight)
	return st
}

// Assumptions are the pricer's disclosed assumptions, as Engine reports them.
func (g *DPGroup) Assumptions() []string { return g.ranks[0].e.Assumptions() }

// EnableKVEvents turns on every rank's KV cache events: the group's results carry them, each
// rank's in rank order (Complete).
func (g *DPGroup) EnableKVEvents() {
	for _, r := range g.ranks {
		r.e.EnableKVEvents()
	}
}

// Ranks is the group's data-parallel width.
func (g *DPGroup) Ranks() int { return len(g.ranks) }

// pendingOrLast is rank r's record of the last iteration, for tests.
func (g *DPGroup) pendingOrLast(r int) *core.RankStep {
	if g.last == nil || r >= len(g.last) {
		return nil
	}
	return &g.last[r]
}

var _ engine.Engine = (*DPGroup)(nil)
