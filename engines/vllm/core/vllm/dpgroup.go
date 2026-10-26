// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/engine/core.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/engine/core_client.py (DPLBAsyncMPClient, the internal load balancer) and
// vllm/v1/engine/coordinator.py (the stats publication and the wave start).

package vllm

import (
	"fmt"
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
type DPGroup struct {
	dp      *core.DP
	ranks   []*rankState
	pricer  Pricer
	rankOf  map[string]int
	pending *dpPending
	nextID  engine.BatchID
	// statsIntervalUs is the coordinator's publication interval; epoch is the interval the
	// balancer's counts were last published in, -1 before the first.
	statsIntervalUs int64
	epoch           int64
	idleWake        engine.Time
	// last is the last completed iteration's per-rank record.
	last []core.RankStep
}

// rankState is one rank: its scheduler, its engine core and the requests it serves.
type rankState struct {
	e *Engine
}

// dpPending is the lockstep iteration in flight: each rank's scheduled batch and the
// iteration's per-rank step record.
type dpPending struct {
	id    engine.BatchID
	steps []core.RankStep
	batch engine.Batch
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
	g := &DPGroup{dp: &core.DP{LB: core.NewLoadBalancer(n)}, pricer: pricer, rankOf: map[string]int{},
		statsIntervalUs: max(st.dp.statsIntervalUs, 1), epoch: -1}
	for i := 0; i < n; i++ {
		e, err := newEngine(st, pricer)
		if err != nil {
			return nil, err
		}
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

// Schedule runs the first half of one lockstep iteration: every rank with requests schedules.
// The iteration is in flight until Complete, whether or not a rank ran tokens, because a rank
// whose wave is running steps a dummy batch even with nothing scheduled. An iteration in which
// no rank has unfinished work and no wave runs is not stepped.
func (g *DPGroup) Schedule(now engine.Time) (engine.Batch, error) {
	if g.pending != nil {
		return engine.Batch{}, engine.ErrNoCapacity
	}
	if !g.stepping() {
		return engine.Batch{}, nil
	}
	steps := g.dp.StepBegin()
	b := engine.Batch{}
	executed := false
	for i, st := range steps {
		if st.Step.Scheduled == nil {
			continue
		}
		rb, err := g.ranks[i].e.batchOf(*st.Step.Scheduled)
		if err != nil {
			return engine.Batch{}, err
		}
		g.ranks[i].e.recordSchedule(*st.Step.Scheduled)
		b.Reqs = append(b.Reqs, rb.Reqs...)
		b.Preempted = append(b.Preempted, rb.Preempted...)
		executed = executed || core.Executed(*st.Step.Scheduled)
	}
	g.nextID++
	b.ID = g.nextID
	g.pending = &dpPending{id: b.ID, steps: steps, batch: b}
	g.idleWake = 0
	// An iteration in which no rank ran tokens is still in flight: every rank stepped a dummy
	// batch (or nothing), which the group's all-reduce waits for.
	_ = executed
	return b, nil
}

// stepping reports whether a lockstep iteration would do anything: a rank has requests or
// unfinished work, or a wave is still running (run_busy_loop's idle check, core.py:2353-2356).
func (g *DPGroup) stepping() bool {
	for i, r := range g.ranks {
		if r.e.s.HasRequests() || r.e.s.HasUnfinished() || g.dp.Waves[i].Running {
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

// rankBatches splits an iteration's batch into the ranks' own batches.
func (g *DPGroup) rankBatches(b engine.Batch) []engine.Batch {
	per := make([]engine.Batch, len(g.ranks))
	for _, s := range b.Reqs {
		r := g.rankOf[s.ID]
		per[r].Reqs = append(per[r].Reqs, s)
	}
	return per
}

// Complete applies the iteration's outcome on every rank, then the dummy batches and the wave's
// all-reduce (core.DP.EndStep), and merges the ranks' results.
func (g *DPGroup) Complete(id engine.BatchID, outcome engine.Outcome, end engine.Time) (engine.Result, error) {
	if g.pending == nil || g.pending.id != id {
		return engine.Result{}, engine.ErrUnknownBatch
	}
	p := g.pending
	g.pending = nil
	res := engine.Result{Emitted: map[string]int{}}
	for i, st := range p.steps {
		if st.Step.Scheduled == nil {
			continue
		}
		rr, fin, err := g.ranks[i].e.completeLockstep(*st.Step.Scheduled, outcome)
		if err != nil {
			return engine.Result{}, err
		}
		p.steps[i].Step.Processed = st.Step.Scheduled
		p.steps[i].Step.Finished = fin
		for k, v := range rr.Emitted {
			res.Emitted[k] = v
		}
		res.FirstTokens = append(res.FirstTokens, rr.FirstTokens...)
		res.Finished = append(res.Finished, rr.Finished...)
		res.KVEvents = append(res.KVEvents, rr.KVEvents...)
	}
	g.dp.EndStep(p.steps)
	g.last = p.steps
	for _, f := range res.Finished {
		g.dp.LB.Finished(f.ID)
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
	return g.ranks[r].e.Finish(id, why, now)
}

// NextWake implements engine.Engine: an idle group with a running wave, or with work it could
// not schedule, steps again at once, as run_busy_loop does; between those, nothing changes.
func (g *DPGroup) NextWake(now engine.Time) (engine.Time, bool) {
	if g.pending == nil && g.stepping() {
		return now + idlePollUs, true
	}
	return 0, false
}

// InFlightLimit implements engine.Engine: one lockstep iteration at a time.
func (g *DPGroup) InFlightLimit() int { return 1 }

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
	if g.pending != nil {
		st.InFlight = 1
	}
	return st
}

// Assumptions are the pricer's disclosed assumptions, as Engine reports them.
func (g *DPGroup) Assumptions() []string { return g.ranks[0].e.Assumptions() }

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

// completeLockstep applies a lockstep iteration's batch to this rank: the outcome's sampled
// tokens, the scheduler's update (the batch is the oldest queued one, as core/engine.Core.StepEnd
// takes it) and the rank's metrics step. It is Complete for a batch the group scheduled.
func (e *Engine) completeLockstep(cb core.Batch, outcome engine.Outcome) (engine.Result, []sched.Finished, error) {
	e.nextID++
	e.pending = append(e.pending, pendingBatch{id: e.nextID, b: cb})
	res, fins, err := e.complete(e.nextID, outcome)
	return res, fins, err
}
