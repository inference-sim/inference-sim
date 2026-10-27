// Moved from inference-sim/sim/dp_lockstep.go at 2d2bf4af994dadeb5cf6c34cc2617319f15458c2.

package contract

import (
	"fmt"

	"github.com/inference-sim/blis-schemas/engine"

	"github.com/inference-sim/blis-vllm/compat/dplb"
	cengine "github.com/inference-sim/blis-vllm/compat/engine"
)

// DPGroup is a data-parallel replica of the compatibility engine behind engine.Engine: dp ranks,
// each a compatibility EngineCore, stepping in lockstep as vLLM's DPEngineCoreProc does
// (vllm/v1/engine/core.py:2251-2350 at v0.31.0), and fed by vLLM's internal DP load balancer
// (compat/dplb). It is the compatibility engine's counterpart of core/vllm.DPGroup, and the
// lockstep step inference-sim's host loop ran (sim.LockstepGroup).
//
// Every rank runs every step of a wave: a rank with requests ready runs them, one without runs a
// dummy batch. The MoE dispatch is a collective over the group, so the step lasts the slowest
// rank's duration. Whether the wave runs another step is the engine's rule (cengine.DPWave:
// vLLM's unfinished-request all-reduce on the wave's first step and every DPSyncInterval-th
// after). A step that runs only dummy batches is not in flight; it holds the GPUs for its
// duration, and the group asks to be woken at its end.
type DPGroup struct {
	ranks  []*Engine
	lb     *dplb.Balancer
	lbView []dplb.Rank
	pricer cengine.DPStepPricer
	wave   cengine.DPWave
	// rankOf is the rank serving each unfinished request; inflight counts them per rank, the
	// front end's exact count the balancer reads.
	rankOf   map[string]int
	inflight []int
	// armed is the group's next step, -1 for none; stepping is set while one runs, when a rank's
	// ask for a step is moot (the step that runs re-arms by the wave's rule).
	armed     int64
	stepping  bool
	busyUntil int64
	flight    *flight
	nextID    engine.BatchID
	// steps counts the group's lockstep steps, and dummy those in which no rank ran a request;
	// active is which ranks had work in the last one.
	steps, dummy int
	active       []bool
}

// NewDPGroup builds a replica of one rank per id, each from cfg as New builds one instance under
// that id.
func NewDPGroup(ids []string, cfg cengine.SimConfig) (*DPGroup, error) {
	dp := len(ids)
	if dp < 2 {
		return nil, fmt.Errorf("contract: %v: a data-parallel group needs >= 2 ranks, got %d", ids, dp)
	}
	g := &DPGroup{lb: dplb.New(dp), rankOf: map[string]int{}, inflight: make([]int, dp), armed: -1}
	for k := 0; k < dp; k++ {
		e, err := New(ids[k], cfg)
		if err != nil {
			return nil, err
		}
		e.group = g
		g.ranks = append(g.ranks, e)
		g.lbView = append(g.lbView, dpRank{g: g, k: k})
	}
	g.pricer = g.ranks[0].core.DPStepPricerOf()
	return g, nil
}

var _ engine.Engine = (*DPGroup)(nil)

// dpRank is one rank as the balancer reads it: its live queues and KV usage, and the group's
// count of its unfinished requests.
type dpRank struct {
	g *DPGroup
	k int
}

func (r dpRank) Counts() dplb.Counts {
	c := r.g.ranks[r.k].core
	usage := 0.0
	if total := c.KVCapacity(); total > 0 {
		usage = float64(c.KVUsed()) / float64(total)
	}
	return dplb.Counts{Waiting: c.QueueDepth(), Running: c.BatchSize(), KVUsage: usage}
}

func (r dpRank) InFlight() int { return r.g.inflight[r.k] }

// frontRank is one rank as the balancer reads it when the host's front end keeps the in-flight
// count: the rank's own live counts, and the front end's count of its unfinished requests on it.
type frontRank struct {
	dpRank
	inFlight int
}

func (r frontRank) InFlight() int { return r.inFlight }

// PickRank is the rank vLLM's internal DP load balancer sends a request that reached the
// replica's endpoint at now to (dplb.Balancer.Pick), for a host that keeps one instance per rank
// and hands the request to the picked rank's instance itself, through its render stage. inFlight
// is the front end's exact count of its unfinished requests on each rank, which vLLM's API-server
// front end keeps (engine_inflight, vllm/v1/engine/core_client.py:1569-1570, :1639, :1655-1660 at
// v0.31.0): raised when it sends a request, lowered when the request finishes. The ranks'
// published counts are the group's own (Counts), sampled once a stats period as the balancer does.
func (g *DPGroup) PickRank(now engine.Time, inFlight []int) (int, error) {
	if len(inFlight) != len(g.ranks) {
		return 0, fmt.Errorf("contract: %d in-flight counts for a group of %d ranks", len(inFlight), len(g.ranks))
	}
	view := make([]dplb.Rank, len(g.ranks))
	for k := range g.ranks {
		view[k] = frontRank{dpRank: dpRank{g: g, k: k}, inFlight: inFlight[k]}
	}
	return g.lb.Pick(view, now), nil
}

// Add routes the request to the rank the balancer picks at now and enqueues it there.
func (g *DPGroup) Add(r engine.Request, now engine.Time) error {
	return g.AddToRank(g.lb.Pick(g.lbView, now), r, now)
}

// AddToRank enqueues the request on rank k, for a host whose router picks the rank itself (vLLM's
// external DP load balancer, every rank its own endpoint). A rank left with work waiting asks
// the group for a step (the host loop's QueuedEvent).
func (g *DPGroup) AddToRank(k int, r engine.Request, now engine.Time) error {
	if k < 0 || k >= len(g.ranks) {
		return fmt.Errorf("contract: no rank %d in a group of %d", k, len(g.ranks))
	}
	if _, dup := g.rankOf[r.ID]; dup {
		return fmt.Errorf("contract: request %s added twice", r.ID)
	}
	e := g.ranks[k]
	if err := e.Add(r, now); err != nil {
		return err
	}
	g.rankOf[r.ID] = k
	g.inflight[k]++
	if e.core.HasWaiting() {
		g.ask(now)
	}
	return nil
}

// RankOf is the rank serving an unfinished request, and whether the group holds it.
func (g *DPGroup) RankOf(id string) (int, bool) {
	k, ok := g.rankOf[id]
	return k, ok
}

// Ranks is the group's rank count.
func (g *DPGroup) Ranks() int { return len(g.ranks) }

// ask arms the group's next step at t, or when the step still executing ends, unless one is
// armed or running (the host loop's scheduleStepAt and LockstepGroup.wake).
func (g *DPGroup) ask(t int64) {
	if g.armed >= 0 || g.stepping {
		return
	}
	g.armed = max(t, g.busyUntil)
}

// Schedule runs the group's armed lockstep step at now, and returns the batch of every rank's
// requests it ran (sim.LockstepGroup.step).
func (g *DPGroup) Schedule(now engine.Time) (engine.Batch, error) {
	if g.flight != nil {
		return engine.Batch{}, engine.ErrNoCapacity
	}
	for _, e := range g.ranks {
		e.now = now
	}
	if g.armed < 0 || g.armed > now {
		return engine.Batch{}, nil
	}
	g.armed, g.stepping = -1, true
	defer func() { g.stepping = false }()
	n := len(g.ranks)
	active := make([]bool, n)
	batches := make([][]*cengine.Request, n)
	for k, e := range g.ranks {
		e.firsts, e.done, e.rec.lookups = e.firsts[:0], e.done[:0], map[string]cengine.PrefixLookup{}
		if !cengine.HasWorkOf(e.core) {
			continue
		}
		active[k] = true
		batches[k] = e.core.LockstepSchedule(now)
	}
	durations := g.price(batches)
	var advance int64 = 1
	for k, e := range g.ranks {
		d := durations[k]
		if active[k] {
			d += e.core.ConsumeTransferLatency()
		}
		advance = max(advance, d)
	}
	// No rank starts another step before this one ends, whatever wakes it.
	g.busyUntil = now + advance
	var b engine.Batch
	var res engine.Result
	res.Emitted = map[string]int{}
	ranAny := false
	for k, e := range g.ranks {
		if !active[k] {
			continue
		}
		ranAny = true
		ran := map[string]bool{}
		for _, req := range e.core.RunningBatch.Requests {
			ran[req.ID] = true
			sc := engine.Scheduled{ID: req.ID, Tokens: req.NumNewTokens, PromptLen: len(req.InputTokens)}
			if l, ok := e.rec.lookups[req.ID]; ok {
				sc.CachedTokens = int(l.HitTokens + l.CPUHitTokens)
				sc.PrefixLookup = true
			}
			b.Reqs = append(b.Reqs, sc)
		}
		for id := range e.running {
			if !ran[id] {
				b.Preempted = append(b.Preempted, id)
			}
		}
		e.core.LockstepFinish(now, advance)
		e.running = map[string]bool{}
		if rb := e.core.RunningBatch; rb != nil {
			for _, req := range rb.Requests {
				e.running[req.ID] = true
			}
		}
		res.FirstTokens = append(res.FirstTokens, e.firsts...)
		res.Finished = append(res.Finished, e.done...)
		res.Handoffs = append(res.Handoffs, e.holdHandoffs(e.done)...)
		res.KVEvents = append(res.KVEvents, e.kvEvents...)
		e.kvEvents = nil
		for id := range ran {
			if req := e.reqs[id]; req != nil {
				if m := outputCount(req) - e.out[id]; m > 0 {
					res.Emitted[id] = m
					e.out[id] += m
				}
			}
		}
		for _, f := range e.done {
			delete(e.reqs, f.ID)
			delete(e.out, f.ID)
			g.forget(f.ID)
		}
	}
	sortStrings(b.Preempted)
	for _, e := range g.ranks {
		// The pulls a rank's schedule admitted (its remote-KV admission, inside scheduleBatch).
		b.TransfersStarted = append(b.TransfersStarted, e.ho.take()...)
	}
	g.steps++
	if !ranAny {
		g.dummy++
	}
	g.active = active
	anyWork := false
	for _, e := range g.ranks {
		anyWork = anyWork || cengine.HasWorkOf(e.core)
	}
	if g.wave.Step(anyWork) {
		g.armed = now + advance
	}
	if len(b.Reqs) == 0 {
		return engine.Batch{Preempted: b.Preempted, TransfersStarted: b.TransfersStarted}, nil
	}
	g.nextID++
	b.ID = g.nextID
	g.flight = &flight{id: b.ID, advance: advance, res: res}
	return b, nil
}

// idleHandoffs admits rank k's waiting P/D handoffs at now when the rank has no other work and no
// step is executing, as an idle decode engine does at once, and returns the pulls it started.
func (g *DPGroup) idleHandoffs(k int, now engine.Time) []engine.Transfer {
	e := g.ranks[k]
	if len(e.ho.waiting) == 0 || e.core.HasWork() || now < g.busyUntil {
		return nil
	}
	e.now = now
	e.admitRemote()
	return e.ho.take()
}

// handoffWake is when rank k's waiting handoffs can next be admitted while it has no other work:
// the end of the step executing.
func (g *DPGroup) handoffWake(k int, now engine.Time) (engine.Time, bool) {
	e := g.ranks[k]
	if len(e.ho.waiting) > 0 && !e.core.HasWork() && g.busyUntil > now {
		return g.busyUntil, true
	}
	return 0, false
}

func (g *DPGroup) price(batches [][]*cengine.Request) []int64 {
	if g.pricer != nil {
		d := g.pricer.StepTimesAcrossDP(batches)
		if len(d) != len(batches) {
			panic(fmt.Sprintf("contract: the DP step pricer returned %d durations for %d ranks", len(d), len(batches)))
		}
		return d
	}
	d := make([]int64, len(batches))
	for k, e := range g.ranks {
		d[k] = e.core.StepTimeAtLeastOne(batches[k])
	}
	return d
}

func (g *DPGroup) forget(id string) {
	if k, ok := g.rankOf[id]; ok {
		delete(g.rankOf, id)
		g.inflight[k]--
	}
}

// StepTime is the lockstep step's duration: the slowest rank's.
func (g *DPGroup) StepTime(b engine.Batch) engine.Duration {
	if g.flight == nil || g.flight.id != b.ID {
		return 1
	}
	return max(g.flight.advance, 1)
}

// Complete reports what the lockstep step made visible.
func (g *DPGroup) Complete(id engine.BatchID, _ engine.Outcome, end engine.Time) (engine.Result, error) {
	if g.flight == nil || g.flight.id != id {
		return engine.Result{}, engine.ErrUnknownBatch
	}
	res := g.flight.res
	g.flight = nil
	for _, e := range g.ranks {
		e.now = end
	}
	return res, nil
}

// TransferDone: a data-parallel group starts no transfers.
func (g *DPGroup) TransferDone(engine.TransferID, bool, engine.Time) error {
	return engine.ErrUnknownTransfer
}

// Finish ends a request on the rank that serves it (Engine.Finish). A timeout that leaves the
// rank with work waiting and nothing running asks for a step, as the host loop's TimeoutEvent
// does.
func (g *DPGroup) Finish(id string, why engine.FinishReason, now engine.Time) error {
	k, ok := g.rankOf[id]
	if !ok {
		return nil
	}
	e := g.ranks[k]
	if err := e.Finish(id, why, now); err != nil {
		return err
	}
	g.forget(id)
	if why != engine.FinishAbort && !e.core.HasRunning() && e.core.HasWaiting() {
		g.ask(now)
	}
	return nil
}

// NextWake is the group's armed step.
func (g *DPGroup) NextWake(now engine.Time) (engine.Time, bool) {
	if g.flight == nil && g.armed > now {
		return g.armed, true
	}
	return 0, false
}

// InFlightLimit is one: the group runs one lockstep step at a time.
func (g *DPGroup) InFlightLimit() int { return 1 }

// Metrics is rank 0's published view; RankMetrics gives each rank's.
func (g *DPGroup) Metrics(now engine.Time) engine.Metrics { return g.ranks[0].Metrics(now) }

// RankMetrics is rank k's published view.
func (g *DPGroup) RankMetrics(k int, now engine.Time) engine.Metrics { return g.ranks[k].Metrics(now) }

// Snapshot is the group's queues and KV pools, summed over the ranks.
func (g *DPGroup) Snapshot() engine.State {
	var s engine.State
	for _, e := range g.ranks {
		r := e.Snapshot()
		s.Waiting += r.Waiting
		s.Running += r.Running
		s.KVBlocksTotal += r.KVBlocksTotal
		s.KVBlocksUsed += r.KVBlocksUsed
		s.KVBlockSize = r.KVBlockSize
	}
	if g.flight != nil {
		s.InFlight = 1
	}
	return s
}

// RankSnapshot is rank k's queues and KV pool.
func (g *DPGroup) RankSnapshot(k int) engine.State { return g.ranks[k].Snapshot() }

// LockstepSteps are the group's lockstep steps so far, and how many of them ran only dummy
// batches.
func (g *DPGroup) LockstepSteps() (steps, dummy int) { return g.steps, g.dummy }

// LastActive is which ranks had work in the group's last lockstep step (a rank without runs a
// dummy batch), nil before the first.
func (g *DPGroup) LastActive() []bool { return append([]bool(nil), g.active...) }

// StepDue reports whether the group's next lockstep step is due at now.
func (g *DPGroup) StepDue(now engine.Time) bool {
	return g.flight == nil && g.armed >= 0 && g.armed <= now
}

// RankView is rank k of a group as an engine.Engine of its own, for a host that keeps one
// instance per rank: under vLLM's external DP load balancer every rank is its own endpoint the
// router picks among, and a P/D pool's ranks stay individually routable. A request added through
// the view runs on its rank. The group steps every rank together, so rank 0's view schedules the
// group's step, reports its wakes and every rank's share of the step, and the other views'
// schedules and wakes are empty; a host completes that one batch and applies each request's
// share to the instance that holds it.
type RankView struct {
	g *DPGroup
	k int
}

// View is rank k's view.
func (g *DPGroup) View(k int) *RankView { return &RankView{g: g, k: k} }

var _ engine.Engine = (*RankView)(nil)

func (v *RankView) Add(r engine.Request, now engine.Time) error { return v.g.AddToRank(v.k, r, now) }

// Schedule admits the rank's waiting handoffs when it is idle, and on rank 0's view runs the
// group's step; every other view's batch runs nothing.
func (v *RankView) Schedule(now engine.Time) (engine.Batch, error) {
	started := v.g.idleHandoffs(v.k, now)
	if v.k != 0 {
		return engine.Batch{TransfersStarted: started}, nil
	}
	b, err := v.g.Schedule(now)
	b.TransfersStarted = append(started, b.TransfersStarted...)
	return b, err
}

func (v *RankView) StepTime(b engine.Batch) engine.Duration { return v.g.StepTime(b) }

func (v *RankView) Complete(id engine.BatchID, o engine.Outcome, end engine.Time) (engine.Result, error) {
	return v.g.Complete(id, o, end)
}

// TransferDone ends one of the rank's transfers (a transfer's ID is its rank's).
func (v *RankView) TransferDone(id engine.TransferID, ok bool, now engine.Time) error {
	return v.g.ranks[v.k].TransferDone(id, ok, now)
}

func (v *RankView) Finish(id string, why engine.FinishReason, now engine.Time) error {
	return v.g.Finish(id, why, now)
}

// NextWake is when the rank's waiting handoffs can be admitted, and on rank 0's view also the
// group's armed step, whichever is first.
func (v *RankView) NextWake(now engine.Time) (engine.Time, bool) {
	at, ok := v.g.handoffWake(v.k, now)
	if v.k != 0 {
		return at, ok
	}
	if gat, gok := v.g.NextWake(now); gok && (!ok || gat < at) {
		return gat, true
	}
	return at, ok
}

func (v *RankView) InFlightLimit() int                     { return 1 }
func (v *RankView) Metrics(now engine.Time) engine.Metrics { return v.g.RankMetrics(v.k, now) }
func (v *RankView) Snapshot() engine.State                 { return v.g.RankSnapshot(v.k) }

// AdmitIdle admits the rank's waiting P/D handoffs at now when it has no other work and no step is
// executing, as an idle decode engine does at once, and returns the pulls it started.
func (v *RankView) AdmitIdle(now engine.Time) []engine.Transfer { return v.g.idleHandoffs(v.k, now) }

// PickRank is the group's rank pick (DPGroup.PickRank).
func (v *RankView) PickRank(now engine.Time, inFlight []int) (int, error) {
	return v.g.PickRank(now, inFlight)
}

// LockstepSteps, LastActive and StepDue are the group's (DPGroup's).
func (v *RankView) LockstepSteps() (steps, dummy int) { return v.g.LockstepSteps() }
func (v *RankView) LastActive() []bool                { return v.g.LastActive() }
func (v *RankView) StepDue(now engine.Time) bool      { return v.g.StepDue(now) }

// Rank is the view's rank, and Group its group.
func (v *RankView) Rank() int       { return v.k }
func (v *RankView) Group() *DPGroup { return v.g }

// The rank's own queries (Engine's): what the host reads per instance.
func (v *RankView) EnableKVEvents()                          { v.g.ranks[v.k].EnableKVEvents() }
func (v *RankView) QueueSnapshots() (waiting, running []int) { return v.g.ranks[v.k].QueueSnapshots() }
func (v *RankView) OffloadCachedTokens(id string) int        { return v.g.ranks[v.k].OffloadCachedTokens(id) }

// DrainWaiting drains rank k's waiting queue (Engine.DrainWaiting); the group forgets the
// drained requests' rank.
func (v *RankView) DrainWaiting() []string {
	ids := v.g.ranks[v.k].DrainWaiting()
	for _, id := range ids {
		v.g.forget(id)
	}
	return ids
}

func (v *RankView) KVThrashing() (rate float64, events, opportunities int64, counted bool) {
	return v.g.ranks[v.k].KVThrashing()
}
func (v *RankView) CachedBlockHashes() []string          { return v.g.ranks[v.k].CachedBlockHashes() }
func (v *RankView) CPUBlockHashes() []string             { return v.g.ranks[v.k].CPUBlockHashes() }
func (v *RankView) TierBlockHashes(name string) []string { return v.g.ranks[v.k].TierBlockHashes(name) }
func (v *RankView) ServerSeriesSpan() (interval, end int64) {
	return v.g.ranks[v.k].ServerSeriesSpan()
}
func (v *RankView) ServerSeriesBlock(id string, end int64) ([]byte, error) {
	return v.g.ranks[v.k].ServerSeriesBlock(id, end)
}
