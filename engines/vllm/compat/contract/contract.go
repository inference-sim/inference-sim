// Original to blis-vllm.

// Package contract puts the compatibility engine (compat/engine) behind blis-schemas'
// engine.Engine, the contract blis-vllm's v0.32.0 engine (core/vllm) implements, so a
// simulator drives every release through one driver (inference-sim GOAL.md phase 2c). This
// adapter carries colocated instances and P/D pools (handoff.go), with no data parallelism.
//
// The engine executes a whole compatibility step when it is scheduled, as compat/engine's own
// host loop does (inference-sim's Simulator.Step runs ExecuteStep, then FinishStep, at the
// step's start), and reports what the step made visible when the simulator completes the batch
// at the step's end. A step that runs nothing still holds the GPU for its duration
// (idleStepTime), so the engine asks to be woken when it ends.
package contract

import (
	"encoding/json"
	"fmt"
	"maps"
	"strings"

	"github.com/inference-sim/blis-schemas/engine"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"
	_ "github.com/inference-sim/blis-vllm/compat/kv" // registers the KV store the engine is built with
)

// TagSLOClass is the engine.Request tag that carries a request's SLO class, which the
// compatibility engine's priority policy reads (SLOPriorityMap). Other engines ignore it.
const TagSLOClass = "slo_class"

// Engine is one instance of the compatibility engine behind engine.Engine.
type Engine struct {
	core    *cengine.EngineCore
	rec     *recorder
	now     int64
	horizon int64
	reqs    map[string]*cengine.Request
	// out is how many output tokens each request has produced, as the engine counts them.
	out map[string]int
	// running is the requests the last step left in the running batch.
	running map[string]bool
	nextID  engine.BatchID
	// flight is the step executed at Schedule, reported at Complete; nil when none.
	flight *flight
	// busyUntil is when the last executed step ends: no step starts before it.
	busyUntil int64
	// wake is a step the engine itself asked for (ArmStep, WakeStep), -1 for none.
	wake int64
	// sleepUntil is the deferral wake the engine sleeps to when every waiting request waits on
	// a KV fetch (deferralWake), 0 for none.
	sleepUntil int64
	// loads are the LoRA adapter loads in flight (ScheduleAdapterLoaded), in the order they
	// started; each completes at its instant, before a step at that instant (completeLoads).
	loads []adapterLoad
	// loadIdle is set while a step left nothing running and a waiting request is held for an
	// adapter load: the host loop schedules no step then (scheduleNextStep), and the next one
	// runs when the load completes, a request arrives, a timeout leaves work waiting, or the
	// engine asks for one (WakeStep, ArmStep).
	loadIdle bool
	// firsts and done collect what a call into the engine made visible.
	firsts []string
	done   []engine.Finished
	// kvEvents are the prefix-cache changes the engine published since the last step returned.
	kvEvents []engine.KVEvent
	inEnque  bool
	dropped  bool
	// ho is the engine's P/D handoffs (handoff.go).
	ho handoffs
}

type adapterLoad struct {
	adapter string
	at      int64
}

type flight struct {
	id      engine.BatchID
	advance int64
	res     engine.Result
}

// New builds the compatibility engine for one instance from its configuration, as
// inference-sim's sim.NewEngine builds it for its own host loop, with this adapter as its host.
// It refuses what the first stage does not carry.
func New(id string, cfg cengine.SimConfig) (*Engine, error) {
	if cfg.LatencyModel == nil {
		return nil, fmt.Errorf("contract: %s: SimConfig.LatencyModel is nil", id)
	}
	kvCfg := cfg.KVCacheConfig
	kvCfg.Speculative = cfg.KV()
	kvCfg.Speculative.MaxModelLen = cfg.MaxModelLen
	kvCfg.Offload = kvCfg.Offload.ForInstance(id)
	newKVStore := cengine.KVStoreConstructor()
	if newKVStore == nil {
		return nil, fmt.Errorf("contract: %s: no KV store constructor is registered", id)
	}
	adapterCost, err := cengine.BuildAdapterCost(cfg)
	if err != nil {
		return nil, fmt.Errorf("contract: %s: adapter cost model: %w", id, err)
	}
	core, err := cengine.NewEngineCore(cfg, newKVStore(kvCfg, cfg.Seed),
		cengine.WithAdapterOverhead(cfg.LatencyModel, adapterCost))
	if err != nil {
		return nil, fmt.Errorf("contract: %s: %w", id, err)
	}
	e := &Engine{core: core, rec: &recorder{}, horizon: cfg.Horizon, reqs: map[string]*cengine.Request{},
		out: map[string]int{}, running: map[string]bool{}, wake: -1}
	core.Host = host{e}
	core.Rec = e.rec
	return e, nil
}

var _ engine.Engine = (*Engine)(nil)

// Add enqueues a request, as the host's QueuedEvent does (EnqueueRequest). A request the engine
// drops at enqueue (its length guards) is refused.
func (e *Engine) Add(r engine.Request, now engine.Time) error {
	if _, dup := e.reqs[r.ID]; dup {
		return fmt.Errorf("contract: request %s added twice", r.ID)
	}
	if r.RemotePrefill {
		return e.addRemote(r, now)
	}
	e.now = now
	req := &cengine.Request{ID: r.ID, InputTokens: tokens(r.Prompt), OutputTokens: tokens(r.Output),
		MaxOutputLen: r.MaxOutputTokens, ArrivalTime: r.ArrivalTime, Adapter: r.Adapter, State: cengine.StateQueued,
		SLOClass: r.Tags[TagSLOClass], RetainKVOnFinish: r.RemoteDecode}
	e.sleepUntil = 0 // an arrival ends a deferral sleep (the host loop's scheduleStepBy)
	e.inEnque, e.dropped = true, false
	e.core.EnqueueRequest(req)
	e.inEnque = false
	if e.core.HasWaiting() {
		// An arrival that leaves work waiting steps an engine idle for an adapter load (the host
		// loop's QueuedEvent, scheduleStepBy).
		e.loadIdle = false
	}
	if e.dropped {
		return fmt.Errorf("contract: the compatibility engine dropped request %s at enqueue", r.ID)
	}
	e.reqs[r.ID] = req
	return nil
}

// Schedule runs one compatibility step at now (Simulator.Step), and returns the batch it ran.
func (e *Engine) Schedule(now engine.Time) (engine.Batch, error) {
	if len(e.loads) > 0 {
		e.now = now
		e.completeLoads(now)
	}
	if e.flight != nil {
		return engine.Batch{}, engine.ErrNoCapacity
	}
	e.now = now
	if e.wake >= 0 && e.wake <= now {
		e.wake = -1
	}
	if e.sleepUntil > now {
		return engine.Batch{}, nil
	}
	e.sleepUntil = 0
	if e.loadIdle {
		return engine.Batch{}, nil
	}
	if now < e.busyUntil {
		return engine.Batch{}, nil
	}
	if len(e.ho.waiting) > 0 && !e.core.HasWork() {
		// An engine with nothing else to do admits its waiting handoffs at once.
		e.admitRemote()
		return engine.Batch{TransfersStarted: e.ho.take()}, nil
	}
	if e.core.NothingToStep() {
		e.core.IdleStep(now)
		return engine.Batch{}, nil
	}
	e.firsts, e.done, e.rec.lookups = e.firsts[:0], e.done[:0], map[string]cengine.PrefixLookup{}
	advance := e.core.ExecuteStep(now)
	e.busyUntil = now + advance
	var b engine.Batch
	ran := map[string]bool{}
	if rb := e.core.RunningBatch; rb != nil {
		for _, req := range rb.Requests {
			ran[req.ID] = true
			sc := engine.Scheduled{ID: req.ID, Tokens: req.NumNewTokens, PromptLen: len(req.InputTokens)}
			if l, ok := e.rec.lookups[req.ID]; ok {
				// The admission's prefix-cache lookup: what the device cache and the KV
				// offload tier supplied of the prompt (PrefixLookup.HitTokens, CPUHitTokens).
				sc.CachedTokens = int(l.HitTokens + l.CPUHitTokens)
			}
			b.Reqs = append(b.Reqs, sc)
		}
	}
	for id := range e.running {
		if !ran[id] {
			b.Preempted = append(b.Preempted, id)
		}
	}
	sortStrings(b.Preempted)
	for _, id := range b.Preempted {
		if req := e.reqs[id]; req != nil && !req.TTFTSet {
			// A request preempted before this engine sampled its first token -- a P/D decode
			// sub-request, whose decode steps the compatibility engine counts without one --
			// prefills again from nothing, and that prefill samples its first token here.
			e.out[id] = 0
		}
	}
	remaining := e.core.FinishStep(now, advance)
	if len(remaining) > 0 {
		e.core.KeepRunning(remaining)
	} else {
		e.core.ClearRunning()
		e.sleepUntil = e.deferralWake(now, now+advance)
		// The host loop schedules no step while a waiting request is held for an adapter load
		// (scheduleNextStep); the load's completion steps the engine (CompleteAdapterLoad).
		e.loadIdle = e.core.HasWaiting() && e.core.AdapterLoading()
	}
	e.core.PublishStep(now + advance)
	e.running = map[string]bool{}
	for _, req := range remaining {
		e.running[req.ID] = true
	}
	res := engine.Result{Emitted: map[string]int{}, FirstTokens: append([]string(nil), e.firsts...),
		Finished: append([]engine.Finished(nil), e.done...), KVEvents: e.kvEvents}
	e.kvEvents = nil
	res.Handoffs = e.holdHandoffs(res.Finished)
	for id := range ran {
		req := e.reqs[id]
		if req == nil {
			continue
		}
		if n := outputCount(req) - e.out[id]; n > 0 {
			res.Emitted[id] = n
			e.out[id] += n
		}
	}
	for _, f := range res.Finished {
		delete(e.reqs, f.ID)
		delete(e.out, f.ID)
	}
	b.TransfersStarted = e.ho.take()
	if len(b.Reqs) == 0 {
		// A step that ran nothing is not in flight, but held the GPU for its duration.
		return engine.Batch{Preempted: b.Preempted, TransfersStarted: b.TransfersStarted}, nil
	}
	e.nextID++
	b.ID = e.nextID
	e.flight = &flight{id: b.ID, advance: advance, res: res}
	return b, nil
}

// deferralWake is when an engine whose step at now left nothing running next steps, when every
// waiting request is deferred on a KV fetch (the host loop's idle wake, scheduleNextStep): the
// next instant a deferral can change state, clamped to the horizon, if that is later than the
// step's end (next). Otherwise 0: the engine polls on, or parks, which polling reproduces. An
// arrival ends the sleep early (Add), as does work the engine wakes itself for (WakeStep).
func (e *Engine) deferralWake(now, next int64) int64 {
	if !e.core.HasWaiting() || e.core.AdapterLoading() || e.core.IsQuiescent() {
		return 0
	}
	wake, ok := e.core.IdleDeferralWake(now)
	if !ok {
		return 0
	}
	if wake = min(wake, e.horizon); wake <= next {
		return 0
	}
	return wake
}

// completeLoads completes, in the order they started, the adapter loads that end by now, each at
// its own instant, as the host loop's AdapterLoadCompletionEvent does before a step at that
// instant (PriorityAdapterLoad < PriorityStep). A load's completion asks for a step when work
// waits (ScheduleStepIfIdle), which ends an idle wait for it.
func (e *Engine) completeLoads(now int64) {
	for len(e.loads) > 0 && e.loads[0].at <= now {
		l := e.loads[0]
		e.loads = e.loads[1:]
		e.core.CompleteAdapterLoad(l.at, l.adapter)
	}
}

// ResidentAdapterIDs are the LoRA adapters resident on the engine, sorted; nil when it serves
// none.
func (e *Engine) ResidentAdapterIDs() []string { return e.core.ResidentAdapterIDs() }

// AdapterCapacity is the engine's resident adapter slots (vLLM max_loras), 0 without LoRA.
func (e *Engine) AdapterCapacity() int { return e.core.AdapterCapacity() }

// AdapterEvents are the engine's cold adapter loads and evictions so far, by adapter (INV-L3).
// The maps are copies.
func (e *Engine) AdapterEvents() (loads, evictions map[string]int64) {
	return maps.Clone(e.rec.loads), maps.Clone(e.rec.evictions)
}

// QueueSnapshots are the engine's waiting and running request counts at each step it executed,
// in order, as its report records them (RecordQueueSnapshot): what a simulator's head-of-line
// blocking detector averages.
func (e *Engine) QueueSnapshots() (waiting, running []int) {
	return append([]int(nil), e.rec.waiting...), append([]int(nil), e.rec.runningN...)
}

// ServerSeriesSpan is the bin width of the engine's server time series (--server-series-interval-s)
// and the last instant it recorded anything, in ticks; both 0 when the engine records none. A
// simulator bins every engine of a run to the latest end, so their arrays line up.
func (e *Engine) ServerSeriesSpan() (interval, end int64) {
	return e.core.ServerSeriesIntervalUs(), e.core.ServerSeriesEnd()
}

// ServerSeriesBlock is the engine's server time series as id, binned to end, as a server_series
// block holding this one engine, in its JSON form (cengine.BuildServerSeries); nil when the
// engine records none. A simulator joins the engines of a run into one block.
func (e *Engine) ServerSeriesBlock(id string, end int64) ([]byte, error) {
	out := cengine.BuildServerSeries([]string{id}, []*cengine.EngineCore{e.core}, []int64{end})
	if out == nil {
		return nil, nil
	}
	return json.Marshal(out)
}

// CachedBlockHashes are the hashes of the blocks the engine's prefix cache can serve, sorted;
// CPUBlockHashes those its CPU KV tier holds, and TierBlockHashes those of the secondary offload
// tiers publishing as medium name. Each is nil for a KV store that does not track them.
func (e *Engine) CachedBlockHashes() []string          { return e.core.CachedBlockHashes() }
func (e *Engine) CPUBlockHashes() []string             { return e.core.CPUBlockHashes() }
func (e *Engine) TierBlockHashes(name string) []string { return e.core.TierBlockHashes(name) }

// OffloadCachedTokens is how much of id's Scheduled.CachedTokens, in the batch Schedule last
// returned, the KV offload tier supplied (PrefixLookup.CPUHitTokens).
func (e *Engine) OffloadCachedTokens(id string) int { return int(e.rec.lookups[id].CPUHitTokens) }

// KVThrashing is the KV store's thrashing rate and, for a store that counts them (the offload
// stores), the thrash events and opportunities behind it (cengine.KVThrashing).
func (e *Engine) KVThrashing() (rate float64, events, opportunities int64, counted bool) {
	rate, c, counted := e.core.KVThrashing()
	return rate, c.Events, c.Opportunities, counted
}

// outputCount is how many output tokens a request has produced: the first token its prefill
// samples, and every decode token since (ProgressIndex past the prompt).
func outputCount(req *cengine.Request) int {
	n := int(max(req.ProgressIndex-req.InputLen(), 0))
	if req.TTFTSet {
		n++
	}
	return n
}

// StepTime is the executed step's duration.
func (e *Engine) StepTime(b engine.Batch) engine.Duration {
	if e.flight == nil || e.flight.id != b.ID {
		return 1
	}
	return max(e.flight.advance, 1)
}

// Complete reports what the executed step made visible. Draft acceptance is the compatibility
// engine's own rule, so the outcome carries nothing it reads.
func (e *Engine) Complete(id engine.BatchID, _ engine.Outcome, end engine.Time) (engine.Result, error) {
	if e.flight == nil || e.flight.id != id {
		return engine.Result{}, engine.ErrUnknownBatch
	}
	res := e.flight.res
	e.flight = nil
	e.now = end
	return res, nil
}

// Finish ends a request the engine did not finish. A timeout cancels it as the engine's
// deadline does (CancelOnTimeout); an abort is a gateway eviction (Evict), whose freed state the
// engine's next step publishes (NoteAbort).
func (e *Engine) Finish(id string, why engine.FinishReason, now engine.Time) error {
	req := e.reqs[id]
	if req == nil {
		e.finishRemote(id, now)
		return nil
	}
	e.now = now
	if why == engine.FinishAbort {
		if e.core.Evict(req) {
			e.core.NoteAbort(now)
		}
	} else {
		e.core.CancelOnTimeout(req)
		if e.loadIdle && !e.core.HasRunning() && e.core.HasWaiting() {
			// The host loop's TimeoutEvent steps an engine that has no step scheduled while work
			// still waits.
			e.loadIdle = false
		}
	}
	delete(e.reqs, id)
	delete(e.out, id)
	delete(e.running, id)
	return nil
}

// NextWake is a deferral wake the engine sleeps to, the end of a step that ran nothing while
// work waits, or a step the engine asked for itself.
//
// An adapter load in flight is a wake at its completion, whether or not the engine is stepping,
// as the host loop's AdapterLoadCompletionEvent is an event of its own.
func (e *Engine) NextWake(now engine.Time) (engine.Time, bool) {
	at, ok := e.nextWake(now)
	if len(e.loads) > 0 && e.loads[0].at > now && (!ok || e.loads[0].at < at) {
		return e.loads[0].at, true
	}
	return at, ok
}

func (e *Engine) nextWake(now engine.Time) (engine.Time, bool) {
	if e.flight == nil && e.sleepUntil > now {
		return e.sleepUntil, true
	}
	if e.flight == nil && !e.loadIdle && (e.core.HasWaiting() || e.core.HasRunning() || len(e.ho.waiting) > 0) &&
		e.busyUntil > now {
		return e.busyUntil, true
	}
	if e.wake > now {
		return e.wake, true
	}
	return 0, false
}

// InFlightLimit is one: the compatibility engine runs one step at a time.
func (e *Engine) InFlightLimit() int { return 1 }

// The metric families Metrics exports: the engine's state as a scrape of its model server at an
// instant sees it. The compatibility engine publishes that view at the end of each step
// (PublishStep), so a scrape between two step ends sees the earlier one's.
const (
	MetricWaiting     = "num_requests_waiting"
	MetricRunning     = "num_requests_running"
	MetricKVUsage     = "kv_cache_usage_perc"
	MetricPreemptions = "num_preemptions"
	// MetricLoRARequests is vLLM's lora_requests_info for an engine serving LoRA adapters: one
	// info sample whose labels LabelRunningAdapters and LabelWaitingAdapters list the adapters
	// with a running or waiting request, sorted and comma-joined, and LabelMaxLoRA the resident
	// adapter slots.
	MetricLoRARequests   = "lora_requests_info"
	LabelRunningAdapters = "running_lora_adapters"
	LabelWaitingAdapters = "waiting_lora_adapters"
	LabelMaxLoRA         = "max_lora"
)

// Metrics is the engine's published view at now (EngineMetricsAt).
func (e *Engine) Metrics(now engine.Time) engine.Metrics {
	m := e.core.EngineMetricsAt(now)
	gauge := func(name string, v float64) engine.Family {
		return engine.Family{Name: name, Kind: engine.Gauge, Samples: []engine.Sample{{Name: name, Value: v}}}
	}
	out := engine.Metrics{At: m.PublishedAt, Families: []engine.Family{
		gauge(MetricWaiting, float64(m.Waiting)),
		gauge(MetricRunning, float64(m.Running)),
		gauge(MetricKVUsage, m.KVUsage),
		{Name: MetricPreemptions, Kind: engine.Counter,
			Samples: []engine.Sample{{Name: MetricPreemptions, Value: float64(m.Preemptions)}}},
	}}
	if slots := e.core.AdapterCapacity(); slots > 0 {
		out.Families = append(out.Families, engine.Family{Name: MetricLoRARequests, Kind: engine.Info,
			Samples: []engine.Sample{{Name: MetricLoRARequests, Value: 1, Labels: map[string]string{
				LabelRunningAdapters: strings.Join(m.RunningAdapters, ","),
				LabelWaitingAdapters: strings.Join(m.WaitingAdapters, ","),
				LabelMaxLoRA:         fmt.Sprint(slots),
			}}}})
	}
	return out
}

// EnableKVEvents makes the engine publish its prefix-cache changes (SetKVEventSink): from the next
// step on, a step's events come back in its Result.KVEvents, stamped at the step's end, and an
// abort's at the next step. A no-op when the engine runs no prefix cache.
func (e *Engine) EnableKVEvents() {
	e.core.SetKVEventSink(func(_ int64, events []cengine.KVEvent) {
		for _, ev := range events {
			e.kvEvents = append(e.kvEvents, engine.KVEvent{Removed: ev.Removed, Hash: ev.Hash,
				Parent: ev.Parent, Tokens: int32s(ev.Tokens), Tier: ev.Tier})
		}
	})
}

// Snapshot is the engine's queues and KV pool.
func (e *Engine) Snapshot() engine.State {
	s := engine.State{Waiting: e.core.WaitQ.Len()}
	if rb := e.core.RunningBatch; rb != nil {
		s.Running = len(rb.Requests)
	}
	if e.flight != nil {
		s.InFlight = 1
	}
	if kv := e.core.KVCache; kv != nil {
		s.KVBlocksTotal, s.KVBlocksUsed, s.KVBlockSize = kv.TotalCapacity(), kv.UsedBlocks(), int(kv.BlockSize())
	}
	return s
}

func (e *Engine) askWake(t int64) {
	if e.wake < 0 || t < e.wake {
		e.wake = t
	}
}

func (e *Engine) noteDone(req *cengine.Request) {
	if e.inEnque {
		e.dropped = true
		return
	}
	why := engine.FinishStop
	if req.LengthCapped {
		why = engine.FinishLength
	}
	e.done = append(e.done, engine.Finished{ID: req.ID, Reason: why, OutputTokens: req.GeneratedTokens})
}

// host is the engine's view of its simulator (cengine.EngineHost): the clock the adapter was
// last called at, and the callbacks the engine makes, kept for the batch they belong to.
type host struct{ e *Engine }

func (h host) CurrentClock() int64                       { return h.e.now }
func (h host) SimHorizon() int64                         { return h.e.horizon }
func (h host) ScheduleTimeout(*cengine.Request, int64)   {} // the simulator times deadlines (Finish)
func (h host) ScheduleScheduled(*cengine.Request, int64) {}
func (h host) ScheduleLeft(*cengine.Request, int64)      {}
func (h host) ScheduleAdapterLoaded(adapter string, t int64) {
	h.e.loads = append(h.e.loads, adapterLoad{adapter: adapter, at: t})
}
func (h host) RequestDone(req *cengine.Request, _ int64)  { h.e.noteDone(req) }
func (h host) FirstTokenAt(req *cengine.Request, _ int64) { h.e.firsts = append(h.e.firsts, req.ID) }
func (h host) ArmStep(t int64) {
	h.e.loadIdle = false // the host loop's scheduleStepAt
	h.e.askWake(t)
}
func (h host) StepArmed() bool {
	return h.e.wake >= 0 || h.e.flight != nil || h.e.sleepUntil > 0
}

// WakeStep asks for a step by t; it ends a deferral sleep that would wake later (the host
// loop's scheduleStepBy), and is moot under one that wakes sooner.
func (h host) WakeStep(t int64) {
	h.e.loadIdle = false // the host loop's scheduleStepBy, which steps an engine with none scheduled
	switch {
	case h.e.sleepUntil > t:
		h.e.sleepUntil = 0
	case h.e.sleepUntil > 0:
		return
	}
	h.e.askWake(t)
}

// recorder is the engine's report record (cengine.Recorder). The simulator keeps its own report
// from what the contract returns; the engine reads back only its preemption count, and the
// adapter reads the step's admission lookups into the batch it returns.
type recorder struct {
	preemptions int64
	// loads and evictions count each adapter's cold loads and evictions (INV-L3).
	loads, evictions map[string]int64
	// lookups are the prefix-cache lookups of the requests the step admitted, by ID.
	lookups map[string]cengine.PrefixLookup
	// waiting and runningN are the engine's queue lengths at each step (QueueSnapshots).
	waiting, runningN []int
}

func (r *recorder) RecordArrival(*cengine.Request, float64) {}
func (r *recorder) RecordDrop(*cengine.Request, bool)       {}
func (r *recorder) RecordInputTokens(int64)                 {}
func (r *recorder) RecordTimedOut(*cengine.Request)         {}
func (r *recorder) RecordPreemption()                       { r.preemptions++ }
func (r *recorder) PreemptionsSoFar() int64                 { return r.preemptions }
func (r *recorder) RecordScheduled(req *cengine.Request, _ int64, l cengine.PrefixLookup) {
	if r.lookups != nil && l.QueryTokens > 0 {
		r.lookups[req.ID] = l
	}
}
func (r *recorder) RecordFirstToken(*cengine.Request) {}
func (r *recorder) RecordQueueSnapshot(waiting, running int) {
	r.waiting, r.runningN = append(r.waiting, waiting), append(r.runningN, running)
}
func (r *recorder) RecordKVUsage(int64, int64, bool)    {}
func (r *recorder) RecordCompletion(cengine.Completion) {}
func (r *recorder) RecordLengthCapped(*cengine.Request) {}
func (r *recorder) RecordKVAllocationFailure()          {}
func (r *recorder) RecordAdapterLoad(id string)         { count(&r.loads, id) }
func (r *recorder) RecordAdapterEviction(id string)     { count(&r.evictions, id) }
func (r *recorder) RecordStillHeldAtEnd(int, int)       {}

func count(m *map[string]int64, id string) {
	if *m == nil {
		*m = map[string]int64{}
	}
	(*m)[id]++
}

func tokens(in []int32) []cengine.TokenID {
	out := make([]cengine.TokenID, len(in))
	for i, v := range in {
		out[i] = cengine.TokenID(v)
	}
	return out
}

func int32s(in []cengine.TokenID) []int32 {
	out := make([]int32, len(in))
	for i, v := range in {
		out[i] = int32(v)
	}
	return out
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}
