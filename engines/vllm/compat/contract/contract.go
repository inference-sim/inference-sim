// Original to blis-vllm.

// Package contract puts the compatibility engine (compat/engine) behind blis-schemas'
// engine.Engine, the contract blis-vllm's v0.32.0 engine (core/vllm) implements, so a
// simulator drives every release through one driver (inference-sim GOAL.md phase 2c). This
// first stage carries one colocated instance with no data parallelism, P/D, KV offload or
// LoRA; New refuses the others, naming them.
//
// The engine executes a whole compatibility step when it is scheduled, as compat/engine's own
// host loop does (inference-sim's Simulator.Step runs ExecuteStep, then FinishStep, at the
// step's start), and reports what the step made visible when the simulator completes the batch
// at the step's end. A step that runs nothing still holds the GPU for its duration
// (idleStepTime), so the engine asks to be woken when it ends.
package contract

import (
	"fmt"

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
	// firsts and done collect what a call into the engine made visible.
	firsts  []string
	done    []engine.Finished
	inEnque bool
	dropped bool
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
	switch {
	case cfg.HasAdapters():
		return nil, fmt.Errorf("contract: %s: LoRA adapters do not run through the engine contract yet", id)
	case cfg.Offload.IsEnabled():
		return nil, fmt.Errorf("contract: %s: a KV offload tier does not run through the engine contract yet", id)
	case cfg.LatencyModel == nil:
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
	e.now = now
	req := &cengine.Request{ID: r.ID, InputTokens: tokens(r.Prompt), OutputTokens: tokens(r.Output),
		MaxOutputLen: r.MaxOutputTokens, ArrivalTime: r.ArrivalTime, Adapter: r.Adapter, State: cengine.StateQueued,
		SLOClass: r.Tags[TagSLOClass]}
	e.inEnque, e.dropped = true, false
	e.core.EnqueueRequest(req)
	e.inEnque = false
	if e.dropped {
		return fmt.Errorf("contract: the compatibility engine dropped request %s at enqueue", r.ID)
	}
	e.reqs[r.ID] = req
	return nil
}

// Schedule runs one compatibility step at now (Simulator.Step), and returns the batch it ran.
func (e *Engine) Schedule(now engine.Time) (engine.Batch, error) {
	if e.flight != nil {
		return engine.Batch{}, engine.ErrNoCapacity
	}
	e.now = now
	if e.wake >= 0 && e.wake <= now {
		e.wake = -1
	}
	if now < e.busyUntil {
		return engine.Batch{}, nil
	}
	if e.core.NothingToStep() {
		e.core.IdleStep(now)
		return engine.Batch{}, nil
	}
	e.firsts, e.done = e.firsts[:0], e.done[:0]
	advance := e.core.ExecuteStep(now)
	e.busyUntil = now + advance
	var b engine.Batch
	ran := map[string]bool{}
	if rb := e.core.RunningBatch; rb != nil {
		for _, req := range rb.Requests {
			ran[req.ID] = true
			b.Reqs = append(b.Reqs, engine.Scheduled{ID: req.ID, Tokens: req.NumNewTokens,
				PromptLen: len(req.InputTokens)})
		}
	}
	for id := range e.running {
		if !ran[id] {
			b.Preempted = append(b.Preempted, id)
		}
	}
	sortStrings(b.Preempted)
	remaining := e.core.FinishStep(now, advance)
	if len(remaining) > 0 {
		e.core.KeepRunning(remaining)
	} else {
		e.core.ClearRunning()
	}
	e.core.PublishStep(now + advance)
	e.running = map[string]bool{}
	for _, req := range remaining {
		e.running[req.ID] = true
	}
	res := engine.Result{Emitted: map[string]int{}, FirstTokens: append([]string(nil), e.firsts...),
		Finished: append([]engine.Finished(nil), e.done...)}
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
	if len(b.Reqs) == 0 {
		// A step that ran nothing is not in flight, but held the GPU for its duration.
		return engine.Batch{Preempted: b.Preempted}, nil
	}
	e.nextID++
	b.ID = e.nextID
	e.flight = &flight{id: b.ID, advance: advance, res: res}
	return b, nil
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

// TransferDone: the first stage starts no transfers.
func (e *Engine) TransferDone(engine.TransferID, bool, engine.Time) error {
	return engine.ErrUnknownTransfer
}

// Finish ends a request the engine did not finish: a timeout or an abort (CancelOnTimeout).
func (e *Engine) Finish(id string, _ engine.FinishReason, now engine.Time) error {
	req := e.reqs[id]
	if req == nil {
		return nil
	}
	e.now = now
	e.core.CancelOnTimeout(req)
	delete(e.reqs, id)
	delete(e.out, id)
	delete(e.running, id)
	return nil
}

// NextWake is the end of a step that ran nothing while work waits, or a step the engine asked
// for itself.
func (e *Engine) NextWake(now engine.Time) (engine.Time, bool) {
	if e.flight == nil && (e.core.HasWaiting() || e.core.HasRunning()) && e.busyUntil > now {
		return e.busyUntil, true
	}
	if e.wake > now {
		return e.wake, true
	}
	return 0, false
}

// InFlightLimit is one: the compatibility engine runs one step at a time.
func (e *Engine) InFlightLimit() int { return 1 }

// Metrics: the simulator reads this engine's state through Snapshot.
func (e *Engine) Metrics(engine.Time) engine.Metrics { return engine.Metrics{} }

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

func (h host) CurrentClock() int64                        { return h.e.now }
func (h host) SimHorizon() int64                          { return h.e.horizon }
func (h host) ScheduleTimeout(*cengine.Request, int64)    {} // the simulator times deadlines (Finish)
func (h host) ScheduleScheduled(*cengine.Request, int64)  {}
func (h host) ScheduleLeft(*cengine.Request, int64)       {}
func (h host) ScheduleAdapterLoaded(string, int64)        {} // LoRA is refused at New
func (h host) RequestDone(req *cengine.Request, _ int64)  { h.e.noteDone(req) }
func (h host) FirstTokenAt(req *cengine.Request, _ int64) { h.e.firsts = append(h.e.firsts, req.ID) }
func (h host) WakeStep(t int64)                           { h.e.askWake(t) }
func (h host) ArmStep(t int64)                            { h.e.askWake(t) }
func (h host) StepArmed() bool                            { return h.e.wake >= 0 || h.e.flight != nil }

// recorder is the engine's report record (cengine.Recorder). The simulator keeps its own report
// from what the contract returns; the engine reads back only its preemption count.
type recorder struct{ preemptions int64 }

func (r *recorder) RecordArrival(*cengine.Request, float64)                       {}
func (r *recorder) RecordDrop(*cengine.Request, bool)                             {}
func (r *recorder) RecordInputTokens(int64)                                       {}
func (r *recorder) RecordTimedOut(*cengine.Request)                               {}
func (r *recorder) RecordPreemption()                                             { r.preemptions++ }
func (r *recorder) PreemptionsSoFar() int64                                       { return r.preemptions }
func (r *recorder) RecordScheduled(*cengine.Request, int64, cengine.PrefixLookup) {}
func (r *recorder) RecordFirstToken(*cengine.Request)                             {}
func (r *recorder) RecordQueueSnapshot(int, int)                                  {}
func (r *recorder) RecordKVUsage(int64, int64, bool)                              {}
func (r *recorder) RecordCompletion(cengine.Completion)                           {}
func (r *recorder) RecordLengthCapped(*cengine.Request)                           {}
func (r *recorder) RecordKVAllocationFailure()                                    {}
func (r *recorder) RecordAdapterLoad(string)                                      {}
func (r *recorder) RecordAdapterEviction(string)                                  {}
func (r *recorder) RecordStillHeldAtEnd(int, int)                                 {}

func tokens(in []int32) []cengine.TokenID {
	out := make([]cengine.TokenID, len(in))
	for i, v := range in {
		out[i] = cengine.TokenID(v)
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
