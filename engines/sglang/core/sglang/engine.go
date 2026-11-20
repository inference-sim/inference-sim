// Original to blis-sglang.

// Package sglang is the engine.Engine of SGLang v0.5.23: core/sched's scheduler and core/cache's
// radix cache behind the contract a simulator drives (blis-substrate contracts/engine/engine.go).
//
// A simulator calls Add when a request arrives, Schedule when it may start a pass, and Complete
// when the pass has ended. Schedule is the part of one iteration of SGLang's event loop that forms
// and launches a batch; Complete is the part that processes its result. With InFlightLimit 1 the
// two follow each other, as in event_loop_normal; with overlap scheduling on, InFlightLimit is 2 and
// the simulator launches the next pass before it completes the last, as event_loop_overlap does
// (managers/scheduler.py:2026-2134@297e54fd). The tokens a request samples are its assigned Output.
//
// What the port does not model is refused by name when a request or a configuration asks for it
// (SGL-6): the engine never runs it as something else.
package sglang

import (
	"errors"
	"fmt"
	"strconv"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/fields"

	"github.com/inference-sim/inference-sim/engines/sglang/core/cache"
	"github.com/inference-sim/inference-sim/engines/sglang/core/metrics"
	"github.com/inference-sim/inference-sim/engines/sglang/core/sched"
)

// Pricer prices a batch: how long its forward pass takes. The engine decides what runs; the cost
// model decides how long it takes (X3).
type Pricer interface {
	StepTime(b engine.Batch) engine.Duration
}

// noBudget is the max_new_tokens of a client that sets none: the scheduler clamps it to the context.
const noBudget = 1 << 30

// Engine is an SGLang scheduler as an engine.Engine.
type Engine struct {
	sc     *sched.Scheduler
	cfg    sched.Config
	pricer Pricer
	list   fields.FieldList
	col    *metrics.Collector

	nextBatch engine.BatchID
	inFlight  []*flight
	seen      map[string]bool // requests whose first admission was already reported
	scripts   map[string]*scripted
	pendingEv []engine.KVEvent
	now       engine.Time
}

type scripted struct {
	budgetEnds bool // the client's budget, not the script, ends the request
	promptLen  int
}

type flight struct {
	id engine.BatchID
	l  *sched.Launched
}

// New builds the engine over a scheduler configuration. The scheduler's pool, request rows and
// radix cache are built from cfg as SGLang's Scheduler.__init__ builds them.
func New(cfg sched.Config, pricer Pricer, list fields.FieldList) (*Engine, error) {
	if pricer == nil {
		return nil, errors.New("sglang: an engine needs a pricer")
	}
	if cfg.DisableConsecutivePrefillOverlap {
		return nil, errors.New("sglang: SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP is not modelled by the engine: SGLang processes the last result between forming a batch and launching it, which the contract's Schedule and Complete cannot express (inference-sim/blis-sglang#4)")
	}
	if len(cfg.EOSTokenIDs) == 0 {
		return nil, errors.New("sglang: the model has no end-of-sequence token, which the engine ends a request with")
	}
	sc, err := Build(cfg, 0)
	if err != nil {
		return nil, err
	}
	return &Engine{sc: sc, cfg: cfg, pricer: pricer, list: list, col: metrics.New(cfg),
		seen: map[string]bool{}, scripts: map[string]*scripted{}}, nil
}

// Build is the scheduler a configuration describes, over the cache framework's own allocator,
// request pool and radix cache.
func Build(cfg sched.Config, sampled int64) (*sched.Scheduler, error) {
	var alloc cache.Allocator
	if cfg.PageSize == 1 {
		alloc = cache.NewTokenAllocator(cfg.MaxTotalNumTokens, nil, cfg.NeedSort)
	} else {
		alloc = cache.NewPagedAllocator(cfg.MaxTotalNumTokens, cfg.PageSize, nil, cfg.NeedSort)
	}
	pool := cache.NewRequestPool(cfg.ReqPoolSize, cfg.MaxContextLen)
	policy := cfg.RadixEvictionPolicy
	if policy == "" {
		policy = "lru"
	}
	tree, err := cache.NewRadixCache(cache.RadixConfig{PageSize: cfg.PageSize, Disable: cfg.Disable, EvictionPolicy: policy,
		KVCacheEvents: cfg.EnableKVEvents, DynamoFormat: cfg.KVEventsDynamo}, alloc, pool)
	if err != nil {
		return nil, err
	}
	return sched.New(cfg, tree, alloc, pool, sampled)
}

func (e *Engine) clock(now engine.Time) {
	e.now = max(e.now, now)
}

// Add queues a request. A request the port does not model is refused by name and leaves no trace.
func (e *Engine) Add(r engine.Request, now engine.Time) error {
	e.clock(now)
	if err := r.Validate(); err != nil {
		return err
	}
	switch {
	case r.Adapter != "":
		return fmt.Errorf("sglang: LoRA adapters are not modelled (inference-sim/blis-sglang#80): request %s names %q", r.ID, r.Adapter)
	case r.RemoteDecode || r.RemotePrefill:
		return fmt.Errorf("sglang: prefill/decode disaggregation is not modelled yet (inference-sim/blis-sglang#64): request %s", r.ID)
	case len(r.Output) == 0:
		return fmt.Errorf("sglang: request %s has no assigned output", r.ID)
	}
	if _, dup := e.scripts[r.ID]; dup {
		return fmt.Errorf("sglang: request %s is already in the engine", r.ID)
	}
	// The assigned output is what the model samples. SGLang stops a request at the model's
	// end-of-sequence token unless the client sets ignore_eos (Tags["ignore_eos"]="true", as a
	// fixed-length benchmark does): the last token of the output is then that token, so the request
	// ends where the output ends and the scheduler plans for it as it plans for any request that may
	// stop early. The client's budget (max_new_tokens) ends it otherwise.
	ignoreEOS := r.Tags["ignore_eos"] == "true"
	prompt := r.Prompt
	maxBudget := r.MaxOutputTokens
	// TokenizerManager._validate_one_request (managers/tokenizer_manager.py:1331-1372@297e54fd),
	// the front of the server a client's request passes before the scheduler: an input at or past the
	// context length, or an input and output budget past it, is an error to the client, or is cut
	// under --allow-auto-truncate. (num_reserved_tokens is 0 without EAGLE, utils.py:484-499.)
	if e.cfg.ContextLen > 0 {
		in := len(prompt)
		if in >= e.cfg.ContextLen {
			if !e.cfg.AllowAutoTruncate {
				return fmt.Errorf("sglang: The input (%d tokens) is longer than the model's context length (%d tokens)", in, e.cfg.ContextLen)
			}
			prompt = prompt[:e.cfg.ContextLen]
			in = len(prompt)
		}
		if maxBudget > 0 && maxBudget+in > e.cfg.ContextLen {
			if !e.cfg.AllowAutoTruncate {
				return fmt.Errorf("sglang: Requested token count exceeds the model's maximum context length of %d tokens. You requested a total of %d tokens: %d tokens from the input messages and %d tokens for the completion. Please reduce the number of tokens in the input messages or the completion to fit within the limit",
					e.cfg.ContextLen, maxBudget+in, in, maxBudget)
			}
			maxBudget = max(0, e.cfg.ContextLen-in)
			if maxBudget == 0 {
				return fmt.Errorf("sglang: request %s: nothing is left of the context after its input", r.ID)
			}
		}
	}
	n, budgetEnds := len(r.Output), false
	maxNew := noBudget
	if maxBudget > 0 {
		maxNew = maxBudget
		if maxBudget < n {
			n, budgetEnds = maxBudget, true
		}
	}
	script := widen(r.Output[:n])
	if ignoreEOS {
		maxNew = min(maxNew, n)
	} else if !budgetEnds {
		script[n-1] = e.cfg.EOSTokenIDs[0]
	}
	a := sched.Arrival{RID: r.ID, Prompt: widen(prompt), MaxNewTokens: maxNew, IgnoreEOS: ignoreEOS, Script: script, CacheSalt: r.CacheSalt}
	if e.cfg.EnablePriorityScheduling {
		p := r.Priority
		if !e.cfg.ScheduleLowPriorityValuesFirst {
			p = -p
		}
		a.Priority = &p
	}
	if err := e.sc.Enqueue(a); err != nil {
		return err
	}
	e.scripts[r.ID] = &scripted{budgetEnds: budgetEnds, promptLen: len(prompt)}
	return nil
}

func widen(xs []int32) []int64 {
	out := make([]int64, len(xs))
	for i, x := range xs {
		out[i] = int64(x)
	}
	return out
}

// InFlightLimit is 1 in the normal loop and 2 under overlap scheduling.
func (e *Engine) InFlightLimit() int {
	if e.cfg.EnableOverlap {
		return 2
	}
	return 1
}

// Schedule forms and launches the next pass.
func (e *Engine) Schedule(now engine.Time) (engine.Batch, error) {
	e.clock(now)
	if len(e.inFlight) >= e.InFlightLimit() {
		return engine.Batch{}, engine.ErrNoCapacity
	}
	l, err := e.sc.Launch()
	if err != nil {
		return engine.Batch{}, err
	}
	reports, preempted := e.sc.TakeReports()
	e.col.Observe(reports)
	e.pendingEv = append(e.pendingEv, kvEvents(reports)...)
	b := engine.Batch{Preempted: preempted}
	for _, id := range e.sc.TakeAborted() {
		b.Finished = append(b.Finished, engine.Finished{ID: id, Reason: engine.FinishAbort})
		delete(e.scripts, id)
	}
	if l == nil {
		return b, nil
	}
	e.nextBatch++
	b.ID = e.nextBatch
	f := l.Forward
	for i, rid := range f.RIDs {
		s := engine.Scheduled{ID: rid}
		switch {
		case (f.Mode == sched.Extend || f.Mode == sched.Mixed) && i < len(f.ExtendSeqLens):
			s.Tokens, s.Computed = f.ExtendSeqLens[i], f.ExtendPrefixLen[i]
		default:
			s.Tokens, s.Computed = 1, f.SeqLens[i]-1
		}
		s.PromptLen = e.scripts[rid].promptLen
		if !e.seen[rid] {
			e.seen[rid] = true
			s.PrefixLookup = !e.cfg.Disable
			s.CachedTokens = s.Computed
		}
		b.Reqs = append(b.Reqs, s)
	}
	e.inFlight = append(e.inFlight, &flight{id: b.ID, l: l})
	return b, nil
}

// StepTime prices a batch that runs requests.
func (e *Engine) StepTime(b engine.Batch) engine.Duration {
	return max(e.pricer.StepTime(b), 1)
}

// Complete processes the result of the oldest launched pass.
func (e *Engine) Complete(id engine.BatchID, _ engine.Outcome, end engine.Time) (engine.Result, error) {
	e.clock(end)
	if len(e.inFlight) == 0 {
		return engine.Result{}, engine.ErrUnknownBatch
	}
	if e.inFlight[0].id != id {
		for _, f := range e.inFlight {
			if f.id == id {
				return engine.Result{}, engine.ErrOutOfOrder
			}
		}
		return engine.Result{}, engine.ErrUnknownBatch
	}
	f := e.inFlight[0]
	c, err := e.sc.Complete(f.l)
	if err != nil {
		return engine.Result{}, err
	}
	e.inFlight = e.inFlight[1:]
	reports, _ := e.sc.TakeReports()
	e.col.Observe(reports)
	res := engine.Result{Emitted: c.Emitted, FirstTokens: c.FirstTokens, KVEvents: append(e.pendingEv, kvEvents(reports)...)}
	e.pendingEv = nil
	for _, fin := range c.Finished {
		reason := engine.FinishLength
		switch {
		case fin.Reason == "abort":
			reason = engine.FinishAbort
		case (fin.ScriptEnded || fin.Reason == "matched_token") && !e.scripts[fin.ID].budgetEnds:
			reason = engine.FinishStop
		}
		res.Finished = append(res.Finished, engine.Finished{ID: fin.ID, Reason: reason, OutputTokens: fin.Output})
		delete(e.scripts, fin.ID)
	}
	return res, nil
}

// kvEvents turns the events the reports carry into the contract's, one per block.
func kvEvents(reports []sched.Report) []engine.KVEvent {
	var out []engine.KVEvent
	for _, r := range reports {
		for _, ev := range r.Events {
			tier := ""
			if ev.Medium != "" && ev.Medium != "GPU" {
				tier = ev.Medium
			}
			switch ev.Type {
			case "BlockStored":
				parent := ""
				if ev.ParentBlockHash != nil {
					parent = strconv.FormatInt(*ev.ParentBlockHash, 10)
				}
				for i, h := range ev.BlockHashes {
					lo, hi := i*ev.BlockSize, min((i+1)*ev.BlockSize, len(ev.TokenIDs))
					toks := make([]int32, 0, hi-lo)
					for _, t := range ev.TokenIDs[lo:hi] {
						toks = append(toks, int32(t))
					}
					hs := strconv.FormatInt(h, 10)
					out = append(out, engine.KVEvent{Hash: hs, Parent: parent, Tokens: toks, Tier: tier})
					parent = hs
				}
			case "BlockRemoved":
				for _, h := range ev.BlockHashes {
					out = append(out, engine.KVEvent{Removed: true, Hash: strconv.FormatInt(h, 10), Tier: tier})
				}
			case "AllBlocksCleared":
				// A cleared cache is reported by the removal of its blocks.
			}
		}
	}
	return out
}

// TransferDone: the engine starts no transfer yet.
func (e *Engine) TransferDone(id engine.TransferID, _ bool, now engine.Time) error {
	e.clock(now)
	return fmt.Errorf("sglang: unknown transfer %d: the engine starts none", id)
}

// Finish ends a request the engine did not decide.
func (e *Engine) Finish(id string, _ engine.FinishReason, now engine.Time) error {
	e.clock(now)
	if _, ok := e.scripts[id]; !ok {
		return nil
	}
	e.sc.Finish(id)
	delete(e.scripts, id)
	return nil
}

// NextWake: nothing changes without an input.
func (e *Engine) NextWake(engine.Time) (engine.Time, bool) { return 0, false }

// Metrics returns the engine's sglang:* series when it runs with --enable-metrics.
func (e *Engine) Metrics(now engine.Time) engine.Metrics {
	if !e.cfg.EnableMetrics {
		return engine.Metrics{At: now}
	}
	out := engine.Metrics{At: now}
	idx := map[string]int{}
	for _, s := range e.col.Series() {
		i, ok := idx[s.Name]
		if !ok {
			kind := engine.Gauge
			if len(s.Name) > 6 && s.Name[len(s.Name)-6:] == "_total" {
				kind = engine.Counter
			}
			out.Families = append(out.Families, engine.Family{Name: s.Name, Kind: kind})
			i = len(out.Families) - 1
			idx[s.Name] = i
		}
		out.Families[i].Samples = append(out.Families[i].Samples, engine.Sample{Name: s.Name, Labels: s.Labels, Value: s.Value})
	}
	return out
}

// Snapshot is the engine's state for a router.
func (e *Engine) Snapshot() engine.State {
	total, used := e.sc.Pool()
	ps := e.cfg.PageSize
	return engine.State{Waiting: e.sc.Waiting(), Running: e.sc.Running(), InFlight: len(e.inFlight),
		KVBlocksTotal: int64(total / ps), KVBlocksUsed: int64((used + ps - 1) / ps), KVBlockSize: ps}
}

// Fields is the field list of the release the engine models.
func (e *Engine) Fields() fields.FieldList { return e.list }
