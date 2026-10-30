// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/engine/core.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/core/sched/scheduler.py (add_request, finish_requests), vllm/v1/engine/input_processor.py
// (the prompt-length check) and vllm/v1/metrics/loggers.py (the exported families).

// Package vllm is vLLM v0.32.0's engine behind blis-schemas' engine.Engine: core/sched's
// Scheduler, core/engine's batch queue and core/metrics' stat logger, driven by a simulator
// through the engine-neutral contract instead of by vLLM's EngineCore loop.
//
// The mapping is EngineCore's, split at the points where vLLM blocks on the GPU:
//
//   - Schedule is schedule() plus the batch enqueue of step_with_batch_queue
//     (vllm/v1/engine/core.py:698-760@5df63a72): it runs the scheduler and returns the batch,
//     which stays in flight until Complete names it.
//   - Complete is the dequeue and update_from_output (:762-783), given the tokens the batch
//     sampled. The simulator supplies the step's outcome; the engine samples the request's
//     assigned output, one token per sampling request plus the drafts the outcome accepted,
//     the last assigned token as the end-of-sequence token, as the shared GPU mock does
//     (oracle/mocks.json).
//   - Finish is finish_requests(ids, FINISHED_ABORTED) (scheduler.py:2649-2712@5df63a72).
//
// A scheduling setting this engine does not model is refused when the engine is built
// (Factory), never ignored; settings that change only cost are the Pricer's: see config.go.
package vllm

import (
	"encoding/binary"
	"fmt"
	"sort"
	"strconv"

	"github.com/inference-sim/blis-schemas/engine"

	core "github.com/inference-sim/blis-vllm/core/engine"
	"github.com/inference-sim/blis-vllm/core/kv"
	"github.com/inference-sim/blis-vllm/core/metrics"
	"github.com/inference-sim/blis-vllm/core/sched"
)

// eosToken is the token the engine samples for a request's last assigned output token, which
// stops it as vLLM's check_stop does (vllm/v1/core/sched/utils.py:104-107@5df63a72). It is the
// shared GPU mock's (tests/v1/core/utils.py:43@5df63a72) and outside every prompt's
// vocabulary an engine.Request can carry, so a prompt token can never stop a request.
const eosToken = int64(1) << 40

// idlePollUs is how long vLLM's engine loop waits before stepping again after a step that ran
// no model while the scheduler still has requests: time.sleep(0.001)
// (vllm/v1/engine/core.py:1603-1619@5df63a72). A schedule that preempted can run nothing and
// still leave work its next schedule admits, so the engine asks to be stepped then.
const idlePollUs = engine.Duration(1000)

// Pricer prices a batch's forward pass. A simulator supplies the latency kernel's pricing; a
// diff test supplies fixed durations (#2021 B8).
type Pricer interface {
	StepTime(b engine.Batch) engine.Duration
}

// FixedPricer prices every step at the same duration, for tests and for a behavioural run that
// holds step time fixed.
type FixedPricer engine.Duration

// StepTime implements Pricer.
func (f FixedPricer) StepTime(engine.Batch) engine.Duration { return max(engine.Duration(f), 1) }

// Engine is one vLLM v0.32.0 engine instance.
type Engine struct {
	s       *sched.Scheduler
	core    *core.Core
	pricer  Pricer
	metrics *metrics.Model

	reqs map[string]*reqState
	// aborted are the requests Finish aborted, or that finished, while a batch in flight scheduled
	// them: their entries in reqs go once no batch in flight does.
	aborted map[string]bool
	// queued reports whether a batch in flight scheduled a request: the adapter's own batches,
	// or a data-parallel group's iterations when the engine is one of its ranks (DPGroup).
	queued  func(id string) bool
	pending []pendingBatch
	nextID  engine.BatchID
	// sched holds each in-flight batch's schedule decisions until its metrics step.
	sched map[int]scheduled
	// series is the server time series, nil when off (EnableServerSeries).
	series *serverSeries
	// adapters numbers LoRA adapters in order of first use (lora_int_id).
	adapters map[string]int
	// maxLoRAs is max_loras, 0 when LoRA is off; specTokens is num_speculative_tokens.
	maxLoRAs, specTokens int
	// async is async scheduling: a decode's drafts are the AsyncScheduler's placeholders, which
	// the worker fills with its own proposals (async_scheduler.py:25-58@5df63a72).
	async bool
	// pp is pipeline_parallel_size: how many stages a batch runs through (StageTimes).
	pp int
	// idleWake is when the engine next polls after an empty schedule that left requests, 0
	// when none is due.
	idleWake engine.Time
	// blocked is set while vLLM's busy loop, with a batch queue, would be blocked on the oldest
	// queued batch's output: no new batch forms until the simulator completes it.
	blocked bool
	// queuedEmpty are the IDs of queued schedules that ran no tokens, oldest first, whose outputs
	// a blocked step applies before the batch it waits on (core/engine's queue order).
	queuedEmpty []int
	// gpu is the sampled tokens of the batch being completed, which Core's GPU interface asks
	// for once per batch.
	gpu batchOutput
}

// reqState is what the engine keeps of a request beside the scheduler's own record.
type reqState struct {
	req       engine.Request
	maxTokens int
	sampled   int  // assigned output tokens sampled so far
	finished  bool // the scheduler or Finish ended it
	// preemptions and emitted are what the finish record reports.
	preemptions int
	emitted     int
}

// pendingBatch is one in-flight batch: the contract's ID and core/engine's batch.
type pendingBatch struct {
	id engine.BatchID
	b  core.Batch
}

// batchOutput is the GPU's answer for the batch being completed.
type batchOutput struct {
	tokens map[string][]int64
}

// Output implements core/engine.GPU.
func (g *batchOutput) Output(core.Batch) map[string][]int64 { return g.tokens }

// newEngine builds an engine from resolved settings.
func newEngine(st settings, pricer Pricer) (*Engine, error) {
	if pricer == nil {
		return nil, fmt.Errorf("vllm: a pricer is required: the latency kernel's, or fixed durations")
	}
	s, err := sched.New(st.sched)
	if err != nil {
		return nil, fmt.Errorf("vllm: %w", err)
	}
	e := &Engine{
		s: s, pricer: pricer, reqs: map[string]*reqState{}, aborted: map[string]bool{},
		metrics:  metrics.New(histogramBounds(st.sched.MaxModelLen)),
		maxLoRAs: st.sched.MaxLoRAs, specTokens: st.sched.Spec.NumTokens, async: st.sched.Async,
		pp: max(st.sched.DecodeStagger, 1),
	}
	e.core = &core.Core{Sched: s, QueueSize: st.queueSize, GPU: &e.gpu}
	return e, nil
}

// histogramBounds are the bucket bounds vLLM v0.32.0 registers for the request-token,
// iteration-token, n and preemption histograms (vllm/v1/metrics/buckets.py:107-128,172-240@5df63a72).
func histogramBounds(maxModelLen int) []metrics.Bucketed {
	tokens := build125(maxModelLen)
	out := []metrics.Bucketed{
		{Family: "vllm:iteration_tokens_total", Bounds: []float64{1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}},
		{Family: "vllm:request_params_n", Bounds: []float64{1, 2, 5, 10, 20}},
		{Family: "vllm:request_num_preemptions", Bounds: []float64{1, 2, 3, 4, 5, 10, 20}},
	}
	for _, name := range []string{"vllm:request_prompt_tokens", "vllm:request_generation_tokens",
		"vllm:request_max_num_generation_tokens", "vllm:request_params_max_tokens",
		"vllm:request_prefill_kv_computed_tokens"} {
		out = append(out, metrics.Bucketed{Family: name, Bounds: tokens})
	}
	return out
}

// build125 is build_1_2_5_buckets (vllm/v1/metrics/buckets.py:172-203@5df63a72).
func build125(maxValue int) []float64 {
	var out []float64
	for scale := 1; ; scale *= 10 {
		for _, m := range []int{1, 2, 5} {
			if m*scale > maxValue {
				return out
			}
			out = append(out, float64(m*scale))
		}
	}
}

// Add implements engine.Engine: add_request, after the input processor's checks.
func (e *Engine) Add(r engine.Request, now engine.Time) error {
	if _, dup := e.reqs[r.ID]; dup {
		return fmt.Errorf("vllm: request %s added twice", r.ID)
	}
	maxLen := e.maxModelLen()
	// The input processor refuses an empty prompt, and one that leaves no room for an output
	// token (vllm/v1/engine/input_processor.py:565-602@5df63a72).
	if len(r.Prompt) == 0 {
		return fmt.Errorf("vllm: request %s: the prompt cannot be empty", r.ID)
	}
	if len(r.Prompt) >= maxLen {
		return fmt.Errorf("vllm: request %s: the prompt (length %d) plus at least one output token is longer "+
			"than the maximum model length of %d", r.ID, len(r.Prompt), maxLen)
	}
	if len(r.Output) == 0 {
		return fmt.Errorf("vllm: request %s has no assigned output: a generating request samples at least one token", r.ID)
	}
	// An unset max_tokens generates up to max_model_len (input_processor.py:476-480@5df63a72).
	maxTokens := r.MaxOutputTokens
	if maxTokens <= 0 {
		maxTokens = maxLen - len(r.Prompt)
	}
	prompt := make([]int64, len(r.Prompt))
	for i, t := range r.Prompt {
		prompt[i] = int64(t)
	}
	sr := &sched.Request{ID: r.ID, Priority: r.Priority, ArrivalTime: float64(r.ArrivalTime) / 1e6,
		Prompt: prompt, MaxTokens: maxTokens, EOS: eosToken}
	if r.CacheSalt != "" {
		sr.Keys.CacheSalt = r.CacheSalt
	}
	if r.Adapter != "" {
		if e.maxLoRAs == 0 {
			// The input processor refuses an adapter when LoRA is off
			// (vllm/v1/engine/input_processor.py:278-286@5df63a72).
			return fmt.Errorf("vllm: request %s names adapter %q but LoRA is not enabled (max_loras unset)", r.ID, r.Adapter)
		}
		// The LoRA block key (vllm/v1/core/kv_cache_utils.py:578-595@5df63a72): its name and
		// path. An engine.Request names the adapter only, so the name stands for both.
		sr.LoRA = e.loraID(r.Adapter)
		sr.LoRAName = r.Adapter
		sr.Keys.Extra = []kv.ExtraKey{{"lora", r.Adapter, r.Adapter}}
	}
	if err := e.s.Add(sr); err != nil {
		return fmt.Errorf("vllm: %w", err)
	}
	e.reqs[r.ID] = &reqState{req: r, maxTokens: maxTokens}
	return nil
}

// loraID is a stable positive lora_int_id per adapter name, in order of first use
// (vllm/lora/request.py:25,38-39@5df63a72 requires one above zero).
func (e *Engine) loraID(name string) int {
	if e.adapters == nil {
		e.adapters = map[string]int{}
	}
	if id, ok := e.adapters[name]; ok {
		return id
	}
	e.adapters[name] = len(e.adapters) + 1
	return e.adapters[name]
}

func (e *Engine) maxModelLen() int {
	// The scheduler refuses a request past it; the engine needs it for the input checks.
	return e.s.MaxModelLen()
}

// Schedule implements engine.Engine.
//
// With one batch in flight it is schedule() of EngineCore.step (vllm/v1/engine/core.py:657-687
// @5df63a72), and the step completes when the simulator completes its batch.
//
// With a batch queue (max_concurrent_batches > 1: async scheduling or PP) it is
// step_with_batch_queue's first half (core.py:698-760@5df63a72), run only where vLLM's busy loop
// would run it. That loop schedules a batch, queues it, and returns without blocking unless the
// queue is full or nothing more can be scheduled (:754-759). A step that would block waits on
// the oldest batch's output, and no new batch forms until that batch completes (:760-783). So
// Schedule returns an empty batch, scheduling nothing, while a step is blocked: the simulator's
// next Complete ends that step, and its following Schedule call is the loop's next step.
func (e *Engine) Schedule(now engine.Time) (engine.Batch, error) {
	if len(e.pending) >= e.InFlightLimit() {
		// The contract's capacity is the batches in flight: an empty schedule holds a slot of
		// vLLM's queue but no forward pass, and is drained before the next batch is completed.
		return engine.Batch{}, engine.ErrNoCapacity
	}
	if e.blocked || e.core.InFlight() >= e.core.QueueSize {
		// vLLM's loop is blocked on the oldest queued batch, or its queue is full of empty
		// schedules waiting behind one: no new batch forms until a Complete.
		return engine.Batch{}, nil
	}
	if e.core.QueueSize > 1 {
		// Each call is one loop step, so a step with nothing to schedule still takes the oldest
		// queued batch: an empty schedule's output is applied, or the loop blocks on a batch in
		// flight (core.py:760-766).
		return e.scheduleQueued(now)
	}
	if !e.s.HasRequests() {
		return engine.Batch{}, nil
	}
	cb := e.core.Schedule()
	return e.admit(cb, now)
}

// scheduleQueued is one step_with_batch_queue (vllm/v1/engine/core.py:698-810@5df63a72) up to the
// point where it waits on a GPU result. The step schedules a batch when the scheduler has
// requests and queues it. When the queue is then full, or nothing more can be scheduled, it pops
// the oldest queued batch and waits for that batch's output (:754-766). A schedule that ran no
// tokens has no forward pass to wait for, so its output is applied at once and the step ends.
// A batch in flight is waited on: the engine is blocked until the simulator completes it.
func (e *Engine) scheduleQueued(now engine.Time) (engine.Batch, error) {
	res, block := e.core.StepBegin()
	var b engine.Batch
	executed := false
	if res.Scheduled != nil {
		cb := *res.Scheduled
		var err error
		if b, err = e.batchOf(cb); err != nil {
			return engine.Batch{}, err
		}
		e.recordSchedule(cb)
		executed = core.Executed(cb)
		if executed {
			e.nextID++
			b.ID = e.nextID
			e.pending = append(e.pending, pendingBatch{id: b.ID, b: cb})
		} else {
			// It holds a queue slot (vLLM queues its future), but it is not in flight through the
			// contract: there is nothing to price.
			e.queuedEmpty = append(e.queuedEmpty, cb.ID)
			b.ID = 0
		}
	}
	if block {
		if oldest, ok := e.core.Oldest(); ok && len(e.queuedEmpty) > 0 && oldest.ID == e.queuedEmpty[0] {
			e.drainEmpty(now)
		} else {
			e.blocked = true
		}
	}
	e.idleWake = 0
	if !executed && e.s.HasRequests() && len(e.pending) == 0 {
		// A step that ran no model sleeps 1 ms before the loop steps again
		// (vllm/v1/engine/core.py:1611-1616@5df63a72).
		e.idleWake = now + idlePollUs
	}
	return b, nil
}

// drainEmpty applies the outputs of queued schedules that ran no tokens, oldest first, while the
// oldest queued batch is one: vLLM's step pops them in turn, each at once.
func (e *Engine) drainEmpty(at engine.Time) {
	for len(e.queuedEmpty) > 0 {
		oldest, ok := e.core.Oldest()
		if !ok || oldest.ID != e.queuedEmpty[0] {
			return
		}
		e.queuedEmpty = e.queuedEmpty[1:]
		res := e.core.StepEnd(core.StepResult{})
		e.finishStep(*res.Processed, nil, res.Finished, at)
	}
}

// admit turns one scheduled batch into the contract's, and keeps it in flight if it runs tokens.
func (e *Engine) admit(cb core.Batch, now engine.Time) (engine.Batch, error) {
	b, err := e.batchOf(cb)
	if err != nil {
		return b, err
	}
	e.recordSchedule(cb)
	if b.Empty() {
		// A batch that runs nothing is not in flight (the contract), and vLLM's step() makes no
		// output for it: update_from_output with nothing scheduled changes no request.
		e.core.Drop(cb)
		e.finishStep(cb, nil, nil, now)
		e.idleWake = 0
		if e.s.HasRequests() && len(e.pending) == 0 {
			e.idleWake = now + idlePollUs
		}
		return b, nil
	}
	e.idleWake = 0
	e.nextID++
	b.ID = e.nextID
	e.pending = append(e.pending, pendingBatch{id: b.ID, b: cb})
	return b, nil
}

// batchOf is a scheduled batch in the contract's terms.
func (e *Engine) batchOf(cb core.Batch) (engine.Batch, error) {
	b := engine.Batch{}
	for _, id := range cb.Out.Order {
		r := e.s.Request(id)
		n := cb.Out.Scheduled[id]
		computed := r.NumComputed - n
		sc := engine.Scheduled{ID: id, Tokens: n, Computed: computed, PromptLen: len(r.Prompt),
			Drafts: len(cb.Out.SpecDecode[id])}
		if hit, ok := cb.Out.PrefixHit[id]; ok {
			sc.CachedTokens = hit
			// A lookup vLLM's prefix-cache stats count: at the request's admission with no
			// computed tokens, when prefix caching is on (record_prefix_cache_stats,
			// vllm/v1/core/sched/scheduler.py:1314-1317 and kv_cache_manager.py:249-262@5df63a72).
			sc.PrefixLookup = e.prefixCaching()
		}
		b.Reqs = append(b.Reqs, sc)
	}
	b.Preempted = append([]string(nil), cb.Out.Preempted...)
	for _, id := range cb.Out.Preempted {
		e.reqs[id].preemptions++
	}
	return b, nil
}

// StepTime implements engine.Engine.
func (e *Engine) StepTime(b engine.Batch) engine.Duration { return max(e.pricer.StepTime(b), 1) }

// StagePricer is a Pricer that splits a batch's price by pipeline stage, summing to StepTime.
type StagePricer interface {
	StageTimes(b engine.Batch) []engine.Duration
}

// StageTimes implements engine.Pipelined: the batch's forward pass split by pipeline stage,
// one entry per stage of pipeline_parallel_size, summing to StepTime. vLLM's batch queue keeps
// max_concurrent_batches batches in flight so each stage works on one (vllm/config/vllm.py:
// 577-588, vllm/v1/engine/core.py:698-810@5df63a72), which a simulator honours by timing each
// batch through the stages. A pricer that does not split gives one stage, the whole step.
func (e *Engine) StageTimes(b engine.Batch) []engine.Duration {
	total := e.StepTime(b)
	p, ok := e.pricer.(StagePricer)
	if !ok || e.pp <= 1 {
		return []engine.Duration{total}
	}
	st := p.StageTimes(b)
	if len(st) != e.pp {
		return []engine.Duration{total}
	}
	var sum engine.Duration
	for i := range st {
		st[i] = max(st[i], 1)
		sum += st[i]
	}
	st[len(st)-1] = max(st[len(st)-1]+total-sum, 1)
	return st
}

// Assumptions are the pricer's disclosed assumptions, when it records any: for the latency
// kernel, that it prices vLLM 0.31.0's engine behaviour for a 0.32.0 scenario. A simulator reads
// them by type-asserting interface{ Assumptions() []string } on the engine.Engine.
func (e *Engine) Assumptions() []string {
	if a, ok := e.pricer.(interface{ Assumptions() []string }); ok {
		return a.Assumptions()
	}
	return nil
}

// Complete implements engine.Engine: the batch's output and update_from_output.
func (e *Engine) Complete(id engine.BatchID, outcome engine.Outcome, end engine.Time) (engine.Result, error) {
	res, _, err := e.complete(id, outcome, end)
	return res, err
}

// complete is Complete, also returning the scheduler's finish records, which a data-parallel
// group's wave step reads (DPGroup).
func (e *Engine) complete(id engine.BatchID, outcome engine.Outcome, end engine.Time) (engine.Result, []sched.Finished, error) {
	idx := -1
	for i, p := range e.pending {
		if p.id == id {
			idx = i
			break
		}
	}
	switch {
	case idx < 0:
		return engine.Result{}, nil, engine.ErrUnknownBatch
	case idx > 0:
		return engine.Result{}, nil, engine.ErrOutOfOrder
	}
	pb := e.pending[0]
	e.pending = e.pending[1:]
	res, fins := e.apply(pb.b, outcome, end, func() []sched.Finished {
		if e.core.QueueSize > 1 {
			// The step that blocked on this batch, or a later one, takes it off the queue
			// (core.py:760-783): it is the oldest queued batch once the empty schedules queued
			// ahead of it are applied.
			e.drainEmpty(end)
			res := e.core.StepEnd(core.StepResult{})
			e.blocked = false
			return res.Finished
		}
		return e.core.Complete(pb.b)
	})
	return res, fins, nil
}

// apply is update_from_output of one batch that ran tokens: the GPU's sampled tokens for it,
// then take, which takes the batch off the queue and applies them, then the result, the
// metrics step and post_step. A data-parallel rank's batch is applied the same way (DPGroup).
func (e *Engine) apply(b core.Batch, outcome engine.Outcome, end engine.Time, take func() []sched.Finished) (engine.Result, []sched.Finished) {
	pb := pendingBatch{b: b}
	// The GPU's answer: each sampling request's next assigned token, plus the drafts the
	// outcome accepted (at most the real drafts it verified; a -1 padding draft is always
	// rejected, while under async scheduling every placeholder is the worker's real draft), the
	// last assigned token as EOS.
	tokens := map[string][]int64{}
	firsts := []string{}
	emitted := map[string]int{}
	for _, rid := range pb.b.Samples {
		st := e.reqs[rid]
		if st == nil || st.finished {
			continue
		}
		n := 1
		if drafts := pb.b.Out.SpecDecode[rid]; len(drafts) > 0 {
			real := len(drafts)
			if !e.async {
				real = 0
				for _, d := range drafts {
					if d >= 0 {
						real++
					}
				}
			}
			n += min(max(outcome.Accepted[rid], 0), real)
		}
		out := st.req.Output
		for j := 0; j < n && st.sampled < len(out); j++ {
			if st.sampled == len(out)-1 {
				tokens[rid] = append(tokens[rid], eosToken)
			} else {
				tokens[rid] = append(tokens[rid], int64(out[st.sampled]))
			}
			st.sampled++
		}
	}
	e.gpu.tokens = tokens
	before := map[string]int{}
	for rid := range tokens {
		before[rid] = len(e.s.Request(rid).Output)
	}
	fins := take()
	e.gpu.tokens = nil
	res := engine.Result{Emitted: map[string]int{}}
	for _, rid := range sortedTokens(tokens) {
		r := e.s.Request(rid)
		if r == nil {
			continue // stopped: counted from its finish record below
		}
		if got := len(r.Output) - before[rid]; got > 0 {
			st := e.reqs[rid]
			if st.emitted == 0 {
				firsts = append(firsts, rid)
			}
			st.emitted += got
			emitted[rid] = got
			res.Emitted[rid] = got
		}
	}
	// A request that stopped took its final tokens with it out of the scheduler's map; count
	// those from the finish record.
	for _, f := range fins {
		st := e.reqs[f.ID]
		if _, counted := res.Emitted[f.ID]; !counted && st != nil {
			if got := f.Output - st.emitted; got > 0 {
				if st.emitted == 0 {
					firsts = append(firsts, f.ID)
				}
				st.emitted += got
				emitted[f.ID] = got
				res.Emitted[f.ID] = got
			}
		}
	}
	sort.Strings(firsts)
	res.FirstTokens = firsts
	for _, f := range fins {
		st := e.reqs[f.ID]
		st.finished = true
		why := engine.FinishStop
		switch f.Status {
		case sched.FinishedLength:
			why = engine.FinishLength
		case sched.FinishedError:
			why = engine.FinishError
		}
		res.Finished = append(res.Finished, engine.Finished{ID: f.ID, Reason: why, OutputTokens: f.Output})
	}
	res.KVEvents = kvEvents(e.indexable(e.s.TakeKVEvents()))
	e.finishStep(pb.b, emitted, fins, end)
	// A request that finished here has had its final step consumed: its result and its metrics
	// step are recorded, so its entry goes (the scheduler freed its own at the finish). Under a
	// batch queue a later batch in flight may still name it: then it stays, finished, and its ID
	// is refused, until that batch completes, as an aborted request does (inFlight).
	for _, f := range fins {
		if e.inFlight(f.ID) {
			e.aborted[f.ID] = true
		} else {
			delete(e.reqs, f.ID)
		}
	}
	// An aborted request stayed for the batches it was in flight with; one is gone once none is.
	for id := range e.aborted {
		if !e.inFlight(id) {
			delete(e.reqs, id)
			delete(e.aborted, id)
		}
	}
	if e.specTokens > 0 && !e.async && len(pb.b.Out.Scheduled) > 0 {
		// EngineCore.post_step (vllm/v1/engine/core.py:689-696@5df63a72): without async
		// scheduling, after a step that ran, the drafter proposes num_speculative_tokens drafts
		// for every request it ran, as the shared GPU mock does (oracle/mocks.json). Only their
		// count reaches a decision; their acceptance is the outcome's (#2021 B3). Under async
		// scheduling post_step proposes nothing: the scheduler set placeholders (core/sched).
		drafts := map[string][]int64{}
		for id := range pb.b.Out.Scheduled {
			drafts[id] = make([]int64, e.specTokens)
		}
		e.s.UpdateDrafts(drafts)
	}
	return res, fins
}

// TransferDone implements engine.Engine. No connector is modelled through this adapter yet,
// so no transfer is ever started and every ID is unknown.
func (e *Engine) TransferDone(engine.TransferID, bool, engine.Time) error {
	return engine.ErrUnknownTransfer
}

// Finish implements engine.Engine: finish_requests(FINISHED_ABORTED), a no-op for an unknown
// or finished request, as vLLM's is.
func (e *Engine) Finish(id string, _ engine.FinishReason, _ engine.Time) error {
	st, ok := e.reqs[id]
	if !ok || st.finished {
		return nil
	}
	st.finished = true
	e.s.Abort([]string{id})
	if e.inFlight(id) {
		// A batch in flight still names it: its entry goes once that batch completes (apply).
		e.aborted[id] = true
	} else {
		delete(e.reqs, id)
	}
	return nil
}

// inFlight reports whether a batch in flight scheduled the request.
func (e *Engine) inFlight(id string) bool {
	if e.queued != nil {
		return e.queued(id)
	}
	for _, p := range e.pending {
		if _, ok := p.b.Out.Scheduled[id]; ok {
			return true
		}
	}
	return false
}

// NextWake implements engine.Engine. vLLM v0.32.0's scheduler, without a connector, makes no
// request wait for an instant; the one wake is its engine loop's poll after a schedule that ran
// nothing while requests remain and no batch is in flight (idlePollUs), such as a schedule whose
// only running request preempted itself, which skips the WAITING phase
// (vllm/v1/core/sched/scheduler.py:900@5df63a72) that the next schedule then runs.
func (e *Engine) NextWake(now engine.Time) (engine.Time, bool) {
	if e.idleWake > now && len(e.pending) == 0 {
		return e.idleWake, true
	}
	return 0, false
}

// InFlightLimit implements engine.Engine: max_concurrent_batches.
func (e *Engine) InFlightLimit() int { return max(e.core.QueueSize, 1) }

// Snapshot implements engine.Engine.
func (e *Engine) Snapshot() engine.State {
	nr, nw := e.s.Counts()
	pool := e.s.KV.Pool
	return engine.State{Waiting: nw, Running: nr, InFlight: len(e.pending),
		KVBlocksTotal: int64(pool.NumBlocks() - 1), KVBlocksUsed: int64(pool.NumBlocks() - 1 - pool.NumFree()),
		KVBlockSize: e.s.BlockSize()}
}

// EnableKVEvents turns on the engine's KV cache events (kv_events_config.enable_kv_cache_events),
// which a simulator calls when a router indexes them: each completed batch then reports its
// prefix-cache changes in Result.KVEvents.
func (e *Engine) EnableKVEvents() { e.s.EnableKVEvents() }

// indexable keeps the events of the KV cache groups a router's prefix index takes. vLLM
// publishes every group's events tagged with the group's kind, and llm-d's indexer drops a
// sliding-window or ring group's stores and removals (pkg/kvevents/pool.go,
// isPrefixIndexableSpecKind); the contract's KVEvent carries no group, so the engine drops them
// here instead, leaving the router's index as vLLM's tagged stream leaves it. A window block
// shares its full-attention block's hash, so a removal passed through would unindex a block the
// full-attention group still caches.
func (e *Engine) indexable(evs []kv.Event) []kv.Event {
	out := evs[:0]
	for _, ev := range evs {
		if e.s.KV.PrefixIndexable(ev.Group) {
			out = append(out, ev)
		}
	}
	return out
}

// kvEvents is the scheduler's KV events in the contract's form: a hash is vLLM's published
// 64-bit integer (maybe_convert_block_hash, vllm/v1/core/kv_cache_utils.py:96-99@5df63a72),
// written in decimal, one event per block.
func kvEvents(evs []kv.Event) []engine.KVEvent {
	var out []engine.KVEvent
	h := func(b kv.BlockHash) string {
		if b == nil {
			return ""
		}
		return strconv.FormatUint(binary.BigEndian.Uint64(b[len(b)-8:]), 10)
	}
	for _, ev := range evs {
		parent := h(ev.Parent)
		for i, bh := range ev.Hashes {
			ke := engine.KVEvent{Removed: ev.Removed, Hash: h(bh), Parent: parent}
			if !ev.Removed && ev.BlockSize > 0 && (i+1)*ev.BlockSize <= len(ev.Tokens) {
				for _, t := range ev.Tokens[i*ev.BlockSize : (i+1)*ev.BlockSize] {
					ke.Tokens = append(ke.Tokens, int32(t))
				}
			}
			out = append(out, ke)
			parent = ke.Hash
		}
	}
	return out
}

func sortedTokens(m map[string][]int64) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

var _ engine.Engine = (*Engine)(nil)
