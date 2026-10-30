// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/engine/core.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/config/vllm.py (max_concurrent_batches) and the DP engine core's wave loop.

// Package engine is vLLM's EngineCore over core/sched: the batch queue that keeps up to
// max_concurrent_batches batches in flight (step_with_batch_queue), the plain step() for a
// queue of one, and the data-parallel wave bookkeeping (DPEngineCoreProc's step counter,
// sync interval and prefill throttling).
//
// The GPU is the caller's: Step hands each batch it schedules to Execute, and when the
// engine needs a batch's output it calls Output with that batch, in the order the batches
// were scheduled. Nothing here reads a clock or starts a goroutine.
package engine

import (
	"fmt"

	"github.com/inference-sim/blis-vllm/core/sched"
)

// MaxConcurrentBatches is VllmConfig.max_concurrent_batches (vllm/config/vllm.py:577-588):
// PP needs pipeline_parallel_size batches in flight to fill the pipeline; async scheduling
// needs two to overlap, and with the V2 model runner pp_size + 1; the V1 runner does not
// support async scheduling with PP, so it keeps pp_size there.
func MaxConcurrentBatches(pp int, async, v2Runner bool) int {
	pp = max(pp, 1)
	if async {
		if v2Runner {
			return pp + 1
		}
		if pp <= 1 {
			return 2
		}
	}
	return pp
}

// CheckConfig refuses an EngineCore configuration vLLM refuses to start: the V1 model runner
// does not support pipeline parallelism with async scheduling
// (_get_v1_model_runner_unsupported_features, vllm/config/vllm.py:2970-2974).
func CheckConfig(pp int, async, v2Runner bool) error {
	if pp > 1 && async && !v2Runner {
		return fmt.Errorf("engine: the V1 model runner does not support pipeline parallelism (%d) "+
			"with async scheduling; vLLM refuses to start (use the V2 model runner)", pp)
	}
	if pp < 1 {
		return fmt.Errorf("engine: pipeline_parallel_size %d is not positive", pp)
	}
	return nil
}

// Batch is one scheduled step with its stable ID: the scheduler's output, as
// step_with_batch_queue queues it with its future.
type Batch struct {
	ID  int
	Out sched.Output
	// Samples are the scheduled requests that sample a token in this batch: those whose
	// step reaches the end of their known tokens (not is_prefill_chunk after
	// _update_after_schedule).
	Samples []string
}

// GPU executes batches. Output returns the tokens a batch sampled, for the requests in
// Samples (one, plus any drafts the target accepted); it is asked for batches in the order
// they were scheduled.
type GPU interface {
	Output(b Batch) map[string][]int64
}

// Core is EngineCore with a batch queue of QueueSize (core.py:216-226).
type Core struct {
	Sched     *sched.Scheduler
	QueueSize int
	GPU       GPU
	// queue holds the in-flight batches, oldest first.
	queue  []Batch
	nextID int
	// Throttle reports whether this step defers prefills (_should_throttle_prefills,
	// core.py:652-655 and 2318-2325); nil never throttles.
	Throttle func() bool
	// aborts are the client aborts queued for the next applied output (Abort).
	aborts []string
}

// StepResult is what one engine step did.
type StepResult struct {
	// Scheduled is the batch this step scheduled, if any.
	Scheduled *Batch
	// Processed is the batch whose output this step applied, if any.
	Processed *Batch
	// Finished are the requests that output finished.
	Finished []sched.Finished
	// Executed is model_executed: the scheduled batch ran tokens.
	Executed bool
}

// Schedules is how many batches the scheduler has made, each one execute_model call
// (core.py:724-729, 667-668), including those that ran no tokens.
func (c *Core) Schedules() int { return c.nextID }

// InFlight is how many batches are queued.
func (c *Core) InFlight() int { return len(c.queue) }

// HasWork is has_work without the DP wave (core.py:1534-1540).
func (c *Core) HasWork() bool { return c.Sched.HasRequests() || len(c.queue) > 0 }

func (c *Core) schedule() Batch {
	throttle := c.Throttle != nil && c.Throttle()
	out := c.Sched.ScheduleThrottled(throttle)
	b := Batch{ID: c.nextID, Out: out}
	c.nextID++
	for _, id := range out.Order {
		if r := c.Sched.Request(id); r != nil && !r.IsPrefillChunk {
			b.Samples = append(b.Samples, id)
		}
	}
	return b
}

// TransferGPU is a GPU whose worker also reports the KV transfers it finished with a batch's
// output: ModelRunnerOutput.kv_connector_output (vllm/v1/outputs.py:215-233@5df63a72), which
// update_from_output applies (vllm/v1/core/sched/scheduler.py:2058-2071, 2342-2343).
type TransferGPU interface {
	GPU
	Transfers(b Batch) sched.TransferOutcome
}

// Abort queues client aborts for the next applied output: EngineCore drains its aborts queue
// after it has a batch's output and before update_from_output (vllm/v1/engine/core.py:680,
// 780-783, 814-822@5df63a72).
func (c *Core) Abort(ids []string) { c.aborts = append(c.aborts, ids...) }

func (c *Core) process(b Batch) []sched.Finished {
	toks := c.GPU.Output(b)
	var o sched.TransferOutcome
	if tg, ok := c.GPU.(TransferGPU); ok {
		o = tg.Transfers(b)
	}
	if len(c.aborts) > 0 {
		c.Sched.Abort(c.aborts)
		c.aborts = nil
	}
	return c.Sched.UpdateWith(b.Out, toks, o)
}

// Step is one engine step: step_with_batch_queue (core.py:698-810) when QueueSize > 1,
// step() (core.py:657-687) otherwise. It is StepBegin, then StepEnd when the step blocks.
func (c *Core) Step() StepResult {
	res, block := c.StepBegin()
	if !block {
		return res
	}
	return c.StepEnd(res)
}

// StepBegin is a step up to the point where vLLM's loop blocks on the GPU: it schedules a batch
// when the scheduler has requests and queues it, and reports whether the step then blocks on
// the oldest queued batch's output (block) or returns without one. A caller that runs the GPU
// on its own clock, as a simulator does through blis-schemas' engine.Engine, calls StepEnd once
// that batch's output exists; Step calls it at once.
//
// With QueueSize 1, step() (core.py:657-687) schedules and then always blocks on the batch it
// scheduled. With a batch queue, step_with_batch_queue (core.py:698-810) blocks unless the queue
// has room and the batch ran tokens or the scheduler has more to schedule (core.py:754-759);
// with no requests it blocks on the oldest queued batch, if any (core.py:760-766).
func (c *Core) StepBegin() (res StepResult, block bool) {
	if c.QueueSize <= 1 {
		if !c.Sched.HasRequests() {
			return StepResult{}, false
		}
		b := c.schedule()
		c.queue = append(c.queue, b)
		return StepResult{Scheduled: &b, Executed: total(b) > 0}, true
	}
	if c.Sched.HasRequests() {
		b := c.schedule()
		res.Scheduled = &b
		res.Executed = total(b) > 0
		c.queue = append(c.queue, b)
		if len(c.queue) < c.QueueSize && (res.Executed || c.Sched.HasRequests()) {
			// Don't block on the next output unless the queue is full or there is nothing
			// more to schedule (core.py:754-759).
			return res, false
		}
	} else if len(c.queue) == 0 {
		return res, false
	}
	return res, true
}

// StepEnd is the rest of a step StepBegin left blocked: it dequeues the oldest batch and
// applies its output (update_from_output, core.py:768-783).
func (c *Core) StepEnd(res StepResult) StepResult {
	b := c.queue[0]
	c.queue = c.queue[1:]
	res.Processed = &b
	res.Finished = c.process(b)
	return res
}

// Oldest is the oldest queued batch, the one a blocked step waits on; ok is false for an empty
// queue.
func (c *Core) Oldest() (b Batch, ok bool) {
	if len(c.queue) == 0 {
		return Batch{}, false
	}
	return c.queue[0], true
}

// Executed reports whether a batch ran tokens (model_executed, core.py:733).
func Executed(b Batch) bool { return total(b) > 0 }

// Schedule is the first half of a step for a caller that runs the batch queue itself, as a
// simulator does through blis-schemas' engine.Engine: it schedules one batch and queues it, and
// the caller completes the batches in order with Complete. Step is both halves under EngineCore's
// own loop; both keep the queue in the order vLLM's batch_queue does
// (vllm/v1/engine/core.py:738-760@5df63a72).
func (c *Core) Schedule() Batch {
	b := c.schedule()
	c.queue = append(c.queue, b)
	return b
}

// Drop takes the most recent batch back off the queue without applying it: a batch that
// scheduled no tokens is not in flight for a caller of Schedule, and vLLM's update_from_output
// of an empty schedule changes no request but one a connector's outcome finishes (a failed load
// under kv_load_failure_policy "fail"), whose finish records it returns.
func (c *Core) Drop(b Batch) []sched.Finished {
	if n := len(c.queue); n > 0 && c.queue[n-1].ID == b.ID {
		c.queue = c.queue[:n-1]
		return c.process(b)
	}
	return nil
}

// Complete is the second half: it applies the oldest queued batch, which must be b, with the
// tokens GPU.Output returns for it, and returns the requests it finished. Batches complete in
// the order they were scheduled.
func (c *Core) Complete(b Batch) []sched.Finished {
	if len(c.queue) == 0 || c.queue[0].ID != b.ID {
		panic(fmt.Sprintf("engine: batch %d completed out of order", b.ID))
	}
	c.queue = c.queue[1:]
	return c.process(b)
}

func total(b Batch) int {
	n := 0
	for _, t := range b.Out.Scheduled {
		n += t
	}
	return n
}
