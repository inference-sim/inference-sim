// Original to blis-sglang.

package sched

import (
	"fmt"
	"slices"
)

// The engine's view of the loop. Step runs one iteration of SGLang's loop whole. An engine driven by
// a simulator splits it where the simulator does: Launch is the part of the iteration that forms
// and launches a batch, and Complete is the part that processes its result once the pass has ended.
// In the normal loop the two follow each other at once; in the overlap loop Launch is called for
// the next batch before Complete is called for the last (the simulator's in-flight limit of two),
// which is the order event_loop_overlap makes (scheduler.py:2061-2134@297e54fd).

// tokenFor is the token the model samples for r next: its script's, or the scheduler's.
func (s *Scheduler) tokenFor(r *Req) int64 {
	if n := len(r.C.OutputIDs); n < len(r.Script) {
		return r.Script[n]
	}
	return s.sampledToken
}

// Launched is a forward pass launched whose result is not yet processed.
type Launched struct {
	// Forward is the pass as the oracle records it.
	Forward Forward
	// Aborted are the requests the scheduler finished at ingest or while forming the batch, with
	// no pass (an input too long, a full queue).
	Aborted []string
	batch   *Batch
	reqs    []*Req
}

// Enqueue is the ingest of one arrival (Scheduler.handle_generate_request). A feature the port
// refuses is returned as an error wrapping ErrRefused, and leaves the request unqueued.
func (s *Scheduler) Enqueue(a Arrival) (err error) {
	defer s.recoverRefusal(&err)
	n := len(s.finishedThisStep)
	s.handleGenerateRequest(a)
	// What finished while the request was ingested finished without a pass.
	s.abortedSinceTake = append(s.abortedSinceTake, s.finishedThisStep[n:]...)
	s.finishedThisStep = s.finishedThisStep[:n]
	return nil
}

func (s *Scheduler) recoverRefusal(err *error) {
	if p := recover(); p != nil {
		if rf, ok := p.(refusal); ok {
			*err = fmt.Errorf("%w: %s", ErrRefused, rf.Error())
			return
		}
		panic(p)
	}
}

// Launch is one iteration of the loop up to the launch: it forms the next batch and counts the
// pass. It returns nil when nothing runs; a nil Launched with aborts is not possible, so the
// aborts of an iteration that runs nothing are returned by TakeAborted.
func (s *Scheduler) Launch() (l *Launched, err error) {
	defer s.recoverRefusal(&err)
	batch := s.getNextBatchToRun()
	if batch != nil {
		l = &Launched{Forward: s.runBatch(batch), batch: batch, reqs: slices.Clone(batch.Reqs)}
		if s.cfg.EnableOverlap {
			s.resultQueue = append(s.resultQueue, batch.snapshot())
		}
	} else if s.isFullyIdle() {
		s.onIdle()
	}
	s.last = batch
	s.step++
	return l, nil
}

// Completion is what a processed result made visible.
type Completion struct {
	// Emitted counts the output tokens each request got in the pass.
	Emitted map[string]int
	// FirstTokens are the requests whose first output token this pass produced.
	FirstTokens []string
	// Finished are the requests that finished, with the reason SGLang gives and their output length.
	Finished []FinishedReq
}

// FinishedReq is a request that finished.
type FinishedReq struct {
	ID     string
	Reason string
	Output int
	// Scripted is true when the request stopped because its script ended.
	ScriptEnded bool
}

// Complete processes the result of the oldest launched pass.
func (s *Scheduler) Complete(l *Launched) (c Completion, err error) {
	defer s.recoverRefusal(&err)
	b := l.batch
	if s.cfg.EnableOverlap {
		if len(s.resultQueue) == 0 {
			return c, fmt.Errorf("sched: no launched pass to complete")
		}
		b = s.resultQueue[0]
		s.resultQueue = s.resultQueue[1:]
	}
	before := map[*Req]int{}
	for _, r := range b.Reqs {
		before[r] = len(r.C.OutputIDs)
	}
	s.finishedThisStep = nil
	s.processBatchResult(b)
	c.Emitted = map[string]int{}
	for _, r := range b.Reqs {
		if d := len(r.C.OutputIDs) - before[r]; d > 0 {
			c.Emitted[r.RID()] += d
			if before[r] == 0 {
				c.FirstTokens = append(c.FirstTokens, r.RID())
			}
		}
	}
	for _, r := range b.Reqs {
		if slices.Contains(s.finishedThisStep, r.RID()) {
			c.Finished = append(c.Finished, FinishedReq{ID: r.RID(), Reason: r.FinishedReason,
				Output: len(r.C.OutputIDs), ScriptEnded: len(r.Script) > 0 && len(r.C.OutputIDs) >= len(r.Script)})
		}
	}
	return c, nil
}

// TakeAborted returns, once, the requests finished since the last call without a pass: aborted at
// ingest, or dropped from a full queue.
func (s *Scheduler) TakeAborted() []string {
	out := s.abortedSinceTake
	s.abortedSinceTake = nil
	return out
}

// Waiting, Running and InFlight describe the scheduler for a router: the waiting queue, the
// running batch, and the passes launched and not yet completed.
func (s *Scheduler) Waiting() int { return len(s.waiting) }
func (s *Scheduler) Running() int { return len(s.running.Reqs) }

// Pool reports the KV pool: its capacity and the tokens in use (neither free nor evictable).
func (s *Scheduler) Pool() (total, used int) {
	total = s.cfg.MaxTotalNumTokens
	return total, total - (s.alloc.AvailableSize() + s.tree.EvictableSize())
}

// Config returns the configuration the scheduler runs with.
func (s *Scheduler) Config() Config { return s.cfg }

// Finish ends a request the engine did not decide (a client abort): the request leaves the queue
// or the running batch and frees what it holds. Finishing an unknown or finished request is a no-op.
func (s *Scheduler) Finish(id string) {
	for i, r := range s.waiting {
		if r.RID() == id {
			s.waiting = slices.Delete(s.waiting, i, i+1)
			return
		}
	}
	for _, r := range s.running.Reqs {
		if r.RID() == id && !r.Finished() {
			r.FinishedReason = "abort"
			r.C.Finished = true
			s.releaseKVCache(r, false)
		}
	}
}

// TakeReports returns, once, the metrics reporter's calls made since the last call, and the
// requests retracted or preempted in that time.
func (s *Scheduler) TakeReports() (reports []Report, preempted []string) {
	reports, preempted = s.reports, s.preemptedSinceTake
	s.reports, s.preemptedSinceTake = nil, nil
	return reports, preempted
}
