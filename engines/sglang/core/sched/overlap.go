// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/managers/scheduler.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).

package sched

// The overlap event loop (S14). SGLang forms and launches the next batch before it processes the
// result of the one before, so the CPU work of one step hides under the GPU work of the other.
// What the scheduler sees of that is only an ordering: a batch's tokens and finishes reach the
// requests one iteration late. The tokens the next decode batch feeds are placeholders resolved on
// the device (FutureMap, managers/overlap_utils.py:281@297e54fd), so no decision reads them, and
// the port needs no device-side relay.

// snapshot is ScheduleBatch.copy (schedule_batch.py:3992-4050): the fields process_batch_result
// reads, with the request list copied so that a later filter of the live batch cannot change it.
func (b *Batch) snapshot() *Batch {
	c := *b
	c.Reqs = append([]*Req(nil), b.Reqs...)
	// seq_lens_cpu is rebound, not changed, by the next decode; the port changes it in place.
	c.SeqLens = append([]int(nil), b.SeqLens...)
	return &c
}

// shouldProcessLastBatchEarly is should_process_last_batch_early (scheduler.py:2327-2348@c4af57ff)
// without data parallel attention, speculation or grammars: with SGLANG_DISABLE_OVERLAP_AFTER_PREFILL
// the result of an extend batch (a mixed batch extends) is processed before the next batch is formed,
// so that the first token of a prefill is not held back a step. It is decided from the last batch
// alone, before the next one exists.
func (s *Scheduler) shouldProcessLastBatchEarly(last *Batch) bool {
	return s.cfg.DisableOverlapAfterPrefill && last != nil && (last.Mode == Extend || last.Mode == Mixed)
}

// popAndProcess is pop_and_process (scheduler.py:2068-2071): process the oldest launched batch.
func (s *Scheduler) popAndProcess() {
	if len(s.resultQueue) == 0 {
		panic("pop from an empty result queue")
	}
	b := s.resultQueue[0]
	s.resultQueue = s.resultQueue[1:]
	s.processBatchResult(b)
}

// overlapIteration is the body of event_loop_overlap (scheduler.py:2255-2325@c4af57ff) after the requests
// arrive. It returns the batch it launched, which becomes last_batch.
func (s *Scheduler) overlapIteration(rec *StepRecord) *Batch {
	// The last batch's result is processed first when it is a prefill's and the environment says so.
	early := s.shouldProcessLastBatchEarly(s.last)
	if early {
		s.popAndProcess()
	}
	batch := s.getNextBatchToRun()
	// Launch the current batch.
	if batch != nil {
		rec.Forwards = append(rec.Forwards, s.runBatch(batch))
		s.resultQueue = append(s.resultQueue, batch.snapshot())
	}
	// Process the last batch.
	if s.last != nil {
		if !early {
			s.popAndProcess()
		}
	} else if batch == nil && s.isFullyIdle() {
		s.onIdle()
	}
	return batch
}

// reqsFinishingInflight is _reqs_finishing_inflight (scheduler.py:4443-4457): the requests whose
// queued result commits their last output by length. The decode batch leaves them out, so the
// step does not decode a token no one will read.
func (s *Scheduler) reqsFinishingInflight(b *Batch) map[*Req]bool {
	if !s.cfg.EnableOverlap || len(s.resultQueue) == 0 {
		return nil
	}
	if len(s.resultQueue) != 1 {
		panic("the overlap loop holds more than one unprocessed result at scheduling time")
	}
	queued := map[*Req]bool{}
	for _, r := range s.resultQueue[0].Reqs {
		queued[r] = true
	}
	out := map[*Req]bool{}
	for _, r := range b.Reqs {
		if queued[r] && r.NextOutputFinishesByLength() {
			out[r] = true
		}
	}
	return out
}
