// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/managers/scheduler.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

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

// isDisableOverlapForBatch is is_disable_overlap_for_batch (scheduler.py:2136-2174) without data
// parallel attention, speculation or grammars: two extend batches in a row (a mixed batch extends)
// do not overlap when the environment says so, so that the first one's result is not held back.
func (s *Scheduler) isDisableOverlapForBatch(batch, last *Batch) bool {
	isExtend := func(b *Batch) bool { return b != nil && (b.Mode == Extend || b.Mode == Mixed) }
	return s.cfg.DisableConsecutivePrefillOverlap && isExtend(batch) && isExtend(last)
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

// overlapIteration is the body of event_loop_overlap (scheduler.py:2073-2133) after the requests
// arrive. It returns the batch it launched, which becomes last_batch.
func (s *Scheduler) overlapIteration(rec *StepRecord) *Batch {
	batch := s.getNextBatchToRun()
	disable := s.isDisableOverlapForBatch(batch, s.last)
	// Without overlap for this batch, the last batch's result is processed first.
	if disable {
		s.popAndProcess()
	}
	// Launch the current batch.
	if batch != nil {
		rec.Forwards = append(rec.Forwards, s.runBatch(batch))
		s.resultQueue = append(s.resultQueue, batch.snapshot())
	}
	// Process the last batch.
	if s.last != nil {
		if !disable {
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
