// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/managers/scheduler_components/metrics_reporter.py at SGLang v0.5.23 (297e54fd105e974cdf401e5b4d20253a22c16f90).

package sched

import "github.com/inference-sim/blis-sglang/core/cache"

// What the scheduler tells its metrics reporter. SGLang's SchedulerMetricsReporter reads these
// values off the scheduler as a batch's result is processed (report_prefill_stats :719-899 and
// report_decode_stats :901-1145@297e54fd); the port hands them over as a Report per call, and
// core/metrics turns them into the sglang:* series.

// Queue is QueueCount (observability/metrics_collector.py:60-77): a queue's size and, with
// priority scheduling, its size by priority.
type Queue struct {
	Total      int         `json:"total"`
	ByPriority map[int]int `json:"by_priority,omitempty"`
}

func (s *Scheduler) queueOf(reqs []*Req) Queue {
	q := Queue{Total: len(reqs)}
	if s.cfg.EnablePriorityScheduling {
		q.ByPriority = map[int]int{}
		for _, r := range reqs {
			q.ByPriority[r.Priority]++
		}
	}
	return q
}

// PrefillStats is PrefillStats.from_adder (metrics_reporter.py:119-160): what the prefill adder
// accounted while it formed the batch.
type PrefillStats struct {
	LogInputTokens     int     `json:"log_input_tokens"`
	LogHitTokens       int     `json:"log_hit_tokens"`
	ReprocessedInput   int     `json:"reprocessed_input_tokens"`
	ReprocessedHit     int     `json:"reprocessed_hit_tokens"`
	LogDeviceHitTokens int     `json:"log_device_hit_tokens"`
	NewTokenRatio      float64 `json:"new_token_ratio"`
	Running            Queue   `json:"running"`
}

// Report is one call of the reporter: a prefill batch's result processed, or a decode batch's.
type Report struct {
	// Prefill is set for a prefill report, Decode for a decode report.
	Prefill *PrefillStats `json:"prefill,omitempty"`
	Decode  *DecodeStats  `json:"decode,omitempty"`
	// Waiting is the waiting queue when the report is made.
	Waiting Queue `json:"waiting"`
	// Available and Evictable are the pool's free and evictable tokens when the report is made.
	Available int `json:"available_tokens"`
	Evictable int `json:"evictable_tokens"`
	// Retracted is the requests the last retraction took, read once by a logged report.
	Retracted int `json:"retracted"`
	// Events are the tree cache's KV events published with this report (publish_kv_events), oldest
	// first; empty when none were queued or publishing is off.
	Events []cache.Event `json:"-"`
	// Idle marks the report on_idle makes: it carries events only.
	Idle bool `json:"idle,omitempty"`
	// Logged is whether this report reached the logging part: always for a prefill batch, every
	// decode_log_interval-th decode batch.
	Logged bool `json:"logged"`
}

// DecodeStats is what report_decode_stats reads of a decode batch.
type DecodeStats struct {
	// Generated is num_generated_tokens of the batch: one per request.
	Generated  int   `json:"generated_tokens"`
	Running    Queue `json:"running"`
	SumSeqLens int   `json:"sum_seq_lens"`
}

func (s *Scheduler) prefillStats(a *adder, running *Batch) *PrefillStats {
	return &PrefillStats{LogInputTokens: a.logInputTokens, LogHitTokens: a.logHitTokens,
		ReprocessedInput: a.reprocessedInputTokens, ReprocessedHit: a.reprocessedHitTokens,
		LogDeviceHitTokens: a.logDeviceHitTokens, NewTokenRatio: a.newTokenRatio, Running: s.queueOf(running.Reqs)}
}

func (s *Scheduler) poolReport() Report {
	return Report{Waiting: s.queueOf(s.waiting), Available: s.alloc.AvailableSize(), Evictable: s.tree.EvictableSize()}
}

// reportPrefill is the call at the end of process_batch_result_prefill (batch_result_processor.py:
// 509-515).
func (s *Scheduler) reportPrefill(b *Batch) {
	if b.Prefill == nil {
		return
	}
	r := s.poolReport()
	r.Prefill = b.Prefill
	r.Logged = true
	r.Retracted = s.numRetracted
	s.numRetracted = 0
	s.publishKVEvents(&r)
	s.reports = append(s.reports, r)
}

// reportDecode is the call at the end of process_batch_result_decode (:1104-1109): every call
// counts the batch's tokens; every decode_log_interval-th also logs the gauges.
func (s *Scheduler) reportDecode(b *Batch) {
	s.forwardCtDecode++
	sum := 0
	for _, l := range b.SeqLens {
		sum += l
	}
	r := s.poolReport()
	r.Decode = &DecodeStats{Generated: len(b.Reqs), Running: s.queueOf(b.Reqs), SumSeqLens: sum}
	interval := max(s.cfg.DecodeLogInterval, 1)
	if s.forwardCtDecode%interval == 0 {
		r.Logged = true
		r.Retracted = s.numRetracted
		s.numRetracted = 0
		s.publishKVEvents(&r)
	}
	s.reports = append(s.reports, r)
}

// publishKVEvents is SchedulerKvEventsPublisher.publish_kv_events (kv_events_publisher.py:98-104):
// the queued events leave in one batch. A prefill report and every logged decode report call it,
// and so does on_idle (into a report of its own, when r is nil).
func (s *Scheduler) publishKVEvents(r *Report) {
	if !s.cfg.EnableKVEvents {
		return
	}
	ev := s.tree.TakeEvents()
	if len(ev) == 0 {
		return
	}
	if r == nil {
		s.reports = append(s.reports, Report{Idle: true, Events: ev})
		return
	}
	r.Events = ev
}
