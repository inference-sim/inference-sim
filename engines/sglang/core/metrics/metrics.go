// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// Derived from python/sglang/srt/observability/metrics_collector.py at SGLang v1.0.0+c4af57ff93f3 (c4af57ff93f301846fa1da4c31f8523c08c7510d).
// Also derived from, at the same release: python/sglang/srt/managers/scheduler_components/metrics_reporter.py
// and python/sglang/srt/managers/scheduler_components/pool_stats_observer.py.

// Package metrics is the sglang:* Prometheus series whose values the scheduler alone decides
// (class "exact" in docs/metrics-classes.json): the queue and pool gauges SchedulerMetricsReporter
// logs and the token counters it increments. It reads the Reports core/sched makes as a batch's
// result is processed and keeps each series as SchedulerMetricsCollector does
// (observability/metrics_collector.py:1417-1429, :1558-1596, :1646-1726@297e54fd).
//
// Series that need time (throughput, latencies, occupancy), the device (graph passes, memory) or a
// feature outside the scheduler (speculation, disaggregation, LoRA, HiCache) are not here: the
// manifest lists each with its class and why.
package metrics

import (
	"fmt"
	"math"
	"sort"
	"strconv"

	"github.com/inference-sim/inference-sim/engines/sglang/core/sched"
)

// Series is one sample of the registry: a series name (a family's name, with _total for a counter),
// its labels and its value.
type Series struct {
	Name   string
	Labels map[string]string
	Value  float64
}

// Collector holds the series. Labels are those of the scheduler of a single device (engine_type
// "unified", every rank 0); a parallel layout labels each rank's own collector.
type Collector struct {
	cfg      sched.Config
	priority bool

	logged   bool // a gauge report has been logged
	prefill  bool // a prefill report has been logged (cache_hit_rate is its)
	known    map[int]bool
	running  sched.Queue
	queue    sched.Queue
	cacheHit float64
	sumSeq   int
	newRatio float64
	avail    int
	evict    int
	retract  int

	realtime  map[string]float64 // mode -> tokens, created on first nonzero increment
	effective map[string]float64 // mode -> tokens, all four created at start
}

// New is SchedulerMetricsCollector at start-up: the constants and the zeroed effective-token counters.
func New(cfg sched.Config) *Collector {
	return &Collector{cfg: cfg, priority: cfg.EnablePriorityScheduling, known: map[int]bool{},
		realtime:  map[string]float64{},
		effective: map[string]float64{"input": 0, "device_hit": 0, "host_hit": 0, "storage_hit": 0}}
}

// Observe applies one step's Reports in order.
func (c *Collector) Observe(reports []sched.Report) {
	for _, r := range reports {
		switch {
		case r.Prefill != nil:
			c.prefillReport(r)
		case r.Decode != nil:
			c.decodeReport(r)
		}
	}
}

func (c *Collector) addRealtime(mode string, n int) {
	if n != 0 {
		c.realtime[mode] += float64(n)
	}
}

// prefillReport is report_prefill_stats (metrics_reporter.py:719-899).
func (c *Collector) prefillReport(r sched.Report) {
	p := r.Prefill
	c.addRealtime("prefill_compute", p.LogInputTokens)
	c.addRealtime("prefill_cache", p.LogHitTokens)
	effIn := p.LogInputTokens - p.ReprocessedInput
	effHit := p.LogHitTokens - p.ReprocessedHit
	total := effIn + effHit
	c.cacheHit = 0
	if total > 0 {
		c.cacheHit = float64(effHit) / float64(total)
	}
	c.prefill = true
	for mode, d := range map[string]int{"input": effIn, "device_hit": p.LogDeviceHitTokens} {
		if d > 0 {
			c.effective[mode] += float64(d)
		}
	}
	c.running, c.queue, c.newRatio = p.Running, r.Waiting, p.NewTokenRatio
	c.setPool(r)
}

// decodeReport is report_decode_stats (:901-1145).
func (c *Collector) decodeReport(r sched.Report) {
	d := r.Decode
	c.addRealtime("decode", d.Generated)
	if !r.Logged {
		return
	}
	c.running, c.queue, c.sumSeq = d.Running, r.Waiting, d.SumSeqLens
	c.setPool(r)
}

func (c *Collector) setPool(r sched.Report) {
	c.logged = true
	c.avail, c.evict, c.retract = r.Available, r.Evictable, r.Retracted
}

// Series returns every series, sorted by name then labels.
func (c *Collector) Series() []Series {
	base := func(kv ...string) map[string]string {
		m := map[string]string{"engine_type": "unified", "moe_ep_rank": "0", "pp_rank": "0", "tp_rank": "0"}
		if c.priority {
			// With priority scheduling every series carries the priority label, "" for a total.
			m["priority"] = ""
		}
		for i := 0; i+1 < len(kv); i += 2 {
			m[kv[i]] = kv[i+1]
		}
		return m
	}
	var out []Series
	add := func(name string, v float64, kv ...string) { out = append(out, Series{name, base(kv...), v}) }
	maxTok := c.cfg.MaxTotalNumTokens
	add("sglang:context_len", float64(c.cfg.ContextLen))
	add("sglang:max_total_num_tokens", float64(maxTok))
	add("sglang:num_pages", float64(maxTok/c.cfg.PageSize))
	add("sglang:page_size", float64(c.cfg.PageSize))
	for mode, v := range c.effective {
		add("sglang:prefill_effective_tokens_total", v, "mode", mode)
	}
	for mode, v := range c.realtime {
		add("sglang:realtime_tokens_total", v, "mode", mode)
	}
	if c.logged {
		used := maxTok - (c.avail + c.evict)
		usage := float64(used) / float64(maxTok)
		gauge := func(name string, v float64) { add(name, v) }
		queueGauge := func(name string, q sched.Queue) {
			if q.ByPriority == nil {
				add(name, float64(q.Total))
				return
			}
			add(name, float64(q.Total))
			for p := range q.ByPriority {
				c.known[p] = true
			}
			for p := range c.known {
				add(name, float64(q.ByPriority[p]), "priority", strconv.Itoa(p))
			}
		}
		queueGauge("sglang:num_running_reqs", c.running)
		queueGauge("sglang:num_queue_reqs", c.queue)
		gauge("sglang:num_grammar_queue_reqs", 0)
		gauge("sglang:cache_hit_rate", c.cacheHit)
		gauge("sglang:decode_sum_seq_lens", float64(c.sumSeq))
		gauge("sglang:token_usage", roundHalfEven2(usage))
		gauge("sglang:full_token_usage", usage)
		gauge("sglang:num_used_tokens", float64(used))
		gauge("sglang:kv_available_tokens", float64(c.avail))
		gauge("sglang:kv_evictable_tokens", float64(c.evict))
		gauge("sglang:kv_used_tokens", float64(used))
		gauge("sglang:new_token_ratio", c.newRatio)
		gauge("sglang:num_retracted_reqs", float64(c.retract))
		gauge("sglang:num_paused_reqs", 0)
		// Set by every log_stats and zero while the feature that moves them is off: the
		// speculative, sliding-window, state-space, disaggregation and routing-key gauges.
		for _, n := range zeroGauges {
			gauge(n, 0)
		}
		for _, n := range zeroQueues {
			queueGauge(n, sched.Queue{})
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Name != out[j].Name {
			return out[i].Name < out[j].Name
		}
		return labelKey(out[i].Labels) < labelKey(out[j].Labels)
	})
	return out
}

var zeroGauges = []string{
	"sglang:swa_token_usage", "sglang:mamba_usage", "sglang:swa_available_tokens", "sglang:swa_evictable_tokens",
	"sglang:swa_used_tokens", "sglang:mamba_available_tokens", "sglang:mamba_evictable_tokens", "sglang:mamba_used_tokens",
	"sglang:spec_accept_length", "sglang:spec_accept_rate", "sglang:spec_cap_length", "sglang:spec_block_accept_length",
	"sglang:spec_num_steps", "sglang:spec_num_draft_tokens", "sglang:num_decode_deferred_kv_release_reqs",
	"sglang:pending_prealloc_token_usage", "sglang:prealloc_full_token_usage", "sglang:prealloc_swa_token_usage",
	"sglang:num_unique_running_routing_keys",
}

var zeroQueues = []string{
	"sglang:num_prefill_bootstrap_queue_reqs", "sglang:num_prefill_inflight_queue_reqs",
	"sglang:num_decode_prealloc_queue_reqs", "sglang:num_decode_transfer_queue_reqs",
	"sglang:num_decode_host_receive_queue_reqs",
}

// roundHalfEven2 is Python's round(x, 2).
func roundHalfEven2(x float64) float64 {
	v, _ := strconv.ParseFloat(fmt.Sprintf("%.2f", x), 64)
	return v
}

func labelKey(l map[string]string) string {
	keys := make([]string, 0, len(l))
	for k := range l {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	s := ""
	for _, k := range keys {
		s += k + "=" + l[k] + ","
	}
	return s
}

var _ = math.NaN
