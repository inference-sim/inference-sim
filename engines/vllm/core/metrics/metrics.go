// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/metrics/loggers.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/metrics/stats.py (PrefixCacheStats, PromptTokenStats, PrefillStats,
// IterationStats) and the PrefillStats bookkeeping of vllm/v1/core/sched/scheduler.py.

// Package metrics is vLLM's Prometheus stat logger for the vllm:* families whose values the
// scheduler alone decides (class "exact" in docs/metrics-manifest.json): request counts, KV
// usage, prefix-cache queries and hits, preemptions, prompt and generation tokens, finished
// requests by reason, and tokens per step. It reads what one engine step decided and keeps
// each series as vLLM's PrometheusStatLogger.record does (loggers.py:1138-1330@5df63a72).
//
// Families that need time (latencies) or supplied inputs (spec-decode acceptance, transfer
// durations) are not here: they are "derived" in the manifest and reported on the
// simulator's values elsewhere.
package metrics

import (
	"fmt"
	"sort"
)

// Admission is one request the scheduler admitted from the waiting queue in a step, after a
// prefix-cache lookup (a request resuming with computed tokens takes none).
type Admission struct {
	ID string
	// PromptTokens is the request's num_tokens at admission: its prompt plus, for a request
	// resuming after preemption, the output it already has.
	PromptTokens int
	// LocalHit is the tokens the local prefix cache supplied.
	LocalHit int
	// Preempted is whether the request was preempted before (num_preemptions > 0).
	Preempted bool
	// CacheLookup is whether a local prefix-cache lookup ran: prefix caching is on
	// (prefix_cache_lookup_enabled, kv_cache_manager.py:249-251@5df63a72). Only a lookup that
	// ran is counted (record_prefix_cache_stats, :253-262), though the prompt is counted
	// either way.
	CacheLookup bool
}

// Finish is one request a step finished.
type Finish struct {
	ID string
	// Reason is vLLM's finish-reason string: stop, length, abort, error or repetition.
	Reason           string
	PromptTokens     int
	GenerationTokens int
	// Preemptions is how many times the request was preempted.
	Preemptions int
	// MaxTokens is the request's max_tokens as the engine received it (SamplingParams
	// .max_tokens, observed when non-zero, loggers.py:1325-1328@5df63a72).
	MaxTokens int
	// CachedTokens is the request's cached prompt tokens as its first output reported them
	// (RequestState.num_cached_tokens, output_processor.py:733-737@5df63a72); Model fills it.
	CachedTokens int
}

// Step is what one engine step decided, as the scheduler's stats and outputs report it.
type Step struct {
	// Running and Waiting are get_request_counts after the step; Deferred is the waiting
	// requests deferred by transient constraints (vllm:num_requests_waiting_by_reason).
	Running, Waiting, Deferred int
	// KVUsage is the KV cache usage after the step.
	KVUsage float64
	// Admitted are the requests admitted from the waiting queue, in admission order.
	Admitted []Admission
	// Scheduled is tokens per request this step.
	Scheduled map[string]int
	// Preempted are the requests preempted this step.
	Preempted []string
	// Emitted is the output tokens each request produced this step; FirstTokens are the
	// requests whose first output token this is.
	Emitted     map[string]int
	FirstTokens []string
	// Finished are the requests the step finished.
	Finished []Finish
}

// Bucketed names a histogram family and its upper bounds, as vLLM registers it.
type Bucketed struct {
	Family string
	Bounds []float64
}

// Model keeps the series of one engine.
type Model struct {
	gauges   map[string]float64
	counters map[string]float64
	hists    map[string]*histogram
	// pending is each admitted request's PrefillStats, set at its first admission and
	// emitted with its first output token (scheduler.py:1074-1080, 2208-2212@5df63a72).
	pending map[string]prefill
	// preemptions are the PREEMPTED events each request holds until its next output takes
	// them (scheduler.py:1609-1610, 2274@5df63a72).
	preemptions map[string]int
	// cached is each request's cached prompt tokens from its first output, until it finishes.
	cached map[string]int
}

type prefill struct{ prompt, localHit int }

type histogram struct {
	bounds []float64
	counts []float64 // cumulative per bound, as Prometheus exposes _bucket
	sum    float64
	n      float64
}

// finishReasons are FinishReason's strings, every one of which vLLM pre-registers at zero
// (loggers.py:819-826@5df63a72; vllm/v1/engine/__init__.py:48-70@5df63a72).
var finishReasons = []string{"stop", "length", "abort", "error", "repetition"}

// promptSources and hitSources are PromptTokenStats.ALL_SOURCES and CacheHitSource
// (vllm/v1/metrics/stats.py; vllm/v1/metrics/cache_hit_source.py:18-24@5df63a72).
var (
	promptSources = []string{"local_compute", "local_cache_hit", "external_kv_transfer"}
	hitSources    = []string{"device", "host", "p2p", "disk", "external_unspecified"}
)

// New returns a model with every series vLLM pre-registers at zero, and the histograms
// whose bounds are given.
func New(hists []Bucketed) *Model {
	m := &Model{gauges: map[string]float64{}, counters: map[string]float64{},
		hists: map[string]*histogram{}, pending: map[string]prefill{}, preemptions: map[string]int{}, cached: map[string]int{}}
	for _, g := range []string{"vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
		"vllm:num_requests_waiting_by_reason{reason=capacity}", "vllm:num_requests_waiting_by_reason{reason=deferred}"} {
		m.gauges[g] = 0
	}
	for _, c := range []string{"vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
		"vllm:external_prefix_cache_queries_total", "vllm:external_prefix_cache_hits_total",
		"vllm:prompt_tokens_total", "vllm:prompt_tokens_cached_total", "vllm:generation_tokens_total",
		"vllm:num_preemptions_total"} {
		m.counters[c] = 0
	}
	for _, r := range finishReasons {
		m.counters["vllm:request_success_total{finished_reason="+r+"}"] = 0
	}
	for _, s := range promptSources {
		m.counters["vllm:prompt_tokens_by_source_total{source="+s+"}"] = 0
	}
	for _, s := range hitSources {
		m.counters["vllm:prompt_tokens_cached_by_source_total{source="+s+"}"] = 0
	}
	for _, h := range hists {
		m.hists[h.Family] = &histogram{bounds: append([]float64(nil), h.Bounds...), counts: make([]float64, len(h.Bounds))}
	}
	return m
}

// Step records one engine step: its scheduler stats, then the iteration stats its outputs
// carry, in the order PrometheusStatLogger.record applies them.
func (m *Model) Step(s Step) {
	// SchedulerStats (loggers.py:1146-1177@5df63a72).
	m.gauges["vllm:num_requests_running"] = float64(s.Running)
	m.gauges["vllm:num_requests_waiting"] = float64(s.Waiting)
	m.gauges["vllm:num_requests_waiting_by_reason{reason=capacity}"] = float64(s.Waiting - s.Deferred)
	m.gauges["vllm:num_requests_waiting_by_reason{reason=deferred}"] = float64(s.Deferred)
	m.gauges["vllm:kv_cache_usage_perc"] = s.KVUsage
	for _, a := range s.Admitted {
		// record_prefix_cache_stats at admission (scheduler.py:1314-1317): a preempted
		// request's lookup goes to preempted_*, which no vllm:* family exports
		// (PrefixCacheStats.record, stats.py:136-145@5df63a72).
		if a.Preempted {
			continue
		}
		if a.CacheLookup {
			m.counters["vllm:prefix_cache_queries_total"] += float64(a.PromptTokens)
			m.counters["vllm:prefix_cache_hits_total"] += float64(a.LocalHit)
			m.counters["vllm:prompt_tokens_cached_by_source_total{source=device}"] += float64(a.LocalHit)
		}
		// PrefillStats.set at the first admission (scheduler.py:1074-1080@5df63a72): a request
		// not preempted before is being admitted for the first time.
		m.pending[a.ID] = prefill{prompt: a.PromptTokens, localHit: a.LocalHit}
	}
	// IterationStats (loggers.py:1259-1330@5df63a72), built from the step's outputs. A
	// preemption is an event on the request, which the request's next emitted output carries
	// to the frontend; IterationStats counts it there (stats.py:574-577@5df63a72).
	for _, id := range s.Preempted {
		m.preemptions[id]++
	}
	emitting := map[string]bool{}
	for id, n := range s.Emitted {
		if n > 0 {
			emitting[id] = true
		}
	}
	for _, f := range s.Finished {
		emitting[f.ID] = true
	}
	for _, id := range sortedBoolKeys(emitting) {
		m.counters["vllm:num_preemptions_total"] += float64(m.preemptions[id])
		delete(m.preemptions, id)
	}
	computed := 0
	for _, id := range s.FirstTokens {
		// The first output carries the request's PrefillStats (PromptTokenStats.update_from_output).
		p, ok := m.pending[id]
		if !ok {
			continue
		}
		delete(m.pending, id)
		m.cached[id] = p.localHit
		m.counters["vllm:prompt_tokens_total"] += float64(p.prompt)
		m.counters["vllm:prompt_tokens_cached_total"] += float64(p.localHit)
		m.counters["vllm:prompt_tokens_by_source_total{source=local_compute}"] += float64(p.prompt - p.localHit)
		m.counters["vllm:prompt_tokens_by_source_total{source=local_cache_hit}"] += float64(p.localHit)
		computed += p.prompt - p.localHit
	}
	generated := 0
	for _, id := range sortedKeys(s.Emitted) {
		generated += s.Emitted[id]
	}
	m.counters["vllm:generation_tokens_total"] += float64(generated)
	if len(s.Emitted) > 0 || len(s.FirstTokens) > 0 {
		// One observation per step that produced outputs: the prompt tokens computed for
		// requests whose first token this is, plus the generated tokens (loggers.py:1272-1275).
		m.observe("vllm:iteration_tokens_total", float64(computed+generated))
	}
	for _, f := range s.Finished {
		f.CachedTokens = m.cached[f.ID]
		delete(m.cached, f.ID)
		m.counters["vllm:request_success_total{finished_reason="+f.Reason+"}"]++
		m.observe("vllm:request_prompt_tokens", float64(f.PromptTokens))
		// The prompt tokens computed rather than cached (loggers.py:1312-1318@5df63a72): the
		// first-admission cache hit the request's PrefillStats recorded.
		m.observe("vllm:request_prefill_kv_computed_tokens", float64(f.PromptTokens-max(f.CachedTokens, 0)))
		m.observe("vllm:request_generation_tokens", float64(f.GenerationTokens))
		m.observe("vllm:request_num_preemptions", float64(f.Preemptions))
		m.observe("vllm:request_max_num_generation_tokens", float64(f.GenerationTokens))
		m.observe("vllm:request_params_n", 1)
		if f.MaxTokens > 0 {
			m.observe("vllm:request_params_max_tokens", float64(f.MaxTokens))
		}
		delete(m.pending, f.ID)
	}
}

// Value is a gauge's or a counter's current value by its series key, as Series names it; 0 for
// a key the model does not keep.
func (m *Model) Value(key string) float64 {
	if v, ok := m.gauges[key]; ok {
		return v
	}
	return m.counters[key]
}

func (m *Model) observe(family string, v float64) {
	h, ok := m.hists[family]
	if !ok {
		h = &histogram{}
		m.hists[family] = h
	}
	h.sum += v
	h.n++
	for i, b := range h.bounds {
		if v <= b {
			h.counts[i]++
		}
	}
}

// Series returns every sample as Prometheus exposes it: counters with their _total suffix,
// histograms as _count, _sum and _bucket{le=...}, by name with sorted labels.
func (m *Model) Series() map[string]float64 {
	out := map[string]float64{}
	for k, v := range m.gauges {
		out[k] = v
	}
	for k, v := range m.counters {
		out[k] = v
	}
	for name, h := range m.hists {
		out[name+"_count"] = h.n
		out[name+"_sum"] = h.sum
		for i, b := range h.bounds {
			out[fmt.Sprintf("%s_bucket{le=%s}", name, formatBound(b))] = h.counts[i]
		}
		if len(h.bounds) > 0 {
			out[name+"_bucket{le=+Inf}"] = h.n
		}
	}
	return out
}

// formatBound writes a bucket bound as prometheus_client does (floatToGoString).
func formatBound(b float64) string {
	if b == float64(int64(b)) {
		return fmt.Sprintf("%.1f", b)
	}
	return fmt.Sprintf("%g", b)
}

func sortedBoolKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func sortedKeys(m map[string]int) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
