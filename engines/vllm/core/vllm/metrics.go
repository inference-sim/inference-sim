// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/sched/scheduler.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).
// Also vllm/v1/metrics/loggers.py (PrometheusStatLogger.record) and vllm/v1/metrics/stats.py.

package vllm

import (
	"sort"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/engine"

	core "github.com/inference-sim/blis-vllm/core/engine"
	"github.com/inference-sim/blis-vllm/core/metrics"
	"github.com/inference-sim/blis-vllm/core/sched"
)

// The engine records its metrics in two halves, as vLLM does: a batch's admissions are its
// schedule's PrefillStats (scheduler.py:1251-1257, 1508-1511@4ba787ec), and its outputs,
// finishes and the scheduler gauges are what update_from_output reports for it
// (:2647-2674@4ba787ec). core/metrics takes both in one Step; the admissions wait here until
// their batch completes.

// scheduled is what a schedule decided that its batch's metrics step reports.
type scheduled struct {
	admitted  []metrics.Admission
	preempted []string
	tokens    map[string]int
}

// lookupInfo is a request's admission lookup: what the local prefix cache and the KV connector
// supplied of its prompt (PrefillStats, scheduler.py:1251-1257@4ba787ec).
type lookupInfo struct {
	set, reported bool
	// local and external are num_new_local_computed_tokens and num_external_computed_tokens.
	local, external int
	// connector is whether a KV connector is configured.
	connector bool
	// sources split external by the CacheHitSource of the tier that supplied it (lower case, as
	// the label's value), as get_external_cache_hit_sources reports them.
	sources map[string]int
	// cached is the tokens the batch reports as supplied by a cache: the local hit and what a
	// KV offload tier supplied; offload is the latter. Tokens a P/D pull supplied are the
	// connector's, not a cache's.
	cached, offload int
}

// noteLookups notes the lookup of every request whose admission took one in a schedule: one
// admitted to run and one parked on an asynchronous KV load, which runs in a later schedule with
// no lookup in it (PrefillStats is set at the admission, scheduler.py:1251-1257@4ba787ec, before
// the load parks the request).
func (e *Engine) noteLookups(b core.Batch) {
	for _, id := range sortedKeys(b.Out.PrefixHit) {
		st := e.reqs[id]
		if st == nil {
			continue
		}
		li := lookupInfo{set: true, local: b.Out.PrefixHit[id]}
		li.external, li.connector = b.Out.PrefixExternal[id]
		li.cached = li.local
		if li.external > 0 {
			li.sources = e.externalSources(e.s.Request(id), li.external)
			for src, n := range li.sources {
				if src != "external_unspecified" {
					li.offload += n
				}
			}
			li.cached += li.offload
		}
		st.lookup = li
	}
}

// externalSources splits a request's external tokens by the tier that supplied them
// (get_external_cache_hit_sources, scheduler.py:1494-1503@4ba787ec): the offloading connector
// names its tiers when it was the one chosen to load the request, and any other connector's
// tokens are EXTERNAL_UNSPECIFIED (base.py:185-186, 550-566).
func (e *Engine) externalSources(r *sched.Request, external int) map[string]int {
	if e.off != nil {
		chosen := e.off.multi == nil
		if !chosen {
			i, ok := e.off.multi.Chosen(r.ID)
			chosen = ok && i == e.off.child
		}
		if chosen {
			out := map[string]int{}
			for src, n := range e.off.conn.ExternalHitSources(r, external) {
				out[strings.ToLower(src)] += n
			}
			return out
		}
	}
	return map[string]int{"external_unspecified": external}
}

// OffloadCachedTokens is how much of the Scheduled.CachedTokens a batch reported for id the KV
// offload tier supplied, which the simulator splits from the device cache's hit.
func (e *Engine) OffloadCachedTokens(id string) int {
	if st := e.reqs[id]; st != nil {
		return st.lookup.offload
	}
	return 0
}

// recordSchedule notes a schedule's admissions, preemptions and tokens for its batch's step.
func (e *Engine) recordSchedule(b core.Batch) {
	sc := scheduled{preempted: append([]string(nil), b.Out.Preempted...), tokens: b.Out.Scheduled}
	for _, id := range sortedKeys(b.Out.PrefixHit) {
		r := e.s.Request(id)
		li := e.reqs[id].lookup
		sc.admitted = append(sc.admitted, metrics.Admission{ID: id, PromptTokens: r.NumTokens(),
			LocalHit: li.local, Preempted: r.NumPreemptions > 0, CacheLookup: e.prefixCaching(),
			External: li.external, Connector: li.connector, ExternalBySource: li.sources})
	}
	if e.sched == nil {
		e.sched = map[int]scheduled{}
	}
	e.sched[b.ID] = sc
}

// finishStep records one update_from_output, which ends at at: the batch's schedule, its outputs
// and finishes, and the scheduler gauges after it.
func (e *Engine) finishStep(b core.Batch, emitted map[string]int, drafts []metrics.Draft, fins []sched.Finished, at engine.Time) {
	sc := e.sched[b.ID]
	delete(e.sched, b.ID)
	st := metrics.Step{Scheduled: sc.tokens, Preempted: sc.preempted, Admitted: sc.admitted,
		Emitted: map[string]int{}, Drafts: drafts}
	for id, n := range emitted {
		st.Emitted[id] = n
	}
	for _, id := range sortedKeys(emitted) {
		if r := e.reqs[id]; r != nil && r.emitted == emitted[id] {
			// This batch emitted the request's first output token.
			st.FirstTokens = append(st.FirstTokens, id)
		}
	}
	for _, f := range fins {
		r := e.reqs[f.ID]
		reason := "stop"
		switch f.Status {
		case sched.FinishedLength:
			reason = "length"
		case sched.FinishedError:
			reason = "error"
		}
		st.Finished = append(st.Finished, metrics.Finish{ID: f.ID, Reason: reason,
			PromptTokens: len(r.req.Prompt), GenerationTokens: r.emitted, Preemptions: r.preemptions,
			MaxTokens: r.maxTokens})
	}
	nr, nw := e.s.Counts()
	st.Running, st.Waiting, st.Deferred = nr, nw, e.s.Deferred()
	st.KVUsage = e.s.KV.Pool.Usage()
	before := e.series.counts(e.metrics)
	e.metrics.Step(st)
	e.series.record(int64(at), before, e.metrics)
}

// prefixCaching reports whether the scheduler's prefix-cache lookups run.
func (e *Engine) prefixCaching() bool { return e.s.PrefixCaching() }

// Metrics implements engine.Engine: every series core/metrics keeps, as Prometheus exposes
// it, grouped into families by name.
func (e *Engine) Metrics(now engine.Time) engine.Metrics {
	series := e.metrics.Series()
	names := make([]string, 0, len(series))
	for k := range series {
		names = append(names, k)
	}
	sort.Strings(names)
	byFamily := map[string]*engine.Family{}
	var order []string
	for _, full := range names {
		sample, family, kind := splitSeries(full)
		f, ok := byFamily[family]
		if !ok {
			f = &engine.Family{Name: family, Kind: kind}
			byFamily[family] = f
			order = append(order, family)
		}
		f.Samples = append(f.Samples, engine.Sample{Name: sample.name, Labels: sample.labels, Value: series[full]})
	}
	out := engine.Metrics{At: now}
	for _, name := range order {
		out.Families = append(out.Families, *byFamily[name])
	}
	return out
}

type sampleName struct {
	name   string
	labels map[string]string
}

// splitSeries splits a series key of core/metrics ("name{k=v,...}") into its sample name and
// labels, and names its family and kind: a counter's family drops its _total suffix, a
// histogram's its _bucket, _sum or _count, as Prometheus names them.
func splitSeries(key string) (sampleName, string, engine.MetricKind) {
	name, labels := key, map[string]string(nil)
	if i := strings.IndexByte(key, '{'); i >= 0 && strings.HasSuffix(key, "}") {
		name = key[:i]
		labels = map[string]string{}
		for _, kv := range strings.Split(key[i+1:len(key)-1], ",") {
			if k, v, ok := strings.Cut(kv, "="); ok {
				labels[k] = v
			}
		}
	}
	for _, suf := range []string{"_bucket", "_sum", "_count"} {
		if base, ok := strings.CutSuffix(name, suf); ok && isHistogram(base) {
			return sampleName{name, labels}, base, engine.Histogram
		}
	}
	if base, ok := strings.CutSuffix(name, "_total"); ok && !isHistogram(name) {
		return sampleName{name, labels}, base, engine.Counter
	}
	return sampleName{name, labels}, name, engine.Gauge
}

// isHistogram reports whether a family is one of the histograms core/metrics keeps.
func isHistogram(family string) bool {
	switch family {
	case "vllm:iteration_tokens_total", "vllm:request_prompt_tokens", "vllm:request_generation_tokens",
		"vllm:request_max_num_generation_tokens", "vllm:request_params_max_tokens",
		"vllm:request_prefill_kv_computed_tokens", "vllm:request_params_n", "vllm:request_num_preemptions":
		return true
	}
	return false
}

func sortedKeys(m map[string]int) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
