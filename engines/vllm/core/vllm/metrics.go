// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/core/sched/scheduler.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/metrics/loggers.py (PrometheusStatLogger.record) and vllm/v1/metrics/stats.py.

package vllm

import (
	"sort"
	"strings"

	"github.com/inference-sim/blis-schemas/engine"

	core "github.com/inference-sim/blis-vllm/core/engine"
	"github.com/inference-sim/blis-vllm/core/metrics"
	"github.com/inference-sim/blis-vllm/core/sched"
)

// The engine records its metrics in two halves, as vLLM does: a batch's admissions are its
// schedule's PrefillStats (scheduler.py:1074-1080, 1314-1317@5df63a72), and its outputs,
// finishes and the scheduler gauges are what update_from_output reports for it
// (:2395-2422@5df63a72). core/metrics takes both in one Step; the admissions wait here until
// their batch completes.

// scheduled is what a schedule decided that its batch's metrics step reports.
type scheduled struct {
	admitted  []metrics.Admission
	preempted []string
	tokens    map[string]int
}

// recordSchedule notes a schedule's admissions, preemptions and tokens for its batch's step.
func (e *Engine) recordSchedule(b core.Batch) {
	sc := scheduled{preempted: append([]string(nil), b.Out.Preempted...), tokens: b.Out.Scheduled}
	for _, id := range b.Out.NewIDs {
		hit, looked := b.Out.PrefixHit[id]
		if !looked {
			continue
		}
		r := e.s.Request(id)
		sc.admitted = append(sc.admitted, metrics.Admission{ID: id, PromptTokens: r.NumTokens(),
			LocalHit: hit, Preempted: r.NumPreemptions > 0, CacheLookup: e.prefixCaching()})
	}
	if e.sched == nil {
		e.sched = map[int]scheduled{}
	}
	e.sched[b.ID] = sc
}

// finishStep records one update_from_output, which ends at at: the batch's schedule, its outputs
// and finishes, and the scheduler gauges after it.
func (e *Engine) finishStep(b core.Batch, emitted map[string]int, fins []sched.Finished, at engine.Time) {
	sc := e.sched[b.ID]
	delete(e.sched, b.ID)
	st := metrics.Step{Scheduled: sc.tokens, Preempted: sc.preempted, Admitted: sc.admitted,
		Emitted: map[string]int{}}
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
