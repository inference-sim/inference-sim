// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/metrics/loggers.py at vLLM v0.32.0 (5df63a7203cdb9328d824a27b682e77c7b9928be).
// Also vllm/v1/core/sched/scheduler.py (make_stats) and vllm/v1/engine/core.py (step).

package vllm

import (
	"encoding/json"
	"fmt"

	"github.com/inference-sim/blis-vllm/core/metrics"
)

// The server time series (a simulator's --server-series-interval-s) is the vllm:* series a
// benchmark client scrapes from /metrics, binned on the simulation clock. PrometheusStatLogger
// .record sets or increments each of them once per engine step (loggers.py:1138-1177,
// 1259-1275@5df63a72), from the SchedulerStats that make_stats builds at the end of
// update_from_output (scheduler.py:2408-2420, 2915-2965@5df63a72) and the IterationStats of
// that step's outputs. A step's change therefore lands at the instant its output is applied,
// its end, and core/metrics' Model records exactly that change per step: the series bins the
// Model's own values, so its bins sum to the engine's Metrics.
//
// Gauges (kv_cache_usage_perc, num_requests_running, num_requests_waiting) hold one step's
// value until the next: each bin carries the time-weighted mean of that step function and, as
// <name>_last, the value set by the last step that ended in the bin or before it. Counters are
// each bin's increments. vLLM labels each engine core's series engine="<index>"
// (loggers.py:499-506@5df63a72), so a data-parallel group reports one engine per rank.

// seriesGauges and seriesCounters are the Model series the server series bins, in
// ServerSeriesEngine's order.
var (
	seriesGauges   = [...]string{"vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting"}
	seriesCounters = [...]string{"vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
		"vllm:prompt_tokens_total", "vllm:prompt_tokens_by_source_total{source=local_compute}",
		"vllm:prompt_tokens_by_source_total{source=local_cache_hit}",
		"vllm:prompt_tokens_by_source_total{source=external_kv_transfer}", "vllm:generation_tokens_total"}
)

// ServerSeriesOutput is a server_series block: the bin width, the clock's origin and each
// engine core's series.
type ServerSeriesOutput struct {
	IntervalS float64              `json:"interval_s"`
	Origin    string               `json:"origin"`
	Engines   []ServerSeriesEngine `json:"engines"`
}

// ServerSeriesEngine is one engine core's binned series. Bin k covers [k, k+1) x interval of
// simulated time from time 0.
type ServerSeriesEngine struct {
	ID   string `json:"id"`
	Bins int    `json:"bins"`

	KVCacheUsage           []float64 `json:"kv_cache_usage"`
	KVCacheUsageLast       []float64 `json:"kv_cache_usage_last"`
	NumRequestsRunning     []float64 `json:"num_requests_running"`
	NumRequestsRunningLast []float64 `json:"num_requests_running_last"`
	NumRequestsWaiting     []float64 `json:"num_requests_waiting"`
	NumRequestsWaitingLast []float64 `json:"num_requests_waiting_last"`

	PrefixCacheQueries             []int64 `json:"prefix_cache_queries"`
	PrefixCacheHits                []int64 `json:"prefix_cache_hits"`
	PromptTokens                   []int64 `json:"prompt_tokens"`
	PromptTokensLocalCompute       []int64 `json:"prompt_tokens_local_compute"`
	PromptTokensLocalCacheHit      []int64 `json:"prompt_tokens_local_cache_hit"`
	PromptTokensExternalKVTransfer []int64 `json:"prompt_tokens_external_kv_transfer"`
	GenerationTokens               []int64 `json:"generation_tokens"`
}

// stepGauge bins a step function: value v holds from time at until the next set.
type stepGauge struct {
	v    float64
	at   int64
	sums []float64 // the integral of v over each bin, in value x microseconds
	last []float64 // the value in effect at each bin's end
}

func (g *stepGauge) grow(k int64) {
	for int64(len(g.sums)) <= k {
		g.sums = append(g.sums, 0)
		g.last = append(g.last, g.v)
	}
}

func (g *stepGauge) advanceTo(t, interval int64) {
	for g.at < t {
		k := g.at / interval
		end := min((k+1)*interval, t)
		g.grow(k)
		g.sums[k] += g.v * float64(end-g.at)
		g.last[k] = g.v
		g.at = end
	}
}

func (g *stepGauge) set(t, interval int64, v float64) {
	g.advanceTo(t, interval)
	g.v = v
	// A step ending exactly at a bin's start is that bin's last sample.
	k := t / interval
	g.grow(k)
	g.last[k] = v
}

// serverSeries is one engine core's recorder; nil when the series is off.
type serverSeries struct {
	interval int64
	end      int64
	gauges   [len(seriesGauges)]stepGauge
	counters [len(seriesCounters)][]int64
}

// counts reads the Model's counters before a step, for the step's increments.
func (s *serverSeries) counts(m *metrics.Model) (v [len(seriesCounters)]float64) {
	if s == nil {
		return v
	}
	for i, k := range seriesCounters {
		v[i] = m.Value(k)
	}
	return v
}

// record bins one step that ended at t: the gauges the Model holds after it, and its counters'
// increments since before.
func (s *serverSeries) record(t int64, before [len(seriesCounters)]float64, m *metrics.Model) {
	if s == nil {
		return
	}
	for i, k := range seriesGauges {
		s.gauges[i].set(t, s.interval, m.Value(k))
	}
	bin := t / s.interval
	for i, k := range seriesCounters {
		n := int64(m.Value(k) - before[i])
		if n == 0 {
			continue
		}
		for int64(len(s.counters[i])) <= bin {
			s.counters[i] = append(s.counters[i], 0)
		}
		s.counters[i][bin] += n
	}
	s.end = max(s.end, t)
}

// output closes the series at end and pads every array to bins.
func (s *serverSeries) output(id string, end int64, bins int) ServerSeriesEngine {
	for i := range s.gauges {
		s.gauges[i].advanceTo(end, s.interval)
	}
	mean := func(g *stepGauge) ([]float64, []float64) {
		m, l := make([]float64, bins), make([]float64, bins)
		for k := range bins {
			lo := int64(k) * s.interval
			hi := min(lo+s.interval, end)
			if k < len(g.sums) && hi > lo {
				m[k], l[k] = g.sums[k]/float64(hi-lo), g.last[k]
			} else {
				m[k], l[k] = g.v, g.v // past the engine's last step the gauge keeps its value
			}
		}
		return m, l
	}
	pad := func(c []int64) []int64 {
		out := make([]int64, bins)
		copy(out, c)
		return out
	}
	e := ServerSeriesEngine{ID: id, Bins: bins}
	e.KVCacheUsage, e.KVCacheUsageLast = mean(&s.gauges[0])
	e.NumRequestsRunning, e.NumRequestsRunningLast = mean(&s.gauges[1])
	e.NumRequestsWaiting, e.NumRequestsWaitingLast = mean(&s.gauges[2])
	c := s.counters
	e.PrefixCacheQueries, e.PrefixCacheHits, e.PromptTokens = pad(c[0]), pad(c[1]), pad(c[2])
	e.PromptTokensLocalCompute, e.PromptTokensLocalCacheHit = pad(c[3]), pad(c[4])
	e.PromptTokensExternalKVTransfer, e.GenerationTokens = pad(c[5]), pad(c[6])
	return e
}

// seriesBlock is engines' series binned to end, as one server_series block in JSON.
func seriesBlock(interval, end int64, ids []string, engines []*serverSeries) ([]byte, error) {
	bins := int(end/interval) + 1
	out := ServerSeriesOutput{IntervalS: float64(interval) / 1e6, Origin: "simulation time 0 (us)"}
	for i, s := range engines {
		out.Engines = append(out.Engines, s.output(ids[i], end, bins))
	}
	return json.Marshal(out)
}

// EnableServerSeries makes the engine record its server time series in bins of intervalUs
// microseconds of simulated time (a simulator's --server-series-interval-s); 0 leaves it off.
func (e *Engine) EnableServerSeries(intervalUs int64) {
	if intervalUs > 0 {
		e.series = &serverSeries{interval: intervalUs}
	}
}

// ServerSeriesSpan is the series' bin width and the last instant it recorded, both 0 when off.
func (e *Engine) ServerSeriesSpan() (interval, end int64) {
	if e.series == nil {
		return 0, 0
	}
	return e.series.interval, e.series.end
}

// ServerSeriesBlock is the series as engine id, binned to end, as a server_series block of this
// one engine in JSON; nil when off.
func (e *Engine) ServerSeriesBlock(id string, end int64) ([]byte, error) {
	if e.series == nil {
		return nil, nil
	}
	return seriesBlock(e.series.interval, max(end, e.series.end), []string{id}, []*serverSeries{e.series})
}

// EnableServerSeries makes every rank record its server time series.
func (g *DPGroup) EnableServerSeries(intervalUs int64) {
	for _, r := range g.ranks {
		r.e.EnableServerSeries(intervalUs)
	}
}

// ServerSeriesSpan is the ranks' bin width and the last instant any rank recorded.
func (g *DPGroup) ServerSeriesSpan() (interval, end int64) {
	for _, r := range g.ranks {
		iv, last := r.e.ServerSeriesSpan()
		interval, end = iv, max(end, last)
	}
	return interval, end
}

// ServerSeriesBlock is the group's series, one engine per rank, each labelled with its rank as
// vLLM labels a DP engine core's series (engine="<index>"); nil when off.
func (g *DPGroup) ServerSeriesBlock(id string, end int64) ([]byte, error) {
	interval, last := g.ServerSeriesSpan()
	if interval <= 0 {
		return nil, nil
	}
	ids := make([]string, len(g.ranks))
	engines := make([]*serverSeries, len(g.ranks))
	for i, r := range g.ranks {
		ids[i], engines[i] = fmt.Sprintf("%s/engine=%d", id, i), r.e.series
	}
	return seriesBlock(interval, max(end, last), ids, engines)
}
