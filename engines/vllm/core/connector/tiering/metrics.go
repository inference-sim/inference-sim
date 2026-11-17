// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_offload/tiering/metrics.py, tiering/manager.py (get_stats, config_info) and
// vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).

package tiering

import (
	"fmt"
	"strings"
)

// The metric families the tiering manager reports (TieringOffloadingMetrics, tiering/base.py:44-67), and
// the CPU tier's (CPUOffloadingMetrics, cpu/common.py:13-18).
const (
	MetricLookupSyncDelay           = "vllm:kv_offload_tiering_lookup_sync_delay_seconds"
	MetricLookupAsyncDelay          = "vllm:kv_offload_tiering_lookup_async_delay_seconds"
	MetricReadBytes                 = "vllm:kv_offload_tiering_read_bytes"
	MetricReadTime                  = "vllm:kv_offload_tiering_read_time"
	MetricWriteBytes                = "vllm:kv_offload_tiering_write_bytes"
	MetricWriteTime                 = "vllm:kv_offload_tiering_write_time"
	MetricPromotionJobFailures      = "vllm:kv_offload_tiering_promotion_job_failures"
	MetricCascadeJobFailures        = "vllm:kv_offload_tiering_cascade_job_failures"
	MetricChunkQueries              = "vllm:kv_offload_tiering_chunk_queries"
	MetricChunkHits                 = "vllm:kv_offload_tiering_chunk_hits"
	MetricPrimaryWriteUsage         = "vllm:kv_offload_tiering_primary_write_usage_perc"
	MetricPrimaryReadUsage          = "vllm:kv_offload_tiering_primary_read_usage_perc"
	MetricPromotionAllocFailures    = "vllm:kv_offload_tiering_promotion_allocation_failures"
	MetricActivePromotionJobs       = "vllm:kv_offload_tiering_active_promotion_jobs"
	MetricActiveCascadeJobs         = "vllm:kv_offload_tiering_active_cascade_jobs"
	MetricBackpressureLatencyEMA    = "vllm:kv_offload_tiering_backpressure_store_latency_ema"
	MetricBackpressureStoresDropped = "vllm:kv_offload_tiering_backpressure_stores_dropped"
	MetricBackpressureBlocksDropped = "vllm:kv_offload_tiering_backpressure_blocks_dropped"
	MetricPromotionLatency          = "vllm:kv_offload_tiering_promotion_latency_seconds"

	MetricCPUUsage      = "vllm:kv_offload_cpu_cache_usage_perc"
	MetricCPUAllocation = "vllm:kv_offload_cpu_allocation_size"
	MetricCPUWriteUsage = "vllm:kv_offload_cpu_cache_write_usage_perc"
	MetricCPUReadUsage  = "vllm:kv_offload_cpu_cache_read_usage_perc"
)

// TierLabelKey is TIER_LABEL, the label name of a tier in config_info (tiering/base.py:40).
const TierLabelKey = "tier"

// PrimaryLabel is the primary tier's label (PRIMARY_TIER_LABEL, tiering/metrics.py:20).
const PrimaryLabel = "0:primary"

type metricKind int

const (
	kindCounter metricKind = iota
	kindGauge
	kindHistogram
)

// series is one label tuple's value: a number, or a histogram's observations.
type series struct {
	num float64
	obs []float64
}

// Stats is OffloadingConnectorStats (offloading/metrics.py:160-340): metric families, each a value per
// label tuple. A counter sums, a gauge keeps the latest, a histogram is the list of its observations.
type Stats struct {
	kinds map[string]metricKind
	vals  map[string]map[string]*series
}

// NewStats is an empty payload.
func NewStats() *Stats {
	return &Stats{kinds: map[string]metricKind{}, vals: map[string]map[string]*series{}}
}

// labelRepr is the Python repr of a label tuple, the key vLLM's payload and its recorded traces use.
func labelRepr(labels []string) string {
	switch len(labels) {
	case 0:
		return "()"
	case 1:
		return "('" + labels[0] + "',)"
	}
	return "('" + strings.Join(labels, "', '") + "')"
}

func (s *Stats) at(name string, kind metricKind, labels []string) *series {
	if _, ok := s.kinds[name]; !ok {
		s.kinds[name] = kind
	}
	m := s.vals[name]
	if m == nil {
		m = map[string]*series{}
		s.vals[name] = m
	}
	k := labelRepr(labels)
	if m[k] == nil {
		m[k] = &series{}
	}
	return m[k]
}

// IncreaseCounter is increase_counter.
func (s *Stats) IncreaseCounter(name string, v float64, labels ...string) {
	s.at(name, kindCounter, labels).num += v
}

// SetGauge is set_gauge.
func (s *Stats) SetGauge(name string, v float64, labels ...string) {
	s.at(name, kindGauge, labels).num = v
}

// ObserveHistogram is observe_histogram.
func (s *Stats) ObserveHistogram(name string, v float64, labels ...string) {
	x := s.at(name, kindHistogram, labels)
	x.obs = append(x.obs, v)
}

// IsEmpty is is_empty (without the info, which a manager's stats never carry).
func (s *Stats) IsEmpty() bool { return s == nil || len(s.vals) == 0 }

// Aggregate is aggregate: counters add, gauges take the other's value, histograms extend.
func (s *Stats) Aggregate(o *Stats) {
	if o.IsEmpty() {
		return
	}
	for name, byLabel := range o.vals {
		kind := o.kinds[name]
		if _, ok := s.kinds[name]; !ok {
			s.kinds[name] = kind
		}
		m := s.vals[name]
		if m == nil {
			m = map[string]*series{}
			s.vals[name] = m
		}
		for k, v := range byLabel {
			cur := m[k]
			if cur == nil {
				cur = &series{}
				m[k] = cur
			}
			switch kind {
			case kindHistogram:
				cur.obs = append(cur.obs, v.obs...)
			case kindCounter:
				cur.num += v.num
			default:
				cur.num = v.num
			}
		}
	}
}

// Values are the payload as the FID-5 traces record it (oracle/kv_interfaces.py stat): family, then
// the label tuple's repr, then the value: a number, or for a histogram its observations. The families
// that vLLM feeds from the clock (names ending _seconds or _time) have no value the port can match
// (the port's clock is the virtual one); a histogram of them is its number of observations
// ({"observations": n}) and a counter or gauge "clock".
func (s *Stats) Values() map[string]map[string]any {
	out := map[string]map[string]any{}
	if s == nil {
		return out
	}
	for name, byLabel := range s.vals {
		clock := strings.HasSuffix(name, "_seconds") || strings.HasSuffix(name, "_time")
		m := map[string]any{}
		for k, v := range byLabel {
			switch {
			case s.kinds[name] == kindHistogram && clock:
				m[k] = map[string]any{"observations": len(v.obs)}
			case s.kinds[name] == kindHistogram:
				obs := make([]any, len(v.obs))
				for i, o := range v.obs {
					obs[i] = o
				}
				m[k] = obs
			case clock:
				m[k] = "clock"
			default:
				m[k] = v.num
			}
		}
		out[name] = m
	}
	return out
}

// Optional parts of a primary tier and of a secondary tier that report metrics. A tier that has none
// reports none (SecondaryTierManager.get_stats and config_info default to None and {}).
type (
	// PrimaryMetrics is a primary tier's get_stats (nil: none) and its one config_info series.
	PrimaryMetrics interface {
		Stats() *Stats
		ConfigInfo() map[string]any
	}
	// TierMetrics is a secondary tier's get_stats (nil: none).
	TierMetrics interface{ Stats() *Stats }
	// TierInfo is a secondary tier's config_info series.
	TierInfo interface{ ConfigInfo() map[string]any }
	// DetectorMetrics is a backpressure detector's part of get_stats (tiering/metrics.py:130-166).
	DetectorMetrics interface {
		StoreLatencyEMA() float64
		PopStoresDropped() (stores, blocks int)
	}
)

// tierState is _TierState (tiering/metrics.py:46-50).
type tierState struct{ promotions, cascades, writeChunks, readChunks int }

// reqLookups is _RequestMetricsState: the lookups observed until the request is allocated, per tier
// label and key, the clock reading when a lookup first returned RETRY (nil once resolved).
type reqLookups struct {
	observed map[string]map[Key]*float64 // nil: the request is allocated
}

// tracker is TieringMetricsTracker (tiering/metrics.py:53-320).
type tracker struct {
	tierTypes  []string
	numPrimary int
	chunkBytes int64
	clock      func() float64
	requests   map[string]*reqLookups
	tiers      []tierState
	stats      *Stats
}

func newTracker(types []string, numPrimary int, chunkBytes int64, clock func() float64) *tracker {
	return &tracker{tierTypes: types, numPrimary: numPrimary, chunkBytes: chunkBytes, clock: clock,
		requests: map[string]*reqLookups{}, tiers: make([]tierState, len(types)), stats: NewStats()}
}

func (t *tracker) label(i int) string { return fmt.Sprintf("%d:%s", i+1, t.tierTypes[i]) }

func (t *tracker) onNewRequest(c *ReqContext) {
	t.requests[c.ReqID] = &reqLookups{observed: map[string]map[Key]*float64{}}
}

func (t *tracker) onRequestAllocated(c *ReqContext) {
	if st := t.requests[c.ReqID]; st != nil {
		st.observed = nil
	}
}

func (t *tracker) onRequestFinished(c *ReqContext) { delete(t.requests, c.ReqID) }

// onLookup is on_lookup (:105-135). The lookup call itself takes no virtual time.
func (t *tracker) onLookup(c *ReqContext, k Key, label string, r Lookup) {
	st := t.requests[c.ReqID]
	if st == nil {
		st = &reqLookups{observed: map[string]map[Key]*float64{}}
		t.requests[c.ReqID] = st
	}
	switch {
	case st.observed != nil && (r == Hit || r == Miss):
		obs := st.observed[label]
		if obs == nil {
			obs = map[Key]*float64{}
			st.observed[label] = obs
		}
		start, seen := obs[k]
		if !seen || start != nil {
			obs[k] = nil
			t.resolved(label, r, start)
		}
	case r == Retry && st.observed != nil:
		obs := st.observed[label]
		if obs == nil {
			obs = map[Key]*float64{}
			st.observed[label] = obs
		}
		if _, seen := obs[k]; !seen {
			now := t.clock()
			obs[k] = &now
		}
	}
}

// resolved is _observe_resolved_lookup (:294-320).
func (t *tracker) resolved(label string, r Lookup, asyncStart *float64) {
	t.stats.IncreaseCounter(MetricChunkQueries, 1, label)
	if r == Hit {
		t.stats.IncreaseCounter(MetricChunkHits, 1, label)
	}
	t.stats.ObserveHistogram(MetricLookupSyncDelay, 0, label)
	if asyncStart != nil {
		t.stats.ObserveHistogram(MetricLookupAsyncDelay, t.clock()-*asyncStart, label)
	}
}

func (t *tracker) onJobRegistered(jm jobMeta) {
	s := &t.tiers[jm.tier]
	n := len(jm.job.ChunkIDs)
	if jm.job.IsPromotion {
		s.promotions++
		s.writeChunks += n
	} else {
		s.cascades++
		s.readChunks += n
	}
}

// onJobFinished is on_job_finished (:137-143).
func (t *tracker) onJobFinished(jm jobMeta, r JobResult) {
	j := jm.job
	label := t.label(jm.tier)
	n := len(j.Keys)
	counted := true
	if !r.Success {
		name := MetricCascadeJobFailures
		if j.IsPromotion {
			name = MetricPromotionJobFailures
		}
		t.stats.IncreaseCounter(name, 1, label)
		if j.IsPromotion && len(r.SuccessfulKeys) > 0 {
			n = len(r.SuccessfulKeys)
		} else {
			counted = false
		}
	}
	if counted {
		bytes, tm := MetricWriteBytes, MetricWriteTime
		if j.IsPromotion {
			bytes, tm = MetricReadBytes, MetricReadTime
		}
		t.stats.IncreaseCounter(bytes, float64(int64(n)*t.chunkBytes), label)
		if r.HasTransferTime {
			t.stats.IncreaseCounter(tm, r.TransferTime, label)
		}
	}
	if j.IsPromotion {
		res := "failure"
		if r.Success {
			res = "success"
		}
		t.stats.ObserveHistogram(MetricPromotionLatency, t.clock()-j.SubmitTime, label, res)
	}
	s := &t.tiers[jm.tier]
	n = len(j.ChunkIDs)
	if j.IsPromotion {
		s.promotions--
		s.writeChunks -= n
	} else {
		s.cascades--
		s.readChunks -= n
	}
}

func (t *tracker) onPromotionAllocationFailure() {
	t.stats.IncreaseCounter(MetricPromotionAllocFailures, 1)
}

// recordBackpressure is record_backpressure (:130-166).
func (t *tracker) recordBackpressure(tiers []Tier) {
	for i, tier := range tiers {
		d, ok := tier.Backpressure().(DetectorMetrics)
		if !ok || tier.Backpressure() == nil {
			continue
		}
		label := t.label(i)
		t.stats.SetGauge(MetricBackpressureLatencyEMA, d.StoreLatencyEMA(), label)
		stores, blocks := d.PopStoresDropped()
		if stores > 0 {
			t.stats.IncreaseCounter(MetricBackpressureStoresDropped, float64(stores), label)
		}
		if blocks > 0 {
			t.stats.IncreaseCounter(MetricBackpressureBlocksDropped, float64(blocks), label)
		}
	}
}

// take is take_stats (:168-183): the active-transfer gauges of the moment, and what was counted since.
func (t *tracker) take() *Stats {
	active := NewStats()
	for i, s := range t.tiers {
		label := t.label(i)
		var w, r float64
		if t.numPrimary > 0 {
			w, r = float64(s.writeChunks)/float64(t.numPrimary), float64(s.readChunks)/float64(t.numPrimary)
		}
		active.SetGauge(MetricPrimaryWriteUsage, w, label)
		active.SetGauge(MetricPrimaryReadUsage, r, label)
		active.SetGauge(MetricActivePromotionJobs, float64(s.promotions), label)
		active.SetGauge(MetricActiveCascadeJobs, float64(s.cascades), label)
	}
	var out *Stats
	if !active.IsEmpty() {
		out = active
	}
	if !t.stats.IsEmpty() {
		if out == nil {
			out = t.stats
		} else {
			out.Aggregate(t.stats)
		}
		t.stats = NewStats()
	}
	return out
}

// idle is assert_idle (:185-193).
func (t *tracker) idle() bool {
	for _, s := range t.tiers {
		if s != (tierState{}) {
			return false
		}
	}
	return true
}

// Stats is get_stats (manager.py:1205-1229): the primary tier's, then the manager's own, then each
// secondary tier's; nil when there are none.
func (m *Manager) Stats() *Stats {
	var out *Stats
	if p, ok := m.primary.(PrimaryMetrics); ok {
		if s := p.Stats(); !s.IsEmpty() {
			out = s
		}
	}
	m.metrics.recordBackpressure(m.tiers)
	if s := m.metrics.take(); s != nil {
		if out == nil {
			out = s
		} else {
			out.Aggregate(s)
		}
	}
	for _, t := range m.tiers {
		tm, ok := t.(TierMetrics)
		if !ok {
			continue
		}
		s := tm.Stats()
		if s.IsEmpty() {
			continue
		}
		if out == nil {
			out = s
		} else {
			out.Aggregate(s)
		}
	}
	return out
}

// ConfigInfo is config_info (manager.py:1231-1248): one series for the primary tier, then one for each
// secondary tier, each with its tier label.
func (m *Manager) ConfigInfo() []map[string]any {
	var primary map[string]any
	if p, ok := m.primary.(PrimaryMetrics); ok {
		primary = p.ConfigInfo()
	}
	out := []map[string]any{withTier(primary, PrimaryLabel)}
	for i, t := range m.tiers {
		var info map[string]any
		if ti, ok := t.(TierInfo); ok {
			info = ti.ConfigInfo()
		}
		out = append(out, withTier(info, m.metrics.label(i)))
	}
	return out
}

func withTier(info map[string]any, label string) map[string]any {
	out := map[string]any{}
	for k, v := range info {
		out[k] = v
	}
	out[TierLabelKey] = label
	return out
}
