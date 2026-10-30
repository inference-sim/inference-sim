package sim

import (
	"math"
	"slices"
	"sort"
)

// PercentileMethodNumpyLinear names the percentile method of every *_p<NN> aggregate:
// sort, then interpolate linearly at rank q/100*(n-1), numpy's default "linear" method and
// aiperf's (aiperf metrics/metric_dicts.py:78-82). CalculatePercentile implements it.
const PercentileMethodNumpyLinear = "numpy_linear"

// MeasuredWindow fixes the measured window's bounds in ticks: aiperf's profiling phase
// start and completion. aiperf divides output tokens by the export context's window
// (metrics/metric_dicts.py:217-238 observation_duration), which records_manager.py:1437-1442
// sets to the profiling phase's start_ns and requests_end_ns, the phase's complete_at
// (timing/phase/progress_tracker.py:225). A replay that runs a warm-up or a grace period
// sets it on the aggregate Metrics before EmitOutput, and the file says so
// (measured_window_source "explicit"); a completion after EndUs is outside the window. Unset,
// the window is derived from the measured requests (first arrival to last completion) and
// labelled "derived". Zero EndUs derives only the end.
type MeasuredWindow struct {
	StartUs int64
	EndUs   int64
}

// InFlightSample is one step of the cluster's engine-side in-flight request count: Count
// holds from AtUs until the next sample.
type InFlightSample struct {
	AtUs  int64
	Count int
}

// StepOwner attributes the next Steps entries of Metrics.AllITLs to one completed request:
// its phase and completion time, which decide whether its steps are in step_interval_*.
type StepOwner struct {
	CompletionUs int64
	Steps        int
	Phase        Phase
}

// The values of measured_window_source.
const (
	WindowSourceExplicit = "explicit"
	WindowSourceDerived  = "derived"
)

// MeasuredMetrics are the aiperf-defined aggregates of the --metrics-path file (#1938,
// #1904), over the requests in the measured window: those whose Phase is measured that
// completed (in the window, when MeasuredWindow sets its end). One value per request;
// percentiles are numpy-linear. The latency families that already had keys in
// MetricsOutput (ttft_*, e2e_*) are written into those keys in the file (see EmitOutput);
// these are the keys that are new.
//
// File-only, like CacheHitRate: stdout keeps the whole-run figures it always printed
// (INV-6). With no phase set and no window, the measured set is every completed request.
type MeasuredMetrics struct {
	PercentileMethod string `json:"percentile_method"`

	TTFTP50Ms float64 `json:"ttft_p50_ms"`
	E2EP50Ms  float64 `json:"e2e_p50_ms"`

	// Per request (latency - TTFT) / (OSL - 1) over requests with OSL >= 2 (aiperf
	// metrics/types/inter_token_latency_metric.py:48-55): aiperf's ITL, per TOKEN, so under
	// speculative decoding it is about the step interval over the accepted length.
	RequestITLMeanMs float64 `json:"request_itl_mean_ms"`
	RequestITLP50Ms  float64 `json:"request_itl_p50_ms"`
	RequestITLP90Ms  float64 `json:"request_itl_p90_ms"`
	RequestITLP99Ms  float64 `json:"request_itl_p99_ms"`
	// 1000 / request_itl_p50_ms and 1000 / request_itl_p90_ms, tokens/s/user.
	InteractivityP50 float64 `json:"interactivity_p50"`
	InteractivityP90 float64 `json:"interactivity_p90"`
	// Mean over requests of 1 / ITL_s (aiperf output_token_throughput_metrics.py:64-78).
	OutputTokPerUserMean float64 `json:"output_tok_per_user_mean"`

	// The per-decode-step series the legacy itl_* keys summarize, over measured requests.
	// Under speculative decoding one entry covers every token a verify step accepted.
	StepIntervalMeanMs float64 `json:"step_interval_mean_ms"`
	StepIntervalP90Ms  float64 `json:"step_interval_p90_ms"`
	StepIntervalP99Ms  float64 `json:"step_interval_p99_ms"`

	// Σ OSL / measured window (aiperf output_token_throughput_metrics.py:40-42), and where
	// the window came from (WindowSourceExplicit or WindowSourceDerived; see MeasuredWindow).
	OutputThroughputTokS float64 `json:"output_throughput_tok_s"`
	MeasuredWindowS      float64 `json:"measured_window_s"`
	MeasuredWindowSource string  `json:"measured_window_source"`

	// Counts in the window. The measured completions have their own key: completed_requests
	// stays the run-wide count INV-1 conserves in the file as on stdout. Cancelled: in flight
	// when the window closed, or completed after its explicit end. Failed: timed out (aiperf
	// records a timeout as an error), dropped, rejected or shed.
	MeasuredCompletedRequests int `json:"measured_completed_requests"`
	CancelledRequests         int `json:"cancelled_requests"`
	ContextOverflowRequests   int `json:"context_overflow_requests"`
	FailedRequests            int `json:"failed_requests"`

	// Prompt tokens the offload tier supplied over prompt tokens queried, measured
	// requests' admissions only (cache_hit_rate in the file is the same population).
	CPUTierHitRate float64 `json:"cpu_tier_hit_rate"`

	GatewayQueueDelayMeanMs float64 `json:"gateway_queue_delay_mean_ms"`
	GatewayQueueDelayP90Ms  float64 `json:"gateway_queue_delay_p90_ms"`

	InFlightRequestsMean float64 `json:"in_flight_requests_mean"`
	InFlightRequestsMax  int     `json:"in_flight_requests_max"`

	KVCapacityTokensPerReplica int64 `json:"kv_capacity_tokens_per_replica"`
}

// measuredLatencies are the latency families whose keys MetricsOutput already has; the file
// carries their measured-window values (see applyMeasured).
type measuredLatencies struct {
	ttftMean, ttftP90, ttftP95, ttftP99 float64
	e2eMean, e2eP90, e2eP95, e2eP99     float64
	cacheHitRate                        float64
	// stepP50 and stepP95 complete the step-interval series' percentiles for the Result
	// document (the file keeps the mean, p90 and p99 it always wrote).
	stepP50, stepP95 float64
	// itlCount and perUserCount are the samples behind request_itl_* and
	// output_tok_per_user_mean, so a reader can tell a zero from no sample.
	itlCount, perUserCount int
	// startUs and endUs bound the window the aggregates were computed over (0, 0 when it
	// is empty).
	startUs, endUs int64
}

// WholeRunLatency keeps, in the --metrics-path file, the whole-run figures stdout prints for
// the keys the file restates over the measured window, for a consumer of the old meaning
// (blis calibrate compares whole-run TTFT and throughput).
type WholeRunLatency struct {
	TTFTMeanMs   float64 `json:"ttft_mean_ms"`
	TTFTP90Ms    float64 `json:"ttft_p90_ms"`
	TTFTP95Ms    float64 `json:"ttft_p95_ms"`
	TTFTP99Ms    float64 `json:"ttft_p99_ms"`
	E2EMeanMs    float64 `json:"e2e_mean_ms"`
	E2EP90Ms     float64 `json:"e2e_p90_ms"`
	E2EP95Ms     float64 `json:"e2e_p95_ms"`
	E2EP99Ms     float64 `json:"e2e_p99_ms"`
	CacheHitRate float64 `json:"cache_hit_rate"`
}

// Measured returns the measured-window aggregates EmitOutput writes into the --metrics-path
// file, for a caller that reads them without writing the file.
func (m *Metrics) Measured() MeasuredMetrics {
	out, _ := m.buildMeasured()
	return out
}

// MeasuredSummary is the measured-window aggregates of MeasuredMetrics together with the
// latency families MetricsOutput already had keys for (ttft_*, e2e_*), over the same
// window, and the window's bounds in ticks. It is what the Result document's metrics are
// read from (#1871).
type MeasuredSummary struct {
	MeasuredMetrics
	TTFTMeanMs, TTFTP90Ms, TTFTP95Ms, TTFTP99Ms float64
	E2EMeanMs, E2EP90Ms, E2EP95Ms, E2EP99Ms     float64
	// CacheHitRate is the measured requests' prompt-lookup hit rate.
	CacheHitRate float64
	// StepIntervalP50Ms and StepIntervalP95Ms complete MeasuredMetrics' step-interval
	// percentiles.
	StepIntervalP50Ms, StepIntervalP95Ms float64
	// RequestITLCount is the number of requests behind request_itl_* (output of two or more
	// tokens), and PerUserCount those behind output_tok_per_user_mean (a positive ITL).
	RequestITLCount, PerUserCount int
	// WindowStartUs and WindowEndUs bound the window; both 0 when it is empty.
	WindowStartUs, WindowEndUs int64
}

// Summary returns the measured-window aggregates and latencies of m.
func (m *Metrics) Summary() MeasuredSummary {
	out, lat := m.buildMeasured()
	return summaryOf(out, lat)
}

// SummaryOver returns the measured-window aggregates of the requests include selects,
// over the fixed window [startUs, endUs]: a split of a run (by model, tenant or adapter)
// computed with the definitions of the whole, over the whole's window. Only the
// per-request families are meaningful in a split; the step, cache and in-flight figures
// are the whole's and are left zero.
func (m *Metrics) SummaryOver(include func(RequestMetrics) bool, startUs, endUs int64) MeasuredSummary {
	sub := &Metrics{
		Requests:               make(map[string]RequestMetrics),
		RequestTTFTs:           m.RequestTTFTs,
		RequestE2Es:            m.RequestE2Es,
		RequestCompletionTimes: m.RequestCompletionTimes,
		MeasuredWindow:         &MeasuredWindow{StartUs: startUs, EndUs: endUs},
	}
	for id, rm := range m.Requests {
		if include(rm) {
			sub.Requests[id] = rm
		}
	}
	out, lat := sub.buildMeasured()
	return summaryOf(out, lat)
}

func summaryOf(out MeasuredMetrics, lat measuredLatencies) MeasuredSummary {
	return MeasuredSummary{
		MeasuredMetrics: out,
		TTFTMeanMs:      lat.ttftMean, TTFTP90Ms: lat.ttftP90, TTFTP95Ms: lat.ttftP95, TTFTP99Ms: lat.ttftP99,
		E2EMeanMs: lat.e2eMean, E2EP90Ms: lat.e2eP90, E2EP95Ms: lat.e2eP95, E2EP99Ms: lat.e2eP99,
		CacheHitRate:      lat.cacheHitRate,
		StepIntervalP50Ms: lat.stepP50, StepIntervalP95Ms: lat.stepP95,
		RequestITLCount: lat.itlCount, PerUserCount: lat.perUserCount,
		WindowStartUs: lat.startUs, WindowEndUs: lat.endUs,
	}
}

// buildMeasured computes the measured-window aggregates from m.
func (m *Metrics) buildMeasured() (MeasuredMetrics, measuredLatencies) {
	out := MeasuredMetrics{PercentileMethod: PercentileMethodNumpyLinear}
	var lat measuredLatencies

	var startUs, endUs int64 = math.MaxInt64, 0
	if w := m.MeasuredWindow; w != nil {
		startUs = w.StartUs
		endUs = w.EndUs
	}
	out.MeasuredWindowSource = WindowSourceDerived
	if m.MeasuredWindow != nil {
		out.MeasuredWindowSource = WindowSourceExplicit
	}
	inWindow := func(completionUs int64) bool {
		return m.MeasuredWindow == nil || m.MeasuredWindow.EndUs <= 0 || completionUs <= m.MeasuredWindow.EndUs
	}

	// perUser is summed in sorted order below, as every other mean is, so the aggregate is a
	// function of the request set and not of the IDs it is iterated in (INV-13).
	var ttfts, e2es, itls, gateway, perUser []float64
	var outputTokens int64
	for _, id := range sortedRequestIDs(m.Requests) {
		rm := m.Requests[id]
		if !rm.Phase.Measured() {
			continue
		}
		e2e := m.RequestE2Es[id]
		completionUs := int64(m.RequestCompletionTimes[id])
		if rm.timedOut && e2e <= 0 {
			continue // failed_requests (RecordTimeout)
		}
		if e2e <= 0 || !inWindow(completionUs) {
			out.CancelledRequests++ // in flight when the window closed, or completed after it
			continue
		}
		out.MeasuredCompletedRequests++
		ttft := m.RequestTTFTs[id]
		// aiperf's request latency ends at the last content chunk
		// (metrics/types/request_latency_metric.py:38-49). vLLM v0.31.0 queues the final
		// token's RequestOutput (v1/engine/output_processor.py:745) before it frees the request
		// and records its finished stats (:759, :766), and the chat stream yields that token's
		// chunk with finish_reason (entrypoints/openai/chat_completion/serving.py:737, :828)
		// before the usage chunk and [DONE] (:891, :931). The completion overhead E2E carries
		// (the kernel's host_completion: finish bookkeeping, serialization, teardown) follows
		// the last chunk, so it is left out.
		latency := e2e - float64(rm.completionOverheadTicks)
		ttfts = append(ttfts, ttft)
		e2es = append(e2es, latency)
		gateway = append(gateway, rm.GatewayQueueDelay*1e3) // ms -> ticks, as the helpers expect
		outputTokens += int64(rm.OutputTokens)
		if m.MeasuredWindow == nil {
			startUs = min(startUs, completionUs-int64(e2e))
		}
		if m.MeasuredWindow == nil || m.MeasuredWindow.EndUs <= 0 {
			endUs = max(endUs, completionUs)
		}
		// aiperf inter_token_latency_metric.py:48-55: (latency - TTFT) / (OSL - 1), OSL >= 2.
		if rm.OutputTokens >= 2 {
			itl := (latency - ttft) / float64(rm.OutputTokens-1)
			itls = append(itls, itl)
			// aiperf output_token_throughput_metrics.py:74-77 raises NoMetricValue for ITL == 0,
			// so such a record has no per-user value: it leaves the sum and the count.
			if itl > 0 {
				perUser = append(perUser, 1e6/itl)
			}
		}
	}
	out.CancelledRequests += m.MeasuredPipelineCancellations
	out.ContextOverflowRequests = m.MeasuredContextOverflows
	out.FailedRequests = m.MeasuredFailures

	sort.Float64s(ttfts)
	sort.Float64s(e2es)
	sort.Float64s(itls)
	sort.Float64s(gateway)
	lat.ttftMean = CalculateMean(ttfts)
	out.TTFTP50Ms = CalculatePercentile(ttfts, 50)
	lat.ttftP90 = CalculatePercentile(ttfts, 90)
	lat.ttftP95 = CalculatePercentile(ttfts, 95)
	lat.ttftP99 = CalculatePercentile(ttfts, 99)
	lat.e2eMean = CalculateMean(e2es)
	out.E2EP50Ms = CalculatePercentile(e2es, 50)
	lat.e2eP90 = CalculatePercentile(e2es, 90)
	lat.e2eP95 = CalculatePercentile(e2es, 95)
	lat.e2eP99 = CalculatePercentile(e2es, 99)

	out.RequestITLMeanMs = CalculateMean(itls)
	out.RequestITLP50Ms = CalculatePercentile(itls, 50)
	out.RequestITLP90Ms = CalculatePercentile(itls, 90)
	out.RequestITLP99Ms = CalculatePercentile(itls, 99)
	out.InteractivityP50 = reciprocalPerSecond(out.RequestITLP50Ms)
	out.InteractivityP90 = reciprocalPerSecond(out.RequestITLP90Ms)
	if len(perUser) > 0 {
		sort.Float64s(perUser)
		var sum float64
		for _, v := range perUser {
			sum += v
		}
		out.OutputTokPerUserMean = sum / float64(len(perUser))
	}
	lat.itlCount, lat.perUserCount = len(itls), len(perUser)

	steps := measuredStepIntervals(m.AllITLs, m.StepOwners, inWindow)
	out.StepIntervalMeanMs = CalculateMean(steps)
	out.StepIntervalP90Ms = CalculatePercentile(steps, 90)
	out.StepIntervalP99Ms = CalculatePercentile(steps, 99)
	lat.stepP50 = CalculatePercentile(steps, 50)
	lat.stepP95 = CalculatePercentile(steps, 95)

	out.GatewayQueueDelayMeanMs = CalculateMean(gateway)
	out.GatewayQueueDelayP90Ms = CalculatePercentile(gateway, 90)

	if endUs > startUs {
		lat.startUs, lat.endUs = startUs, endUs
		out.MeasuredWindowS = float64(endUs-startUs) / 1e6
		out.OutputThroughputTokS = float64(outputTokens) / out.MeasuredWindowS
		out.InFlightRequestsMean, out.InFlightRequestsMax = inFlightOver(m.InFlightSamples, startUs, endUs)
	}

	if q := m.MeasuredPrefixCacheQueryTokens; q > 0 {
		lat.cacheHitRate = float64(m.MeasuredPrefixCacheHitTokens) / float64(q)
		out.CPUTierHitRate = float64(m.MeasuredCPUTierHitTokens) / float64(q)
	}
	out.KVCapacityTokensPerReplica = m.KVCapacityTokensPerReplica
	return out, lat
}

// applyMeasured writes the measured-window values into the keys MetricsOutput already
// carries, for the --metrics-path file, keeping the whole-run values under whole_run.
// cacheHitRate is the whole run's prompt-lookup rate.
func (o *MetricsOutput) applyMeasured(lat measuredLatencies, cacheHitRate float64) {
	o.WholeRun = &WholeRunLatency{
		TTFTMeanMs: o.TTFTMeanMs, TTFTP90Ms: o.TTFTP90Ms, TTFTP95Ms: o.TTFTP95Ms, TTFTP99Ms: o.TTFTP99Ms,
		E2EMeanMs: o.E2EMeanMs, E2EP90Ms: o.E2EP90Ms, E2EP95Ms: o.E2EP95Ms, E2EP99Ms: o.E2EP99Ms,
		CacheHitRate: cacheHitRate,
	}
	o.TTFTMeanMs, o.TTFTP90Ms, o.TTFTP95Ms, o.TTFTP99Ms = lat.ttftMean, lat.ttftP90, lat.ttftP95, lat.ttftP99
	o.E2EMeanMs, o.E2EP90Ms, o.E2EP95Ms, o.E2EP99Ms = lat.e2eMean, lat.e2eP90, lat.e2eP95, lat.e2eP99
	o.CacheHitRate = &lat.cacheHitRate
}

// reciprocalPerSecond is 1000 / ms, the rate of one event every ms milliseconds, or 0 for a
// non-positive interval.
func reciprocalPerSecond(ms float64) float64 {
	if ms <= 0 {
		return 0
	}
	return 1000 / ms
}

// measuredStepIntervals returns, sorted, the AllITLs entries of the measured requests whose
// completion is in the window, walking all in step with its owners.
func measuredStepIntervals(all []int64, owners []StepOwner, inWindow func(int64) bool) []int64 {
	var out []int64
	at := 0
	for _, o := range owners {
		if o.Phase.Measured() && inWindow(o.CompletionUs) {
			out = append(out, all[at:at+o.Steps]...)
		}
		at += o.Steps
	}
	slices.Sort(out)
	return out
}

// inFlightOver returns the time-weighted mean and the maximum of the in-flight step
// function over [startUs, endUs).
func inFlightOver(samples []InFlightSample, startUs, endUs int64) (float64, int) {
	var area float64
	peak := 0
	current := 0
	at := startUs
	for _, s := range samples {
		if s.AtUs <= startUs {
			current = s.Count
			continue
		}
		if s.AtUs >= endUs {
			break
		}
		area += float64(current) * float64(s.AtUs-at)
		peak = max(peak, current)
		current, at = s.Count, s.AtUs
	}
	area += float64(current) * float64(endUs-at)
	peak = max(peak, current)
	return area / float64(endUs-startUs), peak
}
