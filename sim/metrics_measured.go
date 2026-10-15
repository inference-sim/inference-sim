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
// start and completion, the throughput denominator (aiperf records_manager.py:1437-1442,
// metric_dicts.py:217-238). A replay that runs a warm-up or a grace period sets it on the
// aggregate Metrics before EmitOutput; a completion after EndUs is outside the window. Zero
// EndUs leaves the end derived (the last measured completion).
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

	// Σ OSL / measured window (aiperf output_token_throughput_metrics.py:40-42).
	OutputThroughputTokS float64 `json:"output_throughput_tok_s"`
	MeasuredWindowS      float64 `json:"measured_window_s"`

	// Counts in the window. The measured completions have their own key: completed_requests
	// stays the run-wide count INV-1 conserves in the file as on stdout.
	MeasuredCompletedRequests int `json:"measured_completed_requests"`
	CancelledRequests         int `json:"cancelled_requests"`
	ContextOverflowRequests   int `json:"context_overflow_requests"`
	FailedRequests            int `json:"failed_requests"`

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
}

// Measured returns the measured-window aggregates EmitOutput writes into the --metrics-path
// file, for a caller that reads them without writing the file.
func (m *Metrics) Measured() MeasuredMetrics {
	out, _ := m.buildMeasured()
	return out
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
	inWindow := func(completionUs int64) bool {
		return m.MeasuredWindow == nil || m.MeasuredWindow.EndUs <= 0 || completionUs <= m.MeasuredWindow.EndUs
	}

	var ttfts, e2es, itls, gateway []float64
	var perUserSum float64
	var outputTokens int64
	for _, id := range sortedRequestIDs(m.Requests) {
		rm := m.Requests[id]
		if !rm.Phase.Measured() {
			continue
		}
		e2e := m.RequestE2Es[id]
		completionUs := int64(m.RequestCompletionTimes[id])
		if e2e <= 0 || !inWindow(completionUs) {
			out.CancelledRequests++ // timed out, or still in flight when the window closed
			continue
		}
		out.MeasuredCompletedRequests++
		ttft := m.RequestTTFTs[id]
		ttfts = append(ttfts, ttft)
		e2es = append(e2es, e2e)
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
			itl := (e2e - ttft) / float64(rm.OutputTokens-1)
			itls = append(itls, itl)
			// aiperf output_token_throughput_metrics.py:74-78: 1/ITL, skipping ITL == 0.
			if itl > 0 {
				perUserSum += 1e6 / itl
			}
		}
	}
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
	if len(itls) > 0 {
		out.OutputTokPerUserMean = perUserSum / float64(len(itls))
	}

	steps := measuredStepIntervals(m.AllITLs, m.UnmeasuredStepIntervals)
	out.StepIntervalMeanMs = CalculateMean(steps)
	out.StepIntervalP90Ms = CalculatePercentile(steps, 90)
	out.StepIntervalP99Ms = CalculatePercentile(steps, 99)

	out.GatewayQueueDelayMeanMs = CalculateMean(gateway)
	out.GatewayQueueDelayP90Ms = CalculatePercentile(gateway, 90)

	if endUs > startUs {
		out.MeasuredWindowS = float64(endUs-startUs) / 1e6
		out.OutputThroughputTokS = float64(outputTokens) / out.MeasuredWindowS
		out.InFlightRequestsMean, out.InFlightRequestsMax = inFlightOver(m.InFlightSamples, startUs, endUs)
	}

	if m.PrefixCacheLookupTokens > 0 {
		out.CPUTierHitRate = float64(m.CPUTierHitTokens) / float64(m.PrefixCacheLookupTokens)
	}
	out.KVCapacityTokensPerReplica = m.KVCapacityTokensPerReplica
	return out, lat
}

// applyMeasured writes the measured-window values into the keys MetricsOutput already
// carries, for the --metrics-path file.
func (o *MetricsOutput) applyMeasured(lat measuredLatencies) {
	o.TTFTMeanMs, o.TTFTP90Ms, o.TTFTP95Ms, o.TTFTP99Ms = lat.ttftMean, lat.ttftP90, lat.ttftP95, lat.ttftP99
	o.E2EMeanMs, o.E2EP90Ms, o.E2EP95Ms, o.E2EP99Ms = lat.e2eMean, lat.e2eP90, lat.e2eP95, lat.e2eP99
}

// reciprocalPerSecond is 1000 / ms, the rate of one event every ms milliseconds, or 0 for a
// non-positive interval.
func reciprocalPerSecond(ms float64) float64 {
	if ms <= 0 {
		return 0
	}
	return 1000 / ms
}

// measuredStepIntervals returns all minus the multiset unmeasured, sorted. Both are step
// intervals in ticks; unmeasured is a sub-multiset of all.
func measuredStepIntervals(all, unmeasured []int64) []int64 {
	out := slices.Clone(all)
	slices.Sort(out)
	if len(unmeasured) == 0 {
		return out
	}
	drop := slices.Clone(unmeasured)
	slices.Sort(drop)
	kept := out[:0]
	j := 0
	for _, v := range out {
		if j < len(drop) && drop[j] == v {
			j++
			continue
		}
		kept = append(kept, v)
	}
	return kept
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
