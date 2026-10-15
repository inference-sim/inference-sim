package sim

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// measuredFixture is a hand-computed request set (ticks are microseconds). Measured:
//
//	r1: arrival 0.0 s, TTFT 100 ms, E2E 1100 ms, OSL 11 -> ITL (1100-100)/10 = 100 ms
//	r2: arrival 0.5 s, TTFT 200 ms, E2E 1200 ms, OSL 21 -> ITL (1200-200)/20 =  50 ms
//	r3: arrival 1.0 s, TTFT 300 ms, E2E  300 ms, OSL  1 -> no ITL (OSL < 2)
//	r4: arrival 2.0 s, TTFT 400 ms, E2E 2400 ms, OSL  5 -> ITL (2400-400)/4  = 500 ms
//	c1: in flight at the end (no E2E)                   -> cancelled
//
// and one warm-up request, w1, that would move every figure if it counted.
func measuredFixture() *Metrics {
	m := NewMetrics()
	add := func(id string, phase Phase, arrival, ttft, e2e int64, osl int, gatewayMs float64) {
		m.Requests[id] = RequestMetrics{ID: id, ArrivedAt: float64(arrival) / 1e6, Phase: phase,
			OutputTokens: osl, GatewayQueueDelay: gatewayMs}
		m.RequestTTFTs[id] = float64(ttft)
		if e2e > 0 {
			m.RequestE2Es[id] = float64(e2e)
			m.RequestCompletionTimes[id] = float64(arrival + e2e)
			m.CompletedRequests++
			m.TotalOutputTokens += osl
		}
	}
	add("r1", PhaseProfile, 0, 100_000, 1_100_000, 11, 2)
	add("r2", PhaseProfile, 500_000, 200_000, 1_200_000, 21, 4)
	add("r3", PhaseGrace, 1_000_000, 300_000, 300_000, 1, 0)
	add("r4", PhaseUnset, 2_000_000, 400_000, 2_400_000, 5, 0)
	add("c1", PhaseProfile, 3_000_000, 50_000, 0, 0, 0)
	add("w1", PhaseWarmup, 0, 10, 5_000_000, 100, 9)
	m.AllITLs = []int64{10_000, 20_000, 30_000, 20_000, 40_000}
	m.UnmeasuredStepIntervals = []int64{40_000, 20_000} // w1's steps
	m.MeasuredContextOverflows, m.MeasuredFailures = 2, 1
	m.PrefixCacheLookupTokens, m.PrefixCacheHitTokens, m.CPUTierHitTokens = 1000, 600, 250
	m.KVCapacityTokensPerReplica = 12345
	m.InFlightSamples = []InFlightSample{{0, 1}, {1_000_000, 3}, {2_000_000, 2}, {5_000_000, 0}}
	return m
}

func near(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9*math.Max(1, math.Abs(want)) {
		t.Errorf("%s = %.12g, want %.12g", name, got, want)
	}
}

// Every aiperf definition, pinned against the hand-computed fixture: TTFT, request latency,
// per-request ITL over OSL >= 2 (inter_token_latency_metric.py:48-55), interactivity as the
// reciprocal of the ITL percentile, per-user throughput as the mean of reciprocals
// (output_token_throughput_metrics.py:64-78), and output throughput over the window from the
// first measured send to the last measured completion (benchmark_duration_metric.py:38-46).
// The warm-up request is excluded from all of them.
func TestBuildMeasured_PinsTheAiperfDefinitions(t *testing.T) {
	out, lat := measuredFixture().buildMeasured()

	near(t, "ttft mean", lat.ttftMean, 250)
	near(t, "ttft_p50", out.TTFTP50Ms, 250)
	near(t, "ttft p90", lat.ttftP90, 370) // rank 2.7 over [100 200 300 400]
	near(t, "e2e mean", lat.e2eMean, 1250)
	near(t, "e2e_p50", out.E2EP50Ms, 1150) // [300 1100 1200 2400]
	near(t, "e2e p90", lat.e2eP90, 2040)

	near(t, "request_itl_mean", out.RequestITLMeanMs, (100+50+500)/3.0)
	near(t, "request_itl_p50", out.RequestITLP50Ms, 100) // [50 100 500]
	near(t, "request_itl_p90", out.RequestITLP90Ms, 420) // rank 1.8
	near(t, "request_itl_p99", out.RequestITLP99Ms, 492) // rank 1.98
	near(t, "interactivity_p50", out.InteractivityP50, 10)
	near(t, "interactivity_p90", out.InteractivityP90, 1000/420.0)
	near(t, "output_tok_per_user_mean", out.OutputTokPerUserMean, (10+20+2)/3.0)

	near(t, "measured_window_s", out.MeasuredWindowS, 4.4)
	near(t, "output_throughput_tok_s", out.OutputThroughputTokS, (11+21+1+5)/4.4)

	if out.MeasuredCompletedRequests != 4 || out.CancelledRequests != 1 ||
		out.ContextOverflowRequests != 2 || out.FailedRequests != 1 {
		t.Errorf("counts: completed %d cancelled %d context_overflow %d failed %d, want 4 1 2 1",
			out.MeasuredCompletedRequests, out.CancelledRequests, out.ContextOverflowRequests, out.FailedRequests)
	}

	near(t, "step_interval_mean", out.StepIntervalMeanMs, 20) // [10 20 30] ms once w1's steps go
	near(t, "step_interval_p90", out.StepIntervalP90Ms, 28)
	near(t, "gateway_queue_delay_mean", out.GatewayQueueDelayMeanMs, 1.5) // [0 0 2 4]
	near(t, "gateway_queue_delay_p90", out.GatewayQueueDelayP90Ms, 3.4)
	near(t, "in_flight_requests_mean", out.InFlightRequestsMean, (1*1.0+3*1.0+2*2.4)/4.4)
	if out.InFlightRequestsMax != 3 {
		t.Errorf("in_flight_requests_max = %d, want 3", out.InFlightRequestsMax)
	}
	near(t, "cpu_tier_hit_rate", out.CPUTierHitRate, 0.25)
	if out.KVCapacityTokensPerReplica != 12345 || out.PercentileMethod != "numpy_linear" {
		t.Errorf("kv capacity %d, percentile method %q", out.KVCapacityTokensPerReplica, out.PercentileMethod)
	}
}

// An explicit window (a replay's profiling phase) fixes the throughput denominator, and a
// completion after its end was cancelled in the measured run: it leaves every latency
// aggregate and joins cancelled_requests.
func TestBuildMeasured_AnExplicitWindowBoundsTheMeasuredSet(t *testing.T) {
	m := measuredFixture()
	m.MeasuredWindow = &MeasuredWindow{StartUs: 500_000, EndUs: 2_000_000}
	out, lat := m.buildMeasured()

	if out.MeasuredCompletedRequests != 3 || out.CancelledRequests != 2 {
		t.Errorf("completed %d cancelled %d, want r1 r2 r3 completed and c1 r4 cancelled",
			out.MeasuredCompletedRequests, out.CancelledRequests)
	}
	near(t, "measured_window_s", out.MeasuredWindowS, 1.5)
	near(t, "output_throughput_tok_s", out.OutputThroughputTokS, (11+21+1)/1.5)
	near(t, "e2e mean", lat.e2eMean, (1100+1200+300)/3.0)
	near(t, "request_itl_p50", out.RequestITLP50Ms, 75) // [50 100]
	near(t, "in_flight_requests_mean", out.InFlightRequestsMean, (1*0.5+3*1.0)/1.5)
	if out.InFlightRequestsMax != 3 {
		t.Errorf("in_flight_requests_max = %d, want 3", out.InFlightRequestsMax)
	}
}

// With no phase on any request and no window, the measured set is every completed request,
// so step_interval_* is the legacy itl_* series and e2e_* matches stdout's.
func TestBuildMeasured_WithNoPhaseEveryCompletedRequestIsMeasured(t *testing.T) {
	m := measuredFixture()
	delete(m.Requests, "w1")
	delete(m.RequestE2Es, "w1")
	delete(m.RequestTTFTs, "w1")
	m.CompletedRequests--
	m.AllITLs = []int64{10_000, 20_000, 30_000}
	m.UnmeasuredStepIntervals = nil
	legacy := m.BuildOutput("cluster")
	out, lat := m.buildMeasured()
	near(t, "step_interval_mean vs itl_mean", out.StepIntervalMeanMs, legacy.ITLMeanMs)
	near(t, "step_interval_p90 vs itl_p90", out.StepIntervalP90Ms, legacy.ITLP90Ms)
	near(t, "e2e mean vs legacy", lat.e2eMean, legacy.E2EMeanMs)
	near(t, "e2e p99 vs legacy", lat.e2eP99, legacy.E2EP99Ms)
}

// CalculatePercentile is numpy's "linear" method, the one aiperf uses
// (metrics/metric_dicts.py:78-82). The vector and the expected values were generated with
//
//	uv run --no-project --with numpy python -c 'import numpy as np;
//	  v = np.round(np.random.default_rng(1938).lognormal(3, 0.8, 23), 3);
//	  print([np.percentile(v, q, method="linear") for q in (1, 5, 25, 50, 90, 95, 99)])'
func TestCalculatePercentile_MatchesNumpyLinear(t *testing.T) {
	ms := []float64{31.257, 44.11, 21.93, 28.351, 12.507, 52.705, 12.429, 16.204, 13.701, 7.19,
		13.307, 24.972, 23.711, 42.995, 10.085, 10.526, 11.312, 2.788, 19.822, 20.178, 42.741,
		9.007, 11.114}
	ticks := make([]float64, len(ms))
	for i, v := range ms {
		ticks[i] = v * 1000
	}
	sort.Float64s(ticks)
	for _, c := range []struct{ q, want float64 }{
		{1, 3.75644}, {5, 7.371700000000001}, {25, 11.213000000000001}, {50, 16.204},
		{90, 42.944199999999995}, {95, 43.9985}, {99, 50.81410000000001},
	} {
		near(t, fmt.Sprintf("p%g", c.q), CalculatePercentile(ticks, c.q), c.want)
	}
	near(t, "mean", CalculateMean(ticks), 20.997478260869563)
}

// The measured aggregates reach the --metrics-path file only: stdout keeps the whole-run
// figures and none of the new keys (INV-6), while the file's ttft_*/e2e_* are the measured
// window's and completed_requests stays the run-wide count INV-1 conserves.
func TestEmitOutput_MeasuredAggregatesAreFileOnly(t *testing.T) {
	m := measuredFixture()
	output := m.BuildOutput("cluster")
	path := filepath.Join(t.TempDir(), "metrics.json")
	stdout := captureStdout(t, func() {
		if err := m.EmitOutput(output, path); err != nil {
			t.Fatal(err)
		}
	})
	for _, key := range []string{"request_itl_p90_ms", "interactivity_p90", "measured_window_s", "step_interval_p90_ms"} {
		if strings.Contains(stdout, key) {
			t.Errorf("stdout carries file-only key %q", key)
		}
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var file map[string]any
	if err := json.Unmarshal(data, &file); err != nil {
		t.Fatal(err)
	}
	near(t, "file ttft_mean_ms", file["ttft_mean_ms"].(float64), 250)
	if output.TTFTMeanMs == 250 {
		t.Fatal("the fixture's warm-up request must move the whole-run TTFT mean, or this test is vacuous")
	}
	near(t, "file request_itl_p90_ms", file["request_itl_p90_ms"].(float64), 420)
	if got := int(file["completed_requests"].(float64)); got != m.CompletedRequests {
		t.Errorf("file completed_requests = %d, want the run-wide %d", got, m.CompletedRequests)
	}
	if got := int(file["measured_completed_requests"].(float64)); got != 4 {
		t.Errorf("file measured_completed_requests = %d, want 4", got)
	}
}
