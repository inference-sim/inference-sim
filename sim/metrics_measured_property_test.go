package sim

import (
	"fmt"
	"math"
	"sort"
	"testing"

	"pgregory.net/rapid"
)

// genRow is one random request of a measured-window property test, in ticks.
type genRow struct {
	phase              Phase
	arrival, ttft, e2e int64 // e2e == 0: never completed
	osl                int
	gatewayMs          float64
	steps              []int64
	lookup             PrefixLookup
}

func drawRows(rt *rapid.T) []genRow {
	n := rapid.IntRange(1, 40).Draw(rt, "n")
	rows := make([]genRow, n)
	for i := range rows {
		r := genRow{
			phase:     rapid.SampledFrom([]Phase{PhaseUnset, PhaseWarmup, PhaseProfile, PhaseGrace}).Draw(rt, "phase"),
			arrival:   rapid.Int64Range(0, 10_000_000).Draw(rt, "arrival"),
			ttft:      rapid.Int64Range(1, 1_000_000).Draw(rt, "ttft"),
			osl:       rapid.IntRange(0, 60).Draw(rt, "osl"),
			gatewayMs: float64(rapid.IntRange(0, 50).Draw(rt, "gateway")),
		}
		q := rapid.Int64Range(0, 4096).Draw(rt, "query")
		hit := rapid.Int64Range(0, q).Draw(rt, "hit")
		r.lookup = PrefixLookup{QueryTokens: q, HitTokens: hit, CPUHitTokens: rapid.Int64Range(0, q-hit).Draw(rt, "cpuHit")}
		if rapid.IntRange(0, 4).Draw(rt, "completed") > 0 {
			r.e2e = r.ttft + rapid.Int64Range(0, 10_000_000).Draw(rt, "decode")
			for k := 0; k < min(max(r.osl-1, 0), 5); k++ {
				r.steps = append(r.steps, rapid.Int64Range(1, 100_000).Draw(rt, "step"))
			}
		}
		rows[i] = r
	}
	return rows
}

// metricsOf builds the aggregate a cluster would hand EmitOutput for rows.
func metricsOf(rows []genRow) *Metrics {
	m := NewMetrics()
	for i, r := range rows {
		id := fmt.Sprintf("req_%d", i)
		m.Requests[id] = RequestMetrics{ID: id, ArrivedAt: float64(r.arrival) / 1e6, Phase: r.phase,
			OutputTokens: r.osl, GatewayQueueDelay: r.gatewayMs}
		m.RequestTTFTs[id] = float64(r.ttft)
		m.recordPrefixLookup(&Request{ID: id, Phase: r.phase}, r.lookup)
		if r.e2e > 0 {
			m.RequestE2Es[id] = float64(r.e2e)
			m.RequestCompletionTimes[id] = float64(r.arrival + r.e2e)
			m.CompletedRequests++
			m.AllITLs = append(m.AllITLs, r.steps...)
			m.StepOwners = append(m.StepOwners, StepOwner{CompletionUs: r.arrival + r.e2e, Steps: len(r.steps), Phase: r.phase})
		}
	}
	return m
}

func closeTo(a, b float64) bool {
	return math.Abs(a-b) <= 1e-9*math.Max(1, math.Max(math.Abs(a), math.Abs(b)))
}

// INV-M1 (accounting over rows), INV-M2 (interactivity and throughput are exactly their
// definitions), and the percentile laws, over random request sets.
func TestBuildMeasured_InvariantsHoldOverRandomRequestSets(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		rows := drawRows(rt)
		out, lat := metricsOf(rows).buildMeasured()

		measured, tokens := 0, 0
		for _, r := range rows {
			if r.phase.Measured() {
				measured++
				if r.e2e > 0 {
					tokens += r.osl
				}
			}
		}
		if out.MeasuredCompletedRequests+out.CancelledRequests != measured {
			rt.Fatalf("INV-M1: completed %d + cancelled %d != %d measured requests",
				out.MeasuredCompletedRequests, out.CancelledRequests, measured)
		}
		if out.InteractivityP50 != reciprocalPerSecond(out.RequestITLP50Ms) ||
			out.InteractivityP90 != reciprocalPerSecond(out.RequestITLP90Ms) {
			rt.Fatalf("INV-M2: interactivity %v/%v is not 1000/request_itl %v/%v",
				out.InteractivityP50, out.InteractivityP90, out.RequestITLP50Ms, out.RequestITLP90Ms)
		}
		if out.MeasuredWindowS > 0 && !closeTo(out.OutputThroughputTokS*out.MeasuredWindowS, float64(tokens)) {
			rt.Fatalf("INV-M2: throughput %v x window %v != %d measured output tokens",
				out.OutputThroughputTokS, out.MeasuredWindowS, tokens)
		}
		for _, chain := range [][]float64{
			{out.TTFTP50Ms, lat.ttftP90, lat.ttftP95, lat.ttftP99},
			{out.E2EP50Ms, lat.e2eP90, lat.e2eP95, lat.e2eP99},
			{out.RequestITLP50Ms, out.RequestITLP90Ms, out.RequestITLP99Ms},
			{out.StepIntervalP90Ms, out.StepIntervalP99Ms},
		} {
			for i := 1; i < len(chain); i++ {
				if chain[i] < chain[i-1] {
					rt.Fatalf("percentiles not monotone in q: %v", chain)
				}
			}
		}
	})
}

// Metamorphic: scaling every time (arrivals, TTFT, latency) by s scales TTFT, ITL and E2E
// percentiles and the window by s, and interactivity, per-user throughput and output
// throughput by 1/s.
func TestBuildMeasured_ScalingTimeScalesLatenciesAndInvertsRates(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		rows := drawRows(rt)
		s := rapid.Int64Range(2, 50).Draw(rt, "s")
		scaled := make([]genRow, len(rows))
		for i, r := range rows {
			r.arrival, r.ttft, r.e2e = r.arrival*s, r.ttft*s, r.e2e*s
			scaled[i] = r
		}
		a, la := metricsOf(rows).buildMeasured()
		b, lb := metricsOf(scaled).buildMeasured()
		fs := float64(s)
		for name, pair := range map[string][2]float64{
			"ttft_p50": {a.TTFTP50Ms * fs, b.TTFTP50Ms}, "ttft_p99": {la.ttftP99 * fs, lb.ttftP99},
			"e2e_mean": {la.e2eMean * fs, lb.e2eMean}, "e2e_p90": {la.e2eP90 * fs, lb.e2eP90},
			"itl_mean": {a.RequestITLMeanMs * fs, b.RequestITLMeanMs}, "itl_p90": {a.RequestITLP90Ms * fs, b.RequestITLP90Ms},
			"window":            {a.MeasuredWindowS * fs, b.MeasuredWindowS},
			"interactivity_p90": {a.InteractivityP90 / fs, b.InteractivityP90},
			"per_user_mean":     {a.OutputTokPerUserMean / fs, b.OutputTokPerUserMean},
			"throughput":        {a.OutputThroughputTokS / fs, b.OutputThroughputTokS},
		} {
			if !closeTo(pair[0], pair[1]) {
				rt.Fatalf("%s: scaled %v, want %v (s=%d)", name, pair[1], pair[0], s)
			}
		}
	})
}

// Metamorphic: duplicating every request leaves every mean, every minimum and maximum (the
// p0/p100 of a numpy-linear percentile) and the window unchanged, doubles the counts and the
// output throughput. Interior percentiles are deliberately not in this law: numpy's linear
// rank q/100*(n-1) moves with n, so [0, 1] has p90 0.9 and [0, 0, 1, 1] has p90 1.0.
func TestBuildMeasured_DuplicatingEveryRequestKeepsMeansAndExtremes(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		rows := drawRows(rt)
		a, la := metricsOf(rows).buildMeasured()
		b, lb := metricsOf(append(append([]genRow{}, rows...), rows...)).buildMeasured()
		for name, pair := range map[string][2]float64{
			"ttft_mean": {la.ttftMean, lb.ttftMean}, "e2e_mean": {la.e2eMean, lb.e2eMean},
			"itl_mean": {a.RequestITLMeanMs, b.RequestITLMeanMs}, "per_user_mean": {a.OutputTokPerUserMean, b.OutputTokPerUserMean},
			"step_mean": {a.StepIntervalMeanMs, b.StepIntervalMeanMs}, "gateway_mean": {a.GatewayQueueDelayMeanMs, b.GatewayQueueDelayMeanMs},
			"window": {a.MeasuredWindowS, b.MeasuredWindowS}, "throughput x2": {2 * a.OutputThroughputTokS, b.OutputThroughputTokS},
		} {
			if !closeTo(pair[0], pair[1]) {
				rt.Fatalf("%s: duplicated %v, want %v", name, pair[1], pair[0])
			}
		}
		if b.MeasuredCompletedRequests != 2*a.MeasuredCompletedRequests || b.CancelledRequests != 2*a.CancelledRequests {
			rt.Fatalf("counts did not double: %d/%d -> %d/%d", a.MeasuredCompletedRequests, a.CancelledRequests,
				b.MeasuredCompletedRequests, b.CancelledRequests)
		}
		ma, mb := metricsOf(rows), metricsOf(append(append([]genRow{}, rows...), rows...))
		for _, q := range []float64{0, 100} {
			if x, y := extremeITL(ma, q), extremeITL(mb, q); !closeTo(x, y) {
				rt.Fatalf("request ITL p%v moved from %v to %v under duplication", q, x, y)
			}
		}
	})
}

// extremeITL is the p0 or p100 of the measured request ITLs, through the same percentile
// helper the aggregates use.
func extremeITL(m *Metrics, q float64) float64 {
	var itls []float64
	for id, rm := range m.Requests {
		if rm.Phase.Measured() && m.RequestE2Es[id] > 0 && rm.OutputTokens >= 2 {
			itls = append(itls, (m.RequestE2Es[id]-m.RequestTTFTs[id])/float64(rm.OutputTokens-1))
		}
	}
	sort.Float64s(itls)
	return CalculatePercentile(itls, q)
}

// Metamorphic: adding warm-up requests -- with their own steps and prefix-cache lookups --
// changes no measured aggregate, cache_hit_rate and cpu_tier_hit_rate included. This is the
// windowing law: only the phase decides what is measured.
func TestBuildMeasured_AddingWarmupRequestsChangesNothingMeasured(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		rows := drawRows(rt)
		extra := drawRows(rt)
		for i := range extra {
			extra[i].phase = PhaseWarmup
		}
		a, la := metricsOf(rows).buildMeasured()
		b, lb := metricsOf(append(append([]genRow{}, rows...), extra...)).buildMeasured()
		if extraQueries(extra) > 0 && metricsOf(rows).PrefixCacheQueryTokens ==
			metricsOf(append(append([]genRow{}, rows...), extra...)).PrefixCacheQueryTokens {
			rt.Fatal("the warm-up lookups did not reach the whole-run counts; the law would be vacuous")
		}
		if a != b || la != lb {
			rt.Fatalf("warm-up requests moved the measured aggregates:\n%+v %+v\n%+v %+v", a, la, b, lb)
		}
	})
}

func extraQueries(rows []genRow) int64 {
	var q int64
	for _, r := range rows {
		q += r.lookup.QueryTokens
	}
	return q
}
