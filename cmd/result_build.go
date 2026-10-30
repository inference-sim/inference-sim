package cmd

// Building the Result document from a finished run (#1871). The mapping from today's
// MetricsOutput onto the Result's sections is the table in docs/guide/results.md
// ("Where each MetricsOutput field went"); TestResultFieldMap_EveryMetricsOutputFieldHasAHome
// keeps that table complete.

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/measurement"
	"github.com/inference-sim/blis-schemas/spec/result"
	"github.com/inference-sim/blis-schemas/spec/scenario"
	schemaworkload "github.com/inference-sim/blis-schemas/spec/workload"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// resultCite is where every metric definition this document states is implemented.
const resultCite = "inference-sim sim/metrics_measured.go"

// The window bounds the Result states. With no explicit window the measured window is
// derived from the measured requests: the first arrival to the last completion.
const (
	windowStartFirstArrival = "first_arrival_after_warmup"
	windowStopLastComplete  = "last_completion"
	windowStartExplicit     = "profiling_start"
	windowStopExplicit      = "profiling_end"
	// A run with no measured completion has no measured window; its rates are zero over the
	// whole simulated run instead.
	windowStartRun = "simulation_start"
	windowStopRun  = "simulation_end"
)

// resultInputs are the facts about a run that are not in its metrics: what it was given.
type resultInputs struct {
	// Scenario and Deployment are the documents the run read, as decoded.
	Scenario   *scenario.Scenario
	Deployment *deployment.Deployment
	// Policy is the --policy-config bundle as a generic map, nil without one.
	Policy map[string]any
	// Run are the run controls.
	Run result.Run
	// Overrides are the settings simulated differently from the scenario.
	Overrides []sim.ScenarioOverride
	// GPUs is the number of GPUs the run's instances occupy, for output_tok_s_per_gpu.
	GPUs int
	// GoodputTargets are the resolved SLO targets, nil without.
	GoodputTargets map[string]workload.SLODimTargets
	// Saturation is the final label per detector, nil when no detector ran.
	Saturation map[string]string
	// Artifacts are the bulk outputs the run wrote, by role.
	Artifacts map[string]result.Artifact
	// Revisions are the versions the run read.
	Revisions result.Revisions
	// Name is the run's identity; derived when empty.
	Name string
}

// resultMetrics are the run's outputs the document is filled from.
type resultMetrics struct {
	Aggregated *sim.Metrics
	// Ledger is INV-1's ledger, from the cluster's own counters.
	Ledger result.Requests
	// PD is the disaggregation block, nil for a colocated run.
	PD *cluster.PDMetrics
	// Engine, Router and Kernel are each component's report, already keyed as the
	// document carries them.
	Engine map[string]any
	Router map[string]any
	Kernel map[string]any
	// InjectedByClass is the cluster's per-SLO-class arrival count, goodput's denominator.
	InjectedByClass map[string]int64
	// ClockUs is the cluster's clock when the run ended, and HorizonUs the horizon it ran to.
	ClockUs, HorizonUs int64
}

// buildResult assembles the Result document.
func buildResult(in resultInputs, out resultMetrics) *result.Result {
	m := out.Aggregated
	sum := m.Summary()
	r := &result.Result{
		Kind:          result.Kind,
		SchemaVersion: result.SchemaVersion,
		Name:          in.Name,
		Revisions:     in.Revisions,
		Requests:      out.Ledger,
		Engine:        out.Engine,
		Router:        out.Router,
		Kernel:        out.Kernel,
		Artifacts:     in.Artifacts,
	}
	if in.Scenario != nil {
		r.Input.Scenario = *in.Scenario
	}
	if in.Deployment != nil {
		r.Input.Deployment = *in.Deployment
	}
	r.Input.Policy = in.Policy
	r.Input.Run = in.Run
	for _, o := range in.Overrides {
		r.Overrides = append(r.Overrides, result.Override{Setting: o.Setting, Input: o.Scenario, Run: o.Run, Source: o.Source})
	}
	if r.Name == "" {
		r.Name = resultName(r)
	}

	r.Window = windowOf(m, sum, out.ClockUs, out.HorizonUs)
	r.MetricDefs, r.Metrics = standardMetrics(sum, in.GPUs, r.Window.DurationS)
	r.Breakdowns = breakdownsOf(m, sum, r.MetricDefs, out.PD)
	r.Goodput = goodputOf(m, in.GoodputTargets, out.InjectedByClass)
	if len(in.Saturation) > 0 {
		r.Saturation = map[string]any{"final": stringMapAny(in.Saturation)}
	}
	return r
}

// windowOf states the measured window the metrics are over.
func windowOf(m *sim.Metrics, sum sim.MeasuredSummary, clockUs, horizonUs int64) result.Window {
	if sum.MeasuredWindowS <= 0 {
		// No measured request completed, so no window bounds one: the zero rates are stated
		// over the simulated run, which is what such a run measured. An idle engine's own clock
		// never advances, so the span is the cluster's clock when it ran longer; a run that saw
		// nothing at all before its horizon spanned the horizon.
		end := max(m.SimEndedTime, clockUs)
		if end <= 0 && horizonUs > 0 && horizonUs < math.MaxInt64 {
			end = horizonUs
		}
		return result.Window{Start: windowStartRun, Stop: windowStopRun,
			DurationS: float64(end) / 1e6, Requests: sum.MeasuredCompletedRequests}
	}
	w := result.Window{Start: windowStartFirstArrival, Stop: windowStopLastComplete,
		DurationS: sum.MeasuredWindowS, Requests: sum.MeasuredCompletedRequests}
	if m.MeasuredWindow != nil {
		w.Start = windowStartExplicit
		if m.MeasuredWindow.EndUs > 0 {
			w.Stop = windowStopExplicit
		}
	}
	return w
}

// metricDef is a per-request statistic's definition.
func metricDef(f measurement.Formula, st measurement.Statistic, cite string) measurement.MetricDef {
	d := measurement.MetricDef{Formula: f, Statistic: st, Population: measurement.PopulationPerRequest,
		Clock: measurement.ClockSimulated, Unit: f.Unit(), Cite: resultCite + " " + cite}
	if st.Percentile() {
		d.PercentileMethod = measurement.PercentileNumpyLinear
	}
	return d
}

// throughputDef is a rate over the window.
func throughputDef(f measurement.Formula, pop measurement.Population, den measurement.Denominator, cite string) measurement.MetricDef {
	return measurement.MetricDef{Formula: f, Population: pop, Denominator: den,
		Clock: measurement.ClockSimulated, Unit: f.Unit(), Cite: resultCite + " " + cite}
}

// standardMetrics are the standard-named metrics over the measured window. A latency
// family is reported only when it has a sample, so a run that completed nothing states
// no latency rather than a zero that reads as instant. The throughputs need a window.
func standardMetrics(sum sim.MeasuredSummary, gpus int, windowS float64) (map[string]measurement.MetricDef, map[string]float64) {
	defs := map[string]measurement.MetricDef{}
	vals := map[string]float64{}
	put := func(name string, d measurement.MetricDef, v float64) {
		defs[name], vals[name] = d, v
	}
	if windowS > 0 {
		put("output_tok_s", throughputDef(measurement.FormulaOutputThroughput, measurement.PopulationPooledTokens,
			measurement.DenominatorProfilingWindow, "OutputThroughputTokS"), sum.OutputThroughputTokS)
		if gpus > 0 {
			put("output_tok_s_per_gpu", throughputDef(measurement.FormulaOutputThroughput, measurement.PopulationPooledTokens,
				measurement.DenominatorPerGPU, "OutputThroughputTokS over the GPUs the run's instances occupy"),
				sum.OutputThroughputTokS/float64(gpus))
		}
		put("request_throughput", throughputDef(measurement.FormulaRequestThroughput, measurement.PopulationPerRequest,
			measurement.DenominatorProfilingWindow, "MeasuredCompletedRequests over MeasuredWindowS"),
			float64(sum.MeasuredCompletedRequests)/windowS)
	}
	if sum.MeasuredCompletedRequests > 0 {
		for _, l := range []struct {
			name string
			st   measurement.Statistic
			v    float64
			cite string
		}{
			{"ttft_mean_ms", measurement.StatMean, sum.TTFTMeanMs, "TTFT mean"},
			{"ttft_p50_ms", "p50", sum.TTFTP50Ms, "TTFTP50Ms"},
			{"ttft_p90_ms", "p90", sum.TTFTP90Ms, "TTFT p90"},
			{"ttft_p95_ms", "p95", sum.TTFTP95Ms, "TTFT p95"},
			{"ttft_p99_ms", "p99", sum.TTFTP99Ms, "TTFT p99"},
		} {
			put(l.name, metricDef(measurement.FormulaTTFT, l.st, l.cite), l.v)
		}
		for _, l := range []struct {
			name string
			st   measurement.Statistic
			v    float64
			cite string
		}{
			{"e2e_mean_ms", measurement.StatMean, sum.E2EMeanMs, "request latency mean"},
			{"e2e_p50_ms", "p50", sum.E2EP50Ms, "E2EP50Ms"},
			{"e2e_p90_ms", "p90", sum.E2EP90Ms, "request latency p90"},
			{"e2e_p95_ms", "p95", sum.E2EP95Ms, "request latency p95"},
			{"e2e_p99_ms", "p99", sum.E2EP99Ms, "request latency p99"},
		} {
			put(l.name, metricDef(measurement.FormulaE2EL, l.st, l.cite), l.v)
		}
	}
	if sum.RequestITLCount > 0 {
		for _, l := range []struct {
			name string
			st   measurement.Statistic
			v    float64
			cite string
		}{
			{"itl_mean_ms", measurement.StatMean, sum.RequestITLMeanMs, "RequestITLMeanMs"},
			{"itl_p50_ms", "p50", sum.RequestITLP50Ms, "RequestITLP50Ms"},
			{"itl_p90_ms", "p90", sum.RequestITLP90Ms, "RequestITLP90Ms"},
			{"itl_p99_ms", "p99", sum.RequestITLP99Ms, "RequestITLP99Ms"},
		} {
			put(l.name, metricDef(measurement.FormulaTPOTExclFirst, l.st, l.cite), l.v)
		}
		if sum.RequestITLP50Ms > 0 {
			put("interactivity_p50", metricDef(measurement.FormulaInteractivity, "p50", "InteractivityP50"), sum.InteractivityP50)
		}
		if sum.RequestITLP90Ms > 0 {
			put("interactivity_p90", metricDef(measurement.FormulaInteractivity, "p90", "InteractivityP90"), sum.InteractivityP90)
		}
	}
	if sum.PerUserCount > 0 {
		put("interactivity_mean", metricDef(measurement.FormulaUserTPS, measurement.StatMean, "OutputTokPerUserMean"), sum.OutputTokPerUserMean)
	}
	return defs, vals
}

// splitMetrics are the metrics a split (by model, tenant or adapter) restates over the whole
// run's window, with the whole's definitions, and only those the whole states. The per-GPU
// throughput is not split: a key's requests share every GPU.
func splitMetrics(sum sim.MeasuredSummary, defined map[string]measurement.MetricDef) map[string]float64 {
	_, vals := standardMetrics(sum, 0, sum.MeasuredWindowS)
	out := map[string]float64{}
	for k, v := range vals {
		if _, ok := defined[k]; ok {
			out[k] = v
		}
	}
	return out
}

// breakdownsOf splits the run along the axes it exercised: per model, tenant and adapter
// (each the whole's metrics over the requests of that key), P/D, and sessions.
func breakdownsOf(m *sim.Metrics, sum sim.MeasuredSummary, defs map[string]measurement.MetricDef, pd *cluster.PDMetrics) *result.Breakdowns {
	b := &result.Breakdowns{}
	split := func(key func(sim.RequestMetrics) string) map[string]map[string]float64 {
		keys := map[string]bool{}
		for _, rm := range m.Requests {
			if k := key(rm); k != "" {
				keys[k] = true
			}
		}
		if len(keys) == 0 {
			return nil
		}
		out := map[string]map[string]float64{}
		for k := range keys {
			s := m.SummaryOver(func(rm sim.RequestMetrics) bool { return key(rm) == k }, sum.WindowStartUs, sum.WindowEndUs)
			if v := splitMetrics(s, defs); len(v) > 0 {
				out[k] = v
			}
		}
		if len(out) == 0 {
			return nil
		}
		return out
	}
	b.PerModel = split(func(rm sim.RequestMetrics) string { return rm.Model })
	b.PerTenant = split(func(rm sim.RequestMetrics) string { return rm.TenantID })
	b.PerAdapter = split(func(rm sim.RequestMetrics) string { return rm.Adapter })
	b.PD = pdBreakdown(pd)
	b.Sessions = sessionBreakdown(cluster.ComputeSessionMetrics(m))
	if b.PerModel == nil && b.PerTenant == nil && b.PerAdapter == nil && b.PD == nil && b.Sessions == nil {
		return nil
	}
	return b
}

// pdBreakdown is the prefill/decode split, in ms (the cluster reports microseconds).
func pdBreakdown(pd *cluster.PDMetrics) map[string]float64 {
	if pd == nil {
		return nil
	}
	out := map[string]float64{
		"disaggregated_requests":         float64(pd.DisaggregatedCount),
		"dropped_at_decode_kv":           float64(pd.DroppedAtDecodeKV),
		"prefill_sub_request_throughput": pd.PrefillThroughput,
		"decode_sub_request_throughput":  pd.DecodeThroughput,
		"kv_transfers":                   float64(pd.Transfers),
		"prefill_kv_hold_peak_tokens":    float64(pd.PrefillKVHoldPeakTokens),
	}
	// An idle pool makes the imbalance unbounded; the document carries finite values only.
	if pd.LoadImbalanceRatio != math.MaxFloat64 {
		out["load_imbalance_ratio"] = pd.LoadImbalanceRatio
	}
	dist := func(prefix string, d cluster.Distribution) {
		if d.Count == 0 {
			return
		}
		out[prefix+"_mean_ms"] = d.Mean / 1e3
		out[prefix+"_p50_ms"] = d.P50 / 1e3
		out[prefix+"_p95_ms"] = d.P95 / 1e3
		out[prefix+"_p99_ms"] = d.P99 / 1e3
	}
	dist("parent_ttft", pd.ParentTTFT)
	dist("kv_transfer", pd.TransferDuration)
	if pd.TransferDuration.Count > 0 {
		out["kv_transfer_p90_ms"] = pd.TransferP90 / 1e3
	}
	if pd.LinkPeakConcurrent != nil && pd.TransferQueueDelay.Count > 0 {
		dist("kv_transfer_queue_delay", pd.TransferQueueDelay)
		out["kv_transfer_queue_delay_p90_ms"] = pd.TransferQueueDelayP90 / 1e3
		for _, link := range sortedMapKeys(pd.LinkPeakConcurrent) {
			putUnique(out, "link_peak_concurrent_transfers_"+slugPart(link), float64(pd.LinkPeakConcurrent[link]))
		}
	}
	for _, pool := range sortedMapKeys(pd.CPUTierHitRate) {
		putUnique(out, "cpu_tier_hit_rate_"+slugPart(pool), pd.CPUTierHitRate[pool])
	}
	return out
}

// putUnique sets out[key], panicking when two names reduced to the same slug: the second would
// silently replace the first (R1).
func putUnique(out map[string]float64, key string, v float64) {
	if _, ok := out[key]; ok {
		panic(fmt.Sprintf("result breakdown: two names reduce to the key %q", key))
	}
	out[key] = v
}

// sessionBreakdown is the multi-turn session split.
func sessionBreakdown(sm *cluster.SessionMetrics) map[string]float64 {
	if sm == nil {
		return nil
	}
	out := map[string]float64{"sessions": float64(sm.SessionCount)}
	dist := func(prefix string, d cluster.Distribution) {
		if d.Count == 0 {
			return
		}
		out[prefix+"_mean_ms"] = d.Mean
		out[prefix+"_p50_ms"] = d.P50
		out[prefix+"_p95_ms"] = d.P95
		out[prefix+"_p99_ms"] = d.P99
		out[prefix+"_count"] = float64(d.Count)
	}
	dist("ttft_cold", sm.TTFTCold)
	dist("ttft_warm", sm.TTFTWarm)
	dist("session_duration", sm.SessionDuration)
	return out
}

// goodputOf is the SLO outcome over the whole run, as emitGoodput computes it (goodput_rps
// over the run's simulated length), nil without targets.
func goodputOf(m *sim.Metrics, targets map[string]workload.SLODimTargets, injectedByClass map[string]int64) *result.Goodput {
	if len(targets) == 0 {
		return nil
	}
	var mo sim.MetricsOutput
	emitGoodput(&mo, m, injectedByClass, float64(m.SimEndedTime)/1e6, targets)
	g := &result.Goodput{Targets: map[string]schemaworkload.SLODimTargets{}, GoodputRPS: mo.GoodputRPS,
		SLOAttainment: mo.SLOAttainment, PerClass: map[string]result.ClassGoodput{}}
	for cls, t := range targets {
		g.Targets[cls] = schemaworkload.SLODimTargets{TTFTMs: t.TTFTMs, ITLMs: t.ITLMs, E2EMs: t.E2EMs}
	}
	_, per := cluster.SLOAttainmentMultiDim(cluster.BuildLatencyResults(m), injectedByClass, targets)
	for cls, a := range per {
		g.PerClass[cls] = result.ClassGoodput{Completed: completedInClass(m, cls), Met: a.Good}
	}
	return g
}

// completedInClass counts completed requests of an SLO class ("" folds into "default", as
// goodput's lookup does).
func completedInClass(m *sim.Metrics, cls string) int {
	n := 0
	for id := range m.RequestE2Es {
		rm, ok := m.Requests[id]
		if !ok {
			continue
		}
		c := rm.SLOClass
		if c == "" {
			c = "default"
		}
		if c == cls {
			n++
		}
	}
	return n
}

// resultName derives the run's identity from its scenario, deployment, load and seed.
func resultName(r *result.Result) string {
	base := r.Input.Deployment.Name
	if base == "" {
		base = r.Input.Scenario.Name
	}
	if base == "" {
		base = "blis-run"
	}
	load := ""
	switch l := r.Input.Run.Load; {
	case l.Concurrency > 0:
		load = fmt.Sprintf("c%d", l.Concurrency)
	case l.RequestRate > 0:
		load = "r" + strings.ReplaceAll(fmt.Sprintf("%g", l.RequestRate), ".", "p")
	}
	name := fmt.Sprintf("%s.%s-s%d", base, load, r.Input.Run.Seed)
	// Overrides change what ran, so two runs that differ only in them get different names: a
	// short digest of the override rows (a --num-instances sweep no longer collides).
	if len(r.Overrides) > 0 {
		var b strings.Builder
		for _, o := range r.Overrides {
			fmt.Fprintf(&b, "%s\x1f%s\x1f%s\x1f%s\x1e", o.Setting, o.Input, o.Run, o.Source)
		}
		name += "-o" + bytesDigest([]byte(b.String()))[:8]
	}
	return slugPart(name)
}

// slugPart lowercases s and replaces every character a blis-schemas slug does not allow.
func slugPart(s string) string {
	var b strings.Builder
	for _, c := range strings.ToLower(s) {
		switch {
		case c >= 'a' && c <= 'z', c >= '0' && c <= '9', c == '.', c == '_', c == '-':
			b.WriteRune(c)
		default:
			b.WriteByte('-')
		}
	}
	out := strings.Trim(b.String(), "-._")
	if out == "" {
		return "x"
	}
	return out
}

// stringMapAny copies a string map into a generic one, in sorted key order (the encoder
// sorts anyway; the copy is for the type).
func stringMapAny(in map[string]string) map[string]any {
	out := make(map[string]any, len(in))
	for _, k := range sortedMapKeys(in) {
		out[k] = in[k]
	}
	return out
}

// sortedMapKeys is m's keys, sorted.
func sortedMapKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
