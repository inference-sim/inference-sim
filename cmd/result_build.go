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

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	schemaworkload "github.com/inference-sim/blis-substrate/contracts/spec/workload"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// resultInputs are the facts about a run that are not in its metrics: what it was given.
type resultInputs struct {
	// Scenario and Deployment are the documents the run read, as decoded.
	Scenario   *scenario.Scenario
	Deployment *deployment.Deployment
	// Options are the command-line options that set the run, sorted by name (SIM-10).
	Options []result.Option
	// Packaging is the cluster's packaging the kernel resolved (blis-substrate
	// scenario.Cluster.Packaging), recorded when a value came from the chip's catalog entry
	// rather than the scenario (SIM-10). Nil when the scenario states every value, which
	// input.scenario then records as stated.
	Packaging *scenario.Packaging
	// Policy is the --policy-config bundle as a generic map, nil without one.
	Policy map[string]any
	// Generator is the frame's generator: the workload binding, the offered load and the seed
	// (result.Generator; the frame's RNG is set by buildFrame).
	Generator result.Generator
	// NumRequests is the request cap the generator was given, 0 for none; HorizonUs the
	// horizon the run stopped at.
	NumRequests int
	HorizonUs   int64
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
	// Engine is the engine the run used (SIM-1); Revisions the versions it was built from.
	Engine    result.Engine
	Revisions result.Revisions
	// Data is the catalog and registry record; Selected the entries lookups returned.
	Data     result.Data
	Selected result.SelectedEntries
	// Name is the run's identity; derived when empty.
	Name string
}

// resultMetrics are the run's outputs the document is filled from.
type resultMetrics struct {
	Aggregated *sim.Metrics
	// Ledger is the fate count (DEV-9), from the cluster's fate ledger.
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
	// Rows are the per-request rows (DEV-9).
	Rows []result.RequestRow
}

// buildResult assembles the Result document. The frame is filled by buildFrame once the
// metrics are known, since frame.metrics lists them.
func buildResult(in resultInputs, out resultMetrics) *result.Result {
	m := out.Aggregated
	sum := m.Summary()
	r := &result.Result{
		Kind:            result.Kind,
		SchemaVersion:   result.SchemaVersion,
		Name:            in.Name,
		Engine:          in.Engine,
		Data:            in.Data,
		SelectedEntries: in.Selected,
		Revisions:       in.Revisions,
		Requests:        out.Ledger,
		RequestRows:     out.Rows,
		Reports:         result.Reports{Engine: out.Engine, Router: out.Router, Kernel: out.Kernel},
		Artifacts:       in.Artifacts,
	}
	if in.Scenario != nil {
		r.Input.Scenario = *in.Scenario
	}
	if in.Deployment != nil {
		r.Input.Deployment = *in.Deployment
	}
	r.Input.Packaging = in.Packaging
	r.Input.Policy = in.Policy
	r.Input.Options = in.Options
	r.Frame.Generator = in.Generator // buildFrame sets the seed and RNG from the same generator
	for _, o := range in.Overrides {
		r.Overrides = append(r.Overrides, result.Override{Setting: o.Setting, Input: o.Scenario, Run: o.Run, Source: o.Source})
	}
	if r.Name == "" {
		r.Name = resultName(r)
	}

	r.Window = windowOf(m, sum, out.ClockUs, out.HorizonUs)
	r.Metrics = standardMetrics(sum, m, in.GPUs, r.Window.DurationS)
	r.Breakdowns = breakdownsOf(m, sum, r.Metrics, out.PD)
	r.Goodput = goodputOf(m, in.GoodputTargets, out.InjectedByClass, r.Metrics)
	if len(in.Saturation) > 0 {
		r.Reports.Saturation = map[string]any{"final": stringMapAny(in.Saturation)}
	}
	return r
}

// windowStartRule and windowStopRule are the frame's rules for the window windowOf states.
type windowRules struct {
	start result.WindowStart
	stop  result.WindowStop
}

// windowOf states the measured window the metrics are over, in simulated microseconds.
func windowOf(m *sim.Metrics, sum sim.MeasuredSummary, clockUs, horizonUs int64) result.Window {
	if sum.MeasuredWindowS <= 0 {
		// No measured request completed, so no window bounds one: the zero rates are stated
		// over the simulated run, which is what such a run measured. An idle engine's own clock
		// never advances, so the span is the cluster's clock when it ran longer; a run that saw
		// nothing at all before its horizon spanned the horizon.
		end := max(int64(m.SimEndedTime), clockUs)
		if end <= 0 && horizonUs > 0 && horizonUs < math.MaxInt64 {
			end = horizonUs
		}
		return result.Window{StartUs: 0, EndUs: end, DurationS: float64(end) / 1e6,
			Requests: sum.MeasuredCompletedRequests}
	}
	return result.Window{StartUs: sum.WindowStartUs, EndUs: sum.WindowEndUs,
		DurationS: float64(sum.WindowEndUs-sum.WindowStartUs) / 1e6, Requests: sum.MeasuredCompletedRequests}
}

// windowRulesOf names the rules that opened and closed the window windowOf states.
func windowRulesOf(m *sim.Metrics, sum sim.MeasuredSummary) windowRules {
	if sum.MeasuredWindowS <= 0 {
		return windowRules{result.StartSimulation, result.StopSimulation}
	}
	w := windowRules{result.StartFirstArrivalAfterWarmup, result.StopLastCompletion}
	if m.MeasuredWindow != nil {
		w.start = result.StartProfiling
		if m.MeasuredWindow.EndUs > 0 {
			w.stop = result.StopProfiling
		}
	}
	return w
}

// standardMetrics are the run's figures under their metric-registry identifiers
// (contracts/metrics, DEV-8), over the measured window. A latency family is reported only when
// it has a sample, so a run that completed nothing states no latency rather than a zero that
// reads as instant; the rates need a window. Each figure carries the count of samples behind
// it.
//
// Two series were renamed by the registry (docs/guide/results.md has the table): the
// per-request (latency - TTFT) / (output tokens - 1), schema 1's itl_*, is tpot_*; and the
// per-step interval the engine section carried is step_interval_*. Schema 1's itl_* is not
// the registry's itl, which pools every token gap, and BLIS does not compute that pool yet,
// so no itl_* figure is written.
func standardMetrics(sum sim.MeasuredSummary, m *sim.Metrics, gpus int, windowS float64) map[string]result.Figure {
	out := map[string]result.Figure{}
	put := func(id string, v float64, n int) { out[id] = result.Figure{Value: v, Samples: n} }
	n := sum.MeasuredCompletedRequests
	if windowS > 0 {
		put("output_tok_s", sum.OutputThroughputTokS, n)
		if gpus > 0 {
			put("output_tok_s_per_gpu", sum.OutputThroughputTokS/float64(gpus), n)
		}
		put("request_throughput", float64(n)/windowS, n)
	}
	if n > 0 {
		put("ttft_mean_ms", sum.TTFTMeanMs, n)
		put("ttft_p50_ms", sum.TTFTP50Ms, n)
		put("ttft_p90_ms", sum.TTFTP90Ms, n)
		put("ttft_p95_ms", sum.TTFTP95Ms, n)
		put("ttft_p99_ms", sum.TTFTP99Ms, n)
		put("e2e_mean_ms", sum.E2EMeanMs, n)
		put("e2e_p50_ms", sum.E2EP50Ms, n)
		put("e2e_p90_ms", sum.E2EP90Ms, n)
		put("e2e_p95_ms", sum.E2EP95Ms, n)
		put("e2e_p99_ms", sum.E2EP99Ms, n)
	}
	if k := sum.RequestITLCount; k > 0 {
		put("tpot_mean_ms", sum.RequestITLMeanMs, k)
		put("tpot_p50_ms", sum.RequestITLP50Ms, k)
		put("tpot_p90_ms", sum.RequestITLP90Ms, k)
		put("tpot_p99_ms", sum.RequestITLP99Ms, k)
		if sum.RequestITLP50Ms > 0 {
			put("interactivity_p50", sum.InteractivityP50, k)
		}
		if sum.RequestITLP90Ms > 0 {
			put("interactivity_p90", sum.InteractivityP90, k)
		}
	}
	if sum.PerUserCount > 0 {
		put("interactivity_mean", sum.OutputTokPerUserMean, sum.PerUserCount)
	}
	if m != nil {
		// preemptions is a whole-run total over the engines' steps (contracts/metrics): its
		// samples are the steps the engines took, which the per-step interval series counts.
		// A run of no steps states none.
		if steps := len(m.AllITLs); steps > 0 {
			put("preemptions", float64(m.PreemptionCount), steps)
		}
		if steps := sum.StepIntervalCount; steps > 0 {
			put("step_interval_mean_ms", sum.StepIntervalMeanMs, steps)
			put("step_interval_p50_ms", sum.StepIntervalP50Ms, steps)
			put("step_interval_p90_ms", sum.StepIntervalP90Ms, steps)
			put("step_interval_p95_ms", sum.StepIntervalP95Ms, steps)
			put("step_interval_p99_ms", sum.StepIntervalP99Ms, steps)
		}
	}
	return out
}

// splitMetrics are the metrics a split (by model, tenant or adapter) restates over the whole
// run's window, with the whole's identifiers, and only those the whole states. The per-GPU
// throughput and the engine-wide step and preemption figures are not split.
func splitMetrics(sum sim.MeasuredSummary, defined map[string]result.Figure) map[string]float64 {
	out := map[string]float64{}
	for k, f := range standardMetrics(sum, nil, 0, sum.MeasuredWindowS) {
		if _, ok := defined[k]; ok {
			out[k] = f.Value
		}
	}
	return out
}

// breakdownsOf splits the run along the axes it exercised: per model, tenant and adapter
// (each the whole's metrics over the requests of that key), P/D, and sessions.
func breakdownsOf(m *sim.Metrics, sum sim.MeasuredSummary, defs map[string]result.Figure, pd *cluster.PDMetrics) *result.Breakdowns {
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

// goodputOf is the SLO outcome, nil without targets. goodput_rps and slo_attainment go into
// figures under their registry identifiers; the targets and each class's counts stay in the
// goodput section. The values are emitGoodput's, unchanged from schema 1: goodput_rps is over
// the run's simulated length, not the measured window the registry names, and moving it to
// the window is a validation change of its own (docs/guide/results.md, "Known differences").
func goodputOf(m *sim.Metrics, targets map[string]workload.SLODimTargets, injectedByClass map[string]int64,
	figures map[string]result.Figure) *result.Goodput {
	if len(targets) == 0 {
		return nil
	}
	var mo sim.MetricsOutput
	emitGoodput(&mo, m, injectedByClass, float64(m.SimEndedTime)/1e6, targets)
	g := &result.Goodput{Targets: map[string]schemaworkload.SLODimTargets{}, PerClass: map[string]result.ClassGoodput{}}
	for cls, t := range targets {
		g.Targets[cls] = t
	}
	_, per := cluster.SLOAttainmentMultiDim(cluster.BuildLatencyResults(m), injectedByClass, targets)
	met, targeted := 0, 0
	for cls, a := range per {
		g.PerClass[cls] = result.ClassGoodput{Completed: completedInClass(m, cls), Met: a.Good}
		met += a.Good
		targeted += int(injectedByClass[cls])
	}
	figures["goodput_rps"] = result.Figure{Value: mo.GoodputRPS, Samples: met}
	figures["slo_attainment"] = result.Figure{Value: mo.SLOAttainment, Samples: targeted}
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
	switch l := r.Frame.Generator.Load; {
	case l.Concurrency > 0:
		load = fmt.Sprintf("c%d", l.Concurrency)
	case l.RequestRate > 0:
		load = "r" + strings.ReplaceAll(fmt.Sprintf("%g", l.RequestRate), ".", "p")
	}
	name := fmt.Sprintf("%s.%s-s%d", base, load, r.Frame.Generator.Seed)
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
