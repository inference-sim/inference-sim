package cmd

// --view summary (#1871): the plain-text sections run and replay printed after the metrics
// JSON before the Result document existed, now written to stderr. The sections read the same
// figures the Result is built from; the fitness score, the trace summary and the per-SLO
// distributions are views of the run with no home in the document, so the view carries them
// beside it.

import (
	"fmt"
	"io"
	"sort"

	"github.com/inference-sim/blis-schemas/spec/result"

	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/trace"
)

// summaryView is what the summary renders beyond the Result document.
type summaryView struct {
	Raw               *cluster.RawMetrics
	Fitness           *cluster.FitnessResult
	PerSLO            map[string]*cluster.SLOMetrics
	GoodputConfigured bool
	PerModel          map[string]*cluster.ModelMetrics
	PerTenant         map[string]*cluster.TenantMetrics
	Sessions          *cluster.SessionMetrics
	Trace             *trace.TraceSummary
}

// renderSummary writes the text sections to w, in the order the commands printed them.
func renderSummary(w io.Writer, r *result.Result, v *summaryView) {
	if v == nil {
		return
	}
	_, _ = fmt.Fprintf(w, "=== Result %s ===\n", r.Name)
	if f := v.Fitness; f != nil {
		_, _ = fmt.Fprintln(w, "=== Fitness Evaluation ===")
		_, _ = fmt.Fprintf(w, "Score: %.6f\n", f.Score)
		keys := make([]string, 0, len(f.Components))
		for k := range f.Components {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			_, _ = fmt.Fprintf(w, "  %s: %.6f\n", k, f.Components[k])
		}
	}
	if raw := v.Raw; raw != nil {
		printAnomalyCounters(w, raw)
		printKVCacheMetrics(w, raw.PreemptionRate, raw.CacheHitRate, raw.KVThrashingRate)
	}
	printPerSLOMetrics(w, v.PerSLO, v.GoodputConfigured)
	printPerModelMetrics(w, v.PerModel)
	printPerTenantMetrics(w, v.PerTenant)
	printSessionMetrics(w, v.Sessions)
	if v.Raw != nil {
		printPDMetrics(w, v.Raw.PD)
	}
	if ts := v.Trace; ts != nil {
		printTraceSummary(w, ts)
	}
}

// printAnomalyCounters writes the anomaly section when any counter is nonzero.
func printAnomalyCounters(w io.Writer, raw *cluster.RawMetrics) {
	if raw.PriorityInversions == 0 && raw.HOLBlockingEvents == 0 && raw.RejectedRequests == 0 && raw.RoutingRejections == 0 &&
		raw.DroppedUnservable == 0 && raw.LengthCappedRequests == 0 && raw.GatewayQueueDepth == 0 && raw.GatewayQueueRejected == 0 &&
		raw.GatewayEvicted == 0 && raw.GatewayExpired == 0 && raw.EncodeRoutingRejections == 0 && raw.TimedOutRequests == 0 {
		return
	}
	_, _ = fmt.Fprintln(w, "=== Anomaly Counters ===")
	_, _ = fmt.Fprintf(w, "Priority Inversions: %d\n", raw.PriorityInversions)
	_, _ = fmt.Fprintf(w, "HOL Blocking Events: %d\n", raw.HOLBlockingEvents)
	_, _ = fmt.Fprintf(w, "Rejected Requests (Admission): %d\n", raw.RejectedRequests)
	if len(raw.ShedByTier) > 0 {
		tiers := make([]string, 0, len(raw.ShedByTier))
		for k := range raw.ShedByTier {
			tiers = append(tiers, k)
		}
		sort.Strings(tiers) // R2/INV-6: deterministic output order
		for _, tier := range tiers {
			_, _ = fmt.Fprintf(w, "  Shed (%s): %d\n", tier, raw.ShedByTier[tier])
		}
	}
	_, _ = fmt.Fprintf(w, "Rejected Requests (Routing): %d\n", raw.RoutingRejections)
	_, _ = fmt.Fprintf(w, "Dropped Unservable: %d\n", raw.DroppedUnservable)
	_, _ = fmt.Fprintf(w, "Timed Out Requests: %d\n", raw.TimedOutRequests)
	_, _ = fmt.Fprintf(w, "Length-Capped Requests: %d\n", raw.LengthCappedRequests)
	for _, c := range []struct {
		label string
		n     int
	}{
		{"Gateway Queue Depth (horizon)", raw.GatewayQueueDepth},
		{"Gateway Queue Rejected", raw.GatewayQueueRejected},
		{"Gateway Evicted (in-flight)", raw.GatewayEvicted},
		{"Gateway Expired (TTL)", raw.GatewayExpired},
		{"Encode Routing Rejections", raw.EncodeRoutingRejections},
	} {
		if c.n > 0 {
			_, _ = fmt.Fprintf(w, "%s: %d\n", c.label, c.n)
		}
	}
}

// printTraceSummary writes the decision-trace summary (--summarize-trace).
func printTraceSummary(w io.Writer, ts *trace.TraceSummary) {
	_, _ = fmt.Fprintln(w, "=== Trace Summary ===")
	_, _ = fmt.Fprintf(w, "Total Decisions: %d\n", ts.TotalDecisions)
	_, _ = fmt.Fprintf(w, "  Admitted: %d\n", ts.AdmittedCount)
	_, _ = fmt.Fprintf(w, "  Rejected: %d\n", ts.RejectedCount)
	_, _ = fmt.Fprintf(w, "Unique Targets: %d\n", ts.UniqueTargets)
	if len(ts.TargetDistribution) > 0 {
		_, _ = fmt.Fprintln(w, "Target Distribution:")
		keys := make([]string, 0, len(ts.TargetDistribution))
		for k := range ts.TargetDistribution {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			_, _ = fmt.Fprintf(w, "  %s: %d\n", k, ts.TargetDistribution[k])
		}
	}
	_, _ = fmt.Fprintf(w, "Mean Regret: %.6f\n", ts.MeanRegret)
	_, _ = fmt.Fprintf(w, "Max Regret: %.6f\n", ts.MaxRegret)
}
