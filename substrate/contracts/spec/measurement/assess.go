package measurement

import (
	"fmt"
	"sort"
)

// Assessment is a run's computed verdict and the reasons it is not EXACT.
type Assessment struct {
	Verdict    Verdict
	Resolution Resolution
	// Reasons lists each fact that kept the run from EXACT, in document order.
	Reasons []string
}

// Assess computes a run's verdict. permitted is whether the run's licence permits
// scoring, which only the registry knows.
//
// EXACT iff the diff is empty, no provenance row is MISMATCH or UNVERIFIABLE, every
// UNSTATED row is resolved by a cited rule, every metric definition is complete, and the
// licence permits scoring. NOT_REPRODUCIBLE when BLIS cannot build the run: the model,
// engine version, cluster, deployment or workload is missing, or a difference is not
// expressible. PARTIAL otherwise. Resolution is set for EXACT only: RESOLVED when an
// UNSTATED row was resolved, STRICT when none was unstated.
func Assess(r *MeasuredRun, permitted bool) Assessment {
	var a Assessment
	blocked := false
	block := func(format string, args ...any) {
		blocked = true
		a.Reasons = append(a.Reasons, fmt.Sprintf(format, args...))
	}
	for _, m := range []struct {
		field   string
		missing bool
	}{
		{"model", r.Model == ""}, {"engine_version", r.EngineVersion == ""},
		{"cluster", r.Cluster == nil}, {"deployment", r.Deployment == nil},
		{"workload", r.Workload == nil},
	} {
		if m.missing {
			block("%s is not recorded", m.field)
		}
	}

	partial := false
	reason := func(format string, args ...any) {
		partial = true
		a.Reasons = append(a.Reasons, fmt.Sprintf(format, args...))
	}
	for _, d := range r.Diff {
		if d.Kind == DiffNotExpressible {
			block("diff %s: not expressible", d.Path)
			continue
		}
		reason("diff %s: %s", d.Path, d.Kind)
	}
	resolved := false
	for _, row := range r.Provenance {
		switch row.Verdict {
		case RowMismatch, RowUnverifiable:
			reason("provenance %s: %s", row.Setting, row.Verdict)
		case RowUnstated:
			if row.ResolvedBy == "" {
				reason("provenance %s: UNSTATED and unresolved", row.Setting)
			} else {
				resolved = true
			}
		}
	}
	for _, name := range sortedKeys(r.MetricDefs) {
		if missing := r.MetricDefs[name].Unstated(); len(missing) > 0 {
			reason("metric %s: definition does not state %v", name, missing)
		}
	}
	if r.Client.Harness == "" {
		reason("client harness is not recorded")
	}
	if !permitted {
		reason("licence %q does not permit scoring", r.Source.LicenseRef)
	}

	switch {
	case blocked:
		a.Verdict = VerdictNotReproducible
	case partial:
		a.Verdict = VerdictPartial
	default:
		a.Verdict = VerdictExact
		a.Resolution = ResolutionStrict
		if resolved {
			a.Resolution = ResolutionResolved
		}
	}
	return a
}

// Unstated lists the definition fields a metric leaves unstated, so a scorer cannot tell
// whether a BLIS metric is the same quantity.
func (d MetricDef) Unstated() []string {
	var out []string
	if d.Formula == "" {
		out = append(out, "formula")
	}
	if d.Population == "" {
		out = append(out, "population")
	}
	if d.Clock == "" {
		out = append(out, "clock")
	}
	if d.Formula.Throughput() {
		if d.Denominator == "" {
			out = append(out, "denominator")
		}
	} else if d.Statistic == "" {
		out = append(out, "statistic")
	}
	return out
}

// Cases returns one case per point under the assessment a.
func (r *MeasuredRun) Cases(a Assessment) []Case {
	out := make([]Case, 0, len(r.Points))
	for i, pt := range r.Points {
		out = append(out, Case{Run: r.Name, Point: pt.ID, Load: pt.Load, Anchor: i == 0,
			Verdict: a.Verdict, Resolution: a.Resolution})
	}
	return out
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
