package kernelmodel

import (
	"sort"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/vocab"

	"github.com/inference-sim/inference-sim/sim"
)

// enforcedRules are the release rules a run is refused on: deepseek-v41-refuses-mtp by the
// kernel (refuseV4Layouts), speculative-method-known's engine half by checkSpeculativeMethod.
// The kernel applies only the field layer of blis-schemas validation, so every other rule
// of the pack goes unapplied.
var enforcedRules = map[string]bool{
	"deepseek-v41-refuses-mtp": true,
	"speculative-method-known": true,
}

// ruleReport is what the scenario's release rules say about a pool that BLIS runs anyway.
type ruleReport struct {
	skipped  []string
	findings []sim.RuleFinding
}

// reportRules lists the rules of the scenario's engine release that are not enforced, and
// runs the whole pack report-only so the file states what they find. Every finding of a
// run that was not refused is, by construction, one nobody acted on. A release with no pack
// reports blis-schemas' own engine-version-known finding.
func reportRules(in latencykernel.Inputs) ruleReport {
	var r ruleReport
	if pack, ok := in.Rules.(*rules.Pack); ok && pack != nil {
		for _, name := range pack.Names() {
			if !enforcedRules[name] {
				r.skipped = append(r.skipped, name)
			}
		}
	}
	problems := rules.Apply(rules.Input{Scenario: in.Scenario, Deployment: in.Deployment, Model: in.Model})
	for _, p := range problems.All() {
		r.findings = append(r.findings, sim.RuleFinding{
			Rule: p.Rule, Severity: string(p.Severity), Path: p.Path, Message: p.Message,
		})
	}
	return r
}

// evidencer is blis-latency-kernel's Evidence summary, which the concrete kernel carries
// beside the kernel.Kernel interface.
type evidencer interface {
	Evidence() (measured, total int, assumed []string)
}

// Provenance is the pool's evidence for the results file (#1942), from the kernel's own
// Provenance and Evidence. Lists are non-nil so the JSON shape is stable.
func (m *Model) Provenance() sim.KernelPoolProvenance {
	p := sim.KernelPoolProvenance{
		MethodCounts:        map[vocab.Method]int{},
		NotEvidenced:        []kernel.CoefficientOrigin{},
		Assumptions:         []kernel.CoefficientOrigin{},
		SkippedRules:        append([]string{}, m.rules.skipped...),
		SkippedRuleFindings: append([]sim.RuleFinding{}, m.rules.findings...),
	}
	origins := m.k.Provenance()
	for _, o := range origins {
		method := vocab.Method(o.Method)
		p.MethodCounts[method]++
		switch {
		case o.Set == latencykernel.KernelAssumptionSet:
			p.Assumptions = append(p.Assumptions, o)
		case !method.Evidenced() && method != vocab.MethodNotCharged:
			p.NotEvidenced = append(p.NotEvidenced, o)
		}
	}
	p.Total = len(origins)
	if e, ok := m.k.(evidencer); ok {
		p.Evidenced, p.Total, _ = e.Evidence()
	}
	sort.Strings(p.SkippedRules)
	return p
}
