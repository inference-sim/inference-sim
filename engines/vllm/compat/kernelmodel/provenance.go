// Moved from inference-sim/sim/kernelmodel/provenance.go at ca532abca209af41425c007974a5f3ef336f5981.

package kernelmodel

import (
	"sort"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
	latencykernel "github.com/inference-sim/blis-substrate/kernel"

	sim "github.com/inference-sim/blis-vllm/compat/engine"
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
// runs the release's whole rule set report-only, through the rules the kernel was built
// with (in.Rules), so the file states what those skipped rules find. A kernel is never built
// without rules (OpenInputs refuses a release it has none for), so in.Rules is nil only for
// Inputs a caller assembled by hand, which then report nothing.
func reportRules(in latencykernel.Inputs) ruleReport {
	var r ruleReport
	if in.Rules == nil {
		return r
	}
	for _, name := range in.Rules.RuleNames() {
		if !enforcedRules[name] {
			r.skipped = append(r.skipped, name)
		}
	}
	// The rules read a node size from the cluster's resolved packaging, which fromDecoded has
	// already resolved without error; a hand-assembled Inputs that cannot resolve it gives
	// the rules none.
	var pk *scenario.Packaging
	if in.Scenario != nil {
		if p, err := in.Scenario.Cluster.Packaging(in.Chip); err == nil {
			pk = &p
		}
	}
	problems := in.Rules.Check(rules.Input{Scenario: in.Scenario, Deployment: in.Deployment, Model: in.Model, Packaging: pk})
	var all []sim.RuleFinding
	for _, p := range problems.All() {
		all = append(all, sim.RuleFinding{
			Rule: p.Rule, Severity: string(p.Severity), Path: p.Path, Message: p.Message,
		})
	}
	r.findings = skippedFindings(all)
	return r
}

// skippedFindings keeps the findings of rules no one enforces: it drops field-level findings
// (no rule) and those of the rules a run is refused on.
func skippedFindings(all []sim.RuleFinding) []sim.RuleFinding {
	var out []sim.RuleFinding
	for _, f := range all {
		if f.Rule != "" && !enforcedRules[f.Rule] {
			out = append(out, f)
		}
	}
	return out
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
		Substitutions:       append([]sim.Substitution{}, m.substitutions...),
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
	p.EvidenceSource = sim.EvidenceFromProvenance
	if e, ok := m.k.(evidencer); ok {
		p.Evidenced, p.Total, _ = e.Evidence()
		p.EvidenceSource = sim.EvidenceFromKernel
	}
	sort.Strings(p.SkippedRules)
	p.Routing = m.routing
	return p
}
