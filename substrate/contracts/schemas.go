// Package blisschemas is the entry point: it composes field-level validation with
// version-scoped rule validation over a complete set of documents.
//
// The two layers stay visible in the result. A Report separates field problems from
// rule problems, because the two need different responses: a field problem is a
// malformed document and always the author's to fix, where a rule problem may mean
// the document is right and the engine version is wrong, or that a rule needs
// updating for a release it has not seen.
//
// Field validation runs first and rule validation only if it passes. A rule reading
// a malformed document produces findings that are artifacts of the malformation, and
// a reader cannot tell those from real ones.
package contracts

import (
	"fmt"
	"reflect"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/evaluation"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
	"github.com/inference-sim/blis-substrate/contracts/spec/routing"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

// Bundle is a complete set of documents describing one run, and optionally the engine rules
// to check it with. Optional members are nil when a caller is validating a subset: a catalog
// contributor checks a chip alone, where a CI job checking a run supplies everything. The
// Scenario fixes the immutable problem and the Deployment the mutable configuration against
// it; validating both together is what checks that a deployment fits its cluster.
type Bundle struct {
	Scenario     *scenario.Scenario
	Deployment   *deployment.Deployment
	Model        *model.Graph
	Chip         *hardware.Chip
	Fabric       *hardware.Fabric
	Devices      []*hardware.StorageDevice
	Coefficients []*coefficient.Set
	// AcceptanceProfiles are registry acceptance-profile sets, validated field by field.
	AcceptanceProfiles []*acceptance.ProfileSet
	// RoutingProfiles are registry routing-profile sets, validated field by field.
	RoutingProfiles []*routing.ProfileSet
	Workload        *workload.Shape
	// ReplayWorkload is a catalog replay workload, validated field by field.
	ReplayWorkload *workload.CatalogReplay
	Evaluation     *evaluation.Run

	// Rules looks up the engine rules for the scenario's engine_version: the caller's, from the
	// engine model that owns the release, for this schema registers none (GOAL.md phase 2b). It
	// is required whenever the bundle carries a scenario: a pool's engine block is opaque here,
	// so its checks and every version-scoped rule are the rules', and a bundle validated with
	// none would pass unchecked. Rules that answer for another release are refused
	// (engine-version-matches), never applied.
	Rules func(version string) rules.EngineRules
	// RulesVersions lists the versions Rules answers, for the finding when it answers none.
	// nil with Rules set: the finding says only that the version is unknown, listing none.
	RulesVersions func() []string
	// Result is a simulator run's predicted output, validated field by field together with
	// the input it embeds. Rules do not run on that embedded input: to rule-check it, pass
	// &Result.Input.Scenario and &Result.Input.Deployment as Scenario and Deployment too.
	// With Rules set, a Result and no Scenario, the rule layer warns (result-input-rules)
	// rather than read as a rule-checked pass.
	Result *result.Result
}

// Report is the outcome of validating a bundle, with the two layers kept apart.
type Report struct {
	// Field holds structural problems: malformed documents, unrecognized vocabulary,
	// arithmetic that does not close. These are version-independent.
	Field *validate.Problems
	// Rule holds version-scoped problems, each attributed to a named rule.
	Rule *validate.Problems
	// RulesApplied is the engine version whose pack ran, empty if none did.
	RulesApplied string
}

// OK reports whether nothing failed in either layer.
func (r Report) OK() bool { return r.Field.OK() && r.Rule.OK() }

// Problems returns every finding from both layers, field problems first.
func (r Report) Problems() []validate.Problem {
	return append(r.Field.All(), r.Rule.All()...)
}

// Validate checks a bundle: every present document field-by-field, then the
// version-scoped rules if the field layer passed.
func Validate(b Bundle) Report {
	field := &validate.Problems{}

	if b.Scenario != nil {
		field.Merge("scenario", b.Scenario.Validate())
	}
	if b.Deployment != nil {
		field.Merge("deployment", b.Deployment.Validate())
	}
	// A deployment is validated against the cluster that hosts it and the engine version
	// it is tuned for, both of which live on the Scenario. A deployment with no scenario
	// therefore cannot be fully checked — its cluster fit and every version-scoped engine
	// rule would silently not run — and, unlike a chip-only bundle a catalog contributor
	// checks, a deployment-only bundle has no standalone use. So require the scenario
	// rather than return green on a deployment that only looks valid because half its
	// checks were skipped.
	if b.Deployment != nil && b.Scenario == nil {
		field.Field("scenario",
			"required when a deployment is present: a deployment is validated against the cluster and engine version its scenario fixes")
	}
	// A deployment and the cluster it is placed on are two documents, so the checks
	// that couple them — that the pools fill the cluster, that each local data-parallel
	// width divides a node, that no engine needs more GPUs than its pool owns, and that
	// offload tiers draw from the declared storage inventory — can only run when both
	// are present.
	if b.Scenario != nil && b.Deployment != nil {
		field.Merge("deployment", b.Deployment.ValidateAgainstCluster(
			deployment.ClusterConstraints{
				Nodes:       b.Scenario.Cluster.Nodes,
				GPUsPerNode: b.Scenario.Cluster.GPUsPerNode,
				Storage:     b.Scenario.Cluster.Storage,
			}))
	}
	// The engine rules the scenario's release selects, nil when the caller gave none or they
	// know no such release. A pool's engine block is opaque to this schema, so its field checks
	// are the release's (EngineRules.CheckEngineBlock) and run in the field layer, under the
	// same deployment paths a field check of the deployment reports at.
	var er rules.EngineRules
	if b.Scenario != nil && b.Rules != nil {
		if er = b.Rules(b.Scenario.EngineRelease()); isNilRules(er) {
			er = nil
		}
	}
	if b.Deployment != nil && er != nil && er.EngineVersion() == b.Scenario.EngineRelease() {
		for i, pool := range b.Deployment.Pools {
			if pp := er.CheckEngineBlock(fmt.Sprintf("pools[%d]", i), pool.Engine); pp != nil {
				field.Merge("deployment", pp)
			}
		}
		if o := b.Deployment.Offload; o != nil && !o.ConnectorOptions.IsZero() {
			if pp := er.CheckConnectorOptions("offload", o.ConnectorOptions); pp != nil {
				field.Merge("deployment", pp)
			}
		}
	}
	if b.Deployment != nil {
		checkAcceptanceK(field, b, er)
	}
	if b.Model != nil {
		field.Merge("model", b.Model.Validate())
	}
	if b.Chip != nil {
		field.Merge("chip", b.Chip.Validate())
	}
	if b.Fabric != nil {
		field.Merge("fabric", b.Fabric.Validate())
	}
	for _, d := range b.Devices {
		if d != nil {
			field.Merge("device:"+d.Name, d.Validate())
		}
	}
	for _, c := range b.Coefficients {
		if c != nil {
			field.Merge("coefficients:"+c.Name, c.Validate())
		}
	}
	for _, a := range b.AcceptanceProfiles {
		if a != nil {
			field.Merge("acceptance:"+a.Name, a.Validate())
		}
	}
	for _, r := range b.RoutingProfiles {
		if r != nil {
			field.Merge("routing:"+r.Name, r.Validate())
		}
	}
	if b.Result != nil {
		field.Merge("result", b.Result.Validate())
	}
	if b.Workload != nil {
		field.Merge("workload", b.Workload.Validate())
	}
	if b.ReplayWorkload != nil {
		field.Merge("workload", b.ReplayWorkload.Validate())
	}
	if b.Evaluation != nil {
		field.Merge("evaluation", b.Evaluation.Validate())
	}

	rep := Report{Field: field, Rule: &validate.Problems{}}
	// Rules read a document assuming it is well formed. Running them over a
	// malformed one yields findings that are consequences of the malformation, which
	// a reader cannot separate from genuine rule violations.
	if !field.OK() {
		return rep
	}
	if b.Result != nil && b.Rules != nil && b.Scenario == nil {
		rep.Rule.RuleWarnf("result-input-rules",
			"the result's embedded input was not rule-checked: pass its scenario and deployment as "+
				"Bundle.Scenario and Bundle.Deployment to apply the engine rules to them")
	}
	if b.Scenario != nil {
		in := rules.Input{Scenario: b.Scenario, Deployment: b.Deployment, Model: b.Model}
		if b.Rules == nil {
			// A scenario names an engine release, and its engine blocks and version-scoped rules
			// are that release's to check. With no rules given nothing would check them, and a
			// bundle that only looks valid because its checks were skipped must not pass.
			rep.Rule.RuleErrorf("engine-rules-given",
				"no engine rules were given for engine version %q: pass the release's rules "+
					"(Bundle.Rules) from the engine model that owns it; nothing was rule-checked",
				b.Scenario.EngineRelease())
			return rep
		}
		if er == nil {
			// The same finding as a version with no rules: it has not been rule-checked, and
			// saying nothing would read as a pass.
			p := &validate.Problems{}
			if b.RulesVersions != nil {
				p.RuleWarnf("engine-version-known",
					"no rules pack for engine version %q; field validation passed but no version-specific rule was applied. Registered versions: %v",
					b.Scenario.EngineRelease(), b.RulesVersions())
			} else {
				p.RuleWarnf("engine-version-known",
					"no rules pack for engine version %q; field validation passed but no version-specific rule was applied",
					b.Scenario.EngineRelease())
			}
			rep.Rule = p
			return rep
		}
		// Rules for another release are never applied to this one: a lookup that falls back to
		// a nearby release would pass a scenario its own release refuses (rules.go's package
		// comment).
		if v := er.EngineVersion(); v != b.Scenario.EngineRelease() {
			rep.Rule.RuleErrorf("engine-version-matches",
				"the engine rules looked up for engine version %q are for %q; no rule was applied",
				b.Scenario.EngineRelease(), v)
			return rep
		}
		// A Check that finds nothing may say so with a nil Problems; the report keeps its empty
		// layer, so OK and Problems never meet a nil.
		if p := er.Check(in); p != nil {
			rep.Rule = p
		}
		rep.RulesApplied = b.Scenario.EngineRelease()
	}
	return rep
}

// isNilRules reports whether er is nil or holds a nil pointer, map, slice, func or channel of
// any rules type: what a lookup returns, typed, for a version it does not know. A forwarding
// value around a nil is not nil here, and fails when it is asked.
func isNilRules(er rules.EngineRules) bool {
	if er == nil {
		return true
	}
	switch v := reflect.ValueOf(er); v.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Func, reflect.Chan, reflect.Interface:
		return v.IsNil()
	}
	return false
}

// checkAcceptanceK refuses an inline acceptance stated for a K other than one the
// deployment speculates with: the scenario's pinned value, its trace's, its replay's (the
// merged replay, for a reference whose catalog entry the bundle carries), and the
// workload shape's or catalog replay's. Acceptance at one K says nothing exact about another, so a value is never
// rescaled. A value with no speculating pool to apply to is a warning, since it is inert
// rather than wrong.
func checkAcceptanceK(p *validate.Problems, b Bundle, er rules.EngineRules) {
	type stated struct {
		path string
		v    *acceptance.Value
	}
	var values []stated
	if s := b.Scenario; s != nil {
		if s.Acceptance != nil && s.Acceptance.Value != nil {
			values = append(values, stated{"scenario.acceptance.value.num_spec_tokens", s.Acceptance.Value})
		}
		if s.Workload != nil && s.Workload.Trace != nil && s.Workload.Trace.Acceptance != nil {
			values = append(values, stated{"scenario.workload.trace.acceptance.num_spec_tokens", s.Workload.Trace.Acceptance})
		}
		if s.Workload != nil && s.Workload.Replay != nil {
			// A reference takes its acceptance from the catalog entry unless it overrides
			// it, so check the replay it resolves to when the bundle carries that entry.
			r := s.Workload.Replay
			if e := b.ReplayWorkload; r.Workload != "" && e != nil && e.Name == r.Workload {
				if merged, err := r.Resolve(e); err == nil {
					r = merged
				}
			}
			if r.Acceptance != nil {
				values = append(values, stated{"scenario.workload.replay.acceptance.num_spec_tokens", r.Acceptance})
			}
		}
	}
	if b.Workload != nil && b.Workload.Acceptance != nil {
		values = append(values, stated{"workload.acceptance.num_spec_tokens", b.Workload.Acceptance})
	}
	if b.ReplayWorkload != nil && b.ReplayWorkload.Acceptance != nil {
		values = append(values, stated{"workload.acceptance.num_spec_tokens", b.ReplayWorkload.Acceptance})
	}
	if len(values) == 0 {
		return
	}
	if er == nil {
		// Which pools speculate is in their engine blocks, which only the release's rules read;
		// without them the check cannot run, and the rule layer reports why.
		return
	}
	speculates := false
	for i, pool := range b.Deployment.Pools {
		k := er.DraftTokens(pool.Engine)
		if k < 1 {
			continue
		}
		speculates = true
		for _, s := range values {
			if err := acceptance.CheckK(s.v, k); err != nil {
				p.Field(s.path, "deployment pools[%d]: %v", i, err)
			}
		}
	}
	if !speculates {
		for _, s := range values {
			w := &validate.Problems{}
			w.Warnf("an acceptance is stated, but no deployment pool speculates, so it applies to nothing")
			p.Merge(s.path, w)
		}
	}
}
