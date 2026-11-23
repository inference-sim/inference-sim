// Package contracts is the entry point: it composes field-level validation over a complete
// set of documents.
//
// A document is checked here for what it is: counts positive, arithmetic closing, every
// enumerated value one the schema knows. Whether a PARTICULAR engine release would run it is
// not asked here: that is the engine module's refusal (SUB-12's engine layer), made when the
// engine lowers the deployment. The one thing an engine contributes to this validator is the
// reading of what the schema holds unopened (a pool's engine block, an offload's connector
// options), through EngineChecker.
package contracts

import (
	"fmt"
	"reflect"

	"github.com/inference-sim/inference-sim/substrate/contracts/check"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/acceptance"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/coefficient"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/evaluation"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/hardware"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/routing"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/scenario"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/workload"
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

	// Engine looks up the engine's reading of the documents this schema holds unopened, for
	// the engine release the scenario runs under: the caller's, from the engine module that owns the
	// release, for this schema registers none. It is required whenever the bundle carries a
	// scenario: a pool's engine block is opaque here, so its checks are the engine's, and a
	// bundle validated with none would pass unchecked. A checker that answers for another
	// release is refused (engine-version-matches), never applied.
	Engine func(version string) EngineChecker
	// EngineVersions lists the versions Engine answers, for the finding when it answers none.
	// nil with Engine set: the finding says only that the version is unknown, listing none.
	EngineVersions func() []string
	// Result is a simulator run's predicted output, validated field by field together with
	// the input it embeds. The engine's checks do not run on that embedded input: to apply
	// them, pass &Result.Input.Scenario and &Result.Input.Deployment as Scenario and
	// Deployment too. With Engine set, a Result and no Scenario, the report warns
	// (result-input-engine) rather than read as an engine-checked pass.
	Result *result.Result
}

// EngineChecker is the engine module's reading of what this schema holds unopened. The
// engine owns the release, so a pool's engine block and an offload's connector options are
// opened by it, and a document it refuses is refused by the engine that would not run it.
type EngineChecker interface {
	// EngineVersion is the release the checker describes, such as "1.2.0".
	EngineVersion() string
	// CheckEngineBlock is the engine's field checks of one pool's engine block: whether the
	// block decodes, and the bounds its own types declare. at is the pool's path within the
	// deployment, such as "pools[0]", and each finding is at a path below at + ".engine".
	CheckEngineBlock(at string, b deployment.EngineBlock) *validate.Problems
	// CheckConnectorOptions is the engine's checks of an offload's connector options. at is
	// the offload block's path, "offload", and each finding is at a path below
	// at + ".connector_options".
	CheckConnectorOptions(at string, b deployment.EngineBlock) *validate.Problems
	// DraftTokens is the speculative tokens a pool's engine block drafts per step, zero when
	// it drafts none or does not decode.
	DraftTokens(b deployment.EngineBlock) int
}

// Report is the outcome of validating a bundle.
type Report struct {
	// Field holds structural problems: malformed documents, unrecognized vocabulary,
	// arithmetic that does not close, and the engine's reading of its own blocks.
	Field *validate.Problems
}

// OK reports whether nothing failed.
func (r Report) OK() bool { return r.Field.OK() }

// Problems returns every finding.
func (r Report) Problems() []validate.Problem { return r.Field.All() }

// Validate checks a bundle: every present document field-by-field, and the engine's reading
// of the blocks this schema holds unopened.
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
	//
	// The GPUs per node those checks read are the cluster's resolved packaging
	// (scenario.Cluster.Packaging), which needs the chip's catalog entry: the scenario may
	// leave the value to it. Given the chip, the packaging is resolved, and a scenario that
	// contradicts the entry is refused here. Without the chip, a stated value is used, and a
	// scenario that leaves it to the catalog is refused rather than checked with no node size.
	pk := b.packaging(field)
	if b.Scenario != nil && b.Deployment != nil && pk != nil {
		field.Merge("deployment", b.Deployment.ValidateAgainstCluster(
			deployment.ClusterConstraints{
				Nodes:       b.Scenario.Cluster.Nodes,
				GPUsPerNode: pk.GPUsPerNode,
				Storage:     b.Scenario.Cluster.Storage,
			}))
	}
	// The engine checker the scenario's release selects, nil when the caller gave none or it
	// knows no such release. A pool's engine block is opaque to this schema, so its checks
	// are the engine's (EngineChecker.CheckEngineBlock) and run here, under the same
	// deployment paths a field check of the deployment reports at.
	var er EngineChecker
	if b.Scenario != nil && b.Engine != nil {
		if er = b.Engine(b.Scenario.EngineRelease()); isNilChecker(er) {
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
		for i, pool := range b.Deployment.Pools {
			if o := pool.Offload; o != nil && !o.ConnectorOptions.IsZero() {
				if pp := er.CheckConnectorOptions(fmt.Sprintf("pools[%d].offload", i), o.ConnectorOptions); pp != nil {
					field.Merge("deployment", pp)
				}
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

	// The engine's checks need the documents to be well formed, and findings about a
	// malformed one are consequences of the malformation.
	if !field.OK() {
		return Report{Field: field}
	}
	if b.Result != nil && b.Engine != nil && b.Scenario == nil {
		field.RuleWarnf("result-input-engine",
			"the result's embedded input was not engine-checked: pass its scenario and deployment as "+
				"Bundle.Scenario and Bundle.Deployment to apply the engine's checks to them")
	}
	if b.Scenario != nil {
		if b.Engine == nil {
			// A scenario names an engine release, and its engine blocks are that release's to
			// check. With no checker given nothing would check them, and a bundle that only
			// looks valid because its checks were skipped must not pass.
			field.RuleErrorf("engine-given",
				"no engine checker was given for engine version %q: pass the release's checker "+
					"(Bundle.Engine) from the engine module that owns it; nothing was engine-checked",
				b.Scenario.EngineRelease())
		} else if er == nil {
			// Not engine-checked, and saying nothing would read as a pass.
			if b.EngineVersions != nil {
				field.RuleWarnf("engine-version-known",
					"no engine for engine version %q; field validation passed but the engine's checks were not applied. Registered versions: %v",
					b.Scenario.EngineRelease(), b.EngineVersions())
			} else {
				field.RuleWarnf("engine-version-known",
					"no engine for engine version %q; field validation passed but the engine's checks were not applied",
					b.Scenario.EngineRelease())
			}
		} else if v := er.EngineVersion(); v != b.Scenario.EngineRelease() {
			// A checker for another release is never applied to this one: a lookup that falls
			// back to a nearby release would pass a scenario its own release refuses.
			field.RuleErrorf("engine-version-matches",
				"the engine looked up for engine version %q is for %q; its checks were not applied",
				b.Scenario.EngineRelease(), v)
		}
	}
	return Report{Field: field}
}

// isNilChecker reports whether er is nil or holds a nil pointer, map, slice, func or channel of
// any checker type: what a lookup returns, typed, for a version it does not know. A forwarding
// value around a nil is not nil here, and fails when it is asked.
func isNilChecker(er EngineChecker) bool {
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
func checkAcceptanceK(p *validate.Problems, b Bundle, er EngineChecker) {
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
		// Which pools speculate is in their engine blocks, which only the engine reads;
		// without it the check cannot run, and Validate reports why.
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

// packaging returns the cluster's packaging for the checks that need a node size, and
// records at scenario.cluster why there is none.
//
// With the chip it is the resolved packaging, and a cluster that contradicts the chip's
// entry is refused. Without the chip only a stated gpus_per_node is known: it is used for
// the deployment's fit, marked stated, and the rack is left unresolved, because whether a
// stated rack is the entry's cannot be checked without the entry, and no check here reads
// one. A deployment over a cluster that leaves its node size to the catalog is refused
// when the chip is missing, since its fit could not be checked. It is nil when there is no
// scenario, or no node size to check against.
func (b Bundle) packaging(field *validate.Problems) *scenario.Packaging {
	if b.Scenario == nil {
		return nil
	}
	c := b.Scenario.Cluster
	if b.Chip != nil {
		pk, err := c.Packaging(b.Chip)
		if err != nil {
			field.Merge("scenario.cluster", err.(*validate.Problems))
			return nil
		}
		return &pk
	}
	if c.GPUsPerNode > 0 {
		return &scenario.Packaging{GPUsPerNode: c.GPUsPerNode, GPUsPerNodeSource: scenario.SourceStated}
	}
	if b.Deployment != nil {
		field.Field("scenario.cluster.gpus_per_node",
			"is left to the catalog entry of %q, which the bundle does not carry: pass Bundle.Chip, so the packaging resolves and the deployment's fit to the cluster can be checked",
			c.Hardware)
	}
	return nil
}
