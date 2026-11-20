package pack

import (
	"github.com/inference-sim/inference-sim/substrate/contracts/check"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/scenario"
)

// Input is everything a rule may read. A rule sees the whole resolved set of documents,
// because the interesting rules are cross-document: expert divisibility needs the
// deployment's layout and the model's expert count, and neither alone. The Scenario carries
// the immutable problem (engine version, cluster inventory) and the Deployment the mutable
// layout (pools, offload) a rule checks against it.
//
// The rules are this engine's: they say whether a document describes a deployment vLLM of
// this release would run, and the engine module reports what they find (the run's skipped
// rules, and the refusals of its lowering). The substrate's validator runs none of them.
type Input struct {
	Scenario   *scenario.Scenario
	Deployment *deployment.Deployment
	Model      *model.Graph
	// Packaging is the scenario cluster's resolved packaging (scenario.Cluster.Packaging):
	// the GPUs per node and per rack, from the scenario or the chip's catalog entry. A rule
	// that needs a node size reads it here, never from Scenario.Cluster, which is zero where
	// the scenario leaves the value to the catalog. Nil when the caller could not resolve
	// it; a rule then has no node size and checks nothing that needs one.
	Packaging *scenario.Packaging
}

// Rule is one named, version-scoped check.
//
// Name is stable and mentioned in every finding, so a CI job can waive one rule without
// waiving a whole layer, and a reader can tell which release's expectation fired. Because is
// a short statement of why the rule exists, surfaced in documentation rather than in every
// message.
type Rule struct {
	Name    string
	Because string
	Check   func(Input, *validate.Problems)
}

// RuleSet is what a release's rules answer: its rule names and the findings of every rule
// over a document set.
type RuleSet interface {
	// EngineVersion is the release the rules describe.
	EngineVersion() string
	// Check runs every rule of the release over a document set.
	Check(in Input) *validate.Problems
	// RuleNames returns the release's rule names, sorted.
	RuleNames() []string
}
