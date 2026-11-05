// Package rules is the engine-neutral contract for validation that depends on an engine
// release.
//
// The separation from field-level validation is the point of this package. A field check
// asks whether a document is structurally a scenario: are the counts positive, does the
// arithmetic close, is the enumerated value one this schema knows. Those questions have the
// same answer in every engine release, so the answers live with the types and change only
// when the schema does.
//
// A rule asks whether a document describes a deployment a PARTICULAR engine release would
// run. Backend names appear and disappear, defaults flip, a feature gains a fourth
// disqualifying condition. Each of those is a fact about a release, so it lives with the
// engine model that owns the release, and this schema names only the shape such rules take:
// EngineRules, which a caller supplies for the release a scenario names, and the Input and
// Rule a release's checks are written in. A pool's engine block is opaque here
// (deployment.EngineBlock); the release's rules open it.
package rules

import (
	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
)

// Input is everything a rule may read. A rule sees the whole resolved set of documents,
// because the interesting rules are cross-document: expert divisibility needs the
// deployment's layout and the model's expert count, and neither alone. The Scenario carries
// the immutable problem (engine version, cluster inventory) and the Deployment the mutable
// layout (pools, offload) a rule checks against it.
type Input struct {
	Scenario   *scenario.Scenario
	Deployment *deployment.Deployment
	Model      *model.Graph
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

// ArchDeepseekV41 and ArchDeepseekV4 are DeepSeek-V4.1's and DeepSeek-V4's architecture
// names, the values of their model configurations' architectures[0], which several releases'
// rules key on.
const (
	ArchDeepseekV41 = "DeepseekV41ForCausalLM"
	ArchDeepseekV4  = "DeepseekV4ForCausalLM"
)
