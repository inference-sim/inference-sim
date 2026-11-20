// Moved from blis-latency-kernel/example_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

// The Go examples on the documentation site are sections of this file, included between the
// --8<-- markers. Go runs each Example and compares what it prints with its // Output:
// comment, so a page cannot show a call that no longer compiles or a number this kernel no
// longer computes. They read the catalog and registry blis-substrate embeds.

import (
	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-vllm/core/lowering"
	vllmrules "github.com/inference-sim/blis-vllm/core/rulepack"
	"github.com/inference-sim/blis-vllm/core/vllmsettings"

	latencykernel "github.com/inference-sim/blis-substrate/kernel"
	"github.com/inference-sim/blis-substrate/load"
)

// --8<-- [start:repos]
// The roots a scenario's names resolve against (its model, chip, fabric and coefficient
// sets), and the engine rules its engine_version selects: vLLM's, from the engine model.
var repos = latencykernel.Repos{
	Scenarios:     "testdata",
	Data:          exampleData(),
	Rules:         latencykernel.RulesFrom(vllmrules.Lookup),
	RulesVersions: vllmrules.Versions,
	RulesKind:     "vllm",
	DraftModel:    draftModelOf, // the drafter the pool's engine block names
	Lower:         lowerPool,    // the pool's settings, resolved into its plan
}

// draftModelOf is the drafter a pool's engine block names, for Repos.DraftModel.
func draftModelOf(dep *deployment.Deployment, pool int) (string, error) {
	e, err := vllmsettings.Decode(dep.Pools[pool])
	if err != nil || e.Speculative == nil {
		return "", err
	}
	return e.Speculative.DraftModel, nil
}

// lowerPool is blis-vllm's lowering of the pool's engine block, for Repos.Lower.
func lowerPool(in latencykernel.Inputs) (*plan.Lowered, error) {
	s, err := vllmsettings.OfPool(in.Deployment, in.PoolIndex)
	if err != nil {
		return nil, err
	}
	return lowering.Lower(lowering.Input{Scenario: in.Scenario, Deployment: in.Deployment,
		PoolIndex: in.PoolIndex, Model: in.Model, Draft: in.Draft, Chip: in.Chip, Rules: in.Rules,
		Settings: s})
}

// exampleData is the catalog and registry this substrate release embeds, opened and checked by
// its loader.
func exampleData() *load.Data {
	d, err := load.Embedded()
	if err != nil {
		panic(err)
	}
	return d
}
