// Moved from blis-latency-kernel/example_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

// The Go examples on the documentation site are sections of this file, included between the
// --8<-- markers. Go runs each Example and compares what it prints with its // Output:
// comment, so a page cannot show a call that no longer compiles or a number this kernel no
// longer computes. They read the catalog and registry blis-substrate embeds.

import (
	vllmrules "github.com/inference-sim/blis-vllm/rules"

	latencykernel "github.com/inference-sim/blis-substrate/kernel"
	"github.com/inference-sim/blis-substrate/load"
	"github.com/inference-sim/blis-vllm/compat/vllmsettings"
)

// --8<-- [start:repos]
// The roots a scenario's names resolve against (its model, chip, fabric and coefficient
// sets), and the engine rules its engine_version selects: vLLM's, from the engine model.
var repos = latencykernel.Repos{
	Scenarios:      "testdata",
	Data:           exampleData(),
	Rules:          latencykernel.RulesFrom(vllmrules.Lookup),
	RulesVersions:  vllmrules.Versions,
	EngineSettings: vllmsettings.OfPool,
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
