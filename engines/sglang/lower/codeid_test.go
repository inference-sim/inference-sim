// Original to blis-sglang.

package lower

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/coefficient"

	blissglang "github.com/inference-sim/inference-sim/engines/sglang"
)

// The plan states the identity of the engine code that does the host and launch work (Table DC),
// taken from the code-path table this release publishes; the cost model keys engine-path
// coefficients by it.
func TestThePlanStatesTheCodeIDsOfTheEngineCode(t *testing.T) {
	low, err := lowerPool(t, "glm-5", 8, "mem_fraction_static: 0.85\n")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := low.Plan.Host.CodeID, blissglang.CodeIDOf(coefficient.CodePathHost); got != want || got == "" {
		t.Errorf("Host.CodeID = %q, want %q", got, want)
	}
	if got, want := low.Plan.Graphs.CodeID, blissglang.CodeIDOf(coefficient.CodePathGraphs); got != want || got == "" {
		t.Errorf("Graphs.CodeID = %q, want %q", got, want)
	}
}
