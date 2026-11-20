package lowering

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/coefficient"

	blisvllm "github.com/inference-sim/inference-sim/engines/vllm"
	"github.com/inference-sim/inference-sim/engines/vllm/core/vllmsettings"
)

// The plan states the identity of the engine code that does the host and launch work (Table DC),
// taken from the code-path table this release publishes. The cost model keys engine-path
// coefficients by it, so a plan that stated none, or another release's, would price with the
// wrong coefficient or with none. It is stated whatever the graph mode: an eager launch is
// priced by the same path's coefficients.
func TestThePlanStatesTheCodeIDsOfTheEngineCode(t *testing.T) {
	for _, c := range []struct {
		name string
		s    vllmsettings.Settings
	}{
		{"default (full and piecewise graphs)", vllmsettings.Settings{}},
		{"eager", vllmsettings.Settings{EnforceEager: true}},
	} {
		t.Run(c.name, func(t *testing.T) {
			l := lowered(t, moe, c.s)
			if got, want := l.Plan.Host.CodeID, blisvllm.CodeIDOf(coefficient.CodePathHost); got != want || got == "" {
				t.Errorf("Host.CodeID = %q, want %q", got, want)
			}
			if got, want := l.Plan.Graphs.CodeID, blisvllm.CodeIDOf(coefficient.CodePathGraphs); got != want || got == "" {
				t.Errorf("Graphs.CodeID = %q, want %q", got, want)
			}
		})
	}
}
