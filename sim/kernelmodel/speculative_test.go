package kernelmodel

import (
	"testing"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/inference-sim/sim"
)

// openSpeculative opens a committed scenario with its pool given a draft configuration of k
// tokens, through the same New the kernel's own OpenPool ends in.
func openSpeculative(t *testing.T, scenario string, k int) *Model {
	t.Helper()
	in, err := latencykernel.OpenInputs(scenario, repos(), 0)
	if err != nil {
		t.Fatal(err)
	}
	if k > 0 {
		in.Deployment.Pools[0].Engine.Speculative = &deployment.Speculative{Method: "mtp", NumSpecTokens: k}
	}
	kn, err := latencykernel.New(in)
	if err != nil {
		t.Fatalf("New with %d draft tokens: %v", k, err)
	}
	return newModel(kn, identityOf(in))
}

// Under speculation a decode costs its verify width -- 1 + the draft length -- whatever it later
// advanced by: the engine runs every draft position (the blis-schemas contract states
// Scheduled that way), while BLIS's NumNewTokens is the advance (1 + accepted).
//
// Relations, none of which reads the price off the code under test:
//
//   - the price of a decode does not depend on how many drafts were accepted;
//   - a speculative decode never costs less than a plain one, and a longer draft never less
//     than a shorter one;
//   - a request that has just finished its prompt (ProgressIndex == InputLen) is a decode;
//   - a prefill chunk costs the same with and without a draft configuration.
func TestASpeculativeDecodeIsPricedAtItsVerifyWidth(t *testing.T) {
	const scenario = "glm-5-h200-fp8-sglang-tp8.yaml"
	decodeAt := func(progress int64, advance int) []*sim.Request {
		return []*sim.Request{{InputTokens: make([]sim.TokenID, 2048), ProgressIndex: progress, NumNewTokens: advance}}
	}
	prefill := []*sim.Request{{InputTokens: make([]sim.TokenID, 2048), NumNewTokens: 512}}
	plain := openSpeculative(t, scenario, 0)
	prev := plain.StepTime(decodeAt(2100, 1))
	for _, k := range []int{1, 2, 3} {
		m := openSpeculative(t, scenario, k)
		price := m.StepTime(decodeAt(2100, 1))
		for advance := 2; advance <= k+1; advance++ {
			if got := m.StepTime(decodeAt(2100, advance)); got != price {
				t.Errorf("k=%d: a decode advancing %d priced %d, advancing 1 priced %d; acceptance must not move the price",
					k, advance, got, price)
			}
		}
		if price < prev {
			t.Errorf("k=%d: a decode priced %d, below the shorter draft's %d", k, price, prev)
		}
		prev = price
		// The first decode step after the prompt is a decode too.
		if atBoundary := m.StepTime(decodeAt(2048, 1)); atBoundary < plain.StepTime(decodeAt(2048, 1)) {
			t.Errorf("k=%d: the first decode after the prompt priced %d, below a plain decode's %d", k,
				atBoundary, plain.StepTime(decodeAt(2048, 1)))
		}
		if got, want := m.StepTime(prefill), plain.StepTime(prefill); got != want {
			t.Errorf("k=%d: a prefill chunk priced %d with a draft configuration, %d without", k, got, want)
		}
	}
	if last := openSpeculative(t, scenario, 3).StepTime(decodeAt(2100, 1)); last <= plain.StepTime(decodeAt(2100, 1)) {
		t.Errorf("a 3-token draft's decode priced %d, no more than a plain decode's %d; the verify width did not reach the kernel",
			last, plain.StepTime(decodeAt(2100, 1)))
	}
}
