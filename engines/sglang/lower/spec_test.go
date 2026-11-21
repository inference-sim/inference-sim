// Original to blis-sglang.

package lower

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"

	"github.com/inference-sim/inference-sim/engines/sglang/config"
)

func speculate(algorithm string, steps int, multiLayer bool) func(*config.ServerArgs) {
	return func(a *config.ServerArgs) {
		topk, draft := 1, steps+1
		a.SpeculativeAlgorithm, a.SpeculativeEagleTopk = &algorithm, &topk
		a.SpeculativeNumSteps, a.SpeculativeNumDraftTokens = &steps, &draft
		a.EnableMultiLayerEagle = multiLayer
	}
}

// A pool that speculates states its draft passes to the cost model (S30, inference-sim/inference-sim#2355): the chain
// length, which rows the drafter's first pass processes, and which requests the later passes run for.
// The cost model refuses a speculating pool that leaves them unstated, and prices what it is told.
// The EAGLE family lowers alike: the algorithm names the drafter, not the verify path.
func TestASpeculatingPoolStatesItsDraftPasses(t *testing.T) {
	for _, tc := range []struct {
		name       string
		set        func(*config.ServerArgs)
		chain      int
		multiLayer bool
	}{
		{"EAGLE", speculate("EAGLE", 3, false), 3, false},
		{"NEXTN", speculate("NEXTN", 3, false), 3, false},
		{"EAGLE3", speculate("EAGLE3", 2, false), 2, false},
		{"multi-layer EAGLE", speculate("EAGLE", 3, true), 3, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			l := mustLower(t, tc.set)
			d := l.Plan.Drafts
			if d.Chain != tc.chain || d.FirstPassRows != plan.FirstPassScheduled || d.PassRequests != plan.PassRequestsSampling {
				t.Errorf("draft policy %+v, want a chain of %d, first pass over the scheduled rows, later passes for the sampling requests", d, tc.chain)
			}
			if d.TreeTopK != 0 || d.TreeSteps != 0 {
				t.Errorf("a chain states no tree: %+v", d)
			}
			if n := noteNamed(l, "speculative_first_draft_chain"); n == nil {
				t.Error("the first draft chain priced inside the prefill step is a stated approximation, and has no note")
			}
			if n := noteNamed(l, "enable_multi_layer_eagle"); (n != nil) != tc.multiLayer {
				t.Errorf("multi-layer note present=%v, want %v: its under-priced draft passes are stated", n != nil, tc.multiLayer)
			}
		})
	}
	t.Run("no speculation states no drafts", func(t *testing.T) {
		l := mustLower(t, nil)
		if l.Plan.Drafts != (plan.DraftPolicy{}) {
			t.Errorf("a pool that does not speculate states drafts: %+v", l.Plan.Drafts)
		}
		if noteNamed(l, "speculative_first_draft_chain") != nil {
			t.Error("a pool that does not speculate carries the draft notes")
		}
	})
}
