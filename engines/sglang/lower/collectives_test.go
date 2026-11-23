// Original to blis-sglang.

package lower

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
)

// The pool states, per collective node of its model, whether SGLang runs it, so the cost model
// evaluates no condition of its own (SUB-15, SUB-6).

func TestEmitsAnswersEachCondition(t *testing.T) {
	for _, c := range []struct {
		cond   model.EmitCondition
		tp, dp int
		want   bool
	}{
		{model.EmitAlways, 1, 1, true},
		{model.EmitTensorParallel, 1, 1, false},
		{model.EmitTensorParallel, 2, 1, true},
		{model.EmitTensorParallelUnlessSequenceParallelMoE, 2, 1, true},
		{model.EmitTensorParallelUnlessSequenceParallel, 2, 1, true},
		{model.EmitTensorParallelWithoutExpertParallel, 2, 1, true},
		{model.EmitTensorParallelWithoutExpertParallel, 1, 1, false},
		{model.EmitExpertParallel, 2, 1, false},
		{model.EmitExpertParallel, 1, 2, true},
		{model.EmitSequenceParallel, 4, 2, false},
		{model.EmitExpertParallelWithoutDispatch, 4, 2, false},
	} {
		got, err := emits(c.tp, c.dp, c.cond)
		if err != nil || got != c.want {
			t.Errorf("emits(tp %d, dp %d, %q) = %t, %v; want %t", c.tp, c.dp, c.cond, got, err, c.want)
		}
	}
	if _, err := emits(2, 1, "invented_condition"); err == nil {
		t.Error("a condition the engine has no answer for is refused, not answered false")
	}
}

func TestEveryCollectiveNodeOfTheModelHasADecision(t *testing.T) {
	for _, tp := range []int{1, 2} {
		l, err := lowerPool(t, "llama-3.1-8b-instruct", tp, "{}")
		if err != nil {
			t.Fatal(err)
		}
		g := graph(t, "llama-3.1-8b-instruct")
		type key struct{ section, op, role string }
		have := map[key]bool{}
		for _, r := range l.Plan.Collectives.Runs {
			have[key{r.Section, r.Op, r.Role}] = r.Runs
		}
		n := 0
		for _, lk := range g.LayerKinds {
			for _, nd := range lk.Nodes {
				if !nd.Op.Collective() {
					continue
				}
				n++
				got, ok := have[key{plan.LayerSection(lk.ID), string(nd.Op), nd.Role}]
				if !ok {
					t.Errorf("tp %d: no decision for %s %q in kind %s", tp, nd.Op, nd.Role, lk.ID)
				}
				if want := tp > 1; ok && nd.Emit != model.EmitAlways && got != want {
					t.Errorf("tp %d: %s %q (condition %q) runs = %t, want %t", tp, nd.Op, nd.Role, nd.Emit, got, want)
				}
			}
		}
		if n == 0 {
			t.Fatal("the model has no collective node, so the case tests nothing")
		}
	}
}
