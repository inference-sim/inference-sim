// Original to blis-vllm.

package lowering

import (
	"slices"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"

	"github.com/inference-sim/inference-sim/engines/vllm/core/vllmsettings"
)

// Which collectives a layer launches is the engine's lowering of the model (SUB-15): the plan
// says, per collective node of the graph, whether it runs, and the cost model evaluates no
// condition of its own. The graph still names the question each node answers; these cases pin
// the answers.

func TestEmitsAnswersEachConditionFromTheLayout(t *testing.T) {
	tp1 := layout{TP: 1, DP: 1, ExpertWidth: 1, MoEGroupWidth: 1}
	tp2 := layout{TP: 2, DP: 1, ExpertWidth: 1, MoEGroupWidth: 1}
	tp4 := layout{TP: 4, DP: 1, ExpertWidth: 1, MoEGroupWidth: 1}
	tp4dp2 := layout{TP: 4, DP: 2, ExpertWidth: 1, MoEGroupWidth: 8}
	tp4ep := layout{TP: 4, DP: 1, ExpertWidth: 4, MoEGroupWidth: 4}
	tp4dp2ep := layout{TP: 4, DP: 2, ExpertWidth: 8, MoEGroupWidth: 8}
	spMoE := tp4dp2ep
	spMoE.SequenceParallelMoE = true
	spModel := tp4ep
	spModel.ModelSequenceParallel = true
	for _, c := range []struct {
		name string
		lay  layout
		cond model.EmitCondition
		want bool
	}{
		{"always at tp 1", tp1, model.EmitAlways, true},
		{"tensor_parallel at tp 1", tp1, model.EmitTensorParallel, false},
		{"tensor_parallel at tp 2", tp2, model.EmitTensorParallel, true},
		{"tensor_parallel at tp 4", tp4, model.EmitTensorParallel, true},
		{"expert_parallel with one rank", tp1, model.EmitExpertParallel, false},
		{"expert_parallel on a tensor-parallel MoE", tp4, model.EmitExpertParallel, false},
		{"expert_parallel across data-parallel ranks, EP off", tp4dp2, model.EmitExpertParallel, true},
		{"expert_parallel, EP over the tp group only", tp4ep, model.EmitExpertParallel, false},
		{"expert_parallel, EP over tp and dp", tp4dp2ep, model.EmitExpertParallel, true},
		{"expert_parallel, EP over tp with the model's sequence parallelism", spModel, model.EmitExpertParallel, true},
		{"unless_sp_moe without it", tp4dp2ep, model.EmitTensorParallelUnlessSequenceParallelMoE, true},
		{"unless_sp_moe with it", spMoE, model.EmitTensorParallelUnlessSequenceParallelMoE, false},
		{"unless_sp_moe at tp 1", tp1, model.EmitTensorParallelUnlessSequenceParallelMoE, false},
		{"sequence_parallel off", tp4, model.EmitSequenceParallel, false},
		{"sequence_parallel on", spModel, model.EmitSequenceParallel, true},
		{"unless_sequence_parallel off", tp4, model.EmitTensorParallelUnlessSequenceParallel, true},
		{"unless_sequence_parallel on", spModel, model.EmitTensorParallelUnlessSequenceParallel, false},
		{"without_expert_parallel, EP off", tp4, model.EmitTensorParallelWithoutExpertParallel, true},
		{"without_expert_parallel, EP on", tp4ep, model.EmitTensorParallelWithoutExpertParallel, false},
		{"without_expert_parallel at tp 1", tp1, model.EmitTensorParallelWithoutExpertParallel, false},
		{"ep_without_dispatch, EP over tp only", tp4ep, model.EmitExpertParallelWithoutDispatch, true},
		{"ep_without_dispatch, dispatching EP", tp4dp2ep, model.EmitExpertParallelWithoutDispatch, false},
		{"ep_without_dispatch, EP off", tp4, model.EmitExpertParallelWithoutDispatch, false},
	} {
		got, err := c.lay.emits(c.cond)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
		} else if got != c.want {
			t.Errorf("%s: emits(%q) = %t, want %t", c.name, c.cond, got, c.want)
		}
	}
	if _, err := tp4.emits("invented_condition"); err == nil {
		t.Error("a condition the engine has no answer for is refused, not answered false")
	}
}

// Every collective node of every catalog model has exactly one decision in the plan, so the cost
// model never meets a node the engine did not state, and the decision is the node's own
// condition answered under the layout.
func TestEveryCollectiveNodeOfEveryCatalogModelHasADecision(t *testing.T) {
	lay := layout{TP: 4, DP: 2, ExpertWidth: 8, MoEGroupWidth: 8, SequenceParallelMoE: true}
	for _, name := range data.Models() {
		g, err := data.Model(name)
		if err != nil {
			t.Fatal(err)
		}
		runs, err := collectiveRuns(lay, g, g.Speculator, nil, g.LayerKinds)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		type key struct{ section, op, role string }
		have := map[key]bool{}
		for _, r := range runs {
			k := key{r.Section, r.Op, r.Role}
			if _, dup := have[k]; dup {
				t.Errorf("%s: two decisions for %v", name, k)
			}
			have[k] = r.Runs
		}
		want := 0
		check := func(section string, nodes []model.Node) {
			for _, n := range nodes {
				if !n.Op.Collective() {
					continue
				}
				want++
				got, ok := have[key{section, string(n.Op), n.Role}]
				cond, _ := lay.emits(n.Emit)
				if !ok {
					t.Errorf("%s: no decision for %s %q in %s", name, n.Op, n.Role, section)
				} else if got != cond {
					t.Errorf("%s: %s %q in %s runs = %t, its condition %q answers %t", name, n.Op, n.Role, section, got, n.Emit, cond)
				}
			}
		}
		for _, lk := range g.LayerKinds {
			check(plan.LayerSection(lk.ID), lk.Nodes)
		}
		check(plan.SectionHead, g.Head)
		check(plan.SectionEmbedding, g.Embedding)
		if sp := g.Speculator; sp != nil {
			check(plan.SectionDraftEmbedding, sp.Embedding)
			check(plan.SectionDraftHead, sp.Head)
			check(plan.SectionDraftSeqHead, sp.SequentialHead)
		}
		if len(runs) != want {
			t.Errorf("%s: %d decisions for %d collective nodes", name, len(runs), want)
		}
	}
}

// A drafter that is a checkpoint of its own joins the target graph under a prefix, and its
// collectives are decided under that prefix.
func TestADrafterCheckpointsCollectivesAreDecidedUnderTheDraftPrefix(t *testing.T) {
	lay := layout{TP: 2, DP: 1, ExpertWidth: 1, MoEGroupWidth: 1}
	target := &model.Graph{}
	kinds := []model.LayerKind{{ID: "d", Nodes: []model.Node{{Op: model.OpAllReduce, Role: "mlp_out", Emit: model.EmitTensorParallel}}}}
	runs, err := collectiveRuns(lay, target, nil, kinds, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(runs) != 1 || runs[0].Section != plan.LayerSection("draft.d") || !runs[0].Runs {
		t.Errorf("runs = %+v, want one running decision in %q", runs, plan.LayerSection("draft.d"))
	}
}

// The lowering carries the decisions into the plan, and they follow the layout.
func TestLoweringStatesWhichCollectivesRun(t *testing.T) {
	l := lowered(t, dense, vllmsettings.Settings{})
	if len(l.Plan.Collectives.Runs) == 0 {
		t.Fatal("the plan states no collective decisions")
	}
	g, err := data.Model(dense.model)
	if err != nil {
		t.Fatal(err)
	}
	want := 0
	for _, lk := range g.LayerKinds {
		for _, n := range lk.Nodes {
			if n.Op.Collective() {
				want++
			}
		}
	}
	for _, n := range append(slices.Clone(g.Head), g.Embedding...) {
		if n.Op.Collective() {
			want++
		}
	}
	if got := len(l.Plan.Collectives.Runs); got != want {
		t.Errorf("the plan states %d collective decisions for a model with %d collective nodes", got, want)
	}
	any := false
	for _, r := range l.Plan.Collectives.Runs {
		if r.Op == "AllReduce" && r.Role == "mlp_out" {
			any = true
			if !r.Runs {
				t.Errorf("a tensor-parallel dense model does not run its %s %q in %s", r.Op, r.Role, r.Section)
			}
		}
		if r.Op == "All2All" && r.Runs {
			t.Errorf("a dense model at dp 1 runs an all-to-all in %s", r.Section)
		}
	}
	if !any {
		t.Error("no mlp_out all-reduce decision in a dense model's plan")
	}
	one := lowered(t, where{"llama-3.1-70b-instruct", "h100", "tp: 1, pp: 1, dp: 1"}, vllmsettings.Settings{})
	for _, r := range one.Plan.Collectives.Runs {
		if r.Runs {
			t.Errorf("a single-rank pool runs %s %q in %s", r.Op, r.Role, r.Section)
		}
	}
	sp := lowered(t, v41sp, vllmsettings.Settings{})
	if !sp.Plan.Collectives.SequenceParallelModel {
		t.Fatal("model sequence parallelism is off, so the case tests nothing")
	}
	sawGather, sawReduce := false, false
	for _, r := range sp.Plan.Collectives.Runs {
		if r.Op == "AllGather" && r.Role == "attn_in" {
			sawGather = sawGather || r.Runs
		}
		if r.Op == "AllReduce" && r.Role == "attn_out" {
			sawReduce = sawReduce || r.Runs
		}
	}
	if !sawGather || sawReduce {
		t.Errorf("under the model's sequence parallelism attention all-gathers (%t) and no longer all-reduces (%t)", sawGather, sawReduce)
	}
}
