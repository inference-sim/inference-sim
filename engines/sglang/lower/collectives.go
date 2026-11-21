// Original to blis-sglang.

package lower

import (
	"fmt"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"

	"github.com/inference-sim/inference-sim/engines/sglang/config"
)

// emits reports whether SGLang runs a collective whose graph node states condition c, on a pool
// of tp_size tensor-parallel and dp_size data-parallel ranks. The graph names the question and the
// engine answers it (SUB-15); the cost model prices only the collectives the plan says run
// (plan.CollectivePlan.Runs).
//
// This lowering refuses expert parallelism (Args), so no expert group is wider than one rank, and
// it models neither the model's own sequence parallelism nor a sequence-parallel MoE input, so
// neither holds. The MoE then dispatches across the data-parallel ranks whenever there are
// several, as the cost model has always priced it, and a tensor-parallel MoE with one rank
// dispatches nothing.
func emits(tp, dp int, c model.EmitCondition) (bool, error) {
	switch c {
	case model.EmitAlways:
		return true, nil
	case model.EmitTensorParallel, model.EmitTensorParallelUnlessSequenceParallelMoE,
		model.EmitTensorParallelUnlessSequenceParallel, model.EmitTensorParallelWithoutExpertParallel:
		return tp > 1, nil
	case model.EmitExpertParallel:
		return dp > 1, nil
	case model.EmitSequenceParallel, model.EmitExpertParallelWithoutDispatch:
		return false, nil
	}
	return false, fmt.Errorf("the graph states the condition %q, which this engine's lowering "+
		"has no answer for; dropping the node or keeping it would invent a cost", c)
}

// collectiveRuns states, for every collective node of the model (its layer kinds, head,
// embedding and the speculator's embedding and head), whether SGLang runs it.
func collectiveRuns(a *config.ServerArgs, g *model.Graph) ([]plan.CollectiveRun, error) {
	tp, dp := max(a.TpSize, 1), max(a.DpSize, 1)
	var runs []plan.CollectiveRun
	section := func(name string, nodes []model.Node) error {
		for _, n := range nodes {
			if !n.Op.Collective() {
				continue
			}
			r, err := emits(tp, dp, n.Emit)
			if err != nil {
				return fmt.Errorf("%s %s %q: %w", name, n.Op, n.Role, err)
			}
			runs = append(runs, plan.CollectiveRun{Section: name, Op: string(n.Op), Role: n.Role, Runs: r})
		}
		return nil
	}
	for _, lk := range g.LayerKinds {
		if err := section(plan.LayerSection(lk.ID), lk.Nodes); err != nil {
			return nil, err
		}
	}
	parts := []struct {
		name  string
		nodes []model.Node
	}{{plan.SectionHead, g.Head}, {plan.SectionEmbedding, g.Embedding}}
	if sp := g.Speculator; sp != nil {
		parts = append(parts, struct {
			name  string
			nodes []model.Node
		}{plan.SectionDraftEmbedding, sp.Embedding}, struct {
			name  string
			nodes []model.Node
		}{plan.SectionDraftHead, sp.Head}, struct {
			name  string
			nodes []model.Node
		}{plan.SectionDraftSeqHead, sp.SequentialHead})
	}
	for _, s := range parts {
		if err := section(s.name, s.nodes); err != nil {
			return nil, err
		}
	}
	return runs, nil
}
