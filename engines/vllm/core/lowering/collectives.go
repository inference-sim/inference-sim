// Original to blis-vllm.

package lowering

import (
	"fmt"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
)

// emits reports whether the engine runs a collective whose graph node states condition c under
// this layout. The graph names the question and the engine answers it: which collectives vLLM's
// layers launch is the engine's lowering of the model (SUB-15), and the cost model prices only
// the ones the plan says run (plan.CollectivePlan.Runs). The answers were carried over from
// blis-substrate's layout resolution, which evaluated these words in the cost model.
func (l layout) emits(c model.EmitCondition) (bool, error) {
	switch c {
	case model.EmitAlways:
		return true, nil
	case model.EmitTensorParallel:
		return l.TP > 1, nil
	case model.EmitExpertParallel:
		// The MoE dispatch and combine: the engine runs them only where moeDispatches says.
		return l.moeDispatches(), nil
	case model.EmitTensorParallelUnlessSequenceParallelMoE:
		return l.TP > 1 && !l.SequenceParallelMoE, nil
	case model.EmitSequenceParallel:
		return l.ModelSequenceParallel, nil
	case model.EmitTensorParallelUnlessSequenceParallel:
		return l.TP > 1 && !l.ModelSequenceParallel, nil
	case model.EmitTensorParallelWithoutExpertParallel:
		// The experts are sharded by tensor parallelism only while expert parallelism is off;
		// under it every expert is whole on one rank (FusedMoEParallelConfig.make,
		// vllm/model_executor/layers/fused_moe/config.py: ep_size = tp_size, tp_size = 1).
		return l.TP > 1 && l.ExpertWidth <= 1, nil
	case model.EmitExpertParallelWithoutDispatch:
		// Expert parallelism over a group that holds every token: no all-to-all moves them, so
		// each rank's partial expert output is all-reduced across the tensor-parallel group.
		return l.ExpertWidth > 1 && l.TP > 1 && !l.moeDispatches(), nil
	}
	return false, fmt.Errorf("the graph states the condition %q, which this engine's lowering "+
		"has no answer for; dropping the node or keeping it would invent a cost", c)
}

// collectiveRuns states, for every collective node of the model, whether the engine runs it:
// the target graph's layer kinds, head and embedding, and the speculator's embedding and head.
// A drafter that is a checkpoint of its own joins under plan.DraftKindPrefix, as the cost model
// grafts it, and the target's own speculator-only kinds leave with the speculator it replaces.
func collectiveRuns(lay layout, g *model.Graph, sp *model.Speculator, draftKinds []model.LayerKind,
	targetKinds []model.LayerKind) ([]plan.CollectiveRun, error) {
	var runs []plan.CollectiveRun
	section := func(name string, nodes []model.Node) error {
		for _, n := range nodes {
			if !n.Op.Collective() {
				continue
			}
			r, err := lay.emits(n.Emit)
			if err != nil {
				return fmt.Errorf("%s %s %q: %w", name, n.Op, n.Role, err)
			}
			runs = append(runs, plan.CollectiveRun{Section: name, Op: string(n.Op), Role: n.Role, Runs: r})
		}
		return nil
	}
	for _, lk := range targetKinds {
		if err := section(plan.LayerSection(lk.ID), lk.Nodes); err != nil {
			return nil, err
		}
	}
	for _, lk := range draftKinds {
		if err := section(plan.LayerSection(plan.DraftKindPrefix+lk.ID), lk.Nodes); err != nil {
			return nil, err
		}
	}
	if err := section(plan.SectionHead, g.Head); err != nil {
		return nil, err
	}
	if err := section(plan.SectionEmbedding, g.Embedding); err != nil {
		return nil, err
	}
	if sp != nil {
		for _, s := range []struct {
			name  string
			nodes []model.Node
		}{{plan.SectionDraftEmbedding, sp.Embedding}, {plan.SectionDraftHead, sp.Head},
			{plan.SectionDraftSeqHead, sp.SequentialHead}} {
			if err := section(s.name, s.nodes); err != nil {
				return nil, err
			}
		}
	}
	return runs, nil
}
