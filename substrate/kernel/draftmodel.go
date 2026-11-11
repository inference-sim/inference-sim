package latencykernel

import (
	"fmt"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// draftModelName is the drafter the deployment names, or "".
func draftModelName(e enginesettings.Settings) string {
	if e.Speculative == nil {
		return ""
	}
	return e.Speculative.DraftModel
}

// A drafter that is a checkpoint of its own -- Muse Spark's DSpark drafter, an EAGLE3 head --
// is named by the deployment's speculative.draft_model, the engine's speculative_config.model
// (vllm/config/speculative.py:396@5df63a72), and is described by that catalog model's own
// graph, which states itself as a speculator. withDrafter grafts it onto the target graph as
// the target's speculator, so it is planned and priced exactly as a speculator the target
// graph states: its passes per step, its weights and its KV.
//
// The drafter graph's layer kinds join the target's under a prefix, so neither graph's kind
// ids can collide, and its aliases are rewritten to match: a drafter node aliasing its own
// head stays an alias of that node; one aliasing "head.lm_head" names the DRAFTER's head,
// which is the drafter's own weight (Muse Spark's DSpark carries its own lm_head,
// has_own_lm_head, vllm/models/muse_spark/common/dspark.py@0f6b2619), not the target's. A
// drafter that ships no head says shares_target_head, and its alias then stays an alias of the
// target's head, resident once.

// draftKindPrefix keeps a drafter graph's layer kinds apart from the target's.
const draftKindPrefix = "draft."

// withDrafter returns the target graph with the drafter graph's speculator as its own, or the
// target unchanged when draft is nil. A target that already states a speculator, or a drafter
// graph that states none, is refused: either would leave two answers to which drafter runs.
func withDrafter(target, draft *model.Graph) (*model.Graph, error) {
	if draft == nil {
		return target, nil
	}
	if target.Speculator != nil {
		return nil, fmt.Errorf("the deployment names draft_model %s and the target graph %s "+
			"already states a speculator; one drafter runs", draft.Name, target.Name)
	}
	sp := draft.Speculator
	if sp == nil {
		return nil, fmt.Errorf("draft_model %s states no speculator, so its graph does not "+
			"describe a drafter", draft.Name)
	}
	if draft.Global.HiddenSize != target.Global.HiddenSize {
		return nil, fmt.Errorf("draft_model %s has hidden size %d and the target %s %d; the "+
			"drafter reads the target's hidden states", draft.Name, draft.Global.HiddenSize,
			target.Name, target.Global.HiddenSize)
	}
	g := *target
	g.LayerKinds = append([]model.LayerKind{}, target.LayerKinds...)
	rename := func(id string) string { return draftKindPrefix + id }
	// The grafted graph keeps the TARGET's global dtype and scale, so a drafter node that
	// states no weight dtype of its own is stamped with the DRAFTER checkpoint's global ones:
	// the drafter loads its own checkpoint at its own quantization (get_draft_quant_config,
	// vllm/model_executor/models/utils.py:985-1007@v0.32.0), whatever the target's is.
	stamp := func(n *model.Node) {
		if n.Op != model.OpGEMM && n.Op != model.OpGroupedGEMM || n.Aliases != "" ||
			n.WeightDType != "" {
			return
		}
		n.WeightDType = draft.Global.WeightDType
		if n.WeightScale == nil && draft.Global.WeightScale != nil {
			s := *draft.Global.WeightScale
			n.WeightScale = &s
		}
	}
	realias := func(nodes []model.Node) []model.Node {
		out := append([]model.Node(nil), nodes...)
		defer func() {
			// After the aliases are settled: a head alias dropped above is the drafter's own
			// weight, and takes the drafter's dtype too.
			for i := range out {
				stamp(&out[i])
			}
		}()
		for i, n := range out {
			if n.Aliases == "" {
				continue
			}
			j := strings.LastIndex(n.Aliases, ".")
			if j < 0 {
				continue
			}
			scope, role := n.Aliases[:j], n.Aliases[j+1:]
			switch scope {
			case "head":
				// The drafter's own head, grafted below as the speculator's head: the alias
				// points at the target head only where the drafter states none of its own.
				if len(draft.Head) > 0 && !sp.SharesTargetHead {
					out[i].Aliases = ""
				}
			case "embedding":
				out[i].Aliases = "embedding." + role
			default:
				out[i].Aliases = rename(scope) + "." + role
			}
		}
		return out
	}
	for _, lk := range draft.LayerKinds {
		g.LayerKinds = append(g.LayerKinds, model.LayerKind{ID: rename(lk.ID),
			Nodes: realias(lk.Nodes), Edges: lk.Edges})
	}
	renameStack := func(s model.Stack) model.Stack {
		r := model.Stack{Repeat: s.Repeat}
		for _, id := range s.Prologue {
			r.Prologue = append(r.Prologue, rename(id))
		}
		for _, id := range s.Pattern {
			r.Pattern = append(r.Pattern, rename(id))
		}
		for _, id := range s.Epilogue {
			r.Epilogue = append(r.Epilogue, rename(id))
		}
		return r
	}
	if sp.SharesTargetHead {
		for _, n := range sp.Head {
			if strings.HasPrefix(n.Aliases, "head.") {
				t, ok := target.AliasTarget(n.Aliases)
				if !ok {
					return nil, fmt.Errorf("draft_model %s reads its target's %s, and the target %s "+
						"states no such node", draft.Name, n.Aliases, target.Name)
				}
				if n.Op == model.OpGEMM && (t.Op != model.OpGEMM || t.N != n.N || t.K != n.K) {
					return nil, fmt.Errorf("draft_model %s reads its target's %s as a %d x %d head, "+
						"and the target %s holds a %s of %d x %d; the drafter was built for another "+
						"vocabulary or width", draft.Name, n.Aliases, n.N, n.K, target.Name, t.Op, t.N, t.K)
				}
			}
		}
	}
	spec := *sp
	spec.Stack = renameStack(sp.Stack)
	if sp.ContextStack != nil {
		cs := renameStack(*sp.ContextStack)
		spec.ContextStack = &cs
	}
	spec.Embedding = realias(sp.Embedding)
	spec.Head = realias(sp.Head)
	spec.SequentialHead = realias(sp.SequentialHead)
	g.Speculator = &spec
	return &g, nil
}
