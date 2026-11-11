package latencykernel

import (
	"fmt"
	"slices"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// loweredOperations turns the pool's lowered operations (Inputs.Operations, plan.PoolPlan's
// Operations()) into what a coefficient is bound to: each operation's primitive, the kernel that
// executes it, and the classes of work its layer does within the primitive, read from the graph
// the pool runs (the attention kinds the layer has, the recurrent mixer it runs). A coefficient
// whose name states a class (coefficient.ClassOf) is then matched against the kernels of the
// operations that consume it alone, so a kernel that runs only the sliding-window layers does
// not decide the price of the full-attention ones (design VAL-8, 6.2). An operation that names a
// layer the graph does not have is refused: the plan and the graph describe different pools.
func loweredOperations(g *model.Graph, ops []plan.Operation) ([]resolve.Operation, error) {
	target := g.Stack.Expand()
	var draft []string
	if g.Speculator != nil {
		draft = g.Speculator.Stack.Expand()
	}
	kinds := map[string]model.LayerKind{}
	for _, lk := range g.LayerKinds {
		kinds[lk.ID] = lk
	}
	out := make([]resolve.Operation, 0, len(ops))
	for i, op := range ops {
		stack, whose := target, "the target graph"
		if op.Draft {
			stack, whose = draft, "the draft model"
		}
		if op.Layer < 0 || op.Layer >= len(stack) {
			return nil, fmt.Errorf("Inputs.Operations[%d] (%s) names layer %d of %s, which has %d layers",
				i, op.Primitive, op.Layer, whose, len(stack))
		}
		lk, ok := kinds[stack[op.Layer]]
		if !ok {
			return nil, fmt.Errorf("Inputs.Operations[%d] (%s): layer %d of %s is of kind %q, which the graph does not define",
				i, op.Primitive, op.Layer, whose, stack[op.Layer])
		}
		r := resolve.Operation{Primitive: string(op.Primitive), Draft: op.Draft,
			Impl: coefficient.KernelImpl{Library: op.Impl.Library, Variant: op.Impl.Variant, CodeHash: op.Impl.CodeHash}}
		switch op.Primitive {
		case plan.PrimAttentionDecode, plan.PrimAttentionPrefill:
			for _, n := range lk.Nodes {
				if n.Op == model.OpAttention && n.AttentionKind != "" && !slices.Contains(r.Classes, string(n.AttentionKind)) {
					r.Classes = append(r.Classes, string(n.AttentionKind))
				}
			}
		case plan.PrimRecurrentUpdate:
			for _, n := range lk.Nodes {
				if n.RecurrentKind != "" && !slices.Contains(r.Classes, string(n.RecurrentKind)) {
					r.Classes = append(r.Classes, string(n.RecurrentKind))
				}
			}
		}
		out = append(out, r)
	}
	return out, nil
}

// sameImplementations reports whether two by-primitive lists hold the same implementations.
func sameImplementations(a, b map[string][]coefficient.KernelImpl) bool {
	if len(a) != len(b) {
		return false
	}
	for p, x := range a {
		y, ok := b[p]
		if !ok || len(x) != len(y) {
			return false
		}
		sx, sy := slices.Clone(x), slices.Clone(y)
		less := func(u, v coefficient.KernelImpl) int {
			return slices.Compare([]string{u.Library, u.Variant, u.CodeHash}, []string{v.Library, v.Variant, v.CodeHash})
		}
		slices.SortFunc(sx, less)
		slices.SortFunc(sy, less)
		if !slices.Equal(sx, sy) {
			return false
		}
	}
	return true
}
