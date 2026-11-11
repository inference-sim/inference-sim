package latencykernel

import (
	"fmt"
	"slices"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/units"
	"github.com/inference-sim/blis-substrate/kernel/internal/price"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// loweredOperations turns the pool's lowered operations (Inputs.Operations, plan.PoolPlan's
// Operations()) into what a coefficient is bound to: each operation's primitive, the kernel that
// executes it, and the chain of coefficient classes it is priced from (resolve.Operation), so a
// coefficient is matched against the kernels of the operations that read it alone (design VAL-8,
// 6.2).
//
//   - An attention operation's class is the kind of the node it executes, read from the graph the
//     pool runs: the layer's primary attention, or, for a Scorer operation, the layer's
//     block-index scorer (a node that does not hold the engine's KV). The prefill of a
//     non-latent layer is also classed by the format of the cache it reads (PrimitiveShape.Format),
//     which selects the trtllm-gen lane's pair.
//   - A recurrent operation's class is the mixer its layer runs.
//   - A GEMM's chain is the ramps its weight format is priced from on this chip, in the order the
//     kernel falls back (formatFit and addFit): the format the engine states, not the graph's.
//     A GEMM that states no format reads every ramp, the conservative reading.
//
// An operation that names a layer, or a node, the graph does not have is refused: the plan and
// the graph describe different pools.
func loweredOperations(g *model.Graph, ops []plan.Operation, chip hardware.Chip, family int) ([]resolve.Operation, error) {
	target := g.Stack.Expand()
	var draft []string
	if g.Speculator != nil {
		draft = g.Speculator.Stack.Expand()
	}
	kinds := map[string]model.LayerKind{}
	for _, lk := range g.LayerKinds {
		kinds[lk.ID] = lk
	}
	hopper := chip.NVFP4Peak <= 0
	if family > 0 {
		hopper = family == 9
	}
	out := make([]resolve.Operation, 0, len(ops))
	for i, op := range ops {
		stack, whose := target, "the target graph"
		if op.Draft {
			stack, whose = draft, "the draft model"
		}
		at := fmt.Sprintf("Inputs.Operations[%d] (%s)", i, op.Primitive)
		if op.Layer < 0 || op.Layer >= len(stack) {
			return nil, fmt.Errorf("%s names layer %d of %s, which has %d layers", at, op.Layer, whose, len(stack))
		}
		lk, ok := kinds[stack[op.Layer]]
		if !ok {
			return nil, fmt.Errorf("%s: layer %d of %s is of kind %q, which the graph does not define", at, op.Layer, whose, stack[op.Layer])
		}
		r := resolve.Operation{Primitive: string(op.Primitive), Draft: op.Draft, Scorer: op.Scorer,
			Impl: coefficient.KernelImpl{Library: op.Impl.Library, Variant: op.Impl.Variant, CodeHash: op.Impl.CodeHash}}
		switch op.Primitive {
		case plan.PrimAttentionDecode, plan.PrimAttentionPrefill:
			node, ok := attentionNodeOf(lk, op.Scorer)
			if !ok {
				return nil, fmt.Errorf("%s: layer %d of %s has no %s", at, op.Layer, whose, map[bool]string{false: "attention node", true: "block-index scorer"}[op.Scorer])
			}
			if op.Scorer {
				r.Chain = []string{scorerKernel}
				break
			}
			kind := node.AttentionKind
			if c := string(kind); kind == model.AttentionSWA || kind == model.AttentionMLA || kind == model.AttentionSparseMLA {
				r.Chain = append(r.Chain, c)
			}
			if op.Primitive == plan.PrimAttentionPrefill && !latentAttention(kind) {
				r.NonLatentPrefill, r.CacheDtype = true, op.Dtype
				if w := cacheDtypeWord(op.Dtype); w != "" {
					r.Chain = append(r.Chain, "kv_"+w)
				}
			}
		case plan.PrimRecurrentUpdate:
			for _, n := range lk.Nodes {
				if n.RecurrentKind != "" {
					r.Chain = []string{string(n.RecurrentKind)}
					break
				}
			}
		case plan.PrimGEMM, plan.PrimMoEGroupedGEMM:
			f, ok := priceFormatOf(op.Format)
			if !ok {
				r.Wild = true
				break
			}
			r.Chain = gemmChain(f, op.Primitive == plan.PrimMoEGroupedGEMM, chip, hopper)
		}
		out = append(out, r)
	}
	return out, nil
}

// attentionNodeOf is the attention node of a layer kind an operation of the role executes: the
// primary, which holds the engine's KV (else the first attention node), or, for the scorer,
// the first attention node that does not.
func attentionNodeOf(lk model.LayerKind, scorerNode bool) (model.Node, bool) {
	var first, holder, scorer *model.Node
	for i := range lk.Nodes {
		n := &lk.Nodes[i]
		if n.Op != model.OpAttention {
			continue
		}
		if first == nil {
			first = n
		}
		if n.ReadsEngineKV() {
			if holder == nil {
				holder = n
			}
		} else if scorer == nil {
			scorer = n
		}
	}
	switch {
	case scorerNode && scorer != nil:
		return *scorer, true
	case !scorerNode && holder != nil:
		return *holder, true
	case !scorerNode && first != nil:
		return *first, true
	}
	return model.Node{}, false
}

// cacheDtypeWord names the cache element types the trtllm-gen lane has a pair for, from the
// element type the plan states: the lane was fitted on a bfloat16 cache and on an fp8 one. FP16,
// INT8 and a type the plan does not state are not named, so such a cache is not priced from the
// lane; a storage size alone (units.QuantFormat) cannot tell BF16 from FP16.
func cacheDtypeWord(d string) string {
	switch d {
	case "bf16", "fp8":
		return d
	}
	return ""
}

// priceFormatOf reads a weight's stored format, as the plan states it, as the format the kernel
// prices a GEMM from. A format it cannot name is not stated.
func priceFormatOf(f units.QuantFormat) (price.Format, bool) {
	switch {
	case f.PayloadBits == 32 && f.ScaleBits == 0:
		return price.Format{DType: model.DTypeFP32}, true
	case f.PayloadBits == 16 && f.ScaleBits == 0:
		return price.Format{DType: model.DTypeBF16}, true
	case f.PayloadBits == 8:
		return price.Format{DType: model.DTypeFP8, ScaleElements: f.BlockElements}, true
	case f.PayloadBits == 4 && f.ScaleBits == 0:
		return price.Format{DType: model.DTypeINT4}, true
	case f.PayloadBits == 4 && f.BlockElements == 16:
		return price.Format{DType: model.DTypeNVFP4, ScaleElements: 16}, true
	case f.PayloadBits == 4 && f.BlockElements == 32:
		return price.Format{DType: model.DTypeMXFP4, ScaleElements: 32}, true
	}
	return price.Format{}, false
}

// gemmChain is the ramps a GEMM of the format is priced from, in the kernel's order: the one
// formatFit chooses, then the one addFit falls back to when the registry carries none.
func gemmChain(f price.Format, grouped bool, chip hardware.Chip, hopper bool) []string {
	spec := formatFit(f, grouped, chip, hopper && f.DType == model.DTypeFP8 && f.MX())
	chain := []string{spec.suffix}
	if spec.fallback != "" {
		chain = append(chain, spec.fallback)
		if spec.fallback == "w4a16_mxfp4_grouped" {
			chain = append(chain, "bf16")
		}
	}
	return chain
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
