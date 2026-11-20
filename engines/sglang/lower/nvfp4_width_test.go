// Original to blis-sglang.

package lower

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
)

// stackOf is a graph whose one layer kind, in the main stack, holds the nodes.
func stackOf(global model.DType, nodes ...model.Node) *model.Graph {
	g := &model.Graph{}
	g.Global.WeightDType = global
	g.LayerKinds = []model.LayerKind{{ID: "layer", Nodes: nodes}}
	g.Stack.Pattern = []string{"layer"}
	return g
}

func node(op model.Op, w model.DType) model.Node { return model.Node{Op: op, WeightDType: w} }

// A checkpoint stores NVFP4 weights when a GEMM or a grouped GEMM of its main stack says so; another
// dtype, or another op that carries the dtype, does not.
func TestAGraphStoresNVFP4OnlyInAGEMMOrAGroupedGEMMOfItsMainStack(t *testing.T) {
	for _, tc := range []struct {
		name string
		g    *model.Graph
		want bool
	}{
		{"a GEMM in nvfp4", stackOf(model.DTypeBF16, node(model.OpGEMM, model.DTypeNVFP4)), true},
		{"a grouped GEMM in nvfp4", stackOf(model.DTypeBF16, node(model.OpGroupedGEMM, model.DTypeNVFP4)), true},
		{"a GEMM in fp8", stackOf(model.DTypeBF16, node(model.OpGEMM, model.DTypeFP8)), false},
		{"a grouped GEMM in bf16", stackOf(model.DTypeBF16, node(model.OpGroupedGEMM, model.DTypeBF16)), false},
		{"an attention node stating nvfp4", stackOf(model.DTypeBF16, node(model.OpAttention, model.DTypeNVFP4)), false},
		{"an elementwise node stating nvfp4", stackOf(model.DTypeBF16, node(model.OpElementwise, model.DTypeNVFP4)), false},
		{"no node", stackOf(model.DTypeBF16), false},
	} {
		if got := storesNVFP4(tc.g); got != tc.want {
			t.Errorf("%s: storesNVFP4 = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// The routed experts' width is the one width every routed grouped GEMM of the main stack states (the
// checkpoint's where a node states none); experts of two widths, or none, have no width a kernel is
// lowered for.
func TestTheRoutedExpertsWidthIsTheOneEveryRoutedExpertStates(t *testing.T) {
	routed := func(w model.DType) model.Node { return node(model.OpGroupedGEMM, w) }
	shared := true
	sharedExpert := model.Node{Op: model.OpGroupedGEMM, WeightDType: model.DTypeBF16, Shared: &shared}
	for _, tc := range []struct {
		name string
		g    *model.Graph
		want model.DType
	}{
		{"one routed node", stackOf(model.DTypeBF16, routed(model.DTypeFP8)), model.DTypeFP8},
		{"two routed nodes of one width", stackOf(model.DTypeBF16, routed(model.DTypeNVFP4), routed(model.DTypeNVFP4)), model.DTypeNVFP4},
		{"two routed nodes of two widths", stackOf(model.DTypeBF16, routed(model.DTypeFP8), routed(model.DTypeNVFP4)), ""},
		{"three routed nodes, the last unlike the others", stackOf(model.DTypeBF16, routed(model.DTypeFP8), routed(model.DTypeFP8), routed(model.DTypeNVFP4)), ""},
		{"three routed nodes, the first unlike the others", stackOf(model.DTypeBF16, routed(model.DTypeNVFP4), routed(model.DTypeFP8), routed(model.DTypeFP8)), ""},
		{"a node stating none takes the checkpoint's", stackOf(model.DTypeFP8, routed("")), model.DTypeFP8},
		{"a node stating none beside one stating the checkpoint's", stackOf(model.DTypeFP8, routed(""), routed(model.DTypeFP8)), model.DTypeFP8},
		{"a shared expert is not routed", stackOf(model.DTypeBF16, routed(model.DTypeFP8), sharedExpert), model.DTypeFP8},
		{"no routed node", stackOf(model.DTypeBF16, node(model.OpGEMM, model.DTypeFP8)), ""},
	} {
		if got := routedExpertsDType(tc.g); got != tc.want {
			t.Errorf("%s: routedExpertsDType = %q, want %q", tc.name, got, tc.want)
		}
	}
}
