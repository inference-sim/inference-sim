// Original to blis-sglang.

package lower

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

func noteNamed(l *plan.Lowered, name string) *plan.Note {
	for i := range l.Notes {
		if l.Notes[i].Name == name {
			return &l.Notes[i]
		}
	}
	return nil
}

// A checkpoint SGLang serves under --quantization modelopt_fp4 (graphs that store nvfp4 nodes) is lowered with
// the checkpoint's own formats, no experts kernel named and routing fused into FlashInfer's TRT-LLM MoE, and
// the borrowed grouped-expert ramp is a note that carries the known bias. The speculator's bf16 MTP experts
// are no part of the stack a step prices.
func TestAnNVFP4CheckpointIsLoweredAndTheBorrowedExpertRampIsNamed(t *testing.T) {
	for _, tc := range []struct{ graph, chip string }{
		{"glm-5-nvfp4", "b200"}, {"glm-5-nvfp4", "b300"}, {"deepseek-r1-0528-fp4-v2", "b200"}, {"deepseek-r1-0528-fp4-v2", "b300"},
	} {
		in := PoolInput{Deployment: pool(t, 8, "mem_fraction_static: 0.85\n"), Model: graph(t, tc.graph), Chip: chipFile(t, tc.chip)}
		res, err := Resolve(in)
		if err != nil {
			t.Fatalf("%s on %s: %v", tc.graph, tc.chip, err)
		}
		if res.Args.MoeRunnerBackend != "flashinfer_trtllm" {
			t.Fatalf("%s on %s: SGLang resolves the runner %q", tc.graph, tc.chip, res.Args.MoeRunnerBackend)
		}
		// The recipes state --quantization modelopt_fp4 (config accepts the value separately).
		a := res.Args
		q := "modelopt_fp4"
		a.Quantization = &q
		l, err := Lower(&a, in.Model, in.Chip)
		if err != nil {
			t.Fatalf("%s on %s: %v", tc.graph, tc.chip, err)
		}
		if l.Plan.WeightFormat != "" || l.Plan.MoE.Experts != (plan.KernelImpl{}) || l.Plan.MoE.Routing != plan.RoutingFused {
			t.Errorf("%s on %s: weight format %q, experts %+v, routing %q", tc.graph, tc.chip, l.Plan.WeightFormat, l.Plan.MoE.Experts, l.Plan.MoE.Routing)
		}
		n := noteNamed(l, "nvfp4_experts_ramp")
		if n == nil || !strings.Contains(n.Reason, "+10.4 percent") || !strings.Contains(n.Reason, "+19.8 percent") || !strings.Contains(n.Reason, "borrowed") {
			t.Errorf("%s on %s: the borrowed ramp is not named with its known bias: %+v", tc.graph, tc.chip, n)
		}
		if noteNamed(l, "moe_runner_backend") != nil {
			t.Errorf("%s on %s: the block-fp8 note is on an nvfp4 pool", tc.graph, tc.chip)
		}
	}
}

// modelopt_fp4 on a checkpoint that stores no nvfp4 weights would quantize them on load, which is refused by
// name; so is the TRT-LLM runner on experts of a width no ramp prices.
func TestModeloptFP4OnACheckpointWithoutNVFP4WeightsIsRefusedByName(t *testing.T) {
	in := PoolInput{Deployment: pool(t, 8, "mem_fraction_static: 0.85\n"), Model: graph(t, "deepseek-r1-0528"), Chip: chipFile(t, "b200")}
	res, err := Resolve(in)
	if err != nil {
		t.Fatal(err)
	}
	a := res.Args
	q := "modelopt_fp4"
	a.Quantization = &q
	if _, err := Lower(&a, graph(t, "deepseek-r1-0528"), chipFile(t, "b200")); err == nil ||
		!strings.Contains(err.Error(), "quantization=modelopt_fp4") || !strings.Contains(err.Error(), "stores no nvfp4") {
		t.Errorf("got %v, want a refusal naming quantization=modelopt_fp4 and the missing nvfp4 weights", err)
	}
	// bf16 experts under the TRT-LLM runner stay refused by name.
	a2 := res.Args
	if _, err := Lower(&a2, graphWithExperts(t, "deepseek-r1-0528", model.DTypeBF16), chipFile(t, "b200")); err == nil ||
		!strings.Contains(err.Error(), "moe_runner_backend=flashinfer_trtllm") {
		t.Errorf("bf16 experts: got %v, want a refusal naming moe_runner_backend=flashinfer_trtllm", err)
	}
}

// graphWithExperts is the named graph with its routed experts stored as d.
func graphWithExperts(t *testing.T, name string, d model.DType) *model.Graph {
	t.Helper()
	g := graph(t, name)
	for i := range g.LayerKinds {
		for j := range g.LayerKinds[i].Nodes {
			if n := &g.LayerKinds[i].Nodes[j]; n.Op == model.OpGroupedGEMM && !n.IsSharedExpert() {
				n.WeightDType = d
			}
		}
	}
	return g
}
