// Original to blis-sglang.

package lower

import (
	"errors"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"

	"github.com/inference-sim/blis-sglang/config"
)

func r1Args(t *testing.T) config.ServerArgs {
	t.Helper()
	res, err := Resolve(PoolInput{Deployment: pool(t, 8, "mem_fraction_static: 0.82\n"), Model: graph(t, "deepseek-r1-0528"), Chip: h200(t)})
	if err != nil {
		t.Fatal(err)
	}
	return res.Args
}

func lowerWith(t *testing.T, mutate func(*config.ServerArgs)) *plan.Lowered {
	t.Helper()
	a := r1Args(t)
	if mutate != nil {
		mutate(&a)
	}
	l, err := Lower(&a, graph(t, "deepseek-r1-0528"), h200(t))
	if err != nil {
		t.Fatal(err)
	}
	return l
}

func isNotLowered(err error) bool {
	var nl *NotLowered
	return errors.As(err, &nl)
}

// moe_runner_backend flashinfer_trtllm on block-fp8 experts is lowered with no experts kernel named (the
// substrate prices FlashInfer's TRT-LLM grouped GEMM from the weight format) and expert selection fused
// into the kernel; the substitution is a note on the plan. On other experts it is refused by name.
func TestTheTRTLLMExpertsRunnerIsLoweredWithoutANamedKernel(t *testing.T) {
	l := lowerWith(t, func(a *config.ServerArgs) { a.MoeRunnerBackend = "flashinfer_trtllm" })
	if l.Plan.MoE.Experts != (plan.KernelImpl{}) || l.Plan.MoE.Routing != plan.RoutingFused {
		t.Errorf("experts %+v routing %q, want no named kernel and fused routing", l.Plan.MoE.Experts, l.Plan.MoE.Routing)
	}
	noted := false
	for _, n := range l.Notes {
		noted = noted || (n.Name == "moe_runner_backend" && n.Value == "flashinfer_trtllm")
	}
	if !noted {
		t.Error("the TRT-LLM substitution is not a note on the plan")
	}
	// The default runner states Triton's kernel and separate routing, and no such note.
	d := lowerWith(t, nil)
	if d.Plan.MoE.Experts != plan.ExpertsTriton || d.Plan.MoE.Routing != plan.RoutingSeparate {
		t.Errorf("auto: experts %+v routing %q", d.Plan.MoE.Experts, d.Plan.MoE.Routing)
	}
	a := r1Args(t)
	a.MoeRunnerBackend = "flashinfer_trtllm"
	bf16 := graph(t, "deepseek-r1-0528")
	bf16.Global.WeightDType = model.DTypeBF16
	if _, err := Lower(&a, bf16, h200(t)); !isNotLowered(err) {
		t.Errorf("flashinfer_trtllm on bf16 experts must be refused by name, got %v", err)
	}
	a.MoeRunnerBackend = "flashinfer_cutedsl"
	if _, err := Lower(&a, graph(t, "deepseek-r1-0528"), h200(t)); !isNotLowered(err) {
		t.Errorf("another runner must stay refused by name, got %v", err)
	}
}
