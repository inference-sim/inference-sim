package lowering

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"

	"github.com/inference-sim/blis-vllm/core/vllmsettings"
)

// vLLM v0.33.0 runs 128-block FP8 routed experts on SM90 with Triton's fused experts at
// expert-parallel width 1 and FlashInfer's CUTLASS experts above it, unless moe_backend says
// otherwise (fused_moe/oracle/fp8.py:79-130, :274-282@4ba787ec). The plan states the kernel it
// runs, with the build kernel_impls.json lists for the part; on another part or weight format it
// states none.
func TestThePlanStatesTheExpertsKernelVLLMRuns(t *testing.T) {
	tp4 := "tp: 4, pp: 1, dp: 1"
	ep4 := "tp: 1, pp: 1, dp: 4, enable_expert_parallel: true"
	for _, c := range []struct {
		model, chip, parallel, stated string
		want                          plan.KernelImpl
	}{
		{"minimax-m2.5", "h200", tp4, "", plan.ExpertsTriton},
		{"minimax-m2.5", "h200", tp4, "auto", plan.ExpertsTriton},
		{"minimax-m2.5", "h200", ep4, "", plan.ExpertsFlashInferCutlass},
		{"minimax-m2.5", "h200", ep4, "triton", plan.ExpertsTriton},
		{"minimax-m2.5", "h100", tp4, "Flashinfer-Cutlass", plan.ExpertsFlashInferCutlass},
		{"minimax-m2.5", "h200", tp4, "deep_gemm", plan.KernelImpl{Library: "deep_gemm"}},
		{"minimax-m2.5", "b200", tp4, "triton", plan.KernelImpl{}},
		{"gpt-oss-120b", "h200", tp4, "", plan.KernelImpl{}},
	} {
		l, err := tryLower(t, where{c.model, c.chip, c.parallel}, vllmsettings.Settings{MoEBackend: c.stated})
		if err != nil {
			t.Errorf("%s %s %s moe_backend %q: %v", c.model, c.chip, c.parallel, c.stated, err)
			continue
		}
		got := l.Plan.MoE.Experts
		if got.Variant != c.want.Variant || (c.want.Library != "" && got.Library != c.want.Library) {
			t.Errorf("%s %s %s moe_backend %q: experts %+v, want %+v", c.model, c.chip, c.parallel, c.stated, got, c.want)
		}
		// A kernel the release's table lists is stated with its build, the key its coefficients
		// are bound to (VAL-8); the others name no build.
		if listed := c.want.Variant != ""; listed != (got.CodeHash != "" && got.Toolkit != "") {
			t.Errorf("%s %s moe_backend %q: experts %+v, want a build stated: %t", c.model, c.chip, c.stated, got, listed)
		}
	}
}
