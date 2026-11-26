package lower

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
)

// Qwen3.5-MoE served in bf16 under the TRT-LLM runner was refused as bf16 experts. It is lowered
// with fused routing; another architecture's bf16 experts stay refused.
func TestQwen35Bf16ExpertsUnderTheTRTLLMRunnerAreLowered(t *testing.T) {
	a := r1Args(t)
	a.MoeRunnerBackend = "flashinfer_trtllm"
	g := graph(t, "deepseek-r1-0528")
	g.Global.WeightDType = model.DTypeBF16
	g.Global.Architecture = "Qwen3_5MoeForConditionalGeneration"
	_, routing, err := expertsKernel(&a, g)
	if err != nil || routing != plan.RoutingFused {
		t.Errorf("qwen3.5 bf16: routing %q, err %v; want fused and no refusal", routing, err)
	}
	g.Global.Architecture = "DeepseekV3ForCausalLM"
	if _, _, err := expertsKernel(&a, g); !isNotLowered(err) {
		t.Errorf("deepseek bf16 experts must stay refused, got %v", err)
	}
}
