package latencykernel

import (
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/load"
)

// Block-FP8 routed experts on a Blackwell part run FlashInfer's TRT-LLM grouped kernel, which has a per-layer
// device floor the dense FP8 ramp does not carry: AISimulate's table of it prices MiniMax-M2.5's routed layer at
// 28 us for one token at tp 4 and 160 us from 32 tokens, where the dense ramp with the tensor-parallel division
// priced 7 us and 97 us. A decode step of that model on B200 therefore costs more than the same step with the
// plan stating Triton experts (which the registry has no Blackwell ramp for, so it is priced on the fp8 ramp),
// by about a floor per layer at one token.
func TestBlackwellBlockFP8ExpertsCarryTheirKernelsFloor(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	decodes := func(n int) kernel.Batch {
		b := kernel.Batch{DecodeThreshold: 1}
		for range n {
			b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: 1, Computed: 1024, PromptLen: 1024})
		}
		return b
	}
	step := func(chip string, n int, experts plan.KernelImpl) (time.Duration, int) {
		k := mustKernel(t, planCase(t, d, "minimax-m2.5", chip, 1, "tp: 4, pp: 1, dp: 1", func(l *plan.Lowered) {
			l.Plan.MoE.Experts = experts
			l.Plan.Batch.Requests = 64
			l.Plan.Graphs.Ceiling = 64
			l.Plan.Graphs.Buckets = []int{1, 2, 4, 8, 16, 32, 64}
			l.Plan.Graphs.Decode = plan.LaunchGraph
		}))
		return k.StepTime(decodes(n)).NoOverlap, k.plan.TotalLayers
	}
	for _, chip := range []string{"b200", "b300", "gb200-nvl72", "gb300"} {
		stated, layers := step(chip, 1, plan.ExpertsTriton)
		def, _ := step(chip, 1, plan.KernelImpl{})
		gain := def - stated
		t.Logf("%s one decode: %v with Triton stated, %v by default (%d layers, +%v)", chip, stated, def, layers, gain)
		if min := time.Duration(layers) * 15 * time.Microsecond; gain < min {
			t.Errorf("%s: the default block-FP8 experts cost %v more than the fp8-ramp pricing; a floor of about 30us a layer is at least %v", chip, gain, min)
		}
	}
}
