package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/load"
)

// vLLM shards the routed experts' intermediate width across the tensor-parallel ranks
// (intermediate_size_per_partition = intermediate_size // tp_size), so a rank's routed GEMM work is its slice's on
// every ramp, not only on the weight-only and rank-own ones. A compute-bound prefill of a mostly-MoE model on a
// ramp that carried no division (MiniMax-M2.5 block-FP8 on H200 with the plan stating no experts kernel, priced on
// the fp8 ramp) therefore spends well under 0.7 of its tp-4 SM time at tp 8 (the attention and the dense parts that do not
// shrink with tp keep it above a half); with the routed FLOPs charged whole on every rank the routed part does not
// shrink at all and the ratio stays near 0.9.
func TestRoutedExpertFLOPsDivideByTensorParallelOnEveryRamp(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	prefill := kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: 8192, Computed: 0, PromptLen: 8192}}}
	sm := func(tp string) float64 {
		k := mustKernel(t, planCase(t, d, "minimax-m2.5", "h200", 1, "tp: "+tp+", pp: 1, dp: 1", func(l *plan.Lowered) {
			l.Plan.MoE.Experts = plan.KernelImpl{}
		}))
		return k.StepTime(prefill).PerResource[kernel.ResourceSM].Seconds()
	}
	r := sm("8") / sm("4")
	t.Logf("SM time at tp 8 over tp 4: %.3f", r)
	if r > 0.7 {
		t.Errorf("a compute-bound prefill spends %.3f of its tp-4 SM time at tp 8; with every routed FLOP divided by tp it is under 0.7", r)
	}
}
