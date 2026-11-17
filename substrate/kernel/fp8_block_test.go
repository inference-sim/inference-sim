package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/load"
)

// A checkpoint whose weight scales are one per 128x128 block (16,384 elements; MiniMax-M2.5) runs
// vLLM's block-scaled FP8 linear kernels, which the registry measures as the fp8_block ramp
// (blis-substrate#122). The dense projections must read it, so a weaker fp8_block asymptote makes
// a decode step dearer.
func TestBlockScaledFP8DenseProjectionsArePricedOnTheFP8BlockRamp(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, chip := range []string{"b200", "h200"} {
		t.Run(chip, func(t *testing.T) {
			in := planCase(t, d, "minimax-m2.5", chip, 1, "tp: 4, pp: 1, dp: 1", nil)
			weak := in
			weak.Coefficients = perturbed(in.Coefficients, "gemm_eps_max_fp8_block")
			if a, b := blockDecodeStep(t, weak, 64), blockDecodeStep(t, in, 64); a <= b {
				t.Errorf("a 20%% lower fp8_block asymptote prices 64 decodes at %d ns against %d: the dense projections do not read the fp8_block ramp", a, b)
			}
		})
	}
}

// blockDecodeStep is the step time of n one-token decodes at a 2,048-token context.
func blockDecodeStep(t *testing.T, in Inputs, n int) int64 {
	t.Helper()
	k, err := New(in)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	reqs := make([]kernel.ReqShape, n)
	for i := range reqs {
		reqs[i] = kernel.ReqShape{Scheduled: 1, Computed: 2048, PromptLen: 1024}
	}
	return int64(k.StepTime(kernel.Batch{Reqs: reqs, DecodeThreshold: 1}).NoOverlap)
}
