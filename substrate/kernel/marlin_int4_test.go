package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/load"
)

// Kimi-K2.5's routed experts are compressed-tensors int4 (W4A16). vLLM runs them on the Marlin
// W4A16 grouped kernel, so the routed term is priced on the registry's measured Marlin ramp
// (gemm_*_w4a16_mxfp4_grouped) and not on the BF16 dense ramp (blis-substrate#118). A
// weaker Marlin asymptote must therefore make a decode step dearer, at a batch large enough
// for the routed GEMM to be compute-bound.
func TestINT4RoutedExpertsArePricedOnTheMarlinRamp(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	in := planCase(t, d, "kimi-k2.5", "h200", 1, "tp: 8, pp: 1, dp: 1", nil)
	weak := in
	weak.Coefficients = perturbed(in.Coefficients, "gemm_eps_max_w4a16_mxfp4_grouped")
	for _, n := range []int{16, 64} {
		if a, b := decodeStep(t, weak, n), decodeStep(t, in, n); a <= b {
			t.Errorf("%d decodes: a 20%% lower Marlin asymptote prices the step at %d ns against %d: the int4 experts do not read the Marlin ramp", n, a, b)
		}
	}
}
