package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/kernel/internal/price"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
)

// MiniMax-M2's gate is an fp32 GateLinear whole on every rank (minimax_m2.py:106-122). Above 32 tokens
// vLLM runs it as bf16x3 on a Blackwell part and as F.linear on the fp32 weight on a Hopper part, which
// is SGEMM on CUDA cores because the engine's float32 matmul precision defaults to "highest"
// (gate_linear.py:116-136, envs.py:92). The price of an 8,192-token prefill step's gate therefore is
// 12.88 GFLOP a layer at the CUDA-core rate on Hopper (about 190 us at full rate), and three BF16 GEMMs on
// Blackwell; it used to be priced at the TF32 rate (half the BF16 peak) on both.
func TestFP32RouterGateRunsOnTheKernelVLLMPicks(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	gateSeconds := func(chip string) float64 {
		in := planCase(t, d, "minimax-m2.5", chip, 1, "tp: 8, pp: 1, dp: 1", nil)
		in.Draft = nil
		k := mustKernel(t, in)
		var fit gemmFit
		found := false
		for key, f := range k.fits {
			if key.f.DType == model.DTypeFP32 && key.router {
				fit, found = f, true
			}
		}
		if !found {
			t.Fatalf("%s: the fp32 gate has no fit of its own", chip)
		}
		const flops = 2.0 * 8192 * 3072 * 256
		return flops / (fit.peak * fit.epsMax)
	}
	_ = price.Format{}
	if h := gateSeconds("h200") * 1e6; h < 150 || h > 600 {
		t.Errorf("an 8,192-token fp32 gate on H200 takes %.0f us a layer at the fitted ceiling; "+
			"SGEMM on CUDA cores at 67 TFLOP/s is about 190 us a layer at full rate, 300-400 at a typical ramp", h)
	}
	b200, h200 := gateSeconds("b200"), gateSeconds("h200")
	if !(b200 < h200/4) {
		t.Errorf("a Blackwell gate (bf16x3) takes %.0f us and a Hopper one (SGEMM) %.0f us: Blackwell should be several times faster",
			b200*1e6, h200*1e6)
	}
}
