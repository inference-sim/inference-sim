package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/load"
)

// MiniMax-M2's gate is an fp32 GateLinear, whole on every rank (minimax_m2.py:106-122). Above 32 tokens
// vLLM runs it as bf16x3 on a Blackwell part and as F.linear on the fp32 weight on a Hopper part, which is
// SGEMM on CUDA cores because the engine's float32 matmul precision defaults to "highest"
// (gate_linear.py:116-136 and envs.py:92, v0.33.0 4ba787ec). An 8,192-token prefill step's gate is 12.88
// GFLOP a layer over 62 layers: about 12 ms at the H200's 67 TFLOP/s FP32 rate at full rate, and a few
// milliseconds on a Blackwell part at a third of the BF16 peak. It was priced at the TF32 rate (half the BF16
// peak) on both. The gate's share of the step is the step less the same step on the graph without the router.
func TestFP32RouterGateRunsOnTheKernelVLLMPicks(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	prefill := kernel.Batch{Reqs: []kernel.ReqShape{{Scheduled: 8192, Computed: 0, PromptLen: 8192}}}
	decode := kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: 1, Computed: 4000, PromptLen: 4000}}}
	gate := func(chip string, b kernel.Batch) float64 {
		base := quantCase(t, d, "minimax-m2.5", chip)
		old := base
		old.Model = foldedMoE(t, base.Model)
		kNew, kOld := mustKernel(t, withoutDrafts(base)), mustKernel(t, withoutDrafts(old))
		n, o := kNew.StepTime(b), kOld.StepTime(b)
		return (n.PerResource[kernel.ResourceSM] - o.PerResource[kernel.ResourceSM]).Seconds() * 1e3
	}
	if h := gate("h200", prefill); h < 10 || h > 25 {
		t.Errorf("the fp32 gate adds %.1f ms of compute to an 8,192-token prefill step on H200; SGEMM on CUDA cores is 10-25 ms", h)
	}
	if b := gate("b200", prefill); b <= 0 || b > 2.5 {
		t.Errorf("the fp32 gate adds %.2f ms to the same step on B200; bf16x3 is a few ms at most", b)
	}
	if h := gate("h200", decode); h > 0.1 {
		t.Errorf("the fp32 gate adds %.3f ms of compute to a one-token decode step; the specialised kernel is weight-read bound", h)
	}
}
