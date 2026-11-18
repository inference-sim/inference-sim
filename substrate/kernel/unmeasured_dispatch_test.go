package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/load"
)

// A dispatch kernel the registry measures nothing for is priced as the generic routed all-to-all
// and the result says so; a measured one, and the gather-and-scatter dispatch, add no such
// assumption (blis-vllm#75).
func TestAnUnmeasuredDispatchKernelIsPricedAsTheGenericOneAndSaysSo(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	named := func(dispatch plan.KernelImpl) []string {
		in := planCase(t, d, "qwen3-30b-a3b", "h200", 1, "tp: 1, pp: 1, dp: 8, enable_expert_parallel: true",
			func(l *plan.Lowered) { l.Plan.MoE.Dispatch = dispatch })
		var out []string
		for _, a := range mustKernel(t, in).assumptions {
			if strings.HasPrefix(a.Name, "moe_a2a_") {
				out = append(out, a.Name+" | "+a.Scope)
			}
		}
		return out
	}
	for _, lib := range []string{"mori_high_throughput", "nixl_ep", "flashinfer_nvlink_one_sided"} {
		got := named(plan.KernelImpl{Library: lib})
		if len(got) != 1 || !strings.HasPrefix(got[0], "moe_a2a_"+lib+" | generic routed all-to-all") {
			t.Errorf("dispatch %q: assumptions %v, want one naming %q and the generic all-to-all", lib, got, lib)
		}
	}
	if got := named(plan.KernelImpl{}); len(got) != 0 {
		t.Errorf("the gather-and-scatter dispatch records %v, want none", got)
	}
	if got := named(plan.DispatchDeepEPLowLatency); len(got) != 0 {
		t.Errorf("DeepEP low-latency on an H200 node records %v, want none: the registry measures it", got)
	}
}
