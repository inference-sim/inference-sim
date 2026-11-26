package latencykernel

import (
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"
	"github.com/inference-sim/inference-sim/substrate/load"
)

// NVFP4 and BF16 routed experts on a Blackwell part run FlashInfer's TRT-LLM fused-MoE kernel, which has a
// ramp and a per-layer device floor of its own (gemm_*_nvfp4_grouped, gemm_*_bf16_grouped) that the dense
// ramps of those formats do not carry: AISimulate's table of the kernel reads 25 to 35 us for one token
// where the dense ramp, under the same composition, priced a few. The kernel must find those entries on every
// Blackwell part (a missing one is recorded as an assumption and prices on the dense ramp), and a one-token
// step must cost at least one floor.
func TestBlackwellNVFP4AndBF16ExpertsReadTheirOwnRampAndFloor(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	one := kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: 1, Computed: 1024, PromptLen: 1024}}}
	for _, c := range []struct{ model, suffix string }{
		{"minimax-m2.5-nvfp4", "nvfp4_grouped"},
		{"nemotron-3-super-120b-a12b-bf16", "bf16_grouped"},
	} {
		for _, chip := range []string{"b200", "b300", "gb200-nvl72", "gb300"} {
			k := mustKernel(t, planCase(t, d, c.model, chip, 1, "tp: 4, pp: 1, dp: 1", nil))
			for _, a := range k.assumptions {
				if strings.Contains(a.Name, c.suffix) {
					t.Errorf("%s on %s fell back from the %s ramp: %s %s", c.model, chip, c.suffix, a.Name, a.Scope)
				}
			}
			if got := k.StepTime(one).NoOverlap; got < 25*time.Microsecond {
				t.Errorf("%s on %s: a one-token step takes %v, under one grouped floor", c.model, chip, got)
			}
		}
	}
}
