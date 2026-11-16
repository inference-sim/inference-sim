package latencykernel

import (
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// deepseekV4Family reports whether a graph is DeepSeek-V4 or V4.1, whose attention runs on
// that family's own FlashMLA backends rather than the DSA ones. By the architecture the graph names, and nothing else: a 512-wide sparse-MLA
// head is not the family's mark, since GLM-5.3-Flash's sparse layers have one too.
func deepseekV4Family(g *model.Graph) bool {
	switch g.Global.Architecture {
	case "DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM":
		return true
	}
	return false
}

// hopper reports whether the chip is the SM90 family, which decides the kernels vLLM
// selects for several formats and layouts. The catalog states the compute capability; for a
// chip that does not, it is inferred -- data-center Blackwell by native NVFP4, everything
// else as Hopper -- and that inference is recorded the first time it decides anything.
func (k *Kernel) hopper() bool {
	if k.family > 0 {
		return k.family == 9
	}
	h := k.chip.NVFP4Peak <= 0
	if !k.familyAssumed {
		k.familyAssumed = true
		guess := "sm100"
		if h {
			guess = "sm90"
		}
		k.assume("compute_capability", guess,
			"the catalog states no compute capability for "+k.chip.Name+", so its generation "+
				"is inferred from its native-format peaks, which decide kernel "+
				"selection")
	}
	return h
}
