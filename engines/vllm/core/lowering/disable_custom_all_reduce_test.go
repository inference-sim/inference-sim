// Original to blis-vllm.

package lowering

import (
	"testing"

	vllmrules "github.com/inference-sim/inference-sim/engines/vllm/core/rulepack"
	"github.com/inference-sim/inference-sim/engines/vllm/core/vllmsettings"
	"github.com/inference-sim/inference-sim/engines/vllm/internal/upstream"
)

// disable_custom_all_reduce is vLLM's own switch for the custom all-reduce kernel
// (ParallelConfig.disable_custom_all_reduce, vllm/config/parallel.py), and the lowering honours it: true
// moves the reduction from the SM kernel to NCCL, false and unstated keep the kernel where the layout
// supports it, and a stated false does not override a layout that declines it.
func TestDisableCustomAllReduceSwitchesTheKernel(t *testing.T) {
	yes, no := true, false
	rules := vllmrules.Lookup(upstream.Version)
	for _, c := range []struct {
		what    string
		disable *bool
		tp      int
		want    bool
	}{
		{"unstated, tp 8", nil, 8, true},
		{"stated false, tp 8", &no, 8, true},
		{"stated true, tp 8", &yes, 8, false},
		{"stated false, unsupported tp 3", &no, 3, false},
	} {
		t.Run(c.what, func(t *testing.T) {
			got, _ := customAllReduce(vllmsettings.Settings{DisableCustomAllReduce: c.disable},
				layout{TP: c.tp, DP: 1, GPUsPerNode: 8}, false, rules)
			if got != c.want {
				t.Errorf("custom all-reduce %v, want %v", got, c.want)
			}
		})
	}
}
