package kernelmodel

import (
	"testing"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	"github.com/inference-sim/blis-schemas/spec/model"
)

// NIXL's _prefill_backoff (nixl/base_scheduler.py:385-398 at v0.31.0): 1 for a Mamba-state
// model, K-1 for multi-module MTP drafting K, and 0 for every other drafter -- DSpark included,
// whose num_prefill_lookahead_tokens is 1 (config/vllm.py:657-678).
func TestPrefillBackoffOf(t *testing.T) {
	recurrent := &model.Graph{LayerKinds: []model.LayerKind{{Nodes: []model.Node{{Op: model.OpRecurrentUpdate}}}}}
	attention := &model.Graph{LayerKinds: []model.LayerKind{{Nodes: []model.Node{{Op: model.OpGEMM}}}}}
	drafter := func(layers int) *model.Graph {
		g := *attention
		g.Speculator = &model.Speculator{Stack: model.Stack{Pattern: []string{"draft"}, Repeat: layers}}
		return &g
	}
	spec := func(method string, k int) *deployment.Speculative {
		return &deployment.Speculative{Method: method, NumSpecTokens: k}
	}
	for _, tc := range []struct {
		name string
		g    *model.Graph
		sp   *deployment.Speculative
		want int
	}{
		{"attention, no drafter", attention, nil, 0},
		{"recurrent state", recurrent, nil, 1},
		{"DSpark K=5 over 3 draft layers", drafter(3), spec("dspark", 5), 0},
		{"single-module MTP K=3", drafter(1), spec("mtp", 3), 0},
		{"multi-module MTP K=3 over 3 layers", drafter(3), spec("mtp", 3), 2},
		{"multi-module MTP K=2 over 3 layers", drafter(3), spec("mtp", 2), 1},
		{"MTP K=1", drafter(3), spec("mtp", 1), 0},
	} {
		if got := prefillBackoffOf(tc.g, tc.sp); got != tc.want {
			t.Errorf("%s: backoff %d, want %d", tc.name, got, tc.want)
		}
	}
}
