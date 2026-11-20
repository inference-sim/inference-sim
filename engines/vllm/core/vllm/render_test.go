// Original to blis-vllm.

package vllm

import (
	"fmt"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// A data-parallel group is one server: it states whether that server answers the router's render
// calls, as a single engine does (inference-sim#2129). vLLM answers them only with
// --enable-scale-out (vllm/entrypoints/scale_out/factories.py).
func TestDPGroupStatesWhetherItsServerAnswersRenderCalls(t *testing.T) {
	for _, stated := range []string{"", "enable_scale_out: false\n", "enable_scale_out: true\n"} {
		want := stated == "enable_scale_out: true\n"
		src := fmt.Sprintf("block_size: 16\nmax_num_seqs: 8\nmax_num_batched_tokens: 256\nmax_model_len: 256\n"+
			"num_gpu_blocks: 64\nuse_v2_model_runner: false\nasync_scheduling: false\npipeline_parallel_size: 1\n"+
			"data_parallel_size: 2\n%s", stated)
		var node yaml.Node
		if err := yaml.Unmarshal([]byte(src), &node); err != nil {
			t.Fatal(err)
		}
		cfg, err := decodeConfig(*node.Content[0])
		if err != nil {
			t.Fatal(err)
		}
		st, err := resolve(cfg)
		if err != nil {
			t.Fatal(err)
		}
		g, err := newDPGroup(st, FixedPricer(100))
		if err != nil {
			t.Fatal(err)
		}
		r, ok := engine.Engine(g).(engine.RenderServing)
		if !ok {
			t.Fatalf("%T does not state whether it serves render calls", g)
		}
		if got := r.ServesRender(); got != want {
			t.Errorf("stated %q: ServesRender() = %v, want %v", stated, got, want)
		}
	}
}
