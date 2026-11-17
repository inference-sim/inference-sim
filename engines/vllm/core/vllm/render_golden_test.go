// Original to blis-vllm.

package vllm

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
)

// Each golden under testdata/golden-render is vLLM's own decision (oracle/run_render.py runs the
// statements of register_scale_out_api_routers that decide it, unmodified): for the stated
// enable_scale_out, whether the server serves the render endpoints, and the status a render call
// gets. The engine reports the same decision for the same statement (inference-sim#2129).
func TestRenderGoldensAreVLLMsDecision(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join("testdata", "golden-render", "*.json"))
	if err != nil || len(paths) < 3 {
		t.Fatalf("render goldens: %v, %d found; want unset, stated-false and stated-true", err, len(paths))
	}
	saw := map[bool]bool{}
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		var g struct {
			Config struct {
				EnableScaleOut *bool    `json:"enable_scale_out"`
				SupportedTasks []string `json:"supported_tasks"`
			} `json:"config"`
			ServesRender bool `json:"serves_render"`
			RenderStatus int  `json:"render_status"`
		}
		if err := json.Unmarshal(raw, &g); err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		if (g.RenderStatus == 200) != g.ServesRender {
			t.Fatalf("%s: serves_render %v with status %d", p, g.ServesRender, g.RenderStatus)
		}
		src := "block_size: 16\nmax_num_seqs: 8\nmax_num_batched_tokens: 256\nmax_model_len: 256\nnum_gpu_blocks: 64\n" +
			"use_v2_model_runner: false\nasync_scheduling: false\npipeline_parallel_size: 1\n"
		if g.Config.EnableScaleOut != nil {
			src += fmt.Sprintf("enable_scale_out: %t\n", *g.Config.EnableScaleOut)
		}
		var node yaml.Node
		if err := yaml.Unmarshal([]byte(src), &node); err != nil {
			t.Fatal(err)
		}
		cfg, err := decodeConfig(*node.Content[0])
		if err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		st, err := resolve(cfg)
		if err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		e, err := newEngine(st, FixedPricer(100))
		if err != nil {
			t.Fatalf("%s: %v", p, err)
		}
		if got := e.ServesRender(); got != g.ServesRender {
			t.Errorf("%s: the engine serves render calls = %v, vLLM's decision is %v", filepath.Base(p), got, g.ServesRender)
		}
		saw[g.ServesRender] = true
	}
	if !saw[true] || !saw[false] {
		t.Errorf("the goldens cover only %v; want a server that serves render calls and one that does not", saw)
	}
}
