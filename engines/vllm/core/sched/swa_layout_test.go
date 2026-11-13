// Original to blis-vllm.

package sched

import (
	"path/filepath"
	"strings"
	"testing"
)

// swaLayout is what oracle/run_swa_layout.py records beside the trace: vLLM's own resolution of the
// scenario's stated swa_bounded_replay (the statements of DeepseekV4Attention.__init__,
// vllm/models/deepseek_v41/attention.py:529-556@5df63a72) and the layout its SlidingWindowMLASpec
// reports for it.
type swaLayout struct {
	SWALayout struct {
		Stated   *bool `json:"stated"`
		Resolved bool  `json:"resolved_bounded_replay"`
		Runs     []struct {
			BoundedReplay      bool `json:"bounded_replay"`
			PrefixCacheable    bool `json:"prefix_cacheable"`
			PrefixReplayTokens int  `json:"prefix_replay_tokens"`
		} `json:"sliding_window_runs"`
	} `json:"swa_layout"`
}

// TestSWALayoutScenariosAgainstVLLM runs the scenarios that state swa_bounded_replay (and the model
// runner) instead of resolved layers. vLLM resolved each statement into a layout, and its Scheduler
// ran on that layout (testdata/golden-swa/<name>.json, oracle/run_swa_layout.py); the Go scheduler,
// given the layout vLLM resolved, must make the same decisions step by step. The test also holds
// what the golden records of the layout to vLLM's definition (a bounded window is not prefix
// cacheable and replays its window), so the two layouts cannot both be the golden's.
func TestSWALayoutScenariosAgainstVLLM(t *testing.T) {
	scenarios, err := filepath.Glob("../../oracle/scenarios/swa/*.json")
	if err != nil || len(scenarios) == 0 {
		t.Fatalf("no swa_bounded_replay scenarios found (%v)", err)
	}
	for _, path := range scenarios {
		name := strings.TrimSuffix(filepath.Base(path), ".json")
		t.Run(name, func(t *testing.T) {
			var sc scenario
			readJSON(t, path, &sc)
			golden := filepath.Join("testdata", "golden-swa", name+".json")
			var want trace
			readJSON(t, golden, &want)
			var layout swaLayout
			readJSON(t, golden, &layout)
			checkProvenance(t, path, want)
			resolved := layout.SWALayout.Resolved
			if len(layout.SWALayout.Runs) == 0 {
				t.Fatalf("%s records no sliding-window runs", golden)
			}
			for _, r := range layout.SWALayout.Runs {
				if r.BoundedReplay != resolved || r.PrefixCacheable == resolved || (r.PrefixReplayTokens != 0) != resolved {
					t.Errorf("golden run %+v is not the layout of bounded replay %v", r, resolved)
				}
			}
			n := 0
			for i := range sc.Config.KVLayers {
				if sc.Config.KVLayers[i].Spec == "sliding_window_mla" {
					sc.Config.KVLayers[i].BoundedReplay = resolved
					n++
				}
			}
			if n == 0 {
				t.Fatal("the scenario has no sliding_window_mla run")
			}
			if d := earliestDivergence(want, run(t, sc)); d != "" {
				t.Fatalf("earliest divergence from vLLM v0.32.0: %s", d)
			}
		})
	}
}
