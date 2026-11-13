// Original to blis-vllm.

package kernelmodel

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	latencykernel "github.com/inference-sim/blis-substrate/kernel"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// The scenarios under oracle/scenarios/swa state swa_bounded_replay (and the model runner) and
// leave vLLM to resolve the DeepSeek-V4.1 sliding-window caches' layout; core/sched/testdata/
// golden-swa/<name>.json records what vLLM's own statements resolved it to. The pool of the same
// statement must lay its sliding-window runs out the same way, so ignoring or reversing the
// engine's field before KVLayers is emitted fails here. The V1 runner is stated to vLLM as
// use_v2_model_runner false and to the pool as dual batch overlap, a feature vLLM's V2 runner does
// not support (config/vllm.py:3094-3095@5df63a72), which ModelRunner resolves to "v1".
func TestSWALayoutFollowsTheStatedFieldAsVLLMResolvesIt(t *testing.T) {
	scenarios, err := filepath.Glob("../../oracle/scenarios/swa/*.json")
	if err != nil || len(scenarios) == 0 {
		t.Fatalf("no swa_bounded_replay scenarios (%v)", err)
	}
	for _, path := range scenarios {
		name := strings.TrimSuffix(filepath.Base(path), ".json")
		t.Run(name, func(t *testing.T) {
			var sc struct {
				Config struct {
					Stated *bool `json:"swa_bounded_replay"`
					V2     *bool `json:"use_v2_model_runner"`
				} `json:"config"`
			}
			var golden struct {
				SWALayout struct {
					Resolved bool `json:"resolved_bounded_replay"`
				} `json:"swa_layout"`
			}
			readOracleJSON(t, path, &sc)
			readOracleJSON(t, filepath.Join("../../core/sched/testdata/golden-swa", name+".json"), &golden)

			r := repos()
			r.Scenarios = "../../testdata/scenarios"
			in, err := latencykernel.OpenInputs(dspark, r, 0)
			if err != nil {
				t.Fatal(err)
			}
			editPoolEngine(t, &in.Deployment.Pools[0], func(e *engineconfig.Engine) {
				e.Speculative = nil
				e.SWABoundedReplay = sc.Config.Stated
			})
			m, err := fromInputs(in)
			if err != nil {
				t.Fatal(err)
			}
			if sc.Config.V2 != nil && !*sc.Config.V2 {
				// The pool cannot open with a V1-only feature on this model (the kernel refuses
				// it first), so the opened model is told it is on: ModelRunner then answers "v1".
				m.engine.DBO = &engineconfig.DBO{Enabled: true}
			}
			runs, err := m.KVLayers()
			if err != nil {
				t.Fatal(err)
			}
			windows := 0
			for _, run := range runs {
				if run.Spec != "sliding_window_mla" {
					continue
				}
				windows++
				if run.BoundedReplay != golden.SWALayout.Resolved {
					t.Errorf("sliding-window run %+v: bounded replay %v, vLLM resolved the statement to %v",
						run, run.BoundedReplay, golden.SWALayout.Resolved)
				}
			}
			if windows == 0 {
				t.Fatal("the pool has no sliding_window_mla run")
			}
		})
	}
}

func readOracleJSON(t *testing.T, path string, v any) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}
