// Original to blis-vllm.

package vllmengine

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// vLLM serves the router's render endpoints only when the frontend runs with --enable-scale-out
// (engine.enable_scale_out); otherwise every one returns 403
// (vllm/entrypoints/scale_out/factories.py register_scale_out_api_routers). The engine reports
// the result through engine.RenderServing, for a single engine and for a data-parallel group
// (inference-sim#2129).
func TestEngineStatesWhetherItsServerAnswersRenderCalls(t *testing.T) {
	cases := []struct {
		name, scenario, lines string
		want                  bool
	}{
		{"unset", "d034-032.yaml", "", false},
		{"false", "d034-032.yaml", "      enable_scale_out: false\n", false},
		{"true", "d034-032.yaml", "      enable_scale_out: true\n", true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			e, err := newEngine(withEngineLines(t, c.scenario, c.lines))
			if err != nil {
				t.Fatal(err)
			}
			r, ok := e.(engine.RenderServing)
			if !ok {
				t.Fatalf("%T does not state whether it serves render calls", e)
			}
			if got := r.ServesRender(); got != c.want {
				t.Errorf("ServesRender() = %v, want %v", got, c.want)
			}
		})
	}
}
