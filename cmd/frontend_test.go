package cmd

import (
	"testing"

	"github.com/spf13/cobra"
)

// The front-end flags default to vLLM v0.31.0's settings (INV-R1): one render thread per API
// server (renderer_num_workers, vllm/config/model.py:378) and vLLM's per-mode API-server
// count (0 here, resolved per endpoint).
func TestFrontendFlagsDefaultToVLLMs(t *testing.T) {
	for _, c := range []*cobra.Command{runCmd, replayCmd} {
		for name, want := range map[string]string{"renderer-num-workers": "1", "api-server-count": "0"} {
			f := c.Flags().Lookup(name)
			if f == nil {
				t.Fatalf("%s has no --%s", c.Name(), name)
			}
			if f.DefValue != want {
				t.Errorf("%s --%s defaults to %s, want vLLM's %s", c.Name(), name, f.DefValue, want)
			}
		}
	}
	defer func(w, a int) { rendererNumWorkers, apiServerCount = w, a }(rendererNumWorkers, apiServerCount)
	rendererNumWorkers, apiServerCount = 2, 3
	if got := frontendConfigFromCLI(); got.RendererNumWorkers != 2 || got.APIServerCount != 3 {
		t.Errorf("frontendConfigFromCLI = %+v, want the flags' values", got)
	}
}
