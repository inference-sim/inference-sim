package cmd

import (
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim"
)

// The API-server front end (INV-R1). blis-schemas states neither setting, so they are flags
// whose defaults are vLLM v0.31.0's; proposed schema homes: engine.renderer_num_workers and
// parallel.api_server_count.
var (
	rendererNumWorkers int // --renderer-num-workers: vLLM renderer_num_workers (vllm/config/model.py:378)
	apiServerCount     int // --api-server-count: vLLM --api-server-count; 0 = vLLM's per-mode default
)

func registerFrontendFlags(cmd *cobra.Command) {
	cmd.Flags().IntVar(&rendererNumWorkers, "renderer-num-workers", 1, "Render threads per API server (vLLM renderer_num_workers, default 1). "+
		"Each request's admission overhead (chat template + tokenize, priced by the kernel) runs on one, in arrival order, "+
		"so requests queue for it; 0 makes the overhead an uncontended per-request delay")
	cmd.Flags().IntVar(&apiServerCount, "api-server-count", 0, "API servers per endpoint (vLLM --api-server-count). 0 = vLLM's default: "+
		"the data-parallel size behind an internal- or hybrid-balancer replica, else 1 (one per rank under external balancing)")
}

// frontendConfigFromCLI is the single run/replay wiring seam for the front end (INV-13).
func frontendConfigFromCLI() sim.FrontendConfig {
	c := sim.FrontendConfig{RendererNumWorkers: rendererNumWorkers, APIServerCount: apiServerCount}
	if err := c.Validate(); err != nil {
		logrus.Fatalf("--renderer-num-workers / --api-server-count: %v", err)
	}
	return c
}
