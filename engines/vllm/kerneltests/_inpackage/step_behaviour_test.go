// Moved from blis-latency-kernel/step_behaviour_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel

import (
	"github.com/inference-sim/blis-schemas/kernel"
)

// Step-time behaviour, as relations between steps, deployments and graphs.

func prefillOf(tokens, prefix int) kernel.Batch {
	return kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{
		{Scheduled: tokens, Computed: prefix, PromptLen: prefix + tokens}}}
}
