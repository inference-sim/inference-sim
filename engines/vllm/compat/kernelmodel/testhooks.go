// Original to blis-vllm.

package kernelmodel

import (
	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-schemas/kernel"

	sim "github.com/inference-sim/blis-vllm/compat/engine"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// Test hooks for inference-sim's simulator-level tests of this adapter, which price a whole
// simulation and inspect what the kernel was handed. Nothing outside tests calls them.

// BatchShapesForTest is the request shapes the kernel is handed for batch (batchOf).
func (m *Model) BatchShapesForTest(batch []*sim.Request) []kernel.ReqShape {
	return m.batchOf(batch).Reqs
}

// FromInputsForTest builds a Model from inputs a test assembled (fromInputs), the path Open and
// OpenPool end in.
func FromInputsForTest(in latencykernel.Inputs) (*Model, error) { return fromInputs(in) }

// DraftForTest makes in's priced pool draft k tokens a step with method, written into its engine
// block as blis-vllm reads it (DecodeEngine, SetEngine), so a simulator-level test sets a drafter
// without naming vLLM's engine configuration.
func DraftForTest(in *latencykernel.Inputs, method string, k int) error {
	pool := &in.Deployment.Pools[in.PoolIndex]
	e, err := DecodeEngine(*pool)
	if err != nil {
		return err
	}
	e.Speculative = &engineconfig.Speculative{Method: method, NumSpecTokens: k}
	return SetEngine(pool, e)
}
