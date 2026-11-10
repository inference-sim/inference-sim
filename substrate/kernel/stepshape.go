package latencykernel

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
)

// BatchOf is the kernel.Batch one data-parallel rank of a step shape prices: its requests' token
// counts, the decode threshold the engine states, and the tokens the engine padded the rank with.
// It reads the engine's figures and applies no rule of its own: the decode threshold is
// StepShape.DecodeWidth (the contract's neutral 1, never a default of the cost model's), and the
// pad is the launched token count less the scheduled ones (SUB-6).
func BatchOf(s plan.StepShape, rank int) (kernel.Batch, error) {
	if rank < 0 || rank >= len(s.Ranks) {
		return kernel.Batch{}, fmt.Errorf("latencykernel: rank %d of a step shape with %d ranks", rank, len(s.Ranks))
	}
	b := kernel.Batch{DecodeThreshold: int(s.DecodeWidth())}
	var scheduled int
	for _, q := range s.Ranks[rank].Requests {
		b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: int(q.Scheduled), Computed: int(q.Computed),
			PromptLen: int(q.Prompt), CachedTokens: int(q.Cached)})
		scheduled += int(q.Scheduled)
	}
	if rank < len(s.Padded) {
		b.PadTokens = int(s.Padded[rank].Tokens) - scheduled
	}
	return b, b.Validate()
}
