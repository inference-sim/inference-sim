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
// pad is the launched token count less the scheduled ones, and the drafter's launched rows are those
// of its own pass for this rank (SUB-6). A step that runs a layer suffix on a subset of its rows
// carries the layer the suffix starts at, each request's rows, and the rows the engine launched
// beyond them.
func BatchOf(s plan.StepShape, rank int) (kernel.Batch, error) {
	if rank < 0 || rank >= len(s.Ranks) {
		return kernel.Batch{}, fmt.Errorf("latencykernel: rank %d of a step shape with %d ranks", rank, len(s.Ranks))
	}
	b := kernel.Batch{DecodeThreshold: int(s.DecodeWidth()), TailFrom: s.TailFrom}
	var scheduled, tail int
	for _, q := range s.Ranks[rank].Requests {
		b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: int(q.Scheduled), Computed: int(q.Computed),
			PromptLen: int(q.Prompt), CachedTokens: int(q.Cached), TailRows: int(q.TailRows)})
		scheduled += int(q.Scheduled)
		tail += int(q.Scheduled)
		if q.TailRows > 0 {
			tail += int(q.TailRows - q.Scheduled)
		}
	}
	if rank < len(s.Padded) {
		b.PadTokens = int(s.Padded[rank].Tokens) - scheduled
		if t := int(s.Padded[rank].TailTokens); t > 0 {
			b.TailPadTokens = t - tail
		}
	}
	if rank < len(s.Drafts.Passes) {
		b.DraftPaddedTokens = int(s.Drafts.Passes[rank].Padded)
	}
	return b, b.Validate()
}
