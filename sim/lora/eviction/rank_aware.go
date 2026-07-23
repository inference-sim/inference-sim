package eviction

import (
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// rankAware evicts the unpinned adapter with the lowest declared rank — the
// cheapest-to-reload adapter — so the expensive-to-reload (high-rank) adapters
// stay resident and aggregate cold-load penalty is minimized (C-1). Ties on rank
// break by lexicographically smallest id (C-2), making the choice a pure function
// of static adapter metadata and therefore byte-identical across instances (INV-6).
//
// Rank is the monotone reload-cost proxy; this policy is BLIS-native, inspired by
// the rank-aware placement signal in Li2025 (Toppings) — NOT a reproduction of any
// shipped eviction policy (production S-LoRA/vLLM use recency-style eviction).
// Stateless; consumes only the B-3 EvictionContext.RankOf accessor.
type rankAware struct{}

func (rankAware) SelectVictim(ctx sim.EvictionContext) (string, bool) {
	if len(ctx.Candidates) == 0 {
		return "", false // no-victim parity with lru (C-3): caller retries on a later step (INV-8)
	}
	// Single linear scan over candidates tracking argmin(rank ASC, id ASC). An
	// unregistered candidate (RankOf ok=false) is treated as rank +∞ (math.MaxInt),
	// so it is never chosen ahead of any registered adapter (D-4-2). Candidate order
	// does not affect the result — the total order is strict on (rank, id).
	minID := ""
	minRank := 0
	first := true
	for _, id := range ctx.Candidates {
		rank, ok := ctx.RankOf(id)
		if !ok {
			rank = math.MaxInt
		}
		if first || rank < minRank || (rank == minRank && id < minID) {
			minRank = rank
			minID = id
			first = false
		}
	}
	return minID, true
}
