package sim

// defaultLRUCapacity is the default number of block hashes tracked per instance
// in the router-side prefix cache. 10,000 blocks × 16 tokens/block = 160K tokens.
const defaultLRUCapacity = 10000

// newPrefixAffinityScorer creates a prefix-affinity scorer and its observer.
// The scorer returns per-instance scores based on how much of the request's
// prefix each instance has cached. The observer updates the cache index
// after each routing decision.
//
// Signal freshness (R17, INV-7):
//
//	Reads: No RoutingSnapshot fields — uses router-side PrefixCacheIndex only.
//	The cache index is synchronously updated by the observer after each routing
//	decision, making this scorer effectively Tier 1 (always fresh).
//
// Both the scorer and observer share the same PrefixCacheIndex via closure.
// The blockSize should match the simulation's KV cache block size.
func newPrefixAffinityScorer(blockSize int) (scorerFunc, observerFunc) {
	idx := NewPrefixCacheIndex(blockSize, defaultLRUCapacity)

	scorer := func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
		scores := make(map[string]float64, len(snapshots))
		if req == nil {
			return scores
		}
		hashes := idx.ComputeBlockHashes(req.InputTokens)
		totalBlocks := len(hashes)
		for _, snap := range snapshots {
			if totalBlocks == 0 {
				scores[snap.ID] = 0.0
			} else {
				matched := idx.MatchLength(hashes, snap.ID)
				scores[snap.ID] = float64(matched) / float64(totalBlocks)
			}
		}
		return scores
	}

	observer := func(req *Request, targetInstance string) {
		if req == nil {
			return
		}
		hashes := idx.ComputeBlockHashes(req.InputTokens)
		idx.RecordBlocks(hashes, targetInstance)
	}

	return scorer, observer
}
