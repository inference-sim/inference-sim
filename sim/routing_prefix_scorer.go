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
//	decision, making this scorer effectively synchronous (always fresh).
//
// Both the scorer and observer share the same PrefixCacheIndex via closure.
// The blockSize should match the simulation's KV cache block size.
func newPrefixAffinityScorer(blockSize int) (scorerFunc, observerFunc) {
	idx := NewPrefixCacheIndex(blockSize, defaultLRUCapacity)

	// Shared cache: scorer computes hashes once, observer reuses them.
	// Safe because Route() always calls scorer then observer for the same request
	// before moving to the next request (sim/routing.go WeightedScoring.Route).
	// Single-threaded DES event loop guarantees no concurrent Route() calls.
	var cachedHashes []string
	var cachedReqID string

	scorer := func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
		scores := make(map[string]float64, len(snapshots))
		if req == nil {
			return scores
		}
		// Compute block hashes once and cache for the observer
		cachedHashes = idx.ComputeBlockHashes(req.InputTokens)
		cachedReqID = req.ID
		totalBlocks := len(cachedHashes)
		for _, snap := range snapshots {
			if totalBlocks == 0 {
				scores[snap.ID] = 0.0
			} else {
				matched := idx.MatchLength(cachedHashes, snap.ID)
				scores[snap.ID] = float64(matched) / float64(totalBlocks)
			}
		}
		return scores
	}

	observer := func(req *Request, targetInstance string) {
		if req == nil {
			return
		}
		// Reuse hashes from scorer if available for the same request
		hashes := cachedHashes
		if req.ID != cachedReqID {
			hashes = idx.ComputeBlockHashes(req.InputTokens)
		}
		idx.RecordBlocks(hashes, targetInstance)
	}

	return scorer, observer
}
