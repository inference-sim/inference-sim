package sim

import "math"

// newPrecisePrefixCacheScorer creates a scorer that queries actual per-instance
// KV cache state for prefix match counts, then applies min-max normalization.
//
// Signal freshness (R17, INV-7):
//
//	Reads: cacheQueryFn closures — live KVCache.GetCachedBlocks in oracle mode;
//	frozen HashToBlock snapshot in stale mode (delay>0).
//	Freshness depends on --cache-signal-delay:
//	  - delay=0: ground truth (synchronous, no staleness) — oracle mode.
//	  - delay>0 (default 50ms): Demand-triggered staleness via CachedSnapshotProvider cache refresh.
//	    Each routing decision queries a frozen copy of the HashToBlock map,
//	    refreshed every CacheSignalDelay microseconds of sim time.
//	    Default 50ms models aggregate signal staleness from production llm-d.
func newPrecisePrefixCacheScorer(cacheFn cacheQueryFn) (scorerFunc, observerFunc) {
	scorer := func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
		scores := make(map[string]float64, len(snapshots))
		if req == nil || cacheFn == nil {
			for _, snap := range snapshots {
				scores[snap.ID] = 1.0
			}
			return scores
		}
		// Pass 1: compute raw scores and find min/max
		raw := make(map[string]int, len(snapshots))
		minRaw, maxRaw := math.MaxInt, 0
		for _, snap := range snapshots {
			count := 0
			if fn, ok := cacheFn[snap.ID]; ok && fn != nil {
				count = fn(req.InputTokens)
			}
			raw[snap.ID] = count
			if count < minRaw {
				minRaw = count
			}
			if count > maxRaw {
				maxRaw = count
			}
		}
		// Pass 2: min-max normalize (higher cached → higher score)
		for _, snap := range snapshots {
			if maxRaw == minRaw {
				// All-equal (including all-zero): 1.0. Matches llm-d's
				// indexedScoresToNormalizedScoredPods which returns 1.0
				// unconditionally when minScore == maxScore.
				scores[snap.ID] = 1.0
			} else {
				scores[snap.ID] = float64(raw[snap.ID]-minRaw) / float64(maxRaw-minRaw)
			}
		}
		return scores
	}
	return scorer, nil // no observer (BC-8: stateless ground truth)
}
