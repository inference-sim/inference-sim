package sim

import "math"

// newPrecisePrefixCacheScorer creates a scorer that queries actual per-instance
// KV cache state for prefix match counts, then applies min-max normalization.
//
// Signal freshness (R17, INV-7):
//
//	Reads: KVCache.GetCachedBlocks via cacheQueryFn — ground truth (synchronous,
//	no staleness). Each routing decision queries the current KV cache state
//	at the moment of routing.
func newPrecisePrefixCacheScorer(cacheQueryFn cacheQueryFn) (scorerFunc, observerFunc) {
	scorer := func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
		scores := make(map[string]float64, len(snapshots))
		if req == nil || cacheQueryFn == nil {
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
			if fn, ok := cacheQueryFn[snap.ID]; ok && fn != nil {
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
				// All-equal: 1.0 when there is actual cache affinity, 0.5 (neutral)
				// when no instance has any cached blocks — avoids inflating the cache
				// scorer's contribution when there is no cache data to differentiate.
				if maxRaw == 0 {
					scores[snap.ID] = 0.5
				} else {
					scores[snap.ID] = 1.0
				}
			} else {
				scores[snap.ID] = float64(raw[snap.ID]-minRaw) / float64(maxRaw-minRaw)
			}
		}
		return scores
	}
	return scorer, nil // no observer (BC-8: stateless ground truth)
}
