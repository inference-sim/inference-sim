package sim

import "math"

// newPrecisePrefixCacheScorer creates a scorer that queries actual instance KV cache
// state for prefix matching, mirroring llm-d's PrecisePrefixCacheScorer.
//
// Raw score per instance: number of consecutive cached prefix blocks (via cacheQueryFn).
// Normalization: min-max across candidates (highest raw → 1.0, lowest → 0.0, all-equal → 1.0).
// This matches llm-d's indexedScoresToNormalizedScoredPods and BLIS's own scoreQueueDepth.
//
// No router-side index, no approximation. Observer is nil (state is ground truth).
// cacheQueryFn must be non-nil; panics otherwise (factory validation).
func newPrecisePrefixCacheScorer(cacheQueryFn CacheQueryFn) (scorerFunc, observerFunc) {
	if cacheQueryFn == nil {
		panic("precise-prefix-cache scorer requires cacheQueryFn (nil provided); " +
			"this scorer can only be used in cluster mode")
	}

	scorer := func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
		scores := make(map[string]float64, len(snapshots))
		if req == nil || len(snapshots) == 0 {
			return scores
		}

		// Compute raw block counts per instance
		rawScores := make(map[string]int, len(snapshots))
		minRaw, maxRaw := math.MaxInt, 0
		for _, snap := range snapshots {
			fn, ok := cacheQueryFn[snap.ID]
			count := 0
			if ok {
				count = fn(req.InputTokens)
			}
			rawScores[snap.ID] = count
			if count < minRaw {
				minRaw = count
			}
			if count > maxRaw {
				maxRaw = count
			}
		}

		// Min-max normalization: highest → 1.0, lowest → 0.0, all-equal → 1.0
		for _, snap := range snapshots {
			if maxRaw == minRaw {
				scores[snap.ID] = 1.0
			} else {
				scores[snap.ID] = float64(rawScores[snap.ID]-minRaw) / float64(maxRaw-minRaw)
			}
		}
		return scores
	}

	return scorer, nil
}
