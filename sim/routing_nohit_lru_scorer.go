package sim

// newNoHitLRUScorer creates a scorer that distributes cold requests (no cache
// hits on any instance) to least-recently-used endpoints. Warm requests (at
// least one instance has cached blocks) score 0.5 (neutral, defers to other
// scorers).
//
// Signal freshness (R17, INV-7):
//
//	Reads: KVCache.GetCachedBlocks via cacheQueryFn — ground truth (synchronous,
//	no staleness). Used only for warm/cold detection, not scoring magnitude.
//	LRU state is deterministic (updated by observer on cold routing only).
func newNoHitLRUScorer(cacheQueryFn CacheQueryFn) (scorerFunc, observerFunc) {
	// LRU tracking: ordered list of instance IDs, most-recently-used first.
	// Only updated on cold request routing.
	var lruOrder []string // most-recent first
	lruSet := make(map[string]bool)

	// Shared warm/cold flag between scorer and observer (same pattern as
	// cachedHashes/cachedReqID in prefix-affinity scorer). The scorer sets
	// lastWarm; the observer reads it. Safe because the DES is single-threaded
	// and scorer is always called before observer for the same request.
	lastWarm := false
	lastReqID := ""

	scorer := func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
		scores := make(map[string]float64, len(snapshots))

		// Nil cacheQueryFn → neutral (cannot determine hit status)
		if req == nil || cacheQueryFn == nil {
			for _, snap := range snapshots {
				scores[snap.ID] = 0.5
			}
			lastWarm = true // prevent observer from updating LRU
			lastReqID = ""
			return scores
		}

		// Check if any instance has cached blocks (warm detection)
		lastWarm = false
		lastReqID = req.ID
		for _, snap := range snapshots {
			if fn, ok := cacheQueryFn[snap.ID]; ok {
				if fn(req.InputTokens) > 0 {
					lastWarm = true
					break
				}
			}
		}

		if lastWarm {
			// BC-4: warm request → neutral 0.5 for all
			for _, snap := range snapshots {
				scores[snap.ID] = 0.5
			}
			return scores
		}
		// BC-3: cold request → LRU positional scoring
		total := len(snapshots)
		if total == 1 {
			scores[snapshots[0].ID] = 1.0
			return scores
		}
		// Build rank: never-used first (rank 0), then oldest-used to newest-used
		rank := 0
		// Never-used instances (not in lruSet) get lowest rank indices (= highest scores)
		var neverUsed []string
		for _, snap := range snapshots {
			if !lruSet[snap.ID] {
				neverUsed = append(neverUsed, snap.ID)
			}
		}
		for _, id := range neverUsed {
			scores[id] = 1.0 - float64(rank)/float64(total-1)
			rank++
		}
		// Used instances: oldest first (end of lruOrder) to newest (start)
		for i := len(lruOrder) - 1; i >= 0; i-- {
			id := lruOrder[i]
			// Only score if this instance is in the current snapshot set
			found := false
			for _, snap := range snapshots {
				if snap.ID == id {
					found = true
					break
				}
			}
			if found {
				scores[id] = 1.0 - float64(rank)/float64(total-1)
				rank++
			}
		}
		return scores
	}

	observer := func(req *Request, targetInstance string) {
		if req == nil {
			return
		}
		// BC-5: use scorer's warm/cold determination (not re-derived).
		// This avoids disagreement between scorer (checks all instances)
		// and observer (would only check target instance).
		if lastWarm || req.ID != lastReqID {
			return
		}
		// Move targetInstance to front of LRU (most-recently-used)
		if lruSet[targetInstance] {
			// Remove from current position
			for i, id := range lruOrder {
				if id == targetInstance {
					lruOrder = append(lruOrder[:i], lruOrder[i+1:]...)
					break
				}
			}
		}
		lruOrder = append([]string{targetInstance}, lruOrder...)
		lruSet[targetInstance] = true
	}

	return scorer, observer
}
