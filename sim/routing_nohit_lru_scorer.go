package sim

// newNoHitLRUScorer creates a scorer that distributes cold requests (no cache
// hits on any instance) to least-recently-used endpoints. Warm requests (at
// least one instance has cached blocks) score 0.5 (neutral, defers to other
// scorers).
//
// Signal freshness (R17, INV-7):
//
//	Reads: cacheQueryFn closures — live KVCache.GetCachedBlocks in oracle mode;
//	frozen HashToBlock snapshot in stale mode (delay>0) — for warm/cold detection.
//	Freshness depends on --cache-signal-delay:
//	  - delay=0: ground truth (synchronous, no staleness) — oracle mode.
//	  - delay>0 (default 2s): Demand-triggered staleness via StaleCacheIndex snapshot refresh.
//	    Default 2s matches production llm-d's speculative TTL.
//	LRU state is deterministic (updated by observer on cold routing only).
func newNoHitLRUScorer(cacheFn cacheQueryFn) (scorerFunc, observerFunc) {
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
		if req == nil || cacheFn == nil {
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
			if fn, ok := cacheFn[snap.ID]; ok && fn != nil {
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
		if total <= 1 {
			if total == 1 {
				scores[snapshots[0].ID] = 1.0
			}
			return scores
		}

		// Prune stale entries: remove instances no longer in the current snapshot set.
		// Prevents unbounded growth of lruOrder/lruSet under instance churn (autoscaling).
		currentIDs := make(map[string]bool, total)
		for _, snap := range snapshots {
			currentIDs[snap.ID] = true
		}
		pruned := make([]string, 0, len(lruOrder))
		for _, id := range lruOrder {
			if currentIDs[id] {
				pruned = append(pruned, id)
			} else {
				delete(lruSet, id)
			}
		}
		lruOrder = pruned

		// Build rank: never-used first (rank 0), then oldest-used to newest-used.
		// O(N*S) where N=len(lruOrder), S=len(snapshots); acceptable for typical cluster sizes.
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
			scores[id] = 1.0 - float64(rank)/float64(total-1)
			rank++
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
