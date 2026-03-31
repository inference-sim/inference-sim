package sim

// noHitLRUEntry is a doubly-linked list node for the LRU tracker.
type noHitLRUEntry struct {
	id   string
	prev *noHitLRUEntry
	next *noHitLRUEntry
}

// noHitLRU tracks endpoint usage order for cold request distribution.
// Endpoints not in the LRU are considered "never used" and rank highest.
type noHitLRU struct {
	head    *noHitLRUEntry // most recently used
	tail    *noHitLRUEntry // least recently used
	entries map[string]*noHitLRUEntry
}

func newNoHitLRU() *noHitLRU {
	return &noHitLRU{entries: make(map[string]*noHitLRUEntry)}
}

// touch moves an endpoint to the head (most recently used).
func (l *noHitLRU) touch(id string) {
	if e, ok := l.entries[id]; ok {
		l.remove(e)
		l.pushFront(e)
		return
	}
	e := &noHitLRUEntry{id: id}
	l.entries[id] = e
	l.pushFront(e)
}

func (l *noHitLRU) remove(e *noHitLRUEntry) {
	if e.prev != nil {
		e.prev.next = e.next
	} else {
		l.head = e.next
	}
	if e.next != nil {
		e.next.prev = e.prev
	} else {
		l.tail = e.prev
	}
	e.prev = nil
	e.next = nil
}

func (l *noHitLRU) pushFront(e *noHitLRUEntry) {
	e.next = l.head
	e.prev = nil
	if l.head != nil {
		l.head.prev = e
	}
	l.head = e
	if l.tail == nil {
		l.tail = e
	}
}

// rank returns the LRU rank for each snapshot ID, considering only snapshot-visible
// endpoints. Non-snapshot entries in the LRU are skipped to prevent rank inflation
// from filtered-out instances (e.g., non-routable or different-model instances).
//
// Never-used endpoints get rank 0 (highest priority).
// Used endpoints among snapshots rank from 1 (LRU oldest) to N (MRU newest).
// Lower rank = higher score (preferred for cold requests).
func (l *noHitLRU) rank(snapshots []RoutingSnapshot) map[string]int {
	ranks := make(map[string]int, len(snapshots))

	// Build a set of snapshot IDs for fast lookup
	snapshotSet := make(map[string]bool, len(snapshots))
	for _, snap := range snapshots {
		snapshotSet[snap.ID] = true
	}

	// Walk from tail (LRU) to head (MRU), assigning ranks only to
	// endpoints present in the current snapshot set. This ensures
	// ranks are contiguous within the candidate pool.
	r := 1
	for e := l.tail; e != nil; e = e.prev {
		if snapshotSet[e.id] {
			ranks[e.id] = r
			r++
		}
	}
	// Endpoints not in the LRU keep rank 0 (never-used) — the zero value.

	return ranks
}

// newNoHitLRUScorer creates a scorer that distributes cold requests (no cache hits)
// to least-recently-used endpoints, matching llm-d's NoHitLRU scorer.
//
// Warm requests (any candidate instance has cached blocks): all instances score 0.5 (neutral).
// Cold requests (no cached blocks on any candidate): score by LRU position.
// Never-used endpoints score highest. Single endpoint scores 1.0.
//
// LRU is only updated on cold request routing (via observer).
// The scorer and observer share a warm/cold determination via closure — safe because
// the DES is single-threaded and Route() always calls scorer then observer for the
// same request before moving to the next (sim/routing.go WeightedScoring.Route).
// cacheQueryFn must be non-nil; panics otherwise.
func newNoHitLRUScorer(cacheQueryFn CacheQueryFn) (scorerFunc, observerFunc) {
	if cacheQueryFn == nil {
		panic("no-hit-lru scorer requires cacheQueryFn (nil provided); " +
			"this scorer can only be used in cluster mode")
	}

	lru := newNoHitLRU()

	// Shared warm/cold determination between scorer and observer.
	// The scorer sets this; the observer reads it. Safe because DES is single-threaded
	// and Route() calls scorer → observer sequentially for the same request.
	var lastReqID string
	var lastReqWarm bool

	scorer := func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
		scores := make(map[string]float64, len(snapshots))
		if req == nil || len(snapshots) == 0 {
			return scores
		}

		// Check if any candidate instance has cached blocks for this request.
		// Only checks snapshot instances (the routing candidates), not all instances.
		isWarm := false
		for _, snap := range snapshots {
			fn, ok := cacheQueryFn[snap.ID]
			if ok && fn(req.InputTokens) > 0 {
				isWarm = true
				break
			}
		}

		// Cache determination for the observer
		lastReqID = req.ID
		lastReqWarm = isWarm

		if isWarm {
			// Warm request: neutral score for all instances
			for _, snap := range snapshots {
				scores[snap.ID] = 0.5
			}
			return scores
		}

		// Cold request: score by LRU position
		if len(snapshots) == 1 {
			scores[snapshots[0].ID] = 1.0
			return scores
		}

		ranks := lru.rank(snapshots)

		// Positional scoring: rank 0 (never-used/LRU) → highest score
		// score = 1.0 - rank / (totalEndpoints - 1)
		total := len(snapshots)
		for _, snap := range snapshots {
			scores[snap.ID] = 1.0 - float64(ranks[snap.ID])/float64(total-1)
		}
		return scores
	}

	observer := func(req *Request, targetInstance string) {
		if req == nil {
			return
		}
		// Reuse the scorer's warm/cold determination for the same request.
		// Falls back to checking all candidates if IDs don't match (defensive).
		if req.ID == lastReqID {
			if lastReqWarm {
				return
			}
		} else {
			// Defensive fallback: scorer wasn't called for this request (shouldn't happen)
			for _, fn := range cacheQueryFn {
				if fn(req.InputTokens) > 0 {
					return
				}
			}
		}
		lru.touch(targetInstance)
	}

	return scorer, observer
}
