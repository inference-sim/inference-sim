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

// rank returns the LRU rank for each snapshot ID.
// Never-used endpoints get rank 0 (highest priority).
// Used endpoints rank from 1 (least recently used = oldest) to N (most recently used).
// The ranking is: never-used (0) < tail (1) < ... < head (N).
// Lower rank = higher score (preferred for cold requests).
func (l *noHitLRU) rank(snapshots []RoutingSnapshot) map[string]int {
	ranks := make(map[string]int, len(snapshots))

	// Count used endpoints among snapshots for rank offset
	usedCount := 0
	for _, snap := range snapshots {
		if _, ok := l.entries[snap.ID]; ok {
			usedCount++
		}
	}

	// Never-used endpoints get rank 0
	// Used endpoints: walk from tail (LRU) to head (MRU), assign ranks starting at 1
	usedRank := make(map[string]int, usedCount)
	r := 1
	for e := l.tail; e != nil; e = e.prev {
		usedRank[e.id] = r
		r++
	}

	for _, snap := range snapshots {
		if ur, ok := usedRank[snap.ID]; ok {
			ranks[snap.ID] = ur
		}
		// else: rank 0 (never-used)
	}
	return ranks
}

// newNoHitLRUScorer creates a scorer that distributes cold requests (no cache hits)
// to least-recently-used endpoints, matching llm-d's NoHitLRU scorer.
//
// Warm requests (any instance has cached blocks): all instances score 0.5 (neutral).
// Cold requests (no cached blocks on any instance): score by LRU position.
// Never-used endpoints score highest. Single endpoint scores 1.0.
//
// LRU is only updated on cold request routing (via observer).
// cacheQueryFn must be non-nil; panics otherwise.
func newNoHitLRUScorer(cacheQueryFn CacheQueryFn) (scorerFunc, observerFunc) {
	if cacheQueryFn == nil {
		panic("no-hit-lru scorer requires cacheQueryFn (nil provided); " +
			"this scorer can only be used in cluster mode")
	}

	lru := newNoHitLRU()

	scorer := func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
		scores := make(map[string]float64, len(snapshots))
		if req == nil || len(snapshots) == 0 {
			return scores
		}

		// Check if any instance has cached blocks for this request
		isWarm := false
		for _, snap := range snapshots {
			fn, ok := cacheQueryFn[snap.ID]
			if ok && fn(req.InputTokens) > 0 {
				isWarm = true
				break
			}
		}

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
		// Only update LRU for cold requests
		for _, fn := range cacheQueryFn {
			if fn(req.InputTokens) > 0 {
				return // warm request — don't update LRU
			}
		}
		lru.touch(targetInstance)
	}

	return scorer, observer
}
