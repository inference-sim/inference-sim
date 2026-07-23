package lora

import "fmt"

// residentEntry is one adapter slot in the resident set's LRU doubly-linked
// list. pinCount is the number of in-flight requests currently using the
// adapter; a positive count protects the entry from eviction.
type residentEntry struct {
	id       string
	pinCount uint
	prev     *residentEntry // older — toward the LRU head (evicted first)
	next     *residentEntry // newer — toward the MRU tail
}

// residentSet models the finite per-instance GPU adapter slots as a capacity-
// bounded LRU over adapter ids (specs/007-lora-control-plane/data-model.md
// §"Entity: Resident-Adapter Set (per instance)"). It reuses the cpuTier map +
// doubly-linked-list pattern (sim/kv/tiered.go): O(1) store
// and touch; eviction scans from the LRU head past pinned entries — O(1) in the
// common case (few or no pins), O(capacity) worst-case (all entries pinned).
// Adapters required by in-flight requests are pinned
// (reference-counted) and are never chosen as an eviction victim, so the
// capacity bound len(entries) ≤ capacity holds at all times.
//
// Eviction order is a pure function of access order (LRU over touches) with no
// randomness, so a given sequence of operations is deterministic (INV-6).
//
// Its exported methods satisfy sim.ResidentAdapterSet; the type stays unexported
// because callers outside sim/lora only ever hold the interface (wired via
// sim.NewResidentAdapterSetFunc). Not safe for concurrent use: each Simulator
// owns one and mutates it only on the single simulation goroutine.
type residentSet struct {
	capacity int
	entries  map[string]*residentEntry // id → entry, O(1) lookup
	lruHead  *residentEntry            // least recently used (evicted first)
	lruTail  *residentEntry            // most recently used
}

// newResidentSet creates an empty resident set with the given slot capacity.
// A non-positive capacity is a programming error (LoRAConfig.Validate rejects it
// upstream) and panics, mirroring newCpuTier.
func newResidentSet(capacity int) *residentSet {
	if capacity <= 0 {
		panic(fmt.Sprintf("newResidentSet: capacity must be > 0, got %d", capacity))
	}
	return &residentSet{
		capacity: capacity,
		entries:  make(map[string]*residentEntry, capacity),
	}
}

// Len reports the number of currently resident adapters.
func (s *residentSet) Len() int { return len(s.entries) }

// IsResident reports whether id currently occupies a slot.
func (s *residentSet) IsResident(id string) bool {
	_, ok := s.entries[id]
	return ok
}

// Touch moves a resident id to the MRU tail. No-op if id is not resident.
func (s *residentSet) Touch(id string) {
	e, ok := s.entries[id]
	if !ok {
		return
	}
	s.unlink(e)
	s.appendToTail(e)
}

// Store makes id resident at the MRU tail. If id is already resident this is
// equivalent to Touch and returns ("", true). For a new id at capacity, the
// least-recently-used non-pinned entry is evicted first and its id returned. If
// the set is full and every entry is pinned, id is not admitted and ("", false)
// is returned — the caller (cold-load gate) must wait for an in-flight request
// to complete and unpin a slot.
func (s *residentSet) Store(id string) (evicted string, ok bool) {
	if _, exists := s.entries[id]; exists {
		s.Touch(id)
		return "", true
	}
	if len(s.entries) >= s.capacity {
		if evicted, ok = s.EvictLRU(); !ok {
			return "", false // full and every resident adapter is pinned
		}
	}
	e := &residentEntry{id: id}
	s.entries[id] = e
	s.appendToTail(e)
	return evicted, true
}

// EvictLRU removes the least-recently-used non-pinned entry, scanning from the
// LRU head toward the tail so pinned entries are skipped in recency order.
// Returns the evicted id and true, or ("", false) if the set is empty or every
// entry is pinned.
//
// Exported (unlike the enclosing unexported residentSet) because the LoRA cold-load
// gate will drive eviction directly once pin/unpin join the ResidentAdapterSet
// interface; the uppercase name is deliberate, not an accidental export.
func (s *residentSet) EvictLRU() (string, bool) {
	for e := s.lruHead; e != nil; e = e.next {
		if e.pinCount == 0 {
			s.unlink(e)
			delete(s.entries, e.id)
			return e.id, true
		}
	}
	return "", false
}

// Pin increments the in-flight reference count protecting id from eviction.
// No-op if id is not resident.
func (s *residentSet) Pin(id string) {
	if e, ok := s.entries[id]; ok {
		e.pinCount++
	}
}

// Unpin decrements the in-flight reference count; the adapter becomes evictable
// once no in-flight request references it. No-op if id is not resident or its
// count is already zero.
func (s *residentSet) Unpin(id string) {
	if e, ok := s.entries[id]; ok && e.pinCount > 0 {
		e.pinCount--
	}
}

// appendToTail links e at the MRU tail (mirrors cpuTier.appendToTail).
func (s *residentSet) appendToTail(e *residentEntry) {
	e.next = nil
	e.prev = s.lruTail
	if s.lruTail != nil {
		s.lruTail.next = e
	} else {
		s.lruHead = e
	}
	s.lruTail = e
}

// unlink removes e from the LRU doubly-linked list (mirrors cpuTier.unlink).
func (s *residentSet) unlink(e *residentEntry) {
	if e.prev != nil {
		e.prev.next = e.next
	} else {
		s.lruHead = e.next
	}
	if e.next != nil {
		e.next.prev = e.prev
	} else {
		s.lruTail = e.prev
	}
	e.prev = nil
	e.next = nil
}
