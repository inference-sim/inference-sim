package sim

// ResidentAdapterSet is the per-instance view of the finite resident LoRA adapter
// slots: a capacity-bounded LRU over adapter ids. The Simulator uses it to track
// which adapters are "loaded" on the instance as requests are scheduled. The
// concrete implementation (the LRU) lives in sim/lora and is wired in via
// NewResidentAdapterSetFunc, mirroring NewAdapterRegistryFunc / NewLatencyModelFunc,
// so sim/ can own the state without importing sim/lora (Principle I: no reverse
// import).
//
// The interface is scoped to what the scheduling hook and the cold-load
// pre-admission gate consume (R13). The gate (#1466) added AtCapacity and Pin /
// Unpin (to protect an adapter in use by an in-flight request from eviction,
// INV-L5). Victim selection moved to the eviction seam (B-3, #1491): the gate now
// reads UnpinnedCandidates() to build the eviction context and calls Evict(id) to
// remove the chosen victim; the concrete residentSet retains EvictLRU for Store's
// internal replacement and unit tests, but it is no longer part of this interface.
type ResidentAdapterSet interface {
	// IsResident reports whether id currently occupies a slot.
	IsResident(id string) bool
	// Touch moves a resident id to most-recently-used. No-op if id is absent.
	Touch(id string)
	// Store makes id resident at most-recently-used, evicting an entry per the set's
	// replacement policy (LRU here), skipping pinned entries, when it is at capacity.
	// Returns the evicted id ("" if none) and whether id is now resident (false only
	// when the set is full and every slot is pinned). Storing an already-resident id
	// is equivalent to Touch.
	Store(id string) (evicted string, admitted bool)
	// Len returns the number of resident adapters (always ≤ capacity).
	Len() int
	// ResidentIDs returns the ids currently resident. The order is stable and
	// deterministic (LRU→MRU — INV-6); the current sole consumer (the snapshot
	// provider, which collapses the result into RoutingSnapshot.ResidentAdapters as a
	// membership set for the lora-affinity scorer, #1469) does not depend on order,
	// so implementations must not be constrained to preserve it beyond determinism.
	// Returns nil when empty.
	ResidentIDs() []string

	// AtCapacity reports whether every slot is occupied (Len == capacity). The
	// cold-load gate uses it to decide whether a slot must be freed before a load.
	AtCapacity() bool
	// UnpinnedCandidates returns the resident, unpinned adapter ids in LRU→MRU
	// (eviction-priority) order; nil when none are evictable. The eviction seam
	// selects a victim only from these.
	UnpinnedCandidates() []string
	// Evict removes an unpinned resident id, returning true on removal and false
	// if the id is absent or pinned (INV-L5).
	Evict(id string) bool
	// Pin protects id from eviction while an in-flight request references it
	// (reference-counted). No-op if id is absent.
	Pin(id string)
	// Unpin releases one reference; id becomes evictable once no in-flight request
	// references it. No-op if id is absent or its count is already zero.
	Unpin(id string)
}

// NewResidentAdapterSetFunc builds a ResidentAdapterSet with the given per-instance
// slot capacity. Registered by sim/lora's init() (import sim/lora to wire it); nil
// until then, in which case the Simulator leaves the LoRA subsystem inert (INV-6).
var NewResidentAdapterSetFunc func(capacity int) ResidentAdapterSet
