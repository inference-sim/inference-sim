package sim

// ResidentAdapterSet is the per-instance view of the finite resident LoRA adapter
// slots: a capacity-bounded resident set of adapter ids. The Simulator uses it to
// track which adapters are "loaded" on the instance as requests are scheduled. The
// concrete implementation (an LRU) lives in sim/lora and is wired in via
// NewResidentAdapterSetFunc, mirroring NewAdapterRegistryFunc / NewLatencyModelFunc,
// so sim/ can own the state without importing sim/lora (Principle I: no reverse
// import).
//
// The interface is scoped to what the scheduling hook and the cold-load
// pre-admission gate consume (R13). The gate (#1466) added AtCapacity / EvictLRU
// (to commit an eviction victim and reserve a slot at load-start) and Pin / Unpin
// (to protect an adapter in use by an in-flight request from eviction, INV-L5).
type ResidentAdapterSet interface {
	// IsResident reports whether id currently occupies a slot.
	IsResident(id string) bool
	// Touch moves a resident id to most-recently-used. No-op if id is absent.
	Touch(id string)
	// Store makes id resident at most-recently-used, evicting an entry per the set's
	// replacement policy when at capacity, skipping pinned entries.
	// Returns the evicted id ("" if none) and whether id is now resident (false only
	// when the set is full and every slot is pinned). Storing an already-resident id
	// is equivalent to Touch.
	Store(id string) (evicted string, admitted bool)
	// Len returns the number of resident adapters (always ≤ capacity).
	Len() int

	// AtCapacity reports whether every slot is occupied (Len == capacity). The
	// cold-load gate uses it to decide whether a slot must be freed before a load.
	AtCapacity() bool
	// EvictLRU removes the least-recently-used non-pinned adapter and returns its id
	// and true, or ("", false) if the set is empty or every entry is pinned. The gate
	// calls it at load-start to commit the eviction victim and reserve a slot (§7).
	EvictLRU() (string, bool)
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
