package sim

// ResidentAdapterSet is the per-instance view of the finite resident LoRA adapter
// slots: a capacity-bounded LRU over adapter ids. The Simulator uses it to track
// which adapters are "loaded" on the instance as requests are scheduled. The
// concrete implementation (the LRU) lives in sim/lora and is wired in via
// NewResidentAdapterSetFunc, mirroring NewAdapterRegistryFunc / NewLatencyModelFunc,
// so sim/ can own the state without importing sim/lora (Principle I: no reverse
// import).
//
// The interface is intentionally scoped to what the scheduling hook needs today
// (R13). The concrete set additionally supports pin/unpin and explicit eviction;
// those enter this interface when the cold-load gate (#1464) consumes them.
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
}

// NewResidentAdapterSetFunc builds a ResidentAdapterSet with the given per-instance
// slot capacity. Registered by sim/lora's init() (import sim/lora to wire it); nil
// until then, in which case the Simulator leaves the LoRA subsystem inert (INV-6).
var NewResidentAdapterSetFunc func(capacity int) ResidentAdapterSet
