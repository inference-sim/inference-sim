package sim

// AdapterRegistry is the read-only query interface over the pre-declared LoRA adapter
// registry (id -> rank[, base model]). It is a pure-query bridge type owned by sim/
// so control-plane code (routing, cost model, workload validation) can resolve adapter
// identity without importing sim/lora (Principle I: no reverse import).
//
// The concrete implementation lives in sim/lora and is wired in via
// NewAdapterRegistryFunc below, mirroring NewLatencyModelFunc / NewKVCacheStateFunc.
type AdapterRegistry interface {
	// RankOf returns the declared rank of an adapter id, and whether it is registered.
	RankOf(id string) (int, bool)
	// BaseModelOf returns the declared base model of an adapter id (may be ""), and
	// whether the id is registered.
	BaseModelOf(id string) (string, bool)
	// Has reports whether an adapter id is registered.
	Has(id string) bool
	// Len returns the number of registered adapters.
	Len() int
	// IDs returns all registered adapter ids in sorted order (R2, determinism).
	IDs() []string
}

// NewAdapterRegistryFunc builds an AdapterRegistry from declared adapter specs. It is
// registered by sim/lora's init() (import sim/lora to wire it), breaking the import
// cycle between sim/ (interface owner) and sim/lora/ (implementation). Returns an
// error for a malformed registry (duplicate/empty id, non-positive rank).
var NewAdapterRegistryFunc func(adapters []AdapterSpec) (AdapterRegistry, error)

// BuildAdapterRegistry builds the read-only adapter registry when the LoRA
// subsystem is active and sim/lora is linked; otherwise returns (nil, nil) so
// adapter handling stays a no-op (INV-6). The gating condition matches
// NewSimulator's resident-set wiring so the registry is non-nil exactly when the
// resident set and cost model are — the eviction context (D2) can then always
// resolve ranks whenever an eviction can occur.
func BuildAdapterRegistry(cfg SimConfig) (AdapterRegistry, error) {
	if !cfg.HasAdapters() || cfg.AdapterCapacity == nil ||
		NewResidentAdapterSetFunc == nil || NewAdapterRegistryFunc == nil {
		return nil, nil
	}
	return NewAdapterRegistryFunc(cfg.Adapters)
}
