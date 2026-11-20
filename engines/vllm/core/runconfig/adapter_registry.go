// Moved from inference-sim/sim/engine/adapter_registry.go at ca532abca209af41425c007974a5f3ef336f5981.

package runconfig

// AdapterRegistry is the read-only query interface over the pre-declared LoRA adapter
// registry (id -> rank[, base model]). It is a pure-query bridge type owned by sim/
// so control-plane code (routing, cost model, workload validation) can resolve adapter
// identity without importing sim/lora (Principle I: no reverse import).
//
// The concrete implementation is core/lora's Registry.
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
