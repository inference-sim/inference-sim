// Moved from inference-sim/sim/lora/register.go at ca532abca209af41425c007974a5f3ef336f5981.

// register.go wires sim/lora constructors into the sim package's registration
// variables. This init() runs when any package imports sim/lora, breaking the import
// cycle between sim/ (interface owner) and sim/lora/ (implementation). Production code
// (cmd/) imports sim/lora directly; test code in package sim can use a blank import.
package lora

import sim "github.com/inference-sim/blis-vllm/compat/engine"

func init() {
	sim.SetNewAdapterRegistryFunc(func(adapters []sim.AdapterSpec) (sim.AdapterRegistry, error) {
		return NewRegistry(adapters)
	})
	sim.SetNewResidentAdapterSetFunc(func(capacity int) sim.ResidentAdapterSet {
		return newResidentSet(capacity)
	})
	sim.SetNewAdapterCostFunc(func(cfg sim.LoRAConfig) (sim.AdapterCost, error) {
		return NewCostModel(cfg)
	})
}
