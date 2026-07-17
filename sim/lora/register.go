// register.go wires sim/lora constructors into the sim package's registration
// variables. This init() runs when any package imports sim/lora, breaking the import
// cycle between sim/ (interface owner) and sim/lora/ (implementation). Production code
// (cmd/) imports sim/lora directly; test code in package sim can use a blank import.
package lora

import "github.com/inference-sim/inference-sim/sim"

func init() {
	sim.NewAdapterRegistryFunc = func(adapters []sim.AdapterSpec) (sim.AdapterRegistry, error) {
		return NewRegistry(adapters)
	}
	sim.NewResidentAdapterSetFunc = func(capacity int) sim.ResidentAdapterSet {
		return newResidentSet(capacity)
	}
	sim.NewAdapterCostFunc = func(cfg sim.LoRAConfig) (sim.AdapterCost, error) {
		return NewCostModel(cfg)
	}
}
