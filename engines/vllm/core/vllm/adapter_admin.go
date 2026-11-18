// Original to blis-vllm.

package vllm

import (
	"fmt"

	"github.com/inference-sim/blis-vllm/core/lora"
)

// LoadAdapter loads a LoRA adapter into the engine's worker, as vLLM's dynamic load does
// (/v1/load_lora_adapter, with lora_name name; core/lora Cache.Load). A cluster component that
// places adapters calls it; no request is involved. It returns what the worker's adapter cache did.
// An engine without LoRA (max_loras unset) refuses it, as vLLM does without enable_lora.
func (e *Engine) LoadAdapter(name string) ([]lora.Event, error) {
	if e.adapterCache == nil {
		return nil, fmt.Errorf("vllm: adapter %q: LoRA is not enabled (max_loras unset)", name)
	}
	if name == "" {
		return nil, fmt.Errorf("vllm: an adapter needs a name")
	}
	return e.adapterCache.Load(e.loraID(name))
}

// UnloadAdapter unloads a LoRA adapter, as vLLM's dynamic unload does (/v1/unload_lora_adapter;
// core/lora Cache.Unload). It reports whether the adapter was loaded.
func (e *Engine) UnloadAdapter(name string) (bool, []lora.Event, error) {
	if e.adapterCache == nil {
		return false, nil, fmt.Errorf("vllm: adapter %q: LoRA is not enabled (max_loras unset)", name)
	}
	was, ev := e.adapterCache.Unload(e.loraID(name))
	return was, ev, nil
}

// ResidentAdapterIDs is the adapters the worker's CPU cache holds, by name, oldest first: the
// adapters a request can use without a load, which the host publishes to the router (the
// lora-affinity scorer). Nil on an engine without LoRA.
func (e *Engine) ResidentAdapterIDs() []string {
	if e.adapterCache == nil {
		return nil
	}
	names := make(map[int]string, len(e.adapters))
	for n, id := range e.adapters {
		names[id] = n
	}
	var out []string
	for _, id := range e.adapterCache.Registered() {
		out = append(out, names[id])
	}
	return out
}

// AdapterCapacity is how many adapters the worker's CPU cache holds (max_cpu_loras); 0 without LoRA.
func (e *Engine) AdapterCapacity() int {
	if e.adapterCache == nil {
		return 0
	}
	return e.adapterCache.Capacity()
}
