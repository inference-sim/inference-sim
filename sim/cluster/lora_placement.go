package cluster

import (
	"fmt"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// ValidateLoRAPlacement enforces INV-PS2 (pre-placement conservation) for the
// cluster-scoped LoRAAdapterPlacement map (B-5, #1493, DD-B5-f). It is called
// once at cluster construction, before any instance is built, so an invalid
// deployment fails fast via panic (Principle V, library layer) rather than
// producing a silently mis-seeded cluster.
//
// The checks run in a fixed, deterministic order (INV-6): instance indices are
// visited in ascending order, and within each index the per-id checks precede
// the capacity check. The first violation encountered is returned, so the error
// text is a pure function of the config regardless of Go map iteration order.
//
// Checks (DD-B5-f):
//  0. subsystem guard — a non-empty placement requires the LoRA subsystem to be
//     active (registry non-nil); on-demand-only / LoRA-off deployments must not
//     carry placement.
//  1. index range — every key must lie in [0, NumInstances).
//  2. id validity — every adapter id must be non-empty and registered.
//  3. intra-index uniqueness — no id may repeat within one instance's list.
//  4. capacity — an instance's assigned count must not exceed AdapterCapacity.
//
// An empty or absent placement map is always a no-op (returns nil), independent
// of subsystem state — the zero value never surfaces an error (R20).
func ValidateLoRAPlacement(dc DeploymentConfig, registry sim.AdapterRegistry) error {
	if len(dc.LoRAAdapterPlacement) == 0 {
		return nil
	}
	if registry == nil {
		return fmt.Errorf("lora_adapter_placement set but LoRA disabled: "+
			"placement has %d instance assignment(s) yet the adapter subsystem is inactive "+
			"(no adapters/capacity configured)", len(dc.LoRAAdapterPlacement))
	}

	// Visit indices in ascending order so the first reported error is
	// deterministic (INV-6).
	indices := make([]int, 0, len(dc.LoRAAdapterPlacement))
	for idx := range dc.LoRAAdapterPlacement {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	capacity := *dc.AdapterCapacity // non-nil: registry built only when set
	for _, idx := range indices {
		ids := dc.LoRAAdapterPlacement[idx]
		if idx < 0 || idx >= dc.NumInstances {
			return fmt.Errorf("lora_adapter_placement: instance index %d out of range [0, %d)",
				idx, dc.NumInstances)
		}
		seen := make(map[string]struct{}, len(ids))
		for _, id := range ids {
			if id == "" {
				return fmt.Errorf("lora_adapter_placement: instance %d has an empty adapter id", idx)
			}
			if !registry.Has(id) {
				return fmt.Errorf("lora_adapter_placement: instance %d references unregistered adapter %q", idx, id)
			}
			if _, dup := seen[id]; dup {
				return fmt.Errorf("lora_adapter_placement: instance %d lists duplicate adapter %q", idx, id)
			}
			seen[id] = struct{}{}
		}
		if len(ids) > capacity {
			return fmt.Errorf("lora_adapter_placement: instance %d assigned %d adapters, exceeds capacity %d",
				idx, len(ids), capacity)
		}
	}
	return nil
}
