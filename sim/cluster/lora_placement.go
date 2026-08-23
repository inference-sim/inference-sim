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
// Checks (DD-B5-f), now shared with ValidateLoRAPlacementSchedule (Spec 4 Slice B) via
// validatePlacementMap for checks 1-4:
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
	return validatePlacementMap("lora_adapter_placement", dc.LoRAAdapterPlacement, dc, registry)
}

// validatePlacementMap runs checks 1-4 of ValidateLoRAPlacement's list over one placement map.
// label names the config field in every error, so a schedule failure never reads as a
// --lora-adapter-placement failure. Indices are visited in ascending order so the first
// reported error is a pure function of the config (INV-6). Error text is preserved
// byte-for-byte from the pre-extraction ValidateLoRAPlacement body, with the literal
// "lora_adapter_placement" label replaced by the label parameter — existing tests assert
// on these strings.
func validatePlacementMap(label string, placement map[int][]string,
	dc DeploymentConfig, registry sim.AdapterRegistry) error {
	// Visit indices in ascending order so the first reported error is
	// deterministic (INV-6).
	indices := make([]int, 0, len(placement))
	for idx := range placement {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	capacity := *dc.AdapterCapacity // non-nil: registry built only when set
	for _, idx := range indices {
		ids := placement[idx]
		if idx < 0 || idx >= dc.NumInstances {
			return fmt.Errorf("%s: instance index %d out of range [0, %d)",
				label, idx, dc.NumInstances)
		}
		seen := make(map[string]struct{}, len(ids))
		for _, id := range ids {
			if id == "" {
				return fmt.Errorf("%s: instance %d has an empty adapter id", label, idx)
			}
			if !registry.Has(id) {
				return fmt.Errorf("%s: instance %d references unregistered adapter %q", label, idx, id)
			}
			if _, dup := seen[id]; dup {
				return fmt.Errorf("%s: instance %d lists duplicate adapter %q", label, idx, id)
			}
			seen[id] = struct{}{}
		}
		if len(ids) > capacity {
			return fmt.Errorf("%s: instance %d assigned %d adapters, exceeds capacity %d",
				label, idx, len(ids), capacity)
		}
	}
	return nil
}

// ValidateLoRAPlacementSchedule applies ValidateLoRAPlacement's structural checks to EVERY
// entry of DeploymentConfig.PlacementSchedule (Spec 4 Slice B). Called once at cluster
// construction, beside ValidateLoRAPlacement, so an invalid schedule fails fast rather than
// producing a tick that silently proposes nothing actuatable.
//
// Entries are visited in order, so the first reported error is deterministic (INV-6). The
// timestamp ordering and the t=0 agreement with lora_adapter_placement are the CLI's checks
// (cmd.parseLoRAPlacementSchedule and cmd.validateLoRAScheduleFlags); this validator is about
// structure, and it is what makes the per-instance capacity bound enforced by the SIMULATOR
// rather than only by whatever generated the file.
//
// An empty or absent schedule is always a no-op (R20).
func ValidateLoRAPlacementSchedule(dc DeploymentConfig, registry sim.AdapterRegistry) error {
	if len(dc.PlacementSchedule) == 0 {
		return nil
	}
	if registry == nil {
		return fmt.Errorf("lora_placement_schedule set but LoRA disabled: %d entries yet the "+
			"adapter subsystem is inactive (no adapters/capacity configured)", len(dc.PlacementSchedule))
	}
	for i, entry := range dc.PlacementSchedule {
		label := fmt.Sprintf("lora_placement_schedule entry %d (t=%dus)", i, entry.AtUs)
		if err := validatePlacementMap(label, entry.Placement, dc, registry); err != nil {
			return err
		}
	}
	return nil
}
