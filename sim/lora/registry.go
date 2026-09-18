package lora

import (
	"fmt"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// Registry is the concrete pre-declared LoRA adapter registry: an immutable
// id -> AdapterSpec map built once from LoRAConfig.Adapters. It is the single source
// of truth for adapter rank and base model; requests reference adapters by id only.
// All iteration is key-sorted (R2) so any output derived from the registry is
// deterministic. It satisfies sim.AdapterRegistry.
type Registry struct {
	entries map[string]sim.AdapterSpec
}

// NewRegistry builds a Registry from declared specs, rejecting a malformed registry:
// empty id, duplicate id, or non-positive rank (R3). This mirrors LoRAConfig.Validate
// so a registry constructed directly is as safe as one built from a validated config.
func NewRegistry(specs []sim.AdapterSpec) (*Registry, error) {
	entries := make(map[string]sim.AdapterSpec, len(specs))
	for _, s := range specs {
		if s.ID == "" {
			return nil, fmt.Errorf("lora.NewRegistry: adapter with empty id")
		}
		if s.Rank <= 0 {
			return nil, fmt.Errorf("lora.NewRegistry: adapter %q rank must be > 0, got %d", s.ID, s.Rank)
		}
		if _, dup := entries[s.ID]; dup {
			return nil, fmt.Errorf("lora.NewRegistry: duplicate adapter id %q", s.ID)
		}
		entries[s.ID] = s
	}
	return &Registry{entries: entries}, nil
}

// RankOf returns the declared rank of an adapter id and whether it is registered.
func (r *Registry) RankOf(id string) (int, bool) {
	s, ok := r.entries[id]
	if !ok {
		return 0, false
	}
	return s.Rank, true
}

// BaseModelOf returns the declared base model of an adapter id (may be "") and whether
// the id is registered.
func (r *Registry) BaseModelOf(id string) (string, bool) {
	s, ok := r.entries[id]
	if !ok {
		return "", false
	}
	return s.BaseModel, true
}

// Has reports whether an adapter id is registered.
func (r *Registry) Has(id string) bool {
	_, ok := r.entries[id]
	return ok
}

// Len returns the number of registered adapters.
func (r *Registry) Len() int { return len(r.entries) }

// IDs returns all registered adapter ids in sorted order (R2).
func (r *Registry) IDs() []string {
	ids := make([]string, 0, len(r.entries))
	for id := range r.entries {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// CheckReferences reports the first unknown adapter id among the given references, or
// nil if every non-empty reference resolves. An empty id is a base-model-only request
// and is always allowed. The error names all unknown ids (sorted, R2) for a clear
// diagnostic. This is the registry-completeness guard for workload validation.
func (r *Registry) CheckReferences(ids ...string) error {
	var unknown []string
	seen := make(map[string]struct{})
	for _, id := range ids {
		if id == "" {
			continue
		}
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		if !r.Has(id) {
			unknown = append(unknown, id)
		}
	}
	if len(unknown) == 0 {
		return nil
	}
	sort.Strings(unknown)
	return fmt.Errorf("unknown adapter id(s) not in registry: %v", unknown)
}
