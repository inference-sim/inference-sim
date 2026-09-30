package sim

// EvictionPolicy selects which resident adapter to evict when a slot is needed
// for a cold load. It is a pure, stateless query (the resident set owns recency
// ordering; the policy owns no state) — deterministic given its inputs (INV-6).
type EvictionPolicy interface {
	// SelectVictim returns the id to evict, chosen from ctx.Candidates, and true;
	// or ("", false) when no unpinned candidate exists (INV-8: the caller then
	// starts no load this tick and retries once a pin clears).
	SelectVictim(ctx EvictionContext) (victim string, ok bool)
}

// EvictionContext is the read-only view a policy sees at the decision point,
// constructed by the instance simulator (which owns the resident set, cost model,
// and — per D2 — the adapter registry).
type EvictionContext struct {
	// Candidates are the currently-resident, UNPINNED adapter ids in LRU→MRU
	// (eviction-priority) order. A policy MUST choose only from these; choosing
	// otherwise risks evicting a pinned adapter (INV-L5).
	Candidates []string
	// RankOf resolves an adapter id's declared rank and whether it is registered
	// (base-model / unregistered ids report false), sourced from the AdapterRegistry
	// (D2). The function value itself is always present (never nil), so a policy may
	// call it unconditionally; but it reports (0, false) for EVERY id when the
	// registry is absent (LoRA inactive) — a policy must treat a false result as
	// "rank unavailable" and never assume a rank is present. lru ignores it; the
	// rank/cost-aware policy (B-4, #1492) consumes it.
	//
	// RankOf MUST be a pure function of STATIC adapter metadata (the declared
	// registry). It must never depend on runtime history (cache hits, prior
	// evictions, arrival order) or be cached across eviction decisions —
	// otherwise determinism (INV-6) and run/replay parity (INV-13) break. Because
	// each instance builds its registry from the same immutable LoRAConfig.Adapters
	// slice, ranks are byte-identical across instances by construction; a policy
	// may read it freely within a single SelectVictim call.
	RankOf func(id string) (int, bool)
}

// NewEvictionPolicyFunc builds an eviction policy by name. Registered by
// sim/lora/eviction (via sim/lora's init()), breaking the sim ↔ sim/lora import
// cycle — package sim never imports the implementation. Returns an error for an
// unknown name.
var NewEvictionPolicyFunc func(name string) (EvictionPolicy, error)

// ValidEvictionPolicyNamesFunc returns the registered eviction-policy names,
// sorted. Registered by sim/lora/eviction via the same init()-time inversion as
// NewEvictionPolicyFunc — package sim never imports the implementation. Nil when
// LoRA is not linked in; call ValidEvictionPolicyNames for the nil-safe accessor.
var ValidEvictionPolicyNamesFunc func() []string

// ValidEvictionPolicyNames returns the registered eviction-policy names, or a nil
// slice when the LoRA package is not linked in (hook unset). Nil-safe: callers
// building help text or a fail-fast valid-names check need no nil guard, and
// strings.Join(nil, ", ") is "".
func ValidEvictionPolicyNames() []string {
	if ValidEvictionPolicyNamesFunc == nil {
		return nil
	}
	return ValidEvictionPolicyNamesFunc()
}
