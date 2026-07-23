package sim

// CreationPolicy decides adapter residency creation at two entry points (D9): the
// initial-topology seed at t=0 (Initial) and the admit decision when a request
// misses on a non-resident adapter at the cold-load gate (OnResidentMiss). It is
// one two-entry-point seam because both express one coherent placement strategy.
//
// Both methods are PURE queries — deterministic given their inputs (INV-6), with
// no side effects. The caller (the simulator) performs any resident-set mutation;
// a policy only computes what should happen.
//
// Starvation-freedom obligation: because a persistent OnResidentMiss==false holds
// a request at the cold-load gate indefinitely, any policy that can return false
// for an adapter MUST guarantee that adapter eventually becomes resident by
// another route (t=0 Initial seeding, or a later OnResidentMiss==true) — otherwise
// the enqueued request never progresses. The gate makes no liveness promise for a
// policy that starves a request (INV-8 concerns idling on runnable work, and a
// gate-blocked request is deliberately not-yet-runnable). on-demand trivially
// satisfies this (it always admits).
type CreationPolicy interface {
	// Initial returns the adapter ids to seed as resident at t=0 for an instance,
	// given that instance's assigned subset (ctx.Assigned). on-demand returns an
	// empty slice (seeds nothing). The returned ids are seeded WITHOUT a load-count
	// increment and WITHOUT cold-load latency (D4/INV-L3) — t=0 seeding is not a
	// charged load.
	Initial(ctx CreationContext) []string

	// OnResidentMiss reports whether a cold load should be admitted for the missed
	// adapter (ctx.MissedAdapter). on-demand always returns true, preserving the
	// pre-B-5 cold-load behavior. A false result holds the request at the gate this
	// step without starting a load (see the starvation-freedom obligation above).
	OnResidentMiss(ctx CreationContext) bool
}

// CreationContext is the read-only view a CreationPolicy sees, mirroring the
// EvictionContext value-struct pattern (pass-by-value, read-only registry
// accessor, no back-pointer into the simulator) so all three seams share one
// context idiom. It carries no *Request and never exposes Request.OutputTokens
// (INV-9/INV-L6), and exposes no mutators — a policy cannot mutate cluster or
// instance state through it (Principle I).
type CreationContext struct {
	// Assigned is this instance's resolved adapter subset for the Initial path
	// (empty on the miss path). Resolved cluster-side from the placement map; the
	// policy never sees the cross-instance map (Principle I).
	Assigned []string
	// MissedAdapter is the adapter id that missed on the OnResidentMiss path (empty
	// on the Initial path).
	MissedAdapter string
	// Registry is the read-only adapter registry accessor (Has/RankOf). It may be
	// nil only when the subsystem is inert; a policy MUST nil-check before use.
	Registry AdapterRegistry
}

// NewCreationPolicyFunc builds a creation policy by name. Registered by
// sim/lora/creation (via sim/lora's init()), breaking the sim ↔ sim/lora import
// cycle — package sim never imports the implementation. Returns an error for an
// unknown name.
var NewCreationPolicyFunc func(name string) (CreationPolicy, error)

// ValidCreationPolicyNamesFunc returns the registered creation-policy names,
// sorted. Registered by sim/lora/creation via the same init()-time inversion as
// NewCreationPolicyFunc. Nil when LoRA is not linked in; call
// ValidCreationPolicyNames for the nil-safe accessor.
var ValidCreationPolicyNamesFunc func() []string

// ValidCreationPolicyNames returns the registered creation-policy names, or a nil
// slice when the LoRA package is not linked in (hook unset). Nil-safe:
// strings.Join(nil, ", ") is "".
func ValidCreationPolicyNames() []string {
	if ValidCreationPolicyNamesFunc == nil {
		return nil
	}
	return ValidCreationPolicyNamesFunc()
}
