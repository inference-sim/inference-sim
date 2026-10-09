package creation

import (
	"sort"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// tickInertPolicies names every registered creation policy that deliberately does NOT
// implement sim.PeriodicCreationPolicy. Adding a policy without touching this list
// fails TestEveryCreationPolicyDeclaresTickBehaviour below.
//
// This list exists because the tick reaches a policy through a type assertion, so a
// policy that simply forgets the method would silently never tick and the suite would
// stay green. Membership here is a claim: "this policy is gate-only on purpose."
var tickInertPolicies = map[string]string{
	"on-demand":     "shipped default: residency is driven purely by the cold-load gate (INV-L1)",
	"pre-placement": "static placement is fully expressed at t=0 via Initial; nothing to re-decide on a tick",
}

// TestEveryCreationPolicyDeclaresTickBehaviour requires every registered policy to
// either implement the optional periodic interface or be listed as deliberately
// tick-inert, with a reason. It fails in both directions, so the list cannot rot.
func TestEveryCreationPolicyDeclaresTickBehaviour(t *testing.T) {
	names := ValidNames()
	if len(names) == 0 {
		t.Fatal("no creation policies registered — registry init() did not run")
	}
	for _, name := range names {
		p, err := New(name, sim.CreationPolicyConfig{})
		if err != nil {
			t.Fatalf("New(%q): %v", name, err)
		}
		_, ticks := p.(sim.PeriodicCreationPolicy)
		reason, listed := tickInertPolicies[name]
		switch {
		case ticks && listed:
			t.Errorf("policy %q implements PeriodicCreationPolicy but is listed as tick-inert (%q) — remove it from tickInertPolicies", name, reason)
		case !ticks && !listed:
			t.Errorf("policy %q implements neither PeriodicCreationPolicy nor an entry in tickInertPolicies — a policy that silently never ticks is a bug, so declare which it is", name)
		}
	}

	// The list must not name a policy that no longer exists.
	registered := map[string]bool{}
	for _, n := range names {
		registered[n] = true
	}
	stale := []string{}
	for n := range tickInertPolicies {
		if !registered[n] {
			stale = append(stale, n)
		}
	}
	sort.Strings(stale)
	if len(stale) > 0 {
		t.Errorf("tickInertPolicies names unregistered policies %v", stale)
	}
}
