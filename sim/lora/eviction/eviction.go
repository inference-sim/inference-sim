// Package eviction holds the pluggable adapter-eviction policies extracted from
// the resident set's hardcoded LRU (Backend Swap). Policies are registered into
// an unexported registry at init(); New(name) is wired into sim via sim/lora's
// init() so package sim reaches a policy without importing this package.
//
// To add a policy: implement sim.EvictionPolicy (a single SelectVictim method
// choosing a victim from ctx.Candidates), then register it under a name in this
// file's init() via register("my-policy", func() sim.EvictionPolicy { ... }).
// New(name) then resolves it, and the B-4 selector flag exposes it to users. Add
// contract tests mirroring sim/eviction_seam_test.go for the invariants every
// policy must uphold (no-deadlock INV-8, pin-safety INV-L5, determinism INV-6) —
// NOT byte-identity with lru, which only the lru default guarantees.
package eviction

import (
	"fmt"
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
)

type constructor func() sim.EvictionPolicy

// registry maps policy name → constructor. Unexported (R8); all access is via New.
var registry = map[string]constructor{}

func register(name string, c constructor) {
	if name == "" {
		panic("eviction.register: empty policy name")
	}
	if c == nil {
		panic(fmt.Sprintf("eviction.register: nil constructor for %q", name))
	}
	if _, dup := registry[name]; dup {
		panic(fmt.Sprintf("eviction.register: duplicate policy %q", name))
	}
	registry[name] = c
}

// New builds the named eviction policy, or an error listing valid names.
func New(name string) (sim.EvictionPolicy, error) {
	c, ok := registry[name]
	if !ok {
		return nil, fmt.Errorf("unknown eviction policy %q; valid options: %s", name, validNames())
	}
	return c(), nil
}

// validNames returns the registered policy names as a deterministic,
// comma-separated string (R2 / INV-6): a given registry always yields the same
// error text, so New("bogus") is reproducible run-to-run.
func validNames() string {
	names := make([]string, 0, len(registry))
	for name := range registry {
		names = append(names, name)
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
}

func init() {
	register("lru", func() sim.EvictionPolicy { return lru{} })
}

// lru evicts the least-recently-used unpinned adapter — byte-identical to the
// resident set's former EvictLRU at the cold-load gate. Recency-only: it ignores
// ctx.RankOf (BC-2). Stateless.
type lru struct{}

func (lru) SelectVictim(ctx sim.EvictionContext) (string, bool) {
	if len(ctx.Candidates) == 0 {
		return "", false
	}
	return ctx.Candidates[0], true // Candidates are LRU→MRU; [0] is least-recently-used
}
