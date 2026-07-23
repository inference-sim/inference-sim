// Package creation holds the pluggable adapter-creation policies behind the
// sim.CreationPolicy seam (Subsystem Module, design §5.3). Policies are registered
// into an unexported registry at init(); New(name) is wired into sim via sim/lora's
// init() so package sim reaches a policy without importing this package (breaking
// the sim ↔ sim/lora import cycle).
//
// To add a policy: implement sim.CreationPolicy (Initial + OnResidentMiss), then
// register it under a name in this file's init() via
// register("my-policy", func() sim.CreationPolicy { ... }). New(name) then resolves
// it. Add contract tests mirroring sim/creation_seam_test.go for the invariants
// every policy must uphold (seeding is uncharged INV-L3, no busy-loop INV-8,
// purity/determinism INV-6, and the starvation-freedom obligation) — NOT
// byte-identity with on-demand, which only the on-demand default guarantees.
package creation

import (
	"fmt"
	"sort"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
)

type constructor func() sim.CreationPolicy

// registry maps policy name → constructor. Unexported (R8); all access is via New.
var registry = map[string]constructor{}

func register(name string, c constructor) {
	if name == "" {
		panic("creation.register: empty policy name")
	}
	if c == nil {
		panic(fmt.Sprintf("creation.register: nil constructor for %q", name))
	}
	if _, dup := registry[name]; dup {
		panic(fmt.Sprintf("creation.register: duplicate policy %q", name))
	}
	registry[name] = c
}

// New builds the named creation policy, or an error listing valid names. The
// empty name resolves to on-demand — the single canonical empty→on-demand
// fallback site, so LoRAConfig's zero value never surfaces an "unknown policy"
// error (R20).
func New(name string) (sim.CreationPolicy, error) {
	if name == "" {
		name = "on-demand"
	}
	c, ok := registry[name]
	if !ok {
		return nil, fmt.Errorf("unknown creation policy %q; valid options: %s", name, validNames())
	}
	return c(), nil
}

// ValidNames returns the registered policy names, sorted. Reached from package sim
// via the sim.ValidCreationPolicyNamesFunc hook.
func ValidNames() []string {
	names := make([]string, 0, len(registry))
	for name := range registry {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// validNames returns ValidNames() as a deterministic, comma-separated string
// (R2 / INV-6): a given registry always yields the same error text.
func validNames() string {
	return strings.Join(ValidNames(), ", ")
}

func init() {
	register("on-demand", func() sim.CreationPolicy { return onDemand{} })
}
