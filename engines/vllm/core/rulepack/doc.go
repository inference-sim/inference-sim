// Original to blis-vllm.

// Package rulepack holds the vLLM rules packs: one per vLLM release, each the release's accepted
// values, its defaults and its version-scoped checks, implementing blis-schemas'
// rules.EngineRules.
//
// The packs moved here from blis-schemas (inference-sim/inference-sim#2021 B4), which keeps
// only the engine-neutral contract, rules.EngineRules. The pack type is this module's
// (core/rulepack/pack), and the engine block the packs read is too (core/engineconfig). A consumer
// takes a pack from this module and passes it on; the latency kernel takes it from its caller.
//
// It is a package of its own, separate from the engine model, so a cost model can import the
// packs without importing the engine. 0.33 is the one release, the one the engine executes
// (blis-vllm's Upstream), checked against vLLM v0.33.0's own source in CI; a scenario that
// declares another release is refused.
package rulepack

import (
	"sort"

	"github.com/inference-sim/blis-vllm/core/rulepack/pack"
	"github.com/inference-sim/blis-vllm/core/rulepack/v0_33"
)

// Packs returns a fresh copy of every pack, by version.
func Packs() map[string]*pack.Pack {
	return map[string]*pack.Pack{
		v0_33.Version: v0_33.Pack(),
	}
}

// Lookup returns the pack for a version, or nil.
func Lookup(version string) *pack.Pack { return Packs()[version] }

// Versions returns every version this module has a pack for, sorted.
func Versions() []string {
	out := make([]string, 0, 1)
	for v := range Packs() {
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}
