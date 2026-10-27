// Original to blis-vllm.

// Package rules holds the vLLM rules packs: one per vLLM release, each the release's accepted
// values, its defaults and its version-scoped checks, implementing blis-schemas'
// rules.EngineRules.
//
// The packs moved here from blis-schemas (inference-sim/inference-sim#2021 B4), which keeps
// only the engine-neutral contract, rules.EngineRules. The pack type is this module's
// (rules/pack), and the engine block the packs read is too (rules/engineconfig). A consumer
// takes a pack from this module and passes it on; the latency kernel takes it from its caller.
//
// It is a module of its own, separate from the engine model, so a cost model can import the
// packs without importing the engine. 0.32 is the release the engine executes (blis-vllm's
// Upstream), checked against vLLM v0.32.0's own source in CI. 0.29 and 0.31 are validation
// and compatibility data for scenario files written against those releases; the engine
// refuses to execute them.
package rules

import (
	"sort"

	"github.com/inference-sim/blis-vllm/rules/pack"
	"github.com/inference-sim/blis-vllm/rules/v0_29"
	"github.com/inference-sim/blis-vllm/rules/v0_31"
	"github.com/inference-sim/blis-vllm/rules/v0_32"
)

// Packs returns a fresh copy of every pack, by version.
func Packs() map[string]*pack.Pack {
	return map[string]*pack.Pack{
		v0_29.Version: v0_29.Pack(),
		v0_31.Version: v0_31.Pack(),
		v0_32.Version: v0_32.Pack(),
	}
}

// Lookup returns the pack for a version, or nil.
func Lookup(version string) *pack.Pack { return Packs()[version] }

// Versions returns every version this module has a pack for, sorted.
func Versions() []string {
	out := make([]string, 0, 3)
	for v := range Packs() {
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}
