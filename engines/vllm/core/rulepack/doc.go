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
	"fmt"
	"sort"

	"github.com/inference-sim/inference-sim/engines/vllm/core/rulepack/pack"
	"github.com/inference-sim/inference-sim/engines/vllm/core/rulepack/v0_33"
	"github.com/inference-sim/inference-sim/engines/vllm/internal/upstream"
)

// Patch changes a fresh pack as the model code of one backport does (upstream.Backport): it adds
// the entries for the backport's architectures and changes nothing a model of another
// architecture reads.
type Patch func(*pack.Pack)

// patches are the patches of the backports the Upstream record lists, by commit. The pin has no
// backport (upstream.Backports), so it has none.
var patches = map[string]Patch{}

// WithBackports returns p with the patch of each backport in bs applied, in order. A backport
// with no patch in byCommit is an error, as is a patch for a commit bs does not list: the record
// and the rules agree, or the build is refused.
func WithBackports(p *pack.Pack, bs []upstream.Backport, byCommit map[string]Patch) (*pack.Pack, error) {
	listed := map[string]bool{}
	for _, b := range bs {
		listed[b.Commit] = true
		patch, ok := byCommit[b.Commit]
		if !ok {
			return nil, fmt.Errorf("backport %s (%s) has no patch in the rules", b.Commit, b.Subject)
		}
		patch(p)
	}
	for c := range byCommit {
		if !listed[c] {
			return nil, fmt.Errorf("the rules patch commit %s, which the Upstream record does not list as a backport", c)
		}
	}
	return p, nil
}

// Patches returns the patches of the build's backports, by commit.
func Patches() map[string]Patch {
	out := make(map[string]Patch, len(patches))
	for c, p := range patches {
		out[c] = p
	}
	return out
}

// Packs returns a fresh copy of every pack, by version, with the build's backports applied.
func Packs() map[string]*pack.Pack {
	p, err := WithBackports(v0_33.Pack(), upstream.Backports(), patches)
	if err != nil {
		panic("rulepack: " + err.Error())
	}
	return map[string]*pack.Pack{
		v0_33.Version: p,
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
