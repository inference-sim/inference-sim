// Moved from blis-latency-kernel/rules_source_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	"github.com/inference-sim/inference-sim/engines/vllm/internal/upstream"
	"github.com/inference-sim/inference-sim/substrate/contracts/check"
	. "github.com/inference-sim/inference-sim/substrate/kernel"

	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/enginefacts"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"

	"github.com/inference-sim/inference-sim/engines/vllm/core/rulepack/pack"
)

// RulesFrom keeps "no rules for this release" as no rules: a nil *pack.Pack must not become
// a non-nil interface that reads as rules that exist.
func TestRulesFromKeepsAMissingReleaseMissing(t *testing.T) {
	lookup := RulesFrom(func(v string) *pack.Pack {
		if v == "known" {
			return &pack.Pack{Version: "known"}
		}
		return nil
	})
	if r := lookup("unknown"); r != nil {
		t.Errorf("an unknown release gave non-nil rules %#v", r)
	}
	if r := lookup("known"); r == nil || r.EngineVersion() != "known" {
		t.Errorf("a known release gave %v", r)
	}

	// The same holds when the lookup's own result type is an interface whose value holds a
	// nil pointer: comparing it with the interface's zero value would call it present.
	var missing *pack.Pack
	viaInterface := RulesFrom(func(string) enginefacts.EngineRules { return missing })
	if r := viaInterface("unknown"); r != nil {
		t.Errorf("a nil *pack.Pack inside an interface gave non-nil rules %#v", r)
	}
}

// countingRules counts how often the caller's engine-block checks run.
type countingRules struct {
	enginefacts.EngineRules
	checked *int
}

func (c countingRules) CheckEngineBlock(at string, b deployment.EngineBlock) *validate.Problems {
	*c.checked++
	return c.EngineRules.CheckEngineBlock(at, b)
}

// Building a kernel validates its documents with the caller's engine, never the packs a library
// happens to register: Validate runs the checker it is handed (Bundle.Engine), and a release
// it does not know is refused by it. So this pins whose engine reads the pool's engine block
// while the kernel is built.
func TestNewValidatesWithTheCallersRules(t *testing.T) {
	in := fixtureInputs(t, "minimax-m25-h200-ep8.yaml")
	n := 0
	in.Rules = countingRules{in.Rules, &n}
	if _, err := newKernel(in); err != nil {
		t.Fatal(err)
	}
	if want := len(in.Deployment.Pools); n != want {
		t.Errorf("the caller's engine-block checks ran %d times while the kernel was built, want once per pool (%d)", n, want)
	}
}

// A scenario states its engine by kind alone (SIM-1); the release is stamped by the build (LIF-6).
// The rules lookup reads it through EngineRelease, as the kernel does, so a fixture stating only
// the kind finds its rules.
func TestAFixtureStatingOnlyTheEngineBlockFindsItsRules(t *testing.T) {
	in := fixtureInputsAt(t, filepath.Join("testdata", "glm5-h200-tp8.yaml"))
	if got := in.Rules.EngineVersion(); got != upstream.Version {
		t.Errorf("rules are for %q, want %s", got, upstream.Version)
	}
}
