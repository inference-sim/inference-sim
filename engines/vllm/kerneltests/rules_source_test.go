// Moved from blis-latency-kernel/rules_source_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"testing"

	"github.com/inference-sim/blis-schemas/rules"

	"github.com/inference-sim/blis-vllm/rules/pack"
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
	viaInterface := RulesFrom(func(string) rules.EngineRules { return missing })
	if r := viaInterface("unknown"); r != nil {
		t.Errorf("a nil *pack.Pack inside an interface gave non-nil rules %#v", r)
	}
}

// countingRules counts how often the caller's rule checks run.
type countingRules struct {
	rules.EngineRules
	checked *int
}

func (c countingRules) Check(in rules.Input) *rules.Problems {
	*c.checked++
	return c.EngineRules.Check(in)
}

// Building a kernel validates its documents with the caller's rules, never the packs a library
// happens to register: blis-schemas' Validate runs the rules it is handed (Bundle.Rules), and a
// release they do not know is refused by them. Only the field layer can refuse the documents
// (validateDocuments), so this pins whose rules the rule layer reads, not that it refuses.
func TestNewValidatesWithTheCallersRules(t *testing.T) {
	in := fixtureInputs(t, "minimax-m25-h200-ep8.yaml")
	n := 0
	in.Rules = countingRules{in.Rules, &n}
	if _, err := newKernel(in); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("the caller's rule checks ran %d times while the kernel was built, want once", n)
	}
}
