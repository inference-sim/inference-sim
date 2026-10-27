// Moved from inference-sim/blis-schemas/rules/enginerules_test.go at 70246460a379f8b1898fbf6c36a3091364118950.

package pack_test

import (
	"reflect"
	"sort"
	"testing"

	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-vllm/rules/pack"
	"github.com/inference-sim/blis-vllm/rules/v0_29"
	"github.com/inference-sim/blis-vllm/rules/v0_31"
)

// Each EngineRules method answers from the pack's own data, so a consumer moved from the
// fields to the interface sees the same values: the four facts consumers read by type
// assertion today, and the checks.
func TestEngineRulesAnswersFromThePack(t *testing.T) {
	for _, p := range []*pack.Pack{v0_29.Pack(), v0_31.Pack()} {
		t.Run(p.Version, func(t *testing.T) {
			var r rules.EngineRules = p
			if r.EngineVersion() != p.Version {
				t.Errorf("EngineVersion() = %q, want %q", r.EngineVersion(), p.Version)
			}
			if !reflect.DeepEqual(r.RuleNames(), p.Names()) {
				t.Errorf("RuleNames() = %v, want %v", r.RuleNames(), p.Names())
			}
			var want []string
			for m, ok := range p.SpeculativeMethods {
				if ok {
					want = append(want, m)
				}
				if r.SpeculativeMethodAccepted(m) != ok {
					t.Errorf("SpeculativeMethodAccepted(%q) = %v, want %v", m, !ok, ok)
				}
			}
			sort.Strings(want)
			if got := r.SpeculativeMethodNames(); !reflect.DeepEqual(got, want) {
				t.Errorf("SpeculativeMethodNames() = %v, want %v", got, want)
			}
			if r.SpeculativeMethodAccepted("no-such-method") {
				t.Error("an unknown draft method is accepted")
			}
			for m := range p.SpeculativeMethods {
				c, known := r.AsyncCompatibleSpeculative(m)
				if !known || c != p.AsyncCompatibleSpec[m] {
					t.Errorf("AsyncCompatibleSpeculative(%q) = %v, %v; want %v, true", m, c, known, p.AsyncCompatibleSpec[m])
				}
			}
			if r.EngramOnHostByDefault() != p.EngramCPUOffloadDefault {
				t.Errorf("EngramOnHostByDefault() = %v, want %v", r.EngramOnHostByDefault(), p.EngramCPUOffloadDefault)
			}
			before := map[string]bool{}
			for k, v := range p.DCPCommBackends {
				before[k] = v
			}
			got := r.DCPCommBackendNames()
			if !reflect.DeepEqual(got, p.DCPCommBackends) {
				t.Errorf("DCPCommBackendNames() = %v, want %v", got, p.DCPCommBackends)
			}
			if got != nil {
				got["edited-by-the-caller"] = true // the caller's copy, never the pack's
			}
			if !reflect.DeepEqual(p.DCPCommBackends, before) && len(before) > 0 {
				t.Error("DCPCommBackendNames returned the pack's own map: a caller's edit changed the pack")
			}
			for name, mega := range p.MegaMoEBackends {
				if r.MegaMoEBackend(name) != mega {
					t.Errorf("MegaMoEBackend(%q) = %v, want %v", name, !mega, mega)
				}
			}
		})
	}
}

// Check over a document set gives exactly the findings Apply gives for the pack the
// A rule written against the exported Problems alias is a rule this package runs: what lets
// a pack live in another module.
func TestARuleCanBeWrittenOutsideThePackage(t *testing.T) {
	p := &pack.Pack{Version: "test", Rules: []rules.Rule{{
		Name: "always", Because: "a test",
		Check: func(_ rules.Input, out *rules.Problems) { out.RuleWarnf("always", "fired") },
	}}}
	all := p.Check(rules.Input{}).All()
	if len(all) != 1 || all[0].Rule != "always" {
		t.Fatalf("Check found %v, want the one finding the rule reports", all)
	}
}
