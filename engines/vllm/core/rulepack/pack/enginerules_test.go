// Moved from inference-sim/blis-schemas/rules/enginerules_test.go at 70246460a379f8b1898fbf6c36a3091364118950.

package pack_test

import (
	"github.com/inference-sim/inference-sim/substrate/contracts/check"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/engines/vllm/core/rulepack/pack"
	"github.com/inference-sim/inference-sim/engines/vllm/core/rulepack/v0_33"
	"github.com/inference-sim/inference-sim/substrate/contracts/enginefacts"
)

// Each EngineRules method answers from the pack's own data, so a consumer moved from the
// fields to the interface sees the same values: the four facts consumers read by type
// assertion today, and the checks.
func TestEngineRulesAnswersFromThePack(t *testing.T) {
	for _, p := range []*pack.Pack{v0_33.Pack()} {
		t.Run(p.Version, func(t *testing.T) {
			var r enginefacts.EngineRules = p
			if r.EngineVersion() != p.Version {
				t.Errorf("EngineVersion() = %q, want %q", r.EngineVersion(), p.Version)
			}
			if !reflect.DeepEqual(p.RuleNames(), p.Names()) {
				t.Errorf("RuleNames() = %v, want %v", p.RuleNames(), p.Names())
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
	p := &pack.Pack{Version: "test", Rules: []pack.Rule{{
		Name: "always", Because: "a test",
		Check: func(_ pack.Input, out *validate.Problems) { out.RuleWarnf("always", "fired") },
	}}}
	all := p.Check(pack.Input{}).All()
	if len(all) != 1 || all[0].Rule != "always" {
		t.Fatalf("Check found %v, want the one finding the rule reports", all)
	}
}

// The binding to passthrough is stated once, by the pack, and the rules pack and the lowering both
// ask it: it refuses an explicit backend beside a MoE backend that dispatches itself and
// passthrough beside any other, accepts the default and the removed names, and says nothing for a
// release with no such backends.
func TestAll2AllBindingRefusal(t *testing.T) {
	p := v0_33.Pack()
	for _, c := range []struct {
		a2a, moe string
		refused  bool
	}{
		{"", "flashinfer_moe_ep_cutedsl", false},
		{"passthrough", "flashinfer_moe_ep_cutedsl", false},
		{"pplx", "flashinfer_moe_ep_mega_deep_gemm", false},
		{"deepep_low_latency", "flashinfer_moe_ep_cutedsl", true},
		{"mori_high_throughput", "flashinfer-moe-ep-mega-deep-gemm", true},
		{"passthrough", "triton", true},
		{"passthrough", "", true},
		{"deepep_low_latency", "triton", false},
		{"", "", false},
	} {
		msg := p.All2AllBindingRefusal(c.a2a, c.moe)
		if (msg != "") != c.refused {
			t.Errorf("all2all_backend %q beside moe_backend %q: refusal %q, refused want %v", c.a2a, c.moe, msg, c.refused)
		}
		if msg != "" && !strings.Contains(msg, "vllm/config/kernel.py") {
			t.Errorf("refusal %q does not cite vLLM's source", msg)
		}
	}
	none := *p
	none.SelfCommunicatingMoEBackends = nil
	if msg := none.All2AllBindingRefusal("passthrough", "triton"); msg != "" {
		t.Errorf("a pack with no binding refused: %q", msg)
	}
}
