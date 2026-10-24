package kernelmodel

import (
	"strings"
	"testing"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-schemas/rules"
	"github.com/inference-sim/blis-schemas/spec/deployment"
	vllmrules "github.com/inference-sim/blis-vllm/rules"
)

// The engine rules come from the caller, the vLLM rules packs of blis-vllm (inference-sim#2021
// B4): a Repos that states none is given them, so every entry point prices a release against
// the same pack, and a caller that states its own rules is never overridden.
func TestReposGetTheVLLMRulesUnlessTheyStateTheirOwn(t *testing.T) {
	r := withEngineRules(Repos{})
	if r.Rules == nil || r.RulesVersions == nil {
		t.Fatal("a Repos that states no rules was not given the vLLM packs")
	}
	for _, v := range vllmrules.Versions() {
		got := r.Rules(v)
		if got == nil || got.EngineVersion() != v {
			t.Errorf("Rules(%q) = %v, want the %s pack", v, got, v)
		}
	}
	if got := r.Rules("9.9.9"); got != nil {
		t.Errorf("Rules for a release with no pack = %v, want nil (a nil pack must not read as rules)", got)
	}
	called := false
	own := Repos{Rules: func(string) rules.EngineRules { called = true; return nil }}
	if r := withEngineRules(own); r.Rules == nil {
		t.Fatal("a caller's own rules were dropped")
	} else if r.Rules("0.31.0"); !called {
		t.Error("a caller's own rules were replaced by the vLLM packs")
	}
	if DefaultRepos().Rules == nil {
		t.Error("DefaultRepos states no rules, so every test and tool opening through it would be refused")
	}
}

// A rules value that is not a *rules.Pack is read through rules.EngineRules alone, so a
// speculative method is checked whatever the rules' concrete type. Before the switch, any
// other type was refused outright as "not a blis-schemas rules pack".
func TestTheSpeculativeMethodIsCheckedThroughTheInterface(t *testing.T) {
	r := repos()
	r.Scenarios = "../../testdata/scenarios"
	pack := func(v string) rules.EngineRules {
		if p := vllmrules.Lookup(v); p != nil {
			return wrapped{p}
		}
		return nil
	}
	r.Rules = pack
	if _, err := OpenPool(dspark, r, 0); err != nil {
		t.Errorf("DSpark through rules that are not a *rules.Pack: %v", err)
	}
	_, err := openDraftingWith(t, r, "llama-3.1-70b-instruct-h200-tp4-4node.yaml",
		&deployment.Speculative{Method: "bogus", NumSpecTokens: 3})
	if err == nil || !strings.Contains(err.Error(), `"bogus" is not accepted by engine 0.29.0`) {
		t.Errorf("got %v, want the release's refusal of an unknown method", err)
	}
}

// wrapped is a rules value of another concrete type that answers as its pack does. It holds
// the pack behind the interface, not embedded, so no *rules.Pack method or field (Names,
// SpeculativeMethods) is reachable on it: the check must go through rules.EngineRules.
type wrapped struct{ rules.EngineRules }

func openDraftingWith(t *testing.T, r Repos, scenario string, sp *deployment.Speculative) (*Model, error) {
	t.Helper()
	in, err := latencykernel.OpenInputs(scenario, r, 0)
	if err != nil {
		t.Fatal(err)
	}
	in.Deployment.Pools[0].Engine.Speculative = sp
	return fromInputs(in)
}
