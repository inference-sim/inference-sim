package acceptance

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// kimi is three measurements of one target, drafter and method that differ only in the draft
// sampler and the verifier, as InferenceX publishes for Kimi-K3 with DSpark.
func kimi() *ProfileSet {
	p := func(s DraftSampler, v Verifier, al float64) Profile {
		return Profile{Model: "kimi-k3", Drafter: "kimi-k3-dspark", SpeculativeMethod: "dspark", NumSpecTokens: 3,
			Thinking: ThinkingOn, Category: "coding", DraftSampler: s, Verifier: v, AcceptanceLength: al,
			Method: vocab.MethodMeasured, Provenance: measuredProvenance()}
	}
	return &ProfileSet{Kind: Kind, Name: "s", Profiles: []Profile{
		p("", "", 2.6), p(DraftProbabilistic, VerifierBlock, 2.9), p(DraftProbabilistic, "", 2.8)}}
}

func kimiRequest(s DraftSampler, v Verifier) Request {
	return Request{Model: "kimi-k3", Drafter: "kimi-k3-dspark", SpeculativeMethod: "dspark", NumSpecTokens: 3,
		DraftSampler: s, Verifier: v, Registry: []*ProfileSet{kimi()},
		Workload: Workload{Category: "coding", Thinking: ThinkingOn}}
}

func TestProfilesThatDifferOnlyInSamplerOrVerifierAreDistinct(t *testing.T) {
	if p := kimi().Validate(); !p.OK() {
		t.Fatalf("three settings of one target were refused:\n%s", p.Error())
	}
}

func TestTheKeyRendersSamplerAndVerifierOnlyWhenStated(t *testing.T) {
	k := Key{Model: "m", SpeculativeMethod: "dspark", NumSpecTokens: 3, Thinking: ThinkingOn, Category: "coding"}
	if got := k.String(); got != "m/dspark/k3/thinking-on/coding" {
		t.Errorf("a key with neither renders %q", got)
	}
	k.DraftSampler, k.Verifier = DraftProbabilistic, VerifierBlock
	if got := k.String(); got != "m/dspark/k3/thinking-on/coding/draft-probabilistic/verify-block" {
		t.Errorf("a key with both renders %q", got)
	}
}

func TestALookupMatchesTheRunsSamplerAndVerifier(t *testing.T) {
	for _, c := range []struct {
		s  DraftSampler
		v  Verifier
		al float64
	}{{"", "", 2.6}, {DraftProbabilistic, VerifierBlock, 2.9}, {DraftProbabilistic, "", 2.8}} {
		r, err := Resolve(kimiRequest(c.s, c.v))
		if err != nil {
			t.Fatal(err)
		}
		if r.Step != StepExact || r.AcceptanceLength != c.al {
			t.Errorf("sampler %q verifier %q: step %s AL %v, want exact %v", c.s, c.v, r.Step, r.AcceptanceLength, c.al)
		}
	}
}

// A run under a verifier no profile measured borrows the same drafter's other settings, takes the
// lowest value, and names what differed; it never silently takes another verifier's curve.
func TestARunUnderAnotherVerifierBorrowsAndSaysSo(t *testing.T) {
	r, err := Resolve(kimiRequest(DraftProbabilistic, VerifierFly))
	if err != nil {
		t.Fatal(err)
	}
	if r.Step != StepOtherSettings || r.AcceptanceLength != 2.8 || strings.Join(r.Differed, ",") != "verifier" {
		t.Fatalf("step %s AL %v differed %v", r.Step, r.AcceptanceLength, r.Differed)
	}
	r, err = Resolve(kimiRequest(DraftGreedy, VerifierFly))
	if err != nil {
		t.Fatal(err)
	}
	if r.Step != StepOtherSettings || r.AcceptanceLength != 2.6 || strings.Join(r.Differed, ",") != "draft_sampler,verifier" {
		t.Fatalf("step %s AL %v differed %v", r.Step, r.AcceptanceLength, r.Differed)
	}
	req := kimiRequest(DraftGreedy, VerifierFly)
	req.ExactOnly = true
	if _, err := Resolve(req); err == nil {
		t.Error("an exact-only lookup borrowed another verifier's curve")
	}
}

func TestAnUnknownSamplerOrVerifierIsRefused(t *testing.T) {
	for _, c := range []struct {
		field  string
		mutate func(*Profile)
	}{
		{"profiles[0].draft_sampler", func(p *Profile) { p.DraftSampler = "beam" }},
		{"profiles[0].verifier", func(p *Profile) { p.Verifier = "typical" }},
		{"profiles[0].draft_sampler", func(p *Profile) { p.DraftSampler = "Greedy" }},
		{"profiles[0].verifier", func(p *Profile) { p.Verifier = "relaxed" }},
		{"profiles[0].engine", func(p *Profile) { p.Engine = "vLLM" }},
	} {
		s := kimi()
		c.mutate(&s.Profiles[0])
		if p := s.Validate(); p.OK() || !strings.Contains(p.Error(), c.field) {
			t.Errorf("want a problem at %s, got %v", c.field, p.Error())
		}
	}
	if _, err := Resolve(kimiRequest("beam", "")); err == nil || !strings.Contains(err.Error(), "draft sampler") {
		t.Errorf("a request with an unknown sampler: %v", err)
	}
}

// committedSet is the registry's InferenceX set, as committed in this repository.
func committedSet(t *testing.T) *ProfileSet {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "..", "registry", "acceptance", "inferencex-golden-al.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var s ProfileSet
	if err := yaml.Unmarshal(b, &s); err != nil {
		t.Fatal(err)
	}
	return &s
}

// A run identical to its curve's collection matches exactly: DeepSeek-V4.1-Flash was measured on
// vLLM with probabilistic drafting and block verification, and DeepSeek-V4-Pro's MTP curve left
// the sampler at its engine's default (the marker), which the requesting vLLM states as greedy.
// A run that states none of the settings is not taken for one at them: it borrows, and names
// each. Another engine's default does not stand for vLLM's.
func TestARunIdenticalToItsCollectionMatchesExactly(t *testing.T) {
	set := committedSet(t)
	// The engine states its resolved values and its own defaults (here vLLM's, greedy and
	// standard, as blis-vllm states them from its field list).
	req := func(model, method string, s DraftSampler, v Verifier, e string) Request {
		return Request{Model: model, SpeculativeMethod: method, NumSpecTokens: 3,
			DraftSampler: s, Verifier: v, Engine: e, Registry: []*ProfileSet{set},
			DefaultDraftSampler: DraftGreedy, DefaultVerifier: VerifierStandard,
			Workload: Workload{Category: "coding", Thinking: ThinkingOn}}
	}
	for _, c := range []Request{
		req("deepseek-v4.1-flash", "dspark", DraftProbabilistic, VerifierBlock, "vllm"),
		req("deepseek-v4-pro", "mtp", DraftGreedy, VerifierStandard, "vllm"),
	} {
		r, err := Resolve(c)
		if err != nil {
			t.Fatal(err)
		}
		if r.Step != StepExact || r.Borrowed() {
			t.Fatalf("%s at its own settings: step %s differed %v", c.Model, r.Step, r.Differed)
		}
	}
	r, err := Resolve(req("deepseek-v4.1-flash", "dspark", "", "", ""))
	if err != nil {
		t.Fatal(err)
	}
	if r.Step != StepOtherSettings || strings.Join(r.Differed, ",") != "draft_sampler,verifier,engine" {
		t.Fatalf("with none stated: step %s differed %v", r.Step, r.Differed)
	}
	r, err = Resolve(req("deepseek-v4-pro", "mtp", DraftGreedy, VerifierStandard, "sglang"))
	if err != nil {
		t.Fatal(err)
	}
	if r.Step != StepOtherSettings || strings.Join(r.Differed, ",") != "draft_sampler,engine" {
		t.Fatalf("another engine: step %s differed %v", r.Step, r.Differed)
	}
}

func TestAFlyRunMatchesAFlyProfileAndBorrowsOtherwise(t *testing.T) {
	s := kimi()
	fly := s.Profiles[0]
	fly.DraftSampler, fly.Verifier, fly.AcceptanceLength = DraftProbabilistic, VerifierFly, 3.1
	s.Profiles = append(s.Profiles, fly)
	req := kimiRequest(DraftProbabilistic, VerifierFly)
	req.Registry = []*ProfileSet{s}
	r, err := Resolve(req)
	if err != nil {
		t.Fatal(err)
	}
	if r.Step != StepExact || r.AcceptanceLength != 3.1 {
		t.Fatalf("fly: step %s AL %v", r.Step, r.AcceptanceLength)
	}
	r, err = Resolve(kimiRequest(DraftProbabilistic, VerifierFly))
	if err != nil {
		t.Fatal(err)
	}
	if r.Step != StepOtherSettings || strings.Join(r.Differed, ",") != "verifier" {
		t.Fatalf("fly with no fly profile: step %s differed %v", r.Step, r.Differed)
	}
}

func TestTheEngineDefaultMarkerMatchesOnlyTheRequestingEnginesDefault(t *testing.T) {
	s := kimi()
	s.Profiles = s.Profiles[:1]
	s.Profiles[0].DraftSampler, s.Profiles[0].Verifier, s.Profiles[0].Engine = DraftEngineDefault, VerifierStandard, "vllm"
	req := kimiRequest(DraftGreedy, VerifierStandard)
	req.Registry, req.Engine = []*ProfileSet{s}, "vllm"
	if r, err := Resolve(req); err != nil || r.Step != StepOtherSettings {
		t.Fatalf("with no default stated: %v %+v", err, r.Step)
	}
	req.DefaultDraftSampler = DraftGreedy
	if r, err := Resolve(req); err != nil || r.Step != StepExact {
		t.Fatalf("at the engine's stated default: %v %+v", err, r.Step)
	}
	if s.Profiles[0].DraftSampler != DraftEngineDefault {
		t.Fatal("the lookup changed the set")
	}
	req.DefaultDraftSampler = DraftProbabilistic
	if r, err := Resolve(req); err != nil || r.Step != StepOtherSettings {
		t.Fatalf("the engine's default is not the run's sampler: %v %+v", err, r.Step)
	}
	req.DefaultDraftSampler, req.Engine = DraftGreedy, "sglang"
	if r, err := Resolve(req); err != nil || r.Step == StepExact {
		t.Fatalf("another engine's default stood for vllm's: %v %+v", err, r.Step)
	}
	req = kimiRequest(DraftEngineDefault, "")
	if _, err := Resolve(req); err == nil || !strings.Contains(err.Error(), "resolved values") {
		t.Fatalf("a request stating the marker: %v", err)
	}
}
