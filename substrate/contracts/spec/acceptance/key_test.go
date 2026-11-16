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
	r, err := Resolve(kimiRequest(DraftProbabilistic, VerifierRelaxed))
	if err != nil {
		t.Fatal(err)
	}
	if r.Step != StepOtherSettings || r.AcceptanceLength != 2.8 || strings.Join(r.Differed, ",") != "verifier" {
		t.Fatalf("step %s AL %v differed %v", r.Step, r.AcceptanceLength, r.Differed)
	}
	r, err = Resolve(kimiRequest(DraftGreedy, VerifierRelaxed))
	if err != nil {
		t.Fatal(err)
	}
	if r.Step != StepOtherSettings || r.AcceptanceLength != 2.6 || strings.Join(r.Differed, ",") != "draft_sampler,verifier" {
		t.Fatalf("step %s AL %v differed %v", r.Step, r.AcceptanceLength, r.Differed)
	}
	req := kimiRequest(DraftGreedy, VerifierRelaxed)
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

// DeepSeek-V4.1-Flash was measured with probabilistic drafting and block verification. A run at
// those settings matches its profile exactly; a run that states neither is not taken for one at
// them: it borrows, and names both fields.
func TestARunAtADraftersOwnMeasuredSettingsMatchesExactly(t *testing.T) {
	set := committedSet(t)
	req := func(s DraftSampler, v Verifier) Request {
		return Request{Model: "deepseek-v4.1-flash", SpeculativeMethod: "dspark", NumSpecTokens: 3,
			DraftSampler: s, Verifier: v, Registry: []*ProfileSet{set},
			Workload: Workload{Category: "coding", Thinking: ThinkingOn}}
	}
	r, err := Resolve(req(DraftProbabilistic, VerifierBlock))
	if err != nil {
		t.Fatal(err)
	}
	if r.Step != StepExact || r.Borrowed() {
		t.Fatalf("at its own settings: step %s differed %v", r.Step, r.Differed)
	}
	r, err = Resolve(req("", ""))
	if err != nil {
		t.Fatal(err)
	}
	if r.Step != StepOtherSettings || strings.Join(r.Differed, ",") != "draft_sampler,verifier" {
		t.Fatalf("with neither stated: step %s differed %v", r.Step, r.Differed)
	}
}

func TestARelaxedRunMatchesARelaxedProfileAndBorrowsOtherwise(t *testing.T) {
	s := kimi()
	relaxed := s.Profiles[0]
	relaxed.DraftSampler, relaxed.Verifier, relaxed.AcceptanceLength = DraftProbabilistic, VerifierRelaxed, 3.1
	s.Profiles = append(s.Profiles, relaxed)
	req := kimiRequest(DraftProbabilistic, VerifierRelaxed)
	req.Registry = []*ProfileSet{s}
	r, err := Resolve(req)
	if err != nil {
		t.Fatal(err)
	}
	if r.Step != StepExact || r.AcceptanceLength != 3.1 {
		t.Fatalf("relaxed: step %s AL %v", r.Step, r.AcceptanceLength)
	}
	r, err = Resolve(kimiRequest(DraftProbabilistic, VerifierRelaxed))
	if err != nil {
		t.Fatal(err)
	}
	if r.Step != StepOtherSettings || strings.Join(r.Differed, ",") != "verifier" {
		t.Fatalf("relaxed with no relaxed profile: step %s differed %v", r.Step, r.Differed)
	}
}
