// Original to blis-vllm.

package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// museSyntheticScenario is the Muse adaptive scenario with its acceptance pin and speculative block
// rewritten: the block states rejection_sample_method synthetic and a synthetic_acceptance_length
// (blockLine), and the scenario's own pin is as pin gives it (empty removes it).
func museSyntheticScenario(t *testing.T, pin, blockLine string) (Repos, string) {
	t.Helper()
	src, err := os.ReadFile(filepath.Join("..", "..", "vllmengine", "testdata", "muse-gb200-nvl-tep8-dspark-adaptive.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	s := string(src)
	i := strings.Index(s, "acceptance:\n  value:\n")
	j := strings.Index(s, "coefficients:")
	if i < 0 || j < i {
		t.Fatal("the scenario's acceptance pin was not found")
	}
	s = s[:i] + pin + s[j:]
	const block = "        rejection_sample_method: block\n"
	if !strings.Contains(s, block) {
		t.Fatal("the scenario's rejection_sample_method line was not found")
	}
	s = strings.Replace(s, block, "        rejection_sample_method: synthetic\n"+blockLine, 1)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "s.yaml"), []byte(s), 0o600); err != nil {
		t.Fatal(err)
	}
	r := DefaultRepos()
	r.Scenarios = dir
	return r, "s.yaml"
}

// The engine block's synthetic_acceptance_length is the run's supplied acceptance: it reaches the
// shape the simulator resolves acceptance from, at the rank of the scenario's pin. Stated alone it
// becomes the pin; beside a pin that names a profile set it is the pin's value; beside the same
// value it changes nothing; beside another value it is refused.
func TestSyntheticAcceptanceLengthIsTheRunsSuppliedAcceptance(t *testing.T) {
	// No pin: the block's length is the pin.
	r, name := museSyntheticScenario(t, "", "        synthetic_acceptance_length: 3.5\n")
	sh, err := ShapeOf(name, r)
	if err != nil {
		t.Fatal(err)
	}
	v := (*acceptance.Value)(nil)
	if sh.Acceptance != nil {
		v = sh.Acceptance.Value
	}
	if v == nil || v.AcceptanceLength == nil || *v.AcceptanceLength != 3.5 || v.NumSpecTokens != 7 || v.Source != SyntheticAcceptanceSource {
		t.Fatalf("shape's acceptance %+v, want a pinned length 3.5 at 7 drafts from the engine block", sh.Acceptance)
	}

	// The pin states the same value: unchanged, and the scenario's own source is kept.
	same := "acceptance:\n  value:\n    num_spec_tokens: 7\n    acceptance_length: 3.5\n    source: \"the scenario's own\"\n"
	r, name = museSyntheticScenario(t, same, "        synthetic_acceptance_length: 3.5\n")
	if sh, err = ShapeOf(name, r); err != nil || sh.Acceptance == nil || sh.Acceptance.Value.Source != "the scenario's own" {
		t.Errorf("the same value in both: %+v (%v), want the scenario's pin unchanged", sh.Acceptance, err)
	}

	// The pin states another: refused, naming both.
	r, name = museSyntheticScenario(t, strings.Replace(same, "3.5", "2.84", 1), "        synthetic_acceptance_length: 3.5\n")
	if _, err = ShapeOf(name, r); err == nil || !strings.Contains(err.Error(), "contradicts the acceptance the scenario pins") {
		t.Errorf("a conflicting pin: %v, want a refusal", err)
	}

	// A block that states no length leaves the pin as the scenario gave it.
	r, name = museSyntheticScenario(t, same, "")
	if sh, err = ShapeOf(name, r); err != nil || sh.Acceptance == nil || *sh.Acceptance.Value.AcceptanceLength != 3.5 {
		t.Errorf("no length in the block: %+v (%v)", sh.Acceptance, err)
	}
}

// The resolver, given the pin AcceptanceWithEngine builds, returns the length with vLLM's
// minimum-variance rates, derived rather than stated, at the scenario layer; the command line
// still outranks it.
func TestSyntheticAcceptanceLengthResolvesToVLLMsSchedule(t *testing.T) {
	in, e := museAdaptiveInputs(t)
	in.Scenario.Acceptance = nil
	l := 3.5
	sp := *e.Speculative
	sp.RejectionSampleMethod, sp.SyntheticAcceptanceLength = "synthetic", &l
	e.Speculative = &sp
	got, err := resolveAcceptance(in, e, repos().Data, AcceptanceOverrides{})
	if err != nil {
		t.Fatal(err)
	}
	want, _ := acceptance.MinimumVarianceRates(3.5, 7)
	if got.AcceptanceLength != 3.5 || got.Layer != "scenario" || !got.RatesDerived || len(got.PerPositionRates) != 7 {
		t.Fatalf("resolved %+v, want length 3.5 derived at the scenario layer", got)
	}
	for i := range want {
		if got.PerPositionRates[i] != want[i] {
			t.Fatalf("rates %v, want vLLM's schedule %v", got.PerPositionRates, want)
		}
	}
	cli := 5.0
	got, err = resolveAcceptance(in, e, repos().Data, AcceptanceOverrides{CLI: cliLength(7, &cli)})
	if err != nil || got.AcceptanceLength != 5 {
		t.Errorf("with a command-line length: %+v (%v), want 5", got, err)
	}
}

// A block's synthetic length replaces a scenario pin that selects a measured profile, whichever way the pin
// selects one, and the pin it builds is one the substrate accepts: the substrate refuses a value stated
// beside a profile set, a fallback category or exact_only, so copying the pin and adding the value (as this
// merge once did) turned each valid profile-selecting scenario into an invalid one.
func TestSyntheticAcceptanceLengthReplacesAProfileSelection(t *testing.T) {
	length := 3.5
	e := engineconfig.Engine{Speculative: &engineconfig.Speculative{Method: "eagle3", NumSpecTokens: 3, SyntheticAcceptanceLength: &length}}
	for name, pin := range map[string]*acceptance.Pin{
		"a profile set":     {Profiles: "measured-eagle3"},
		"a fallback":        {FallbackCategory: "chat"},
		"exact only":        {ExactOnly: true},
		"a set and a limit": {Profiles: "measured-eagle3", FallbackCategory: "chat", ExactOnly: true},
	} {
		if errs := pin.Validate().Errors(); len(errs) != 0 {
			t.Fatalf("%s: the scenario's own pin is not valid: %v", name, errs)
		}
		got, err := AcceptanceWithEngine(pin, e)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		if got == nil || got.Value == nil || got.Value.AcceptanceLength == nil || *got.Value.AcceptanceLength != 3.5 || got.Value.NumSpecTokens != 3 {
			t.Errorf("%s: pin %+v, want the block's length 3.5 at 3 drafts", name, got)
			continue
		}
		if got.Profiles != "" || got.FallbackCategory != "" || got.ExactOnly {
			t.Errorf("%s: pin %+v keeps the profile selection beside the value", name, got)
		}
		if errs := got.Validate().Errors(); len(errs) != 0 {
			t.Errorf("%s: the pin built is invalid: %v", name, errs)
		}
	}
}
