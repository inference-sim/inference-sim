package build

import (
	"math"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/load"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/blis-vllm/core/engineconfig"
)

// inference-sim/inference-sim#2136: a command-line rate stated as natural is spread with the
// registry's measured decay for the method; synthetic, or unstated, keeps the engine's schedule.
func TestACommandLineRatesRealisationReachesTheResolution(t *testing.T) {
	data, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	terms := acceptance.Workload{Category: "coding", Thinking: acceptance.ThinkingOn}
	resolve := func(r acceptance.Realisation) acceptance.Resolution {
		got, ok, err := ResolveAcceptance(AcceptanceInputs{Scenario: "s", Shape: kernelmodel.Shape{Model: "deepseek-v4.1-flash"},
			Data: data, Workload: &terms, Flags: AcceptanceFlags{Rate: 0.3, RateSet: true, Realisation: r}}, 5, "dspark")
		if err != nil || !ok {
			t.Fatalf("realisation %q: %v %v", r, ok, err)
		}
		return got.Res
	}
	nat := resolve(acceptance.RealisationNatural)
	if nat.Schedule != acceptance.ScheduleNaturalDecay || nat.Shape == nil || nat.Shape.SpeculativeMethod != "dspark" {
		t.Fatalf("natural: schedule %q shape %+v", nat.Schedule, nat.Shape)
	}
	if s := 1 + sum(nat.PerPositionRates); math.Abs(s-nat.AcceptanceLength) > 1e-9 || nat.PerPositionRates[4] <= 0 {
		t.Errorf("natural rates %v do not decay to the stated length %v", nat.PerPositionRates, nat.AcceptanceLength)
	}
	for _, r := range []acceptance.Realisation{acceptance.RealisationSynthetic, ""} {
		if got := resolve(r); got.Schedule != acceptance.ScheduleMinimumVariance {
			t.Errorf("realisation %q: schedule %q, want the engine's minimum-variance schedule", r, got.Schedule)
		}
	}
}

func sum(xs []float64) float64 {
	var s float64
	for _, x := range xs {
		s += x
	}
	return s
}

// A run's draft sampler and verifier reach the lookup: DeepSeek-V4.1-Flash, measured with
// probabilistic drafting and block verification, matches exactly at those settings, and a run
// that states neither borrows, naming both.
func TestARunsSamplerAndVerifierReachTheLookup(t *testing.T) {
	data, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	terms := acceptance.Workload{Category: "coding", Thinking: acceptance.ThinkingOn}
	in := AcceptanceInputs{Scenario: "s", Shape: kernelmodel.Shape{Model: "deepseek-v4.1-flash"}, Data: data, Workload: &terms,
		DraftSampler: acceptance.DraftProbabilistic, Verifier: acceptance.VerifierBlock, Engine: "vllm"}
	got, ok, err := ResolveAcceptance(in, 3, "dspark")
	if err != nil || !ok || got.Res.Step != acceptance.StepExact {
		t.Fatalf("at the profile's settings: %v %v step %q", ok, err, got.Res.Step)
	}
	in.DraftSampler, in.Verifier = "", ""
	got, ok, err = ResolveAcceptance(in, 3, "dspark")
	if err != nil || !ok || got.Res.Step != acceptance.StepOtherSettings {
		t.Fatalf("with neither stated: %v %v step %q", ok, err, got.Res.Step)
	}
}

// A length the engine block forces (speculative.synthetic_acceptance_length) is a synthetic
// value: the scenario layer it becomes states that realisation, and it resolves with the engine's
// minimum-variance schedule. A scenario that pins the same length as natural contradicts it.
func TestAnEngineForcedLengthIsSynthetic(t *testing.T) {
	data, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	l := 2.5
	e := engineconfig.Engine{Speculative: &engineconfig.Speculative{Method: "mtp", NumSpecTokens: 3,
		RejectionSampleMethod: "synthetic", SyntheticAcceptanceLength: &l}}
	same := l
	for name, pin := range map[string]*acceptance.Pin{
		"no pin":          nil,
		"the same length": {Value: &acceptance.Value{NumSpecTokens: 3, AcceptanceLength: &same, Source: "scenario"}},
	} {
		merged, err := kernelmodel.AcceptanceWithEngine(pin, e)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if merged.Value == nil || merged.Value.Realisation != acceptance.RealisationSynthetic {
			t.Errorf("%s: the forced value %+v does not state realisation synthetic", name, merged.Value)
		}
		got, ok, err := ResolveAcceptance(AcceptanceInputs{Scenario: "s", Data: data, Engine: "vllm",
			Shape: kernelmodel.Shape{Model: "deepseek-v4-pro", Acceptance: merged}}, 3, "mtp")
		if err != nil || !ok {
			t.Fatalf("%s: %v %v", name, ok, err)
		}
		if got.Res.Schedule != acceptance.ScheduleMinimumVariance || got.Res.AcceptanceLength != l {
			t.Errorf("%s: schedule %q length %v, want the engine's minimum-variance schedule at %v", name, got.Res.Schedule, got.Res.AcceptanceLength, l)
		}
	}
	natural := l
	if _, err := kernelmodel.AcceptanceWithEngine(&acceptance.Pin{Value: &acceptance.Value{NumSpecTokens: 3,
		AcceptanceLength: &natural, Realisation: acceptance.RealisationNatural}}, e); err == nil {
		t.Error("a scenario pinning the forced length as natural resolved; want a refusal")
	}
}

// A run at a profile's own collection settings, served by vLLM, matches that profile exactly:
// DeepSeek-V4-Pro MTP at K 3, thinking on, coding, measured with greedy drafting and standard
// verification. A deployment that states neither setting runs vLLM's defaults, which are those,
// so it matches exactly too; one that states block verification does not.
func TestARunAtAProfilesSettingsOnVLLMMatchesExactly(t *testing.T) {
	data, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	terms := acceptance.Workload{Category: "coding", Thinking: acceptance.ThinkingOn}
	base := AcceptanceInputs{Scenario: "s", Shape: kernelmodel.Shape{Model: "deepseek-v4-pro"}, Data: data, Workload: &terms}
	for name, sp := range map[string]*engineconfig.Speculative{
		"stating both":    {Method: "mtp", NumSpecTokens: 3, DraftSampleMethod: "greedy", RejectionSampleMethod: "standard"},
		"stating neither": {Method: "mtp", NumSpecTokens: 3},
	} {
		in := base.VLLMInputs(sp)
		if in.DraftSampler != acceptance.DraftGreedy || in.Verifier != acceptance.VerifierStandard || in.Engine != "vllm" {
			t.Errorf("%s: resolved %q %q on %q, want greedy standard on vllm", name, in.DraftSampler, in.Verifier, in.Engine)
		}
		got, ok, err := ResolveAcceptance(in, 3, "mtp")
		if err != nil || !ok {
			t.Fatalf("%s: %v %v", name, ok, err)
		}
		if got.Res.Step != acceptance.StepExact || len(got.Res.Differed) != 0 {
			t.Errorf("%s: step %q differed %v, want an exact match", name, got.Res.Step, got.Res.Differed)
		}
	}
	got, ok, err := ResolveAcceptance(base.VLLMInputs(&engineconfig.Speculative{Method: "mtp", NumSpecTokens: 3,
		RejectionSampleMethod: "block"}), 3, "mtp")
	if err != nil || !ok || got.Res.Step == acceptance.StepExact {
		t.Errorf("block verification: %v %v step %q, want a borrowed value", ok, err, got.Res.Step)
	}
}
