package build

import (
	"math"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/load"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
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
		DraftSampler: acceptance.DraftProbabilistic, Verifier: acceptance.VerifierBlock}
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
