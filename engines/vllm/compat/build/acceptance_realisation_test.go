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
