package acceptance

import (
	"math"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

func TestNaturalRatesKeepTheLengthAndDecay(t *testing.T) {
	for _, c := range []struct {
		al       float64
		k        int
		exponent float64
	}{{3.1, 7, 1.17}, {4.05, 7, 1.24}, {1.5, 1, 1}, {3.9, 8, 2}, {1.01, 3, 0.5}} {
		r, err := NaturalRates(c.al, c.k, c.exponent)
		if err != nil {
			t.Fatalf("%+v: %v", c, err)
		}
		if err := CheckRates(r, c.k); err != nil {
			t.Fatalf("%+v: not a schedule: %v", c, err)
		}
		if got := 1 + sum(r); math.Abs(got-c.al) > 1e-9 {
			t.Errorf("%+v: 1 + sum(rates) = %v, want %v", c, got, c.al)
		}
		for i := 1; i < len(r); i++ {
			if !(r[i] < r[i-1]) {
				t.Errorf("%+v: rate %d (%v) does not fall below rate %d (%v)", c, i, r[i], i-1, r[i-1])
			}
		}
	}
}

// A rate-only value at K=7 and 30%: the engine's schedule is a step, which no measured curve
// is; the natural one puts acceptance at every position.
func TestNaturalRatesAreNotTheStep(t *testing.T) {
	al := LengthFromRate(0.3, 7)
	step, _ := MinimumVarianceRates(al, 7)
	nat, err := NaturalRates(al, 7, 1.17)
	if err != nil {
		t.Fatal(err)
	}
	if step[3] != 0 || nat[6] <= 0 {
		t.Fatalf("step %v, natural %v", step, nat)
	}
	if nat[0] >= step[0] {
		t.Errorf("the natural first position %v is not below the step's %v", nat[0], step[0])
	}
}

func TestNaturalRatesOfAConstantAcceptanceAreGeometric(t *testing.T) {
	r, err := NaturalRates(3, 6, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i < len(r); i++ {
		if math.Abs(r[i]/r[i-1]-r[0]) > 1e-9 {
			t.Fatalf("exponent 1 is not geometric: %v", r)
		}
	}
}

func TestNaturalRatesAtTheBounds(t *testing.T) {
	zero, _ := NaturalRates(1, 4, 1.2)
	one, _ := NaturalRates(5, 4, 1.2)
	for i := range 4 {
		if zero[i] != 0 || one[i] != 1 {
			t.Fatalf("al 1 gives %v, al k+1 gives %v", zero, one)
		}
	}
	if _, err := NaturalRates(2, 4, 2.5); err == nil {
		t.Error("an exponent above the bound was accepted")
	}
	if _, err := NaturalRates(6, 4, 1); err == nil {
		t.Error("an acceptance length above K+1 was accepted")
	}
}

func shaped() *ProfileSet {
	s := validProfileSet()
	s.Shapes = []Shape{
		{SpeculativeMethod: "dspark", DecayExponent: 1.17, FittedFrom: []string{"a.yaml"}, Rule: "fit",
			Method: vocab.MethodCalibrated},
		{DecayExponent: 1.27, FittedFrom: []string{"a.yaml", "b.yaml"}, Rule: "fit", Method: vocab.MethodAssumed},
	}
	return s
}

func natural(rate float64, k int) *Value {
	return &Value{NumSpecTokens: k, AcceptanceRate: &rate, Realisation: RealisationNatural}
}

func TestANaturalRateOnlyValueTakesItsMethodsShape(t *testing.T) {
	set := shaped()
	r, err := Resolve(Request{Model: "deepseek-v4.1-flash", SpeculativeMethod: "dspark",
		NumSpecTokens: 7, Registry: []*ProfileSet{set}, CLI: natural(0.3, 7)})
	if err != nil {
		t.Fatal(err)
	}
	want, _ := NaturalRates(LengthFromRate(0.3, 7), 7, 1.17)
	if r.Schedule != ScheduleNaturalDecay || !r.RatesDerived || r.Shape == nil ||
		r.Shape.SpeculativeMethod != "dspark" || r.ShapeSet != set.Name {
		t.Fatalf("resolution %+v", r)
	}
	for i := range want {
		if math.Abs(r.PerPositionRates[i]-want[i]) > 1e-12 {
			t.Fatalf("rates %v, want %v", r.PerPositionRates, want)
		}
	}
	if math.Abs(1+sum(r.PerPositionRates)-r.AcceptanceLength) > 1e-9 {
		t.Errorf("1 + sum(rates) %v is not the pinned length %v", 1+sum(r.PerPositionRates), r.AcceptanceLength)
	}
}

func TestANaturalValueOfAnotherMethodTakesThePooledShape(t *testing.T) {
	r, err := Resolve(Request{Model: "m", SpeculativeMethod: "eagle3", NumSpecTokens: 3,
		Registry: []*ProfileSet{shaped()}, Scenario: &Pin{Value: natural(0.5, 3)}})
	if err != nil {
		t.Fatal(err)
	}
	if r.Schedule != ScheduleNaturalDecay || r.Shape.SpeculativeMethod != "" || r.Shape.DecayExponent != 1.27 {
		t.Fatalf("resolution %+v, shape %+v", r, r.Shape)
	}
}

func TestANaturalValueWithNoShapeIsRefused(t *testing.T) {
	_, err := Resolve(Request{Model: "m", SpeculativeMethod: "dspark", NumSpecTokens: 7,
		Registry: []*ProfileSet{validProfileSet()}, CLI: natural(0.3, 7)})
	if err == nil || !strings.Contains(err.Error(), "states none for \"dspark\"") ||
		!strings.Contains(err.Error(), "synthetic") {
		t.Fatalf("got %v", err)
	}
}

func TestASyntheticOrUnstatedValueKeepsTheEnginesSchedule(t *testing.T) {
	for _, real := range []Realisation{RealisationSynthetic, ""} {
		v := natural(0.3, 7)
		v.Realisation = real
		r, err := Resolve(Request{Model: "m", SpeculativeMethod: "dspark", NumSpecTokens: 7,
			Registry: []*ProfileSet{shaped()}, CLI: v})
		if err != nil {
			t.Fatal(err)
		}
		want, _ := MinimumVarianceRates(LengthFromRate(0.3, 7), 7)
		if r.Schedule != ScheduleMinimumVariance || r.Shape != nil || !equalRates(r.PerPositionRates, want) {
			t.Errorf("realisation %q: %+v", real, r)
		}
	}
}

func TestANaturalValueWithRatesKeepsThem(t *testing.T) {
	v := &Value{NumSpecTokens: 3, PerPositionRates: []float64{0.8, 0.5, 0.2}, Realisation: RealisationNatural}
	r, err := Resolve(Request{Model: "m", SpeculativeMethod: "dspark", NumSpecTokens: 3,
		Registry: []*ProfileSet{shaped()}, CLI: v})
	if err != nil {
		t.Fatal(err)
	}
	if r.Schedule != ScheduleStated || r.RatesDerived || !equalRates(r.PerPositionRates, v.PerPositionRates) {
		t.Fatalf("%+v", r)
	}
}

func TestAProfileResolutionNamesItsSchedule(t *testing.T) {
	r, err := Resolve(Request{Model: "deepseek-v4.1-flash", SpeculativeMethod: "dspark", NumSpecTokens: 5,
		Registry: []*ProfileSet{shaped()}, Workload: Workload{Category: "coding", Thinking: ThinkingOff}})
	if err != nil {
		t.Fatal(err)
	}
	if r.Schedule != ScheduleMinimumVariance || !r.RatesDerived {
		t.Fatalf("%+v", r)
	}
}

func TestShapesAreValidated(t *testing.T) {
	cases := []struct {
		name, field string
		mutate      func(*ProfileSet)
	}{
		{"exponent below", "shapes[0].decay_exponent", func(s *ProfileSet) { s.Shapes[0].DecayExponent = 0.4 }},
		{"exponent above", "shapes[0].decay_exponent", func(s *ProfileSet) { s.Shapes[0].DecayExponent = 2.1 }},
		{"no sources", "shapes[0].fitted_from", func(s *ProfileSet) { s.Shapes[0].FittedFrom = nil }},
		{"empty source", "shapes[0].fitted_from[0]", func(s *ProfileSet) { s.Shapes[0].FittedFrom = []string{""} }},
		{"no rule", "shapes[0].rule", func(s *ProfileSet) { s.Shapes[0].Rule = "" }},
		{"two for one method", "shapes[1].speculative_method", func(s *ProfileSet) { s.Shapes[1].SpeculativeMethod = "dspark" }},
		{"no method", "shapes[0].method", func(s *ProfileSet) { s.Shapes[0].Method = "" }},
		{"called measured", "shapes[0].method", func(s *ProfileSet) { s.Shapes[0].Method = vocab.MethodMeasured }},
	}
	if p := shaped().Validate(); !p.OK() {
		t.Fatalf("a valid shaped set was rejected:\n%s", p.Error())
	}
	for _, c := range cases {
		s := shaped()
		c.mutate(s)
		p := s.Validate()
		if p.OK() || !strings.Contains(p.Error(), c.field) {
			t.Errorf("%s: want a problem at %s, got %v", c.name, c.field, p.Error())
		}
	}
}

func TestAnUnknownRealisationIsRefused(t *testing.T) {
	v := natural(0.3, 4)
	v.Realisation = "forced"
	if p := v.Validate(); p.OK() || !strings.Contains(p.Error(), "realisation") {
		t.Fatalf("got %v", p.Error())
	}
}

func equalRates(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-12 {
			return false
		}
	}
	return true
}
