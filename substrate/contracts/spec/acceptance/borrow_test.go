package acceptance

import (
	"strings"
	"testing"
)

// prof builds a measured profile for the borrowing tests. al is the acceptance length at k.
func prof(model, drafter, method string, k int, al float64, mut ...func(*Profile)) Profile {
	p := Profile{Model: model, Drafter: drafter, SpeculativeMethod: method, NumSpecTokens: k,
		Thinking: ThinkingOff, Category: "coding", AcceptanceLength: al, Method: "measured",
		Provenance: &Provenance{Source: "s", Hardware: "b300", Date: "2026-10-01",
			Sampling: &Sampling{Temperature: fp(1)}}}
	for _, m := range mut {
		m(&p)
	}
	return p
}

func fp(f float64) *float64 { return &f }

func set(name string, ps ...Profile) *ProfileSet {
	return &ProfileSet{Kind: Kind, Name: name, Profiles: ps}
}

func req(model, drafter, method string, k int, sets ...*ProfileSet) Request {
	return Request{Model: model, Drafter: drafter, SpeculativeMethod: method, NumSpecTokens: k,
		Hardware: "b300", Registry: sets, Workload: Workload{Category: "coding", Thinking: ThinkingOff}}
}

// An exact key is found as before: the new fields change nothing for it.
func TestAnExactKeyIsFoundAsBefore(t *testing.T) {
	r, err := Resolve(req("m", "", "mtp", 3, set("s", prof("m", "", "mtp", 3, 2.5), prof("m", "", "mtp", 5, 3.1))))
	if err != nil {
		t.Fatal(err)
	}
	if r.Step != StepExact || r.Borrowed() || len(r.Differed) != 0 || r.KRule != "" || r.AcceptanceLength != 2.5 || len(r.Used) != 1 {
		t.Errorf("an exact key: %+v", r)
	}
}

// D8 step 1: the same target and drafter at another K.
func TestBorrowingAcrossK(t *testing.T) {
	// K 2 and K 6 are measured; the rates at K 6 are stated, those at K 2 are not.
	k2 := prof("m", "", "mtp", 2, 2.0)
	k6 := prof("m", "", "mtp", 6, 4.0)
	withRates := prof("m", "", "mtp", 4, 3.0, func(p *Profile) { p.PerPositionRates = []float64{1, 0.6, 0.3, 0.1} })
	cases := []struct {
		name  string
		k     int
		sets  []Profile
		al    float64
		rule  KRule
		rates []float64 // nil: derived
	}{
		{"between two measured values it interpolates linearly", 4, []Profile{k2, k6}, 3.0, KInterpolated, nil},
		{"a quarter of the way between", 3, []Profile{k2, k6}, 2.5, KInterpolated, nil},
		{"above the largest it takes the largest's value", 8, []Profile{k2, k6}, 4.0, KAboveLargest, nil},
		{"above the largest with stated rates, the extra positions accept nothing", 6, []Profile{withRates}, 3.0, KAboveLargest, []float64{1, 0.6, 0.3, 0.1, 0, 0}},
		{"below the smallest with stated rates, the first K positions", 2, []Profile{withRates}, 2.6, KBelowSmallestRates, []float64{1, 0.6}},
		{"below the smallest with no stated rates, 1 + K(AL-1)/K0", 1, []Profile{k6}, 1 + 1*(4.0-1)/6, KBelowSmallestProportional, nil},
		{"below the smallest, two positions of six", 2, []Profile{k6}, 1 + 2*(4.0-1)/6, KBelowSmallestProportional, nil},
	}
	for _, c := range cases {
		r, err := Resolve(req("m", "", "mtp", c.k, set("s", c.sets...)))
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if r.Step != StepOtherK || !r.Borrowed() || !near(r.AcceptanceLength, c.al) || r.KRule != c.rule {
			t.Errorf("%s: step %s AL %v rule %s; want %s %v %s", c.name, r.Step, r.AcceptanceLength, r.KRule, StepOtherK, c.al, c.rule)
		}
		if len(r.Differed) != 1 || r.Differed[0] != "num_spec_tokens" {
			t.Errorf("%s: differed %v, want [num_spec_tokens]", c.name, r.Differed)
		}
		if c.rates != nil {
			if len(r.PerPositionRates) != len(c.rates) {
				t.Errorf("%s: rates %v, want %v", c.name, r.PerPositionRates, c.rates)
			}
			for i := range c.rates {
				if i < len(r.PerPositionRates) && !near(r.PerPositionRates[i], c.rates[i]) {
					t.Errorf("%s: rates %v, want %v", c.name, r.PerPositionRates, c.rates)
				}
			}
			if r.RatesDerived {
				t.Errorf("%s: stated rates reported as derived", c.name)
			}
		}
		// The result is a valid acceptance for the run's K.
		if err := CheckLength(r.AcceptanceLength, c.k); err != nil {
			t.Errorf("%s: %v", c.name, err)
		}
		if err := CheckRates(r.PerPositionRates, c.k); err != nil {
			t.Errorf("%s: %v", c.name, err)
		}
	}
	// An interpolation reads, and reports, both profiles it lies between.
	r, _ := Resolve(req("m", "", "mtp", 4, set("s", k2, k6)))
	if len(r.Used) != 2 {
		t.Errorf("an interpolation reports %d profiles, want the two it lies between", len(r.Used))
	}
}

// The proportional rule for a K below the smallest cannot exceed what the profile's own rates
// would give, so it does not overstate drafting's gain (D8).
func TestTheProportionalRuleDoesNotOverstate(t *testing.T) {
	for _, c := range []struct {
		k0 int
		al float64
	}{{4, 3.0}, {6, 4.0}, {8, 2.1}, {3, 4.0}} {
		p := prof("m", "", "mtp", c.k0, c.al)
		rates, err := MinimumVarianceRates(c.al, c.k0)
		if err != nil {
			t.Fatal(err)
		}
		for k := 1; k < c.k0; k++ {
			r, err := Resolve(req("m", "", "mtp", k, set("s", p)))
			if err != nil {
				t.Fatal(err)
			}
			ownRates := 1.0
			for _, x := range rates[:k] {
				ownRates += x
			}
			// The profile's own rates, in the most favourable arrangement for drafting (all
			// the mass at the front), give the most it could advance at K.
			if r.AcceptanceLength > ownRates+1e-9 {
				t.Errorf("K0 %d AL %v at K %d: borrowed %v exceeds the %v its own front-loaded rates give", c.k0, c.al, k, r.AcceptanceLength, ownRates)
			}
		}
	}
}

// D8 step 2: another drafter of the same target, the same method preferred, then the lowest
// value at the run's K.
func TestBorrowingAnotherDrafter(t *testing.T) {
	own := prof("m", "", "mtp", 3, 2.0)
	eagle := prof("m", "m-eagle", "eagle3", 3, 2.6)
	mtpB := prof("m", "m-draftb", "mtp", 3, 2.4)
	mtpC := prof("m", "m-draftc", "mtp", 3, 2.2)
	// The run's drafter is a third one with no profile: the same method comes first, and of
	// those the lowest value at K.
	r, err := Resolve(req("m", "m-new", "mtp", 3, set("s", eagle, mtpB, mtpC)))
	if err != nil {
		t.Fatal(err)
	}
	if r.Step != StepOtherDrafter || r.Profile.Drafter != "m-draftc" || !near(r.AcceptanceLength, 2.2) {
		t.Errorf("same method, lowest: step %s drafter %q AL %v; want other_drafter m-draftc 2.2", r.Step, r.Profile.Drafter, r.AcceptanceLength)
	}
	if len(r.Differed) != 1 || r.Differed[0] != "drafter" {
		t.Errorf("differed %v, want [drafter]", r.Differed)
	}
	// The same method is preferred even when another method's value is lower: the preference
	// comes before the lowest value, so it is not an accident of the numbers.
	r, err = Resolve(req("m", "m-new", "mtp", 3, set("s", prof("m", "m-eagle", "eagle3", 3, 1.9), mtpB)))
	if err != nil {
		t.Fatal(err)
	}
	if r.Profile.Drafter != "m-draftb" || !near(r.AcceptanceLength, 2.4) {
		t.Errorf("same method over a lower value: got %q %v; want m-draftb 2.4", r.Profile.Drafter, r.AcceptanceLength)
	}
	// With no profile of the same method, another method is taken, the lowest of them.
	r, err = Resolve(req("m", "m-new", "mtp", 3, set("s", eagle, prof("m", "m-other", "dspark", 3, 2.9))))
	if err != nil {
		t.Fatal(err)
	}
	if r.Profile.Drafter != "m-eagle" || !near(r.AcceptanceLength, 2.6) {
		t.Errorf("no same-method drafter: got %q %v; want m-eagle 2.6", r.Profile.Drafter, r.AcceptanceLength)
	}
	if want := []string{"drafter", "speculative_method"}; strings.Join(r.Differed, ",") != strings.Join(want, ",") {
		t.Errorf("differed %v, want %v", r.Differed, want)
	}
	// The model's own speculator is a different drafter from a separate one: a run with the
	// own speculator and only a separate drafter's profile borrows it, and says so.
	r, err = Resolve(req("m", "", "mtp", 3, set("s", mtpB)))
	if err != nil || r.Step != StepOtherDrafter || r.Differed[0] != "drafter" {
		t.Errorf("own speculator with only a separate drafter's profile: %+v %v", r, err)
	}
	// Step 1 comes before step 2: the same drafter at another K beats another drafter at the
	// run's K.
	r, err = Resolve(req("m", "", "mtp", 3, set("s", prof("m", "", "mtp", 5, 3.3), mtpC)))
	if err != nil || r.Step != StepOtherK || r.Profile.Drafter != "" {
		t.Errorf("step 1 must come first: %+v %v", r, err)
	}
	_ = own
}

// D8 step 3: the same method on another target whose graph records the same architecture.
func TestBorrowingAnotherTarget(t *testing.T) {
	arch := func(a string) func(*Profile) { return func(p *Profile) { p.Architecture = a } }
	a := prof("a", "", "mtp", 3, 2.5, arch("DeepseekV3ForCausalLM"))
	b := prof("b", "", "mtp", 3, 2.7, arch("DeepseekV3ForCausalLM"))
	other := prof("c", "", "mtp", 3, 3.5, arch("LlamaForCausalLM"))
	rq := req("m", "", "mtp", 3, set("s", a, other))
	rq.Architecture = "DeepseekV3ForCausalLM"
	r, err := Resolve(rq)
	if err != nil {
		t.Fatal(err)
	}
	if r.Step != StepOtherTarget || r.Profile.Model != "a" || len(r.Differed) != 1 || r.Differed[0] != "model" {
		t.Errorf("same architecture: %+v", r)
	}
	// Only the same method counts at this step.
	rq = req("m", "", "eagle3", 3, set("s", a))
	rq.Architecture = "DeepseekV3ForCausalLM"
	if _, err := Resolve(rq); err == nil {
		t.Error("another method on another target was borrowed")
	}
	// Without the run's architecture, or when the profile states none, this step is skipped.
	rq = req("m", "", "mtp", 3, set("s", a))
	if _, err := Resolve(rq); err == nil {
		t.Error("a target was borrowed with no architecture to match")
	}
	rq = req("m", "", "mtp", 3, set("s", prof("a", "", "mtp", 3, 2.5)))
	rq.Architecture = "DeepseekV3ForCausalLM"
	if _, err := Resolve(rq); err == nil {
		t.Error("a profile that states no architecture was borrowed across targets")
	}
	// Two targets of the same architecture tie; the tie-break picks the same one in either order.
	for _, order := range [][]Profile{{a, b}, {b, a}} {
		rq = req("m", "", "mtp", 3, set("s", order...))
		rq.Architecture = "DeepseekV3ForCausalLM"
		r, err = Resolve(rq)
		if err != nil {
			t.Fatal(err)
		}
		if r.Profile.Model != "a" {
			t.Errorf("tie between a and b resolved to %q in one order; the choice must not depend on order", r.Profile.Model)
		}
	}
}

// The tie order is the acceptance reference's: the run's hardware, then the newest source, then
// the set name. The first rule that separates two candidates decides.
func TestTheTieBreakOrder(t *testing.T) {
	arch := func(p *Profile) { p.Architecture = "X" }
	hw := func(h string) func(*Profile) { return func(p *Profile) { p.Provenance.Hardware = h } }
	date := func(d string) func(*Profile) { return func(p *Profile) { p.Provenance.Date = d } }
	run := func(sa, sb *ProfileSet) string {
		rq := req("m", "", "mtp", 3, sa, sb)
		rq.Architecture = "X"
		r, err := Resolve(rq)
		if err != nil {
			t.Fatal(err)
		}
		return r.Profile.Model
	}
	// Hardware beats a newer date.
	if got := run(set("s", prof("a", "", "mtp", 3, 2, arch, hw("b300"), date("2026-01-01"))),
		set("t", prof("b", "", "mtp", 3, 2, arch, hw("h100"), date("2026-12-01")))); got != "a" {
		t.Errorf("the run's hardware must come first, got %q", got)
	}
	// With the same hardware, the newest source.
	if got := run(set("s", prof("a", "", "mtp", 3, 2, arch, date("2026-01-01"))),
		set("t", prof("b", "", "mtp", 3, 2, arch, date("2026-12-01")))); got != "b" {
		t.Errorf("the newest source must come second, got %q", got)
	}
	// With the same date, the set name.
	if got := run(set("zeta", prof("a", "", "mtp", 3, 2, arch)), set("alpha", prof("b", "", "mtp", 3, 2, arch))); got != "b" {
		t.Errorf("the set name must come third, got %q", got)
	}
	// A profile with no date is older than any dated one.
	if got := run(set("s", prof("a", "", "mtp", 3, 2, arch, date(""))), set("t", prof("b", "", "mtp", 3, 2, arch, date("2020-01-01")))); got != "b" {
		t.Errorf("an undated profile is older than any dated one, got %q", got)
	}
}

// Two profiles that share a key but were measured under different sampling or settings are not
// interchangeable, and the lookup refuses; so does an exact key found in two sets.
func TestAmbiguityIsRefused(t *testing.T) {
	a := prof("m", "", "mtp", 3, 2.5)
	b := prof("m", "", "mtp", 3, 2.6, func(p *Profile) { p.Provenance.Sampling = &Sampling{Temperature: fp(0)} })
	// An exact key in two sets: refused (the existing rule).
	if _, err := Resolve(req("m", "", "mtp", 3, set("s", a), set("t", b))); err == nil || !strings.Contains(err.Error(), "matches in both") {
		t.Errorf("an exact key in two sets: %v", err)
	}
	// The same key at another K, in two sets with different sampling: refused while borrowing.
	_, err := Resolve(req("m", "", "mtp", 4, set("s", a), set("t", b)))
	if err == nil || !strings.Contains(err.Error(), "different sampling or settings") {
		t.Errorf("a borrowed key in two sets under different sampling: %v", err)
	}
	// The same key in two sets under the same conditions is one measurement, and is not
	// ambiguous; the tie-break picks the newer copy.
	c := prof("m", "", "mtp", 3, 2.5, func(p *Profile) { p.Provenance.Date = "2026-11-01" })
	if _, err := Resolve(req("m", "", "mtp", 4, set("s", a), set("t", c))); err != nil {
		t.Errorf("one measurement in two sets under the same conditions: %v", err)
	}
	// Different engine settings count too.
	d := prof("m", "", "mtp", 3, 2.5, func(p *Profile) { p.Provenance.Settings = map[string]string{"rejection_sample_method": "block"} })
	if _, err := Resolve(req("m", "", "mtp", 4, set("s", a), set("t", d))); err == nil {
		t.Error("different engine settings must make a shared key ambiguous")
	}
}

// exact_only, from the request or the scenario's pin, forbids borrowing; and with no profile on
// any step the run is refused.
func TestExactOnlyAndRefusal(t *testing.T) {
	s := set("s", prof("m", "", "mtp", 5, 3.1))
	rq := req("m", "", "mtp", 3, s)
	rq.ExactOnly = true
	if _, err := Resolve(rq); err == nil || !strings.Contains(err.Error(), "exact_only forbids borrowing") {
		t.Errorf("exact_only on the request: %v", err)
	}
	rq = req("m", "", "mtp", 3, s)
	rq.Scenario = &Pin{ExactOnly: true}
	if _, err := Resolve(rq); err == nil || !strings.Contains(err.Error(), "exact_only forbids borrowing") {
		t.Errorf("exact_only on the scenario's pin: %v", err)
	}
	// An exact key is still found with exact_only.
	rq = req("m", "", "mtp", 5, s)
	rq.ExactOnly = true
	if r, err := Resolve(rq); err != nil || r.Step != StepExact {
		t.Errorf("an exact key under exact_only: %+v %v", r, err)
	}
	// A different category or thinking mode is never borrowed: every step keeps both.
	rq = req("m", "", "mtp", 5, s)
	rq.Workload.Category = "math"
	if _, err := Resolve(rq); err == nil {
		t.Error("a profile of another category was borrowed")
	}
	rq = req("m", "", "mtp", 5, s)
	rq.Workload.Thinking = ThinkingOn
	if _, err := Resolve(rq); err == nil {
		t.Error("a profile of another thinking mode was borrowed")
	}
	// No profile on any step.
	if _, err := Resolve(req("other", "", "mtp", 3, s)); err == nil {
		t.Error("a run with no profile on any step was not refused")
	}
}

// A stated value is used as stated and never borrows or consults profiles.
func TestAStatedValueNeverBorrows(t *testing.T) {
	rq := req("m", "", "mtp", 3, set("s", prof("m", "", "mtp", 5, 3.1)))
	rq.CLI = &Value{NumSpecTokens: 3, AcceptanceLength: fp(2.2)}
	r, err := Resolve(rq)
	if err != nil || r.Layer != LayerCLI || r.Step != "" || r.Borrowed() || r.AcceptanceLength != 2.2 {
		t.Errorf("a command-line value: %+v %v", r, err)
	}
}

// The drafter is part of the key: the same target with two drafters has two profiles, and the
// run's own drafter is found exactly when it has one.
func TestTheDrafterIsPartOfTheKey(t *testing.T) {
	own, sep := prof("m", "", "mtp", 3, 2.0), prof("m", "m-drafter", "mtp", 3, 2.8)
	if own.Key() == sep.Key() || own.Key().String() == sep.Key().String() {
		t.Fatalf("two drafters share a key: %s %s", own.Key(), sep.Key())
	}
	s := set("s", own, sep)
	for drafter, want := range map[string]float64{"": 2.0, "m-drafter": 2.8} {
		r, err := Resolve(req("m", drafter, "mtp", 3, s))
		if err != nil || r.Step != StepExact || r.AcceptanceLength != want {
			t.Errorf("drafter %q: %+v %v; want exact %v", drafter, r, err, want)
		}
	}
	// A key with no drafter renders as it always has, so old keys still name the same profile.
	if got, want := own.Key().String(), "m/mtp/k3/thinking-off/coding"; got != want {
		t.Errorf("a key with no drafter renders %q, want %q", got, want)
	}
}

// Step 2 admits another drafter only. The same drafter under another method is not another
// drafter, and no other step takes it (steps 1 and 3 keep the method), so the lookup refuses.
func TestTheSameDrafterUnderAnotherMethodIsNotBorrowed(t *testing.T) {
	for _, drafter := range []string{"", "m-draft"} {
		_, err := Resolve(req("m", drafter, "mtp", 3, set("s", prof("m", drafter, "eagle3", 3, 2.6))))
		if err == nil || !strings.Contains(err.Error(), "no profile for") {
			t.Errorf("drafter %q: a same-drafter, other-method profile satisfied the lookup: %v", drafter, err)
		}
		// Not at another K either.
		if _, err = Resolve(req("m", drafter, "mtp", 3, set("s", prof("m", drafter, "eagle3", 5, 3.1)))); err == nil {
			t.Errorf("drafter %q: other method at another K satisfied the lookup", drafter)
		}
	}
	// A genuinely different drafter still qualifies, of whatever method, beside the same
	// drafter's other-method profile, which is ignored.
	r, err := Resolve(req("m", "", "mtp", 3, set("s", prof("m", "", "eagle3", 3, 1.5), prof("m", "m-eagle", "eagle3", 3, 2.6))))
	if err != nil || r.Step != StepOtherDrafter || r.Profile.Drafter != "m-eagle" || !near(r.AcceptanceLength, 2.6) {
		t.Errorf("a different drafter beside the same drafter's other method: %+v %v", r, err)
	}
}
