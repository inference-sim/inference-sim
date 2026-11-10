package acceptance

import (
	"fmt"
	"reflect"
	"sort"
	"strings"
)

// Step is how a profile lookup found its value (design D8). A value found by an exact key is
// StepExact; the others borrow, in the order D8 states and the lookup tries them.
type Step string

const (
	// StepExact: a measured profile has the run's exact key.
	StepExact Step = "exact"
	// StepOtherK: the same target and drafter, measured at another K.
	StepOtherK Step = "same_drafter_other_k"
	// StepOtherDrafter: another drafter of the same target.
	StepOtherDrafter Step = "other_drafter"
	// StepOtherTarget: the same method on another target of the same architecture.
	StepOtherTarget Step = "other_target"
)

// KRule is how a borrowed value was brought to the run's K. It is empty when no K changed.
type KRule string

const (
	// KInterpolated: linear between the two measured values of K that bracket the run's.
	KInterpolated KRule = "interpolated"
	// KAboveLargest: the run's K is above every measured K, so the largest's value is taken.
	KAboveLargest KRule = "above_largest"
	// KBelowSmallestRates: below the smallest measured K, the first K of its stated
	// per-position rates.
	KBelowSmallestRates KRule = "below_smallest_rates"
	// KBelowSmallestProportional: below the smallest measured K, with no stated rates,
	// 1 + K(AL-1)/K0 for the profile's AL and K0. It cannot exceed what the profile's own
	// rates would give, so it does not overstate drafting's gain.
	KBelowSmallestProportional KRule = "below_smallest_proportional"
)

// Used is a profile a resolution read, with the set that holds it.
type Used struct {
	Set     string
	Profile *Profile
}

// group is the profiles of one target, drafter and method at one category and thinking mode,
// across the measured values of K and across the sets that hold them.
type group struct {
	model, drafter, method string
	byK                    map[int]*held
}

type held struct {
	set string
	p   *Profile
}

// value is a group's acceptance at the run's K, with how it got there.
type value struct {
	al      float64
	rates   []float64 // stated or padded rates, or nil to derive
	rule    KRule
	primary held   // the profile that sets the value; the tie-break reads it
	used    []held // every profile read, primary first
}

// borrow runs steps 1 to 3 of D8 for key, whose exact profile was not found. It returns
// (_, false, nil) when no step finds one, and an error when two profiles that must agree do not.
func borrow(sets []*ProfileSet, req Request, key Key) (Resolution, bool, error) {
	groups, err := gatherGroups(sets, key)
	if err != nil {
		return Resolution{}, false, err
	}
	same := func(g *group) bool {
		return g.model == key.Model && g.drafter == key.Drafter && g.method == key.SpeculativeMethod
	}
	// Step 1: the same target and drafter at another K.
	for _, g := range groups {
		if same(g) && len(g.byK) > 0 {
			v, err := valueAt(g, key.NumSpecTokens)
			if err != nil {
				return Resolution{}, false, err
			}
			return build(StepOtherK, key, g, v, req), true, nil
		}
	}
	// Step 2: another drafter of the same target. Prefer the same method, then the lowest
	// value at the run's K. The same drafter under another method is not another drafter, and
	// no step takes it: the method is part of what a profile measures.
	var cands []*group
	for _, g := range groups {
		if g.model == key.Model && g.drafter != key.Drafter {
			cands = append(cands, g)
		}
	}
	if best, v, err := pick(cands, key, req, true); err != nil {
		return Resolution{}, false, err
	} else if best != nil {
		return build(StepOtherDrafter, key, best, v, req), true, nil
	}
	// Step 3: the same method on another target whose graph records the same architecture.
	// No value is preferred among them: they are equally close, and the tie-break decides.
	if req.Architecture != "" {
		cands = nil
		for _, g := range groups {
			if g.model != key.Model && g.method == key.SpeculativeMethod && architectureOf(g) == req.Architecture {
				cands = append(cands, g)
			}
		}
		if best, v, err := pick(cands, key, req, false); err != nil {
			return Resolution{}, false, err
		} else if best != nil {
			return build(StepOtherTarget, key, best, v, req), true, nil
		}
	}
	return Resolution{}, false, nil
}

// gatherGroups collects every profile of key's category and thinking mode, grouped by target,
// drafter and method. Two profiles of one group and one K from different sets must have been
// measured under the same sampling and settings, or the lookup refuses: they share a key and
// are not interchangeable.
func gatherGroups(sets []*ProfileSet, key Key) ([]*group, error) {
	idx := map[[3]string]*group{}
	var out []*group
	for _, s := range sets {
		if s == nil {
			continue
		}
		for i := range s.Profiles {
			p := &s.Profiles[i]
			if p.Thinking != key.Thinking || p.Category != key.Category {
				continue
			}
			gk := [3]string{p.Model, p.Drafter, p.SpeculativeMethod}
			g := idx[gk]
			if g == nil {
				g = &group{model: p.Model, drafter: p.Drafter, method: p.SpeculativeMethod, byK: map[int]*held{}}
				idx[gk] = g
				out = append(out, g)
			}
			cur := g.byK[p.NumSpecTokens]
			if cur == nil {
				g.byK[p.NumSpecTokens] = &held{s.Name, p}
				continue
			}
			if !sameConditions(cur.p, p) {
				return nil, fmt.Errorf("%s is measured in both %q and %q under different sampling or settings, "+
					"so the two are not interchangeable; name one set in the scenario", p.Key(), cur.set, s.Name)
			}
			if before(p, s.Name, cur.p, cur.set, "") {
				g.byK[p.NumSpecTokens] = &held{s.Name, p}
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		return a.model+"\x00"+a.drafter+"\x00"+a.method < b.model+"\x00"+b.drafter+"\x00"+b.method
	})
	return out, nil
}

// sameConditions reports whether two profiles were measured under the same sampling and engine
// settings, which acceptance depends on.
func sameConditions(a, b *Profile) bool {
	var as, bs *Sampling
	var aset, bset map[string]string
	if a.Provenance != nil {
		as, aset = a.Provenance.Sampling, a.Provenance.Settings
	}
	if b.Provenance != nil {
		bs, bset = b.Provenance.Sampling, b.Provenance.Settings
	}
	if len(aset) == 0 && len(bset) == 0 {
		aset, bset = nil, nil
	}
	return reflect.DeepEqual(as, bs) && reflect.DeepEqual(aset, bset)
}

// architectureOf is the architecture a group's profiles state, or "" when they state none or
// disagree.
func architectureOf(g *group) string {
	arch := ""
	for _, h := range g.byK {
		switch {
		case h.p.Architecture == "":
			return ""
		case arch == "":
			arch = h.p.Architecture
		case arch != h.p.Architecture:
			return ""
		}
	}
	return arch
}

// valueAt brings a group to K by D8's rules for the first step: an exact K is itself; between
// measured values it interpolates linearly; above the largest it takes the largest's value;
// below the smallest it takes that profile's stated rates for the first K positions, or
// 1 + K(AL-1)/K0 when it states none. It never extrapolates.
func valueAt(g *group, k int) (value, error) {
	var ks []int
	for kk := range g.byK {
		ks = append(ks, kk)
	}
	sort.Ints(ks)
	if h, ok := g.byK[k]; ok {
		return value{al: h.p.AcceptanceLength, rates: h.p.PerPositionRates, primary: *h, used: []held{*h}}, nil
	}
	lo, hi := 0, 0 // the largest measured K below k and the smallest above, 0 for none
	for _, kk := range ks {
		if kk < k {
			lo = kk
		} else if hi == 0 {
			hi = kk
		}
	}
	var v value
	switch {
	case lo > 0 && hi > 0:
		a, b := g.byK[lo], g.byK[hi]
		frac := float64(k-lo) / float64(hi-lo)
		v = value{al: a.p.AcceptanceLength + (b.p.AcceptanceLength-a.p.AcceptanceLength)*frac,
			rule: KInterpolated, primary: *a, used: []held{*a, *b}}
	case lo > 0:
		a := g.byK[lo]
		v = value{al: a.p.AcceptanceLength, rule: KAboveLargest, primary: *a, used: []held{*a}}
		if len(a.p.PerPositionRates) > 0 {
			v.rates = append(append([]float64(nil), a.p.PerPositionRates...), make([]float64, k-lo)...)
		}
	default:
		b := g.byK[hi]
		if len(b.p.PerPositionRates) > 0 {
			r := append([]float64(nil), b.p.PerPositionRates[:k]...)
			v = value{al: 1 + sum(r), rates: r, rule: KBelowSmallestRates, primary: *b, used: []held{*b}}
		} else {
			v = value{al: 1 + float64(k)*(b.p.AcceptanceLength-1)/float64(hi),
				rule: KBelowSmallestProportional, primary: *b, used: []held{*b}}
		}
	}
	if err := CheckLength(v.al, k); err != nil {
		return value{}, fmt.Errorf("borrowing %s at K %d: %w", g.model, k, err)
	}
	return v, nil
}

// pick chooses among candidate groups by D8's step 2 (preferSameMethod and the lowest value) or
// step 3 (no preference), then breaks any remaining tie in the substrate's order: the profile
// measured on the run's hardware, then the newest, then the set name, then the key.
func pick(cands []*group, key Key, req Request, step2 bool) (*group, value, error) {
	type scored struct {
		g *group
		v value
	}
	var all []scored
	for _, g := range cands {
		if len(g.byK) == 0 {
			continue
		}
		v, err := valueAt(g, key.NumSpecTokens)
		if err != nil {
			return nil, value{}, err
		}
		all = append(all, scored{g, v})
	}
	if len(all) == 0 {
		return nil, value{}, nil
	}
	if step2 {
		var sameMethod []scored
		for _, s := range all {
			if s.g.method == key.SpeculativeMethod {
				sameMethod = append(sameMethod, s)
			}
		}
		if len(sameMethod) > 0 {
			all = sameMethod
		}
		lowest := all[0].v.al
		for _, s := range all {
			if s.v.al < lowest {
				lowest = s.v.al
			}
		}
		var low []scored
		for _, s := range all {
			if s.v.al == lowest {
				low = append(low, s)
			}
		}
		all = low
	}
	sort.SliceStable(all, func(i, j int) bool {
		a, b := all[i].v.primary, all[j].v.primary
		return before(a.p, a.set, b.p, b.set, req.Hardware)
	})
	return all[0].g, all[0].v, nil
}

// before is the substrate's tie-break order (design D8, the acceptance reference): the profile
// measured on the run's hardware comes first, then the newest source, then the set name, then
// the key. A profile with no date is older than any dated one.
func before(a *Profile, aSet string, b *Profile, bSet, hardware string) bool {
	onHW := func(p *Profile) bool {
		return hardware != "" && p.Provenance != nil && strings.EqualFold(p.Provenance.Hardware, hardware)
	}
	if onHW(a) != onHW(b) {
		return onHW(a)
	}
	date := func(p *Profile) string {
		if p.Provenance == nil {
			return ""
		}
		return p.Provenance.Date
	}
	if da, db := date(a), date(b); da != db {
		return da > db
	}
	if aSet != bSet {
		return aSet < bSet
	}
	return a.Key().String() < b.Key().String()
}

// build makes the resolution for a borrowed value, naming the step and each key field that
// differed from the run's.
func build(step Step, key Key, g *group, v value, req Request) Resolution {
	rates, derived := v.rates, false
	if len(rates) == 0 {
		rates, _ = MinimumVarianceRates(v.al, key.NumSpecTokens)
		derived = true
	}
	var differed []string
	if g.model != key.Model {
		differed = append(differed, "model")
	}
	if g.drafter != key.Drafter {
		differed = append(differed, "drafter")
	}
	if g.method != key.SpeculativeMethod {
		differed = append(differed, "speculative_method")
	}
	if v.rule != "" {
		differed = append(differed, "num_spec_tokens")
	}
	p := v.primary.p
	r := Resolution{AcceptanceLength: v.al, PerPositionRates: rates, RatesDerived: derived,
		Set: v.primary.set, Profile: p, Step: step, Differed: differed, KRule: v.rule}
	for _, h := range v.used {
		r.Used = append(r.Used, Used{Set: h.set, Profile: h.p})
	}
	if p.Provenance != nil {
		r.Source = p.Provenance.Source
		if req.Hardware != "" && p.Provenance.Hardware != "" && !strings.EqualFold(req.Hardware, p.Provenance.Hardware) {
			r.MeasuredOn = p.Provenance.Hardware
		}
	}
	return r
}
