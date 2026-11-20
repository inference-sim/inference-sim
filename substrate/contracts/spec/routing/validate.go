package routing

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// Validate checks a profile set: identity and role, each profile's key, a strictly
// increasing curve of finite ratios with token imbalance at least 1, the evidence fields,
// and that no key repeats.
func (s *ProfileSet) Validate() *validate.Problems {
	p := &validate.Problems{}
	if s.Kind != Kind {
		p.Field("kind", "must be %q, got %q", Kind, s.Kind)
	}
	if s.Name == "" {
		p.Field("name", "required: a set's identity is its name")
	}
	if s.Role != RoleDefault && s.Role != RoleSensitivity {
		p.Field("role", "must be %q or %q, got %q", RoleDefault, RoleSensitivity, s.Role)
	}
	if len(s.Profiles) == 0 {
		p.Field("profiles", "a set with no profiles carries nothing")
	}
	seen := map[Key]int{}
	for i, pr := range s.Profiles {
		at := fmt.Sprintf("profiles[%d]", i)
		pr.validate(p, at)
		if first, dup := seen[pr.Key()]; dup {
			p.Field(at, "repeats the key %s of profiles[%d]", pr.Key(), first)
			continue
		}
		seen[pr.Key()] = i
	}
	return p
}

func (pr Profile) validate(p *validate.Problems, at string) {
	if pr.Model == "" {
		p.Field(at+".model", "required: the catalog model name")
	}
	if !ValidCategory(pr.Category) {
		p.Field(at+".category", "%q is not a workload category: lower-case letters, digits, '_' and '-'", pr.Category)
	}
	if !pr.Phase.Valid() {
		p.Field(at+".phase", "must be %q or %q, got %q", PhaseDecode, PhasePrefill, pr.Phase)
	}
	if pr.Placement != PlacementLinear {
		p.Field(at+".placement", "only %q placement is described, got %q", PlacementLinear, pr.Placement)
	}
	if pr.ExpertParallel < 2 {
		p.Field(at+".expert_parallel", "must be at least 2: one rank has no cross-rank imbalance, got %d", pr.ExpertParallel)
	}
	if len(pr.Points) == 0 {
		p.Field(at+".points", "a curve needs at least one point")
	}
	prev := 0
	for i, pt := range pr.Points {
		pat := fmt.Sprintf("%s.points[%d]", at, i)
		if pt.BatchTokens <= prev {
			p.Field(pat+".batch_tokens", "must be positive and strictly increasing, got %d after %d", pt.BatchTokens, prev)
		}
		prev = pt.BatchTokens
		if p.FiniteField(pat+".token_imbalance", pt.TokenImbalance) && pt.TokenImbalance < 1 {
			p.Field(pat+".token_imbalance", "must be at least 1 (the busiest rank holds at least the mean), got %v", pt.TokenImbalance)
		}
		if p.FiniteField(pat+".expert_imbalance", pt.ExpertImbalance) && pt.ExpertImbalance <= 0 {
			p.Field(pat+".expert_imbalance", "must be positive, got %v", pt.ExpertImbalance)
		}
		if pt.TokenP10 != nil && pt.TokenP90 != nil && *pt.TokenP10 > *pt.TokenP90 {
			p.Field(pat, "token_p10 %v exceeds token_p90 %v", *pt.TokenP10, *pt.TokenP90)
		}
	}
	m := vocab.Method(pr.Method)
	switch {
	case !m.Valid():
		p.Field(at+".method", "%q is not one of %v", pr.Method, vocab.AllMethods())
		return
	case m == vocab.MethodNotCharged || m == vocab.MethodCalibrated:
		p.Field(at+".method", "%q does not describe a routing curve read from a capture", pr.Method)
		return
	}
	if m != vocab.MethodMeasured && pr.Rationale == "" {
		p.Field(at+".rationale", "method %q requires a rationale", pr.Method)
	}
	if pr.Provenance == nil {
		if m != vocab.MethodAssumed {
			p.Field(at+".provenance", "method %q requires a provenance", pr.Method)
		}
		return
	}
	if pr.Provenance.Source == "" {
		p.Field(at+".provenance.source", "required: the dataset the routes were read from")
	}
	if pr.Provenance.Tier != 0 && pr.Provenance.Tier != 1 && pr.Provenance.Tier != 2 {
		p.Field(at+".provenance.tier", "must be 1 or 2, got %d", pr.Provenance.Tier)
	}
}

// Validate checks an inline value.
func (v *Value) Validate() *validate.Problems {
	p := &validate.Problems{}
	if p.FiniteField("token_imbalance", v.TokenImbalance) && v.TokenImbalance < 1 {
		p.Field("token_imbalance", "must be at least 1, got %v", v.TokenImbalance)
	}
	if v.ExpertImbalance != nil && p.FiniteField("expert_imbalance", *v.ExpertImbalance) &&
		*v.ExpertImbalance <= 0 {
		p.Field("expert_imbalance", "must be positive, got %v", *v.ExpertImbalance)
	}
	return p
}

// Validate checks a scenario pin: exactly one of a set name and a value.
func (pin *Pin) Validate() *validate.Problems {
	p := &validate.Problems{}
	switch {
	case pin.Profiles == "" && pin.Value == nil:
		p.Errorf("a pin names a profile set or states a value")
	case pin.Profiles != "" && pin.Value != nil:
		p.Errorf("a pin names a profile set or states a value, not both")
	case pin.Value != nil && pin.FallbackCategory != "":
		p.Field("fallback_category", "applies to a named set, not to a value")
	case pin.Value != nil:
		p.Merge("value", pin.Value.Validate())
	}
	if pin.FallbackCategory != "" && !ValidCategory(pin.FallbackCategory) {
		p.Field("fallback_category", "%q is not a workload category", pin.FallbackCategory)
	}
	return p
}
