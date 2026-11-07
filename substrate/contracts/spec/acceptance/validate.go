package acceptance

import (
	"fmt"
	"math"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// Validate performs field-level validation of a profile set: the required fields are
// present, every acceptance length lies in [1, K+1], stated rates are a valid schedule
// that sums to AL-1, the evidence fields agree with the method, and no key repeats.
func (s *ProfileSet) Validate() *validate.Problems {
	p := &validate.Problems{}
	if s.Kind != Kind {
		p.Field("kind", "must be %q, got %q", Kind, s.Kind)
	}
	if s.Name == "" {
		p.Field("name", "required: a set's identity is its name")
	}
	for i, c := range s.Caveats {
		if c == "" {
			p.Field(fmt.Sprintf("caveats[%d]", i), "must not be empty")
		}
	}
	if len(s.Profiles) == 0 {
		p.Field("profiles", "a set with no profiles carries nothing")
	}
	// Two profiles with one key would leave a resolver to pick whichever came last.
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
	if pr.SpeculativeMethod == "" {
		p.Field(at+".speculative_method", "required")
	}
	switch {
	case pr.Thinking == "":
		p.Field(at+".thinking", "required: one of %v", AllThinkingModes())
	case !pr.Thinking.Valid():
		p.Field(at+".thinking", "%q is not one of %v", pr.Thinking, AllThinkingModes())
	}
	if !ValidCategory(pr.Category) {
		p.Field(at+".category",
			"%q is not a workload category: lower-case letters, digits, '_' and '-'", pr.Category)
	}
	if pr.NumSpecTokens < 1 {
		p.Field(at+".num_spec_tokens", "must be at least 1, got %d", pr.NumSpecTokens)
	} else {
		checkLengthAndRates(p, at+".acceptance_length", at+".per_position_rates",
			pr.AcceptanceLength, pr.PerPositionRates, pr.NumSpecTokens)
	}
	pr.validateEvidence(p, at)
}

// checkLengthAndRates checks an acceptance length against K and, when rates are stated,
// the rates against K and the length, reporting at the two named fields. k must be at
// least 1.
func checkLengthAndRates(p *validate.Problems, alAt, ratesAt string, al float64, rates []float64, k int) {
	alOK := p.FiniteField(alAt, al)
	if alOK {
		if err := CheckLength(al, k); err != nil {
			p.Field(alAt, "%v", err)
			alOK = false
		}
	}
	if len(rates) == 0 {
		return
	}
	if err := CheckRates(rates, k); err != nil {
		p.Field(ratesAt, "%v", err)
		return
	}
	if alOK && math.Abs(sum(rates)-(al-1)) > RatesSumTolerance {
		p.Field(ratesAt,
			"sum to %.4g, but acceptance_length %v states %.4g accepted drafts per step (tolerance %v)",
			sum(rates), al, al-1, RatesSumTolerance)
	}
}

func (pr Profile) validateEvidence(p *validate.Problems, at string) {
	switch {
	case !pr.Method.Valid():
		p.Field(at+".method", "%q is not one of %v", pr.Method, vocab.AllMethods())
		return
	case pr.Method == vocab.MethodNotCharged:
		p.Field(at+".method",
			"not_charged declares a deliberate zero cost; an acceptance length is not a cost")
		return
	case pr.Method == vocab.MethodCalibrated:
		p.Field(at+".method",
			"calibrated is a coefficient back-solved through the cost model; an acceptance length is read from a run, not fitted through one")
		return
	}
	if pr.Method != vocab.MethodMeasured && pr.Rationale == "" {
		p.Field(at+".rationale", "method %q requires a rationale", pr.Method)
	}
	if pr.Method == vocab.MethodTransferred {
		if pr.TransferredFrom == "" {
			p.Field(at+".transferred_from", "method transferred requires transferred_from")
		}
		if strings.TrimSpace(pr.TransferRule) == "" {
			p.Field(at+".transfer_rule", "method transferred requires transfer_rule, the stated rule that moved the value")
		}
	} else if pr.TransferredFrom != "" || pr.TransferRule != "" {
		p.Field(at+".transferred_from", "transferred_from and transfer_rule apply only to method transferred, got method %q", pr.Method)
	}
	if pr.Provenance == nil {
		if pr.Method != vocab.MethodAssumed {
			p.Field(at+".provenance", "method %q requires a provenance", pr.Method)
		}
		return
	}
	pr.Provenance.validate(p, at+".provenance", pr.Method == vocab.MethodMeasured)
}

func (v Provenance) validate(p *validate.Problems, at string, measured bool) {
	if v.Source == "" {
		p.Field(at+".source", "required: the document the value was read from")
	}
	if v.Prompts < 0 {
		p.Field(at+".prompts", "must not be negative")
	}
	if v.TP < 0 {
		p.Field(at+".tp", "must not be negative")
	}
	for key, val := range v.Settings {
		if key == "" || val == "" {
			p.Field(at+".settings", "a setting needs a name and a value, got %q: %q", key, val)
		}
	}
	// A measurement is reusable only as far as what it was measured under is known.
	if measured {
		if v.Hardware == "" {
			p.Field(at+".hardware", "required for a measured profile: reuse on other hardware is recorded against it")
		}
		if v.EngineImage == "" {
			p.Field(at+".engine_image", "required for a measured profile")
		}
		if v.Sampling == nil {
			p.Field(at+".sampling", "required for a measured profile: acceptance depends on it")
		}
	}
	if v.Sampling != nil {
		v.Sampling.validate(p, at+".sampling")
	}
}

func (s Sampling) validate(p *validate.Problems, at string) {
	if s.Temperature == nil {
		p.Field(at+".temperature", "required; 0 states greedy decoding")
	} else if p.FiniteField(at+".temperature", *s.Temperature) && *s.Temperature < 0 {
		p.Field(at+".temperature", "must not be negative, got %v", *s.Temperature)
	}
	if s.TopP != nil && p.FiniteField(at+".top_p", *s.TopP) && (*s.TopP <= 0 || *s.TopP > 1) {
		p.Field(at+".top_p", "must lie in (0, 1], got %v", *s.TopP)
	}
	if s.TopK != nil && *s.TopK < -1 {
		p.Field(at+".top_k", "must be at least -1 (-1 or 0 disables it), got %d", *s.TopK)
	}
	if s.MaxTokens < 0 {
		p.Field(at+".max_tokens", "must not be negative")
	}
}

// Validate performs field-level validation of an inline value: it names a K, states its
// acceptance once, and what it states is consistent and lies in [1, K+1].
func (v *Value) Validate() *validate.Problems {
	p := &validate.Problems{}
	if v.NumSpecTokens < 1 {
		p.Field("num_spec_tokens",
			"must be at least 1: a value is stated for the K it was measured at, got %d", v.NumSpecTokens)
		return p
	}
	k := v.NumSpecTokens
	switch {
	case v.AcceptanceLength != nil && v.AcceptanceRate != nil:
		p.Errorf("acceptance_length and acceptance_rate are two spellings of one number; state one")
		return p
	case v.AcceptanceLength == nil && v.AcceptanceRate == nil && len(v.PerPositionRates) == 0:
		p.Errorf("states no acceptance: set acceptance_length, acceptance_rate or per_position_rates")
		return p
	}
	if v.AcceptanceRate != nil {
		rate := *v.AcceptanceRate
		if !p.FiniteField("acceptance_rate", rate) {
			return p
		}
		if rate < 0 || rate > 1 {
			p.Field("acceptance_rate", "must lie in [0, 1]: it is the accepted fraction (AL-1)/K, got %v", rate)
			return p
		}
	}
	if v.AcceptanceLength == nil && v.AcceptanceRate == nil {
		if err := CheckRates(v.PerPositionRates, k); err != nil {
			p.Field("per_position_rates", "%v", err)
		}
		return p
	}
	alAt := "acceptance_length"
	if v.AcceptanceRate != nil {
		alAt = "acceptance_rate"
	}
	checkLengthAndRates(p, alAt, "per_position_rates", v.Length(), v.PerPositionRates, k)
	return p
}

// Validate performs field-level validation of a scenario's acceptance pin.
func (pin *Pin) Validate() *validate.Problems {
	p := &validate.Problems{}
	switch {
	case pin.Profiles != "" && pin.Value != nil:
		p.Errorf("pins acceptance by a profile set or by a value, not both")
	case pin.Profiles == "" && pin.Value == nil && pin.FallbackCategory == "":
		p.Errorf("pins nothing: name a profile set, state a value, or request a fallback_category; or omit acceptance")
	}
	if pin.Value != nil {
		if pin.FallbackCategory != "" {
			p.Field("fallback_category", "has no effect with an inline value, which no category selects")
		}
		p.Merge("value", pin.Value.Validate())
	}
	if pin.FallbackCategory != "" && !ValidCategory(pin.FallbackCategory) {
		p.Field("fallback_category",
			"%q is not a workload category: lower-case letters, digits, '_' and '-'", pin.FallbackCategory)
	}
	return p
}

// ValidateWorkloadTerms checks the acceptance terms a workload states, recording under p:
// a category, when stated, is well formed, a thinking mode is recognized, and an
// explicit value is valid. Each term is optional.
func ValidateWorkloadTerms(p *validate.Problems, category string, thinking Thinking, v *Value) {
	if category != "" && !ValidCategory(category) {
		p.Field("category",
			"%q is not a workload category: lower-case letters, digits, '_' and '-'", category)
	}
	if thinking != "" && !thinking.Valid() {
		p.Field("thinking", "%q is not one of %v", thinking, AllThinkingModes())
	}
	if v != nil {
		p.Merge("acceptance", v.Validate())
	}
}

// CheckK refuses a value stated for a K other than the deployment's. A value is not
// rescaled to another K: acceptance at K=3 says nothing exact about K=5.
func CheckK(v *Value, k int) error {
	if v != nil && v.NumSpecTokens != k {
		return fmt.Errorf(
			"the value is stated for num_spec_tokens %d, but the deployment speculates %d; "+
				"acceptance does not carry across K", v.NumSpecTokens, k)
	}
	return nil
}
