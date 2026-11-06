package acceptance

import (
	"fmt"
	"strings"
)

// Layer names where a resolved acceptance came from. The layers are ordered lowest to
// highest precedence: registry, workload, scenario, cli.
type Layer string

const (
	// LayerRegistry: the registry's profile for the run's key, the default.
	LayerRegistry Layer = "registry"
	// LayerWorkload: a value the workload measured and states itself.
	LayerWorkload Layer = "workload"
	// LayerScenario: the scenario's pin, by a named profile set or an inline value.
	LayerScenario Layer = "scenario"
	// LayerCLI: a value given on the command line, for sweeps and what-ifs.
	LayerCLI Layer = "cli"
)

// Workload is what a workload says about acceptance: the category and thinking mode that
// select a profile, and optionally a value of its own.
type Workload struct {
	Category string
	Thinking Thinking
	Value    *Value
}

// Request is everything that bears on a run's acceptance. Model, SpeculativeMethod and
// NumSpecTokens come from the scenario and the deployment; Hardware, from the scenario's
// cluster, is used only to record a profile measured elsewhere.
type Request struct {
	Model             string
	SpeculativeMethod string
	NumSpecTokens     int
	Hardware          string

	// Registry is the default layer: the registry sets a consumer consults.
	Registry []*ProfileSet
	Workload Workload
	// Scenario is the scenario's pin, and ScenarioSet the loaded set its Profiles names.
	Scenario    *Pin
	ScenarioSet *ProfileSet
	CLI         *Value
	// FallbackCategory requests, from the command line, a category to substitute when the
	// workload's has none. The scenario's own fallback_category, when set, takes
	// precedence.
	FallbackCategory string
}

// Resolution is a run's acceptance and its origin, for the run to record.
type Resolution struct {
	AcceptanceLength float64
	// PerPositionRates are the stated rates, or the minimum-variance schedule when
	// RatesDerived.
	PerPositionRates []float64
	RatesDerived     bool
	Layer            Layer

	// Set and Profile are the profile set and entry a profile layer resolved to.
	Set     string
	Profile *Profile
	// Source cites the evidence: the profile's provenance source or the value's source.
	Source string
	// FallbackFrom is the workload category that had no curve when FallbackCategory
	// replaced it, and empty otherwise. The workload stating no category is recorded as
	// "(none)".
	FallbackFrom string
	// MeasuredOn is the hardware a profile was measured on when it differs from the
	// request's, and empty otherwise.
	MeasuredOn string
}

// AcceptanceRate is the accepted fraction of drafts, (AL-1)/K.
func (r Resolution) AcceptanceRate() float64 {
	return RateFromLength(r.AcceptanceLength, len(r.PerPositionRates))
}

// Resolve picks a run's acceptance from the highest layer that states one: the command
// line, then the scenario, then the workload, then the registry. Layers are not merged;
// the winning layer supplies the whole value.
//
// It refuses rather than guesses. A value stated for another K is refused, and so is a
// profile lookup that matches nothing, unless a fallback category was requested, in which
// case the substitution is recorded. A lookup that matches in two registry sets is
// ambiguous and refused.
func Resolve(req Request) (Resolution, error) {
	k := req.NumSpecTokens
	if k < 1 {
		return Resolution{}, fmt.Errorf("num_spec_tokens must be at least 1, got %d: a run without speculation has no acceptance", k)
	}
	if req.CLI != nil {
		return fromValue(LayerCLI, req.CLI, k)
	}
	fallback := req.FallbackCategory
	if pin := req.Scenario; pin != nil {
		if p := pin.Validate(); !p.OK() {
			return Resolution{}, fmt.Errorf("scenario acceptance: %s", p.Error())
		}
		if pin.FallbackCategory != "" {
			fallback = pin.FallbackCategory
		}
		if pin.Value != nil {
			return fromValue(LayerScenario, pin.Value, k)
		}
		if pin.Profiles != "" {
			if req.ScenarioSet == nil || req.ScenarioSet.Name != pin.Profiles {
				return Resolution{}, fmt.Errorf("the scenario names acceptance profile set %q, which was not supplied", pin.Profiles)
			}
			return lookup(LayerScenario, []*ProfileSet{req.ScenarioSet}, req, fallback)
		}
	}
	if req.Workload.Value != nil {
		return fromValue(LayerWorkload, req.Workload.Value, k)
	}
	return lookup(LayerRegistry, req.Registry, req, fallback)
}

func fromValue(layer Layer, v *Value, k int) (Resolution, error) {
	if p := v.Validate(); !p.OK() {
		return Resolution{}, fmt.Errorf("%s acceptance: %s", layer, p.Error())
	}
	if err := CheckK(v, k); err != nil {
		return Resolution{}, fmt.Errorf("%s acceptance: %w", layer, err)
	}
	rates, derived, err := v.Rates()
	if err != nil {
		return Resolution{}, fmt.Errorf("%s acceptance: %w", layer, err)
	}
	return Resolution{AcceptanceLength: v.Length(), PerPositionRates: rates,
		RatesDerived: derived, Layer: layer, Source: v.Source}, nil
}

func lookup(layer Layer, sets []*ProfileSet, req Request, fallback string) (Resolution, error) {
	if req.Model == "" || req.SpeculativeMethod == "" {
		return Resolution{}, fmt.Errorf("%s acceptance: a profile lookup needs the model and the speculative method", layer)
	}
	if req.Workload.Thinking == "" {
		return Resolution{}, fmt.Errorf("%s acceptance: the workload states no thinking mode, which selects the profile; state one (%v) or override the acceptance",
			layer, AllThinkingModes())
	}
	if !req.Workload.Thinking.Valid() {
		return Resolution{}, fmt.Errorf("%s acceptance: thinking %q is not one of %v", layer, req.Workload.Thinking, AllThinkingModes())
	}
	key := Key{Model: req.Model, SpeculativeMethod: req.SpeculativeMethod,
		NumSpecTokens: req.NumSpecTokens, Thinking: req.Workload.Thinking,
		Category: req.Workload.Category}
	fallbackFrom := ""
	set, prof, err := find(sets, key)
	if err != nil {
		return Resolution{}, fmt.Errorf("%s acceptance: %w", layer, err)
	}
	if prof == nil && fallback != "" && fallback != key.Category {
		fallbackFrom = key.Category
		if fallbackFrom == "" {
			fallbackFrom = "(none)"
		}
		key.Category = fallback
		if set, prof, err = find(sets, key); err != nil {
			return Resolution{}, fmt.Errorf("%s acceptance: %w", layer, err)
		}
	}
	if prof == nil {
		return Resolution{}, fmt.Errorf("%s acceptance: no profile for %s in %s; %s",
			layer, key, setNames(sets), missHint(req.Workload.Category, fallback))
	}
	rates, derived, err := prof.Rates()
	if err != nil {
		return Resolution{}, fmt.Errorf("%s acceptance: profile %s: %w", layer, key, err)
	}
	r := Resolution{AcceptanceLength: prof.AcceptanceLength, PerPositionRates: rates,
		RatesDerived: derived, Layer: layer, Set: set, Profile: prof,
		FallbackFrom: fallbackFrom}
	if prof.Provenance != nil {
		r.Source = prof.Provenance.Source
		if req.Hardware != "" && prof.Provenance.Hardware != "" &&
			!strings.EqualFold(req.Hardware, prof.Provenance.Hardware) {
			r.MeasuredOn = prof.Provenance.Hardware
		}
	}
	return r, nil
}

func missHint(category, fallback string) string {
	switch {
	case category == "" && fallback == "":
		return "the workload states no category; state one, request a fallback_category, or override the acceptance"
	case fallback == "":
		return "no fallback_category was requested, and none is applied silently"
	default:
		return fmt.Sprintf("the fallback category %q has none either", fallback)
	}
}

// find returns the one profile matching key across sets, refusing a key two sets match.
func find(sets []*ProfileSet, key Key) (string, *Profile, error) {
	var foundSet string
	var found *Profile
	for _, s := range sets {
		if s == nil {
			continue
		}
		for i := range s.Profiles {
			if s.Profiles[i].Key() != key {
				continue
			}
			if found != nil {
				return "", nil, fmt.Errorf("%s matches in both %q and %q; name one set in the scenario",
					key, foundSet, s.Name)
			}
			foundSet, found = s.Name, &s.Profiles[i]
		}
	}
	return foundSet, found, nil
}

func setNames(sets []*ProfileSet) string {
	var names []string
	for _, s := range sets {
		if s != nil {
			names = append(names, s.Name)
		}
	}
	if len(names) == 0 {
		return "no profile set (none was supplied)"
	}
	return fmt.Sprintf("set(s) %v", names)
}
