package routing

import (
	"fmt"
	"math"
	"sort"
)

// Layer names where a resolved imbalance came from, lowest to highest precedence.
type Layer string

const (
	// LayerDefault: nothing stated one, so the ranks are priced as evenly loaded. Recorded,
	// so a run that priced no imbalance says so.
	LayerDefault Layer = "default"
	// LayerRegistry: a default-role registry set's profile for the run's key.
	LayerRegistry Layer = "registry"
	// LayerWorkload: a value the workload states itself.
	LayerWorkload Layer = "workload"
	// LayerScenario: the scenario's pin, by a named set (default or sensitivity) or a value.
	LayerScenario Layer = "scenario"
	// LayerCLI: a value given on the command line.
	LayerCLI Layer = "cli"
)

// Workload is what a workload says about routing: the category that selects a profile, and
// optionally a value of its own.
type Workload struct {
	Category string
	Value    *Value
}

// Request is everything that bears on a run's routing imbalance.
type Request struct {
	Model          string
	ExpertParallel int
	// EPLB is whether the deployment rearranges experts. The curves describe linear
	// placement only, so with EPLB a profile lookup does not apply.
	EPLB     bool
	Hardware string

	Registry    []*ProfileSet
	Workload    Workload
	Scenario    *Pin
	ScenarioSet *ProfileSet
	CLI         *Value
}

// Resolution is a run's imbalance and its origin. Exactly one of Value and the two curves
// is set, except at the default layer, where neither is and both ratios are 1.
type Resolution struct {
	Layer   Layer
	Set     string
	Decode  *Profile
	Prefill *Profile
	Value   *Value
	Source  string
	// Reason explains the default layer: why no profile applied.
	Reason string
	// MeasuredOn is the hardware the profile was captured on when it differs from the run's.
	MeasuredOn string
	// FallbackFrom is the workload's category when the pin's fallback replaced it.
	FallbackFrom string
}

// Lookup is a point read off a resolution, and whether the batch lay outside the curve's
// measured range (the end value is then held).
type Lookup struct {
	TokenImbalance  float64
	ExpertImbalance float64
	Clamped         bool
}

// At returns the imbalance for a step of batch tokens in the given phase. A phase with no
// curve uses the other phase's curve; a resolution with only a decode curve prices a
// prefill chunk at decode's ratio, which understates it (a chunk's tokens route alike).
func (r Resolution) At(phase Phase, batch int) Lookup {
	if r.Value != nil {
		return Lookup{TokenImbalance: r.Value.TokenImbalance, ExpertImbalance: r.Value.Expert()}
	}
	p := r.Decode
	if phase == PhasePrefill && r.Prefill != nil || p == nil {
		p = r.Prefill
	}
	if p == nil {
		return Lookup{TokenImbalance: 1, ExpertImbalance: 1}
	}
	return p.At(batch)
}

// At interpolates the curve in log(batch), holding the end values outside it.
func (p Profile) At(batch int) Lookup {
	pts := p.Points
	if len(pts) == 0 {
		return Lookup{TokenImbalance: 1, ExpertImbalance: 1}
	}
	if batch <= pts[0].BatchTokens {
		return Lookup{pts[0].TokenImbalance, pts[0].ExpertImbalance, batch < pts[0].BatchTokens}
	}
	last := pts[len(pts)-1]
	if batch >= last.BatchTokens {
		return Lookup{last.TokenImbalance, last.ExpertImbalance, batch > last.BatchTokens}
	}
	i := sort.Search(len(pts), func(i int) bool { return pts[i].BatchTokens >= batch })
	a, b := pts[i-1], pts[i]
	t := (math.Log(float64(batch)) - math.Log(float64(a.BatchTokens))) /
		(math.Log(float64(b.BatchTokens)) - math.Log(float64(a.BatchTokens)))
	return Lookup{
		TokenImbalance:  a.TokenImbalance + t*(b.TokenImbalance-a.TokenImbalance),
		ExpertImbalance: a.ExpertImbalance + t*(b.ExpertImbalance-a.ExpertImbalance),
	}
}

// Resolve picks a run's routing imbalance from the highest layer that states one: the
// command line, then the scenario, then the workload, then the default-role registry sets,
// then the documented default (evenly loaded). Layers are not merged.
//
// A run without expert parallelism has no cross-rank imbalance and resolves to the default.
// A scenario naming a set that was not supplied, or a set without the run's key, is refused:
// a named set is a request, not a hint. A registry miss is not refused; it falls to the
// default and the reason is recorded.
func Resolve(req Request) (Resolution, error) {
	if req.ExpertParallel < 2 {
		return Resolution{Layer: LayerDefault, Reason: "no expert parallelism"}, nil
	}
	if req.CLI != nil {
		return fromValue(LayerCLI, req.CLI)
	}
	if pin := req.Scenario; pin != nil {
		if p := pin.Validate(); !p.OK() {
			return Resolution{}, fmt.Errorf("scenario routing: %s", p.Error())
		}
		if pin.Value != nil {
			return fromValue(LayerScenario, pin.Value)
		}
		if req.ScenarioSet == nil || req.ScenarioSet.Name != pin.Profiles {
			return Resolution{}, fmt.Errorf("the scenario names routing profile set %q, which was not supplied", pin.Profiles)
		}
		if req.EPLB {
			return Resolution{}, fmt.Errorf("the scenario names routing profile set %q, whose curves are for linear placement, but the deployment enables EPLB", pin.Profiles)
		}
		r, ok, err := lookup(LayerScenario, []*ProfileSet{req.ScenarioSet}, req)
		if err != nil {
			return Resolution{}, err
		}
		if fb := pin.FallbackCategory; !ok && fb != "" && fb != req.Workload.Category {
			from := req.Workload.Category
			if from == "" {
				from = "(none)"
			}
			req.Workload.Category = fb
			if r, ok, err = lookup(LayerScenario, []*ProfileSet{req.ScenarioSet}, req); err != nil {
				return Resolution{}, err
			}
			r.FallbackFrom = from
		}
		if !ok {
			return Resolution{}, fmt.Errorf("routing profile set %q has no profile for %s",
				pin.Profiles, key(req, PhaseDecode))
		}
		return r, nil
	}
	if req.Workload.Value != nil {
		return fromValue(LayerWorkload, req.Workload.Value)
	}
	if req.EPLB {
		return Resolution{Layer: LayerDefault, Reason: "EPLB is enabled and no profile describes its placement"}, nil
	}
	var defaults []*ProfileSet
	for _, s := range req.Registry {
		if s != nil && s.Role == RoleDefault {
			defaults = append(defaults, s)
		}
	}
	r, ok, err := lookup(LayerRegistry, defaults, req)
	if err != nil {
		return Resolution{}, err
	}
	if !ok {
		return Resolution{Layer: LayerDefault, Reason: fmt.Sprintf(
			"no default-role routing profile for %s", key(req, PhaseDecode))}, nil
	}
	return r, nil
}

func fromValue(layer Layer, v *Value) (Resolution, error) {
	if p := v.Validate(); !p.OK() {
		return Resolution{}, fmt.Errorf("%s routing: %s", layer, p.Error())
	}
	return Resolution{Layer: layer, Value: v, Source: v.Source}, nil
}

func key(req Request, phase Phase) Key {
	return Key{Model: req.Model, Category: req.Workload.Category, Phase: phase,
		Placement: PlacementLinear, ExpertParallel: req.ExpertParallel}
}

// lookup finds the decode and prefill curves for the run's key. Either may be absent; with
// neither, ok is false. A key present in two sets is refused.
func lookup(layer Layer, sets []*ProfileSet, req Request) (Resolution, bool, error) {
	if req.Model == "" || req.Workload.Category == "" {
		return Resolution{}, false, nil
	}
	r := Resolution{Layer: layer}
	for _, phase := range []Phase{PhaseDecode, PhasePrefill} {
		set, prof, err := find(sets, key(req, phase))
		if err != nil {
			return Resolution{}, false, fmt.Errorf("%s routing: %w", layer, err)
		}
		if prof == nil {
			continue
		}
		if r.Set != "" && r.Set != set {
			return Resolution{}, false, fmt.Errorf("%s routing: decode and prefill curves for %s come from different sets (%q, %q)",
				layer, key(req, phase), r.Set, set)
		}
		r.Set = set
		if phase == PhaseDecode {
			r.Decode = prof
		} else {
			r.Prefill = prof
		}
		if prof.Provenance != nil {
			r.Source = prof.Provenance.Source
			if req.Hardware != "" && prof.Provenance.Hardware != "" &&
				prof.Provenance.Hardware != req.Hardware {
				r.MeasuredOn = prof.Provenance.Hardware
			}
		}
	}
	return r, r.Decode != nil || r.Prefill != nil, nil
}

func find(sets []*ProfileSet, k Key) (string, *Profile, error) {
	var foundSet string
	var found *Profile
	for _, s := range sets {
		if s == nil {
			continue
		}
		for i := range s.Profiles {
			if s.Profiles[i].Key() != k {
				continue
			}
			if found != nil {
				return "", nil, fmt.Errorf("%s matches in both %q and %q; name one set in the scenario",
					k, foundSet, s.Name)
			}
			foundSet, found = s.Name, &s.Profiles[i]
		}
	}
	return foundSet, found, nil
}
