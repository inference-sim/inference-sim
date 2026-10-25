// Moved from inference-sim/cmd/speculative_acceptance.go at 9c6ba5fa0054368bd1cb3007ced919cd69c5c369.

package build

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	blisschemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/spec/acceptance"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
)

// Speculative acceptance (#1939): how far a verification step advances a request is resolved
// by blis-schemas' acceptance.Resolve from four layers, lowest to highest -- the registry's
// profile for (model, method, K, thinking, category), the workload's own value, the
// scenario's pin, the command line -- and recorded in the results file. This file only
// gathers the layers; precedence, the K check, AL in [1, K+1] and the refusal of an unmatched
// lookup are the resolver's (blis-schemas#65).

// The acceptance flags, named here so the refusals name them.
const (
	AcceptanceRateFlag     = "speculative-acceptance-rate"
	AcceptanceLengthFlag   = "speculative-acceptance-length"
	AcceptanceThinkingFlag = "speculative-thinking"
	AcceptanceCategoryFlag = "speculative-category"
	AcceptanceFallbackFlag = "speculative-fallback-category"

	// registryAcceptanceDir is the registry's acceptance-profile namespace, the counterpart of
	// coefficients/ (blis-schemas registry layout).
	registryAcceptanceDir = "acceptance"
)

// AcceptanceFlags are the command line's acceptance layer and selectors, each with whether it
// was given.
type AcceptanceFlags struct {
	Rate, Length                         float64
	RateSet, LengthSet                   bool
	Thinking, Category, FallbackCategory string
	ThinkingSet, CategorySet             bool
	// FallbackSet records that --speculative-fallback-category was given.
	FallbackSet bool
}

// AcceptanceInputs are the resolution's inputs: the scenario (its name, shape and pin), the
// registry it names, the hardware, what the simulated workload states (nil defers to the
// scenario's workload binding), and the command line.
type AcceptanceInputs struct {
	Scenario string
	Shape    kernelmodel.Shape
	Registry string
	Hardware string
	Workload *acceptance.Workload
	Flags    AcceptanceFlags
}

// Acceptance is a run's resolved acceptance.
type Acceptance struct {
	Res acceptance.Resolution
	// CLIRate is the --speculative-acceptance-rate value when that flag decided, so the
	// simulated rate is the flag's own number rather than its round trip through AL.
	CLIRate *float64
	// Ignored lists the selector flags given for a run that does not speculate, which the
	// caller warns about.
	Ignored []string
}

// ResolveAcceptance resolves the run's acceptance for a pool drafting k tokens with method.
// It returns false when the run does not speculate. A refusal is an error.
func ResolveAcceptance(in AcceptanceInputs, k int, method string) (Acceptance, bool, error) {
	f := in.Flags
	if k == 0 {
		if f.LengthSet {
			return Acceptance{}, false, fmt.Errorf("--%s set (%v) but scenario %q drafts no tokens; there is no acceptance to state",
				AcceptanceLengthFlag, f.Length, in.Scenario)
		}
		var out Acceptance
		for _, s := range []struct {
			name string
			set  bool
		}{{AcceptanceThinkingFlag, f.ThinkingSet}, {AcceptanceCategoryFlag, f.CategorySet}, {AcceptanceFallbackFlag, f.FallbackSet}} {
			if s.set {
				out.Ignored = append(out.Ignored, s.name)
			}
		}
		return out, false, nil
	}

	req := acceptance.Request{
		Model:             in.Shape.Model,
		SpeculativeMethod: method,
		NumSpecTokens:     k,
		Hardware:          in.Hardware,
		Workload:          acceptanceWorkloadTerms(in),
		Scenario:          in.Shape.Acceptance,
		FallbackCategory:  f.FallbackCategory,
	}
	var out Acceptance
	switch {
	case f.RateSet:
		r := f.Rate
		req.CLI = &acceptance.Value{NumSpecTokens: k, AcceptanceRate: &r, Source: "--" + AcceptanceRateFlag}
		out.CLIRate = &r
	case f.LengthSet:
		l := f.Length
		req.CLI = &acceptance.Value{NumSpecTokens: k, AcceptanceLength: &l, Source: "--" + AcceptanceLengthFlag}
	}
	sets, err := loadRegistryAcceptanceSets(in.Registry)
	if err != nil {
		return Acceptance{}, false, fmt.Errorf("speculative acceptance: %w", err)
	}
	req.Registry = sets
	if pin := req.Scenario; pin != nil && pin.Profiles != "" {
		req.ScenarioSet = findSet(sets, pin.Profiles)
	}
	res, err := acceptance.Resolve(req)
	if err != nil {
		return Acceptance{}, false, fmt.Errorf("scenario %q drafts %d %q tokens, and its acceptance does not resolve: %v. "+
			"State it with --%s or --%s, select a registry profile with --%s and --%s, or pin one in "+
			"the scenario", in.Scenario, k, method, err, AcceptanceLengthFlag, AcceptanceRateFlag,
			AcceptanceThinkingFlag, AcceptanceCategoryFlag)
	}
	out.Res = res
	return out, true, nil
}

// acceptanceWorkloadTerms is the workload layer: what the simulated workload states (a
// catalog preset, or the scenario's trace or replay binding), with the command line's
// selectors overriding its category and thinking mode.
func acceptanceWorkloadTerms(in AcceptanceInputs) acceptance.Workload {
	var w acceptance.Workload
	switch b := in.Shape.WorkloadBinding; {
	case in.Workload != nil:
		w = *in.Workload
	case b != nil && b.Trace != nil:
		w = b.Trace.AcceptanceTerms()
	case b != nil && b.Replay != nil:
		w = b.Replay.AcceptanceTerms()
	}
	if in.Flags.ThinkingSet {
		w.Thinking = acceptance.Thinking(in.Flags.Thinking)
	}
	if in.Flags.CategorySet {
		w.Category = in.Flags.Category
	}
	return w
}

// loadRegistryAcceptanceSets loads and validates every acceptance-profile set under the
// registry's acceptance/ namespace, in path order. A registry with no such namespace has no
// sets; a set that fails to load or validate is an error, never skipped.
func loadRegistryAcceptanceSets(registry string) ([]*acceptance.ProfileSet, error) {
	dir := filepath.Join(registry, registryAcceptanceDir)
	if _, err := os.Stat(dir); errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	var paths []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.Type().IsRegular() && (strings.HasSuffix(p, ".yaml") || strings.HasSuffix(p, ".yml")) {
			paths = append(paths, p)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", dir, err)
	}
	sort.Strings(paths)
	sets := make([]*acceptance.ProfileSet, 0, len(paths))
	for _, p := range paths {
		s, err := blisschemas.LoadAcceptanceProfileSet(p)
		if err != nil {
			return nil, fmt.Errorf("acceptance profile set %s: %w", p, err)
		}
		if problems := s.Validate(); !problems.OK() {
			return nil, fmt.Errorf("acceptance profile set %s: %s", p, problems.Error())
		}
		sets = append(sets, s)
	}
	return sets, nil
}

func findSet(sets []*acceptance.ProfileSet, name string) *acceptance.ProfileSet {
	for _, s := range sets {
		if s.Name == name {
			return s
		}
	}
	return nil
}
