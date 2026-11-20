// Moved from inference-sim/cmd/speculative_acceptance.go at 9c6ba5fa0054368bd1cb3007ced919cd69c5c369.

package build

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/load"

	"github.com/inference-sim/blis-vllm/core/engineconfig"
	"github.com/inference-sim/blis-vllm/core/kernelmodel"
)

// Speculative acceptance (#1939): how far a verification step advances a request is resolved
// by blis-schemas' acceptance.Resolve from four layers, lowest to highest -- the registry's
// profile for (model, method, K, thinking, category), the workload's own value, the
// scenario's pin, the command line -- and recorded in the results file. This file only
// gathers the layers; precedence, the K check, AL in [1, K+1] and the refusal of an unmatched
// lookup are the resolver's (https://github.com/inference-sim/blis-schemas/issues/65).

// The acceptance flags, named here so the refusals name them.
const (
	AcceptanceRateFlag     = "speculative-acceptance-rate"
	AcceptanceLengthFlag   = "speculative-acceptance-length"
	AcceptanceThinkingFlag = "speculative-thinking"
	AcceptanceCategoryFlag = "speculative-category"
	AcceptanceFallbackFlag = "speculative-fallback-category"
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
	// Realisation is how the run the rate or length describes realised its acceptance
	// (natural or synthetic); empty leaves it unstated, which keeps the engine's schedule.
	Realisation acceptance.Realisation
}

// AcceptanceInputs are the resolution's inputs: the scenario (its name, shape and pin), the
// run's data (whose registry holds the acceptance profiles), the hardware, what the simulated workload states (nil defers to the
// scenario's workload binding), and the command line.
type AcceptanceInputs struct {
	Scenario string
	Shape    kernelmodel.Shape
	Data     *load.Data
	Hardware string
	Workload *acceptance.Workload
	Flags    AcceptanceFlags
	// Drafter is the catalog name of the pool's separate drafter checkpoint
	// (speculative.draft_model), or "" for the model's own in-graph speculator. A profile measured
	// with a drafter matches only a run that names the same one.
	Drafter string
	// DraftSampler and Verifier are the run's draft sampler and verifier as the engine resolves
	// them (speculative.draft_sample_method and rejection_sample_method, each defaulted as vLLM
	// defaults it: kernelmodel.ResolvedSampling, which VLLMInputs applies), or "" for unstated
	// (design D8, inference-sim/blis-substrate#174).
	DraftSampler acceptance.DraftSampler
	Verifier     acceptance.Verifier
	// Engine is the lower-case name of the engine serving the run (Engine, "vllm"). A profile
	// measured on another engine does not match the run exactly.
	Engine string
	// DefaultDraftSampler and DefaultVerifier are the engine's own defaults for those settings
	// (DefaultDraftSampler, DefaultVerifier). A profile whose source left a setting at its
	// default states the marker "default", which the lookup reads as these when the profile's
	// engine is the run's.
	DefaultDraftSampler acceptance.DraftSampler
	DefaultVerifier     acceptance.Verifier
}

// Engine is the engine name this module models, as acceptance profiles record it.
const Engine = kernelmodel.Engine

// vLLM's defaults for the drafting and verification settings (kernelmodel.DefaultDraftSampler,
// kernelmodel.DefaultVerifier).
const (
	DefaultDraftSampler = kernelmodel.DefaultDraftSampler
	DefaultVerifier     = kernelmodel.DefaultVerifier
)

// VLLMInputs fills in the engine's part of the inputs for a pool whose speculative block is sp:
// the engine, its defaults, and the resolved draft sampler and verifier.
func (in AcceptanceInputs) VLLMInputs(sp *engineconfig.Speculative) AcceptanceInputs {
	in.Engine, in.DefaultDraftSampler, in.DefaultVerifier = Engine, DefaultDraftSampler, DefaultVerifier
	in.DraftSampler, in.Verifier = kernelmodel.ResolvedSampling(sp)
	return in
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
		Model:               in.Shape.Model,
		SpeculativeMethod:   method,
		NumSpecTokens:       k,
		Hardware:            in.Hardware,
		Drafter:             in.Drafter,
		DraftSampler:        in.DraftSampler,
		Verifier:            in.Verifier,
		Engine:              in.Engine,
		DefaultDraftSampler: in.DefaultDraftSampler,
		DefaultVerifier:     in.DefaultVerifier,
		Architecture:        targetArchitecture(in.Data, in.Shape.Model),
		Workload:            acceptanceWorkloadTerms(in),
		Scenario:            in.Shape.Acceptance,
		FallbackCategory:    f.FallbackCategory,
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
	if req.CLI != nil {
		req.CLI.Realisation = f.Realisation
	}
	if in.Data == nil {
		return Acceptance{}, false, fmt.Errorf("scenario %q drafts %d %q tokens: no catalog and registry to "+
			"resolve its acceptance from (AcceptanceInputs.Data is unset)", in.Scenario, k, method)
	}
	sets := in.Data.AcceptanceSets()
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
	in.Data.RecordAcceptance(&res)
	out.Res = res
	return out, true, nil
}

// targetArchitecture is the architecture of the target model as its graph records it
// (global.architecture), which a lookup that borrows across targets reads. It is empty when the
// graph cannot be had; that skips the borrowing step, so the lookup can only refuse, never resolve
// to another target's value.
func targetArchitecture(data *load.Data, modelName string) string {
	if data == nil {
		return ""
	}
	g, err := data.Model(modelName)
	if err != nil || g == nil {
		return ""
	}
	return g.Global.Architecture
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

func findSet(sets []*acceptance.ProfileSet, name string) *acceptance.ProfileSet {
	return kernelmodel.FindAcceptanceSet(sets, name)
}
