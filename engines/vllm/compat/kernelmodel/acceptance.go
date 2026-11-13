// Original to blis-vllm.

package kernelmodel

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
	latencykernel "github.com/inference-sim/blis-substrate/kernel"
	"github.com/inference-sim/blis-substrate/load"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// Speculative acceptance reaches the kernel only for DSpark adaptive verification
// (blis-substrate kernel/adaptive.go): vLLM's verifier then sizes its draft budget from each request's
// survival probability (AdaptiveVerificationManager.get_num_tokens,
// vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:282-350@5df63a72), which the kernel
// derives from the run's acceptance. It is resolved here, per pool, by blis-schemas'
// acceptance.Resolve from the same layers the simulator's own resolution reads (compat/build
// ResolveAcceptance): the registry's profiles, the workload, the scenario's pin and the
// caller's overrides (SetAcceptanceOverrides). Without adaptive verification nothing here runs,
// so a pool that does not trim prices exactly as before.

// AcceptanceOverrides are the acceptance layers a caller supplies beyond the scenario file: a
// command-line value, a workload that is not the scenario's own binding (a catalog preset, with
// any command-line category or thinking mode applied), and a command-line fallback category.
type AcceptanceOverrides struct {
	CLI              *acceptance.Value
	Workload         *acceptance.Workload
	FallbackCategory string
}

// SyntheticAcceptanceSource names the engine block's synthetic_acceptance_length as the source of a
// pinned acceptance value.
const SyntheticAcceptanceSource = "engine.speculative.synthetic_acceptance_length"

// AcceptanceWithEngine is the scenario's acceptance pin with the pool's engine-block acceptance
// stated in it. A block that states speculative.synthetic_acceptance_length together with
// rejection_sample_method synthetic tells vLLM the mean acceptance length every verification
// step has; here that is the run's supplied acceptance profile (the registry's "map onto a supplied
// profile"), and it takes the rank of the scenario's own pin: above the registry and the workload,
// below the command line. vLLM turns the length into per-position rates by the minimum-variance
// schedule (SpeculativeConfig._acceptance_length_to_rates), which the resolver derives for a
// length alone (acceptance.MinimumVarianceRates). A pin that already states a value must state
// this one: two different acceptance values for one run are refused, not resolved by order. A pin
// that selects a profile (a profile set, a fallback category, exact_only) is replaced by the length:
// the run forces its acceptance, so no measured profile applies. A block that states none returns
// the pin as given.
func AcceptanceWithEngine(pin *acceptance.Pin, e engineconfig.Engine) (*acceptance.Pin, error) {
	sp := e.Speculative
	if sp == nil || sp.SyntheticAcceptanceLength == nil {
		return pin, nil
	}
	l := *sp.SyntheticAcceptanceLength
	if pin == nil {
		return &acceptance.Pin{Value: &acceptance.Value{NumSpecTokens: sp.NumSpecTokens, AcceptanceLength: &l,
			Source: SyntheticAcceptanceSource}}, nil
	}
	if v := pin.Value; v != nil {
		if v.NumSpecTokens != sp.NumSpecTokens || v.AcceptanceLength == nil || *v.AcceptanceLength != l {
			return nil, fmt.Errorf("engine.speculative.synthetic_acceptance_length %v at num_spec_tokens %d contradicts the "+
				"acceptance the scenario pins (%s); state one, or the same value in both", l, sp.NumSpecTokens, describeValue(v))
		}
		return pin, nil
	}
	// The block's length is a stated value, and vLLM with a synthetic length forces that acceptance rather
	// than drawing on any measured profile, so the scenario's profile selection (a profile set, a
	// fallback category, exact_only) has nothing to select and is dropped: the substrate refuses a pin that
	// states a value together with any of them. The value is all the merged pin states.
	return &acceptance.Pin{Value: &acceptance.Value{NumSpecTokens: sp.NumSpecTokens, AcceptanceLength: &l,
		Source: SyntheticAcceptanceSource}}, nil
}

// describeValue is a pinned value for a refusal.
func describeValue(v *acceptance.Value) string {
	switch {
	case v.AcceptanceLength != nil:
		return fmt.Sprintf("acceptance_length %v at num_spec_tokens %d", *v.AcceptanceLength, v.NumSpecTokens)
	case v.AcceptanceRate != nil:
		return fmt.Sprintf("acceptance_rate %v at num_spec_tokens %d", *v.AcceptanceRate, v.NumSpecTokens)
	default:
		return fmt.Sprintf("per-position rates at num_spec_tokens %d", v.NumSpecTokens)
	}
}

// resolveRequest is acceptance.Resolve; a test replaces it to see the request a pool makes.
var resolveRequest = acceptance.Resolve

// acceptanceOverrides is set by the command layer before a pool opens (SetAcceptanceOverrides).
var acceptanceOverrides AcceptanceOverrides

// SetAcceptanceOverrides records the caller's acceptance layers for the pools opened after it.
func SetAcceptanceOverrides(o AcceptanceOverrides) { acceptanceOverrides = o }

// adaptiveVerification reports whether the pool's engine block turns DSpark adaptive
// verification on (speculative.enable_adaptive_verification, vllm/config/speculative.py:551
// @5df63a72; unstated is false).
func adaptiveVerification(e engineconfig.Engine) bool {
	sp := e.Speculative
	return sp != nil && sp.EnableAdaptiveVerification != nil && *sp.EnableAdaptiveVerification
}

// resolveAcceptance resolves one pool's speculative acceptance for the kernel, or nil when the
// pool does not verify adaptively. A layer that does not resolve is an error: the kernel cannot
// size the verify budget without it.
func resolveAcceptance(in latencykernel.Inputs, e engineconfig.Engine, data *load.Data, o AcceptanceOverrides) (*acceptance.Resolution, error) {
	if !adaptiveVerification(e) || e.Speculative.NumSpecTokens <= 0 {
		return nil, nil
	}
	pin, err := AcceptanceWithEngine(in.Scenario.Acceptance, e)
	if err != nil {
		return nil, err
	}
	req := acceptance.Request{
		Model:             in.Scenario.Model,
		Drafter:           e.Speculative.DraftModel,
		Architecture:      targetArchitecture(in, data),
		SpeculativeMethod: e.Speculative.Method,
		NumSpecTokens:     e.Speculative.NumSpecTokens,
		Hardware:          in.Chip.Name,
		Workload:          bindingAcceptanceTerms(in.Scenario.Workload),
		Scenario:          pin,
		CLI:               o.CLI,
		FallbackCategory:  o.FallbackCategory,
	}
	if o.Workload != nil {
		req.Workload = *o.Workload
	}
	sets := data.AcceptanceSets()
	req.Registry = sets
	if pin := req.Scenario; pin != nil && pin.Profiles != "" {
		req.ScenarioSet = FindAcceptanceSet(sets, pin.Profiles)
	}
	r, err := resolveRequest(req)
	if err != nil {
		return nil, fmt.Errorf("speculative_config.enable_adaptive_verification sizes the verify budget from the run's "+
			"acceptance, and it does not resolve: %w", err)
	}
	data.RecordAcceptance(&r)
	return &r, nil
}

// targetArchitecture is the architecture of the pool's target model as its graph records it
// (global.architecture): the key a lookup that borrows across targets reads (acceptance.Request
// Architecture). It is empty when the graph cannot be had, which skips the borrowing step and so
// can only make a lookup refuse, never resolve to another target's value.
func targetArchitecture(in latencykernel.Inputs, data *load.Data) string {
	if in.Model != nil {
		return in.Model.Global.Architecture
	}
	if data == nil {
		return ""
	}
	g, err := data.Model(in.Scenario.Model)
	if err != nil || g == nil {
		return ""
	}
	return g.Global.Architecture
}

// bindingAcceptanceTerms is what a scenario's workload binding states about acceptance.
func bindingAcceptanceTerms(b *workload.Binding) acceptance.Workload {
	switch {
	case b == nil:
		return acceptance.Workload{}
	case b.Trace != nil:
		return b.Trace.AcceptanceTerms()
	case b.Replay != nil:
		return b.Replay.AcceptanceTerms()
	}
	return acceptance.Workload{}
}

// FindAcceptanceSet is the loaded set named name, nil when none is.
func FindAcceptanceSet(sets []*acceptance.ProfileSet, name string) *acceptance.ProfileSet {
	for _, s := range sets {
		if s.Name == name {
			return s
		}
	}
	return nil
}
