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
	req := acceptance.Request{
		Model:             in.Scenario.Model,
		Drafter:           e.Speculative.DraftModel,
		Architecture:      targetArchitecture(in, data),
		SpeculativeMethod: e.Speculative.Method,
		NumSpecTokens:     e.Speculative.NumSpecTokens,
		Hardware:          in.Chip.Name,
		Workload:          bindingAcceptanceTerms(in.Scenario.Workload),
		Scenario:          in.Scenario.Acceptance,
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
		return nil, fmt.Errorf("speculative.enable_adaptive_verification sizes the verify budget from the run's "+
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
