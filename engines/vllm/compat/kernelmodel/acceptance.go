// Original to blis-vllm.

package kernelmodel

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	blisschemas "github.com/inference-sim/blis-schemas"
	"github.com/inference-sim/blis-schemas/spec/acceptance"
	"github.com/inference-sim/blis-schemas/spec/workload"

	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// Speculative acceptance reaches the kernel only for DSpark adaptive verification
// (blis-latency-kernel#89): vLLM's verifier then sizes its draft budget from each request's
// survival probability (AdaptiveVerificationManager.get_num_tokens,
// vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:282-350@5df63a72), which the kernel
// derives from the run's acceptance. It is resolved here, per pool, by blis-schemas'
// acceptance.Resolve from the same layers the simulator's own resolution reads (compat/build
// ResolveAcceptance): the registry's profiles, the workload, the scenario's pin and the
// caller's overrides (SetAcceptanceOverrides). Without adaptive verification nothing here runs,
// so a pool that does not trim prices exactly as before.

// registryAcceptanceDir is the registry's acceptance-profile namespace, beside coefficients/
// and routing/.
const registryAcceptanceDir = "acceptance"

// AcceptanceOverrides are the acceptance layers a caller supplies beyond the scenario file: a
// command-line value, a workload that is not the scenario's own binding (a catalog preset, with
// any command-line category or thinking mode applied), and a command-line fallback category.
type AcceptanceOverrides struct {
	CLI              *acceptance.Value
	Workload         *acceptance.Workload
	FallbackCategory string
}

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
func resolveAcceptance(in latencykernel.Inputs, e engineconfig.Engine, registry string, o AcceptanceOverrides) (*acceptance.Resolution, error) {
	if !adaptiveVerification(e) || e.Speculative.NumSpecTokens <= 0 {
		return nil, nil
	}
	req := acceptance.Request{
		Model:             in.Scenario.Model,
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
	sets, err := LoadRegistryAcceptanceSets(registry)
	if err != nil {
		return nil, fmt.Errorf("speculative acceptance: %w", err)
	}
	req.Registry = sets
	if pin := req.Scenario; pin != nil && pin.Profiles != "" {
		req.ScenarioSet = FindAcceptanceSet(sets, pin.Profiles)
	}
	r, err := acceptance.Resolve(req)
	if err != nil {
		return nil, fmt.Errorf("speculative.enable_adaptive_verification sizes the verify budget from the run's "+
			"acceptance, and it does not resolve: %w", err)
	}
	return &r, nil
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

// LoadRegistryAcceptanceSets loads and validates every acceptance-profile set under the
// registry's acceptance/ namespace, in path order. A registry with no such namespace has no
// sets; a set that fails to load or validate is an error, never skipped.
func LoadRegistryAcceptanceSets(registry string) ([]*acceptance.ProfileSet, error) {
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

// FindAcceptanceSet is the loaded set named name, nil when none is.
func FindAcceptanceSet(sets []*acceptance.ProfileSet, name string) *acceptance.ProfileSet {
	for _, s := range sets {
		if s.Name == name {
			return s
		}
	}
	return nil
}
