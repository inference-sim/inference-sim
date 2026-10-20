package cmd

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
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kernelmodel"
)

// Speculative acceptance (#1939): how far a verification step advances a request is resolved
// by blis-schemas' acceptance.Resolve from four layers, lowest to highest -- the registry's
// profile for (model, method, K, thinking, category), the workload's own value, the
// scenario's pin, the command line -- and recorded in the results file. This file only
// gathers the layers; precedence, the K check, AL in [1, K+1] and the refusal of an unmatched
// lookup are the resolver's (blis-schemas#65).

const (
	acceptanceRateFlag     = "speculative-acceptance-rate"
	acceptanceLengthFlag   = "speculative-acceptance-length"
	acceptanceThinkingFlag = "speculative-thinking"
	acceptanceCategoryFlag = "speculative-category"
	acceptanceFallbackFlag = "speculative-fallback-category"

	// registryAcceptanceDir is the registry's acceptance-profile namespace, the counterpart of
	// coefficients/ (blis-schemas registry layout).
	registryAcceptanceDir = "acceptance"
)

var (
	speculativeAcceptanceLength float64 // --speculative-acceptance-length (AL in [1, K+1])
	speculativeThinking         string  // --speculative-thinking (on|off): selects the profile
	speculativeCategory         string  // --speculative-category: selects the profile
	speculativeFallbackCategory string  // --speculative-fallback-category: recorded substitute

	// scenarioShape is the scenario's deployment shape and its acceptance and workload terms,
	// read once by adoptKernelDeployment.
	scenarioShape kernelmodel.Shape

	// workloadAcceptance is what the workload the run simulates states about acceptance -- a
	// catalog preset's category, thinking mode and own value -- when it states anything; nil
	// defers to the scenario's workload binding. A workload source that is not the scenario's
	// binding sets it before the speculative config resolves.
	workloadAcceptance *acceptance.Workload

	// acceptanceRecord is the resolved acceptance, recorded in the results file; nil when the
	// run does not speculate.
	acceptanceRecord *sim.AcceptanceRecord
)

// registerAcceptanceFlags registers the acceptance flags on run and replay alike (INV-13). The
// names are literals, as the physics-literal guard reads registrations (#1819).
func registerAcceptanceFlags(cmd *cobra.Command) {
	cmd.Flags().Float64Var(&speculativeAcceptance, "speculative-acceptance-rate", 0.0, "Speculative decoding: the accepted fraction of drafts, (AL-1)/K, in [0,1]. Overrides every other acceptance layer (registry profile, workload, scenario). Not with --"+acceptanceLengthFlag+".")
	cmd.Flags().Float64Var(&speculativeAcceptanceLength, "speculative-acceptance-length", 0.0, "Speculative decoding: the mean acceptance length AL, tokens per verification step counting the bonus token, in [1, K+1]. Overrides every other acceptance layer. Not with --"+acceptanceRateFlag+".")
	cmd.MarkFlagsMutuallyExclusive(acceptanceRateFlag, acceptanceLengthFlag)
	cmd.Flags().StringVar(&speculativeThinking, "speculative-thinking", "", "Speculative decoding: whether the target runs with thinking on ("+strings.Join(acceptance.AllThinkingModes(), "|")+"); with --"+acceptanceCategoryFlag+" it selects the registry's acceptance profile, overriding the workload's own selectors.")
	cmd.Flags().StringVar(&speculativeCategory, "speculative-category", "", "Speculative decoding: the workload category (e.g. coding) that selects the registry's acceptance profile, overriding the workload's own.")
	cmd.Flags().StringVar(&speculativeFallbackCategory, "speculative-fallback-category", "", "Speculative decoding: a category whose profile to use, recorded as a fallback, when the workload's category has none. Without it an unmatched category is refused. Unlike the acceptance value, where the CLI is highest, a scenario pin's own fallback_category wins over this flag, as blis-schemas' acceptance.Resolve applies them.")
}

// acceptanceResolution is a run's resolved acceptance and the config it implies.
type acceptanceResolution struct {
	res acceptance.Resolution
	// cliRate is the --speculative-acceptance-rate value when that flag decided, so the
	// simulated rate is the flag's own number rather than its round trip through AL.
	cliRate *float64
}

// resolveAcceptance resolves the run's acceptance for a pool drafting k tokens with method.
// It returns false when the run does not speculate. CLI boundary: a refusal is fatal.
func resolveAcceptance(cmd *cobra.Command, k int, method string) (acceptanceResolution, bool) {
	rateSet := cmd.Flags().Changed(acceptanceRateFlag)
	lengthSet := cmd.Flags().Changed(acceptanceLengthFlag)
	if k == 0 {
		if lengthSet {
			logrus.Fatalf("--%s set (%v) but scenario %q drafts no tokens; there is no acceptance to state",
				acceptanceLengthFlag, speculativeAcceptanceLength, kernelScenario)
		}
		for _, f := range []string{acceptanceThinkingFlag, acceptanceCategoryFlag, acceptanceFallbackFlag} {
			if cmd.Flags().Changed(f) {
				logrus.Warnf("--%s is ignored: scenario %q does not speculate", f, kernelScenario)
			}
		}
		return acceptanceResolution{}, false
	}

	req := acceptance.Request{
		Model:             scenarioShape.Model,
		SpeculativeMethod: method,
		NumSpecTokens:     k,
		Hardware:          gpu,
		Workload:          acceptanceWorkloadTerms(cmd),
		Scenario:          scenarioShape.Acceptance,
		FallbackCategory:  speculativeFallbackCategory,
	}
	var out acceptanceResolution
	switch {
	case rateSet:
		r := speculativeAcceptance
		req.CLI = &acceptance.Value{NumSpecTokens: k, AcceptanceRate: &r, Source: "--" + acceptanceRateFlag}
		out.cliRate = &r
	case lengthSet:
		l := speculativeAcceptanceLength
		req.CLI = &acceptance.Value{NumSpecTokens: k, AcceptanceLength: &l, Source: "--" + acceptanceLengthFlag}
	}
	sets, err := loadRegistryAcceptanceSets(kernelRegistry)
	if err != nil {
		logrus.Fatalf("speculative acceptance: %v", err)
	}
	req.Registry = sets
	if pin := req.Scenario; pin != nil && pin.Profiles != "" {
		req.ScenarioSet = findSet(sets, pin.Profiles)
	}
	res, err := acceptance.Resolve(req)
	if err != nil {
		logrus.Fatalf("scenario %q drafts %d %q tokens, and its acceptance does not resolve: %v. "+
			"State it with --%s or --%s, select a registry profile with --%s and --%s, or pin one in "+
			"the scenario", kernelScenario, k, method, err, acceptanceLengthFlag, acceptanceRateFlag,
			acceptanceThinkingFlag, acceptanceCategoryFlag)
	}
	out.res = res
	return out, true
}

// acceptanceWorkloadTerms is the workload layer: what the simulated workload states (a
// catalog preset, or the scenario's trace or replay binding), with the command line's
// selectors overriding its category and thinking mode.
func acceptanceWorkloadTerms(cmd *cobra.Command) acceptance.Workload {
	var w acceptance.Workload
	switch b := scenarioShape.WorkloadBinding; {
	case workloadAcceptance != nil:
		w = *workloadAcceptance
	case b != nil && b.Trace != nil:
		w = b.Trace.AcceptanceTerms()
	case b != nil && b.Replay != nil:
		w = b.Replay.AcceptanceTerms()
	}
	if cmd.Flags().Changed(acceptanceThinkingFlag) {
		w.Thinking = acceptance.Thinking(speculativeThinking)
	}
	if cmd.Flags().Changed(acceptanceCategoryFlag) {
		w.Category = speculativeCategory
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

// acceptanceRecordOf is the results-file record of a resolution as the run simulates it: the
// length and rate are the config's own, so the record is what advanced requests.
func acceptanceRecordOf(r acceptance.Resolution, c sim.SpeculativeConfig) *sim.AcceptanceRecord {
	rec := &sim.AcceptanceRecord{
		Length:      c.EffectiveTokensPerStep(),
		Rate:        c.Acceptance,
		SourceLayer: string(r.Layer),
		Entry: sim.AcceptanceEntry{
			Set:              r.Set,
			Source:           r.Source,
			FallbackFrom:     r.FallbackFrom,
			MeasuredOn:       r.MeasuredOn,
			PerPositionRates: r.PerPositionRates,
			RatesDerived:     r.RatesDerived,
		},
	}
	if p := r.Profile; p != nil {
		rec.Entry.Key = p.Key().String()
		rec.Entry.Method = string(p.Method)
		rec.Entry.CopiedFrom = p.CopiedFrom
		if p.Provenance != nil {
			rec.Entry.Run = p.Provenance.Run
		}
	}
	return rec
}

// logAcceptance reports the resolution on stderr, and warns where it substitutes: a fallback
// category, or a profile measured on other hardware.
func logAcceptance(r *sim.AcceptanceRecord, k int) {
	from := r.SourceLayer
	if r.Entry.Key != "" {
		from += " profile " + r.Entry.Key + " in " + r.Entry.Set
	}
	if r.Entry.Source != "" {
		from += " (" + r.Entry.Source + ")"
	}
	logrus.Infof("speculative acceptance: AL %.4f tokens/step at K=%d (rate %.4f), from the %s",
		r.Length, k, r.Rate, from)
	if r.Entry.FallbackFrom != "" {
		logrus.Warnf("speculative acceptance: category %q has no profile; using the requested fallback, recorded in the results file",
			r.Entry.FallbackFrom)
	}
	if r.Entry.MeasuredOn != "" {
		logrus.Warnf("speculative acceptance: the profile was measured on %s, not %s", r.Entry.MeasuredOn, gpu)
	}
}
