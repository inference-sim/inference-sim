package cmd

import (
	"strings"

	"github.com/inference-sim/blis-schemas/spec/acceptance"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/blis-vllm/compat/kernelmodel"
	"github.com/inference-sim/inference-sim/sim"
	_ "github.com/inference-sim/inference-sim/sim/vendored" // the adapter's vendoring root
)

// Speculative acceptance (#1939): how far a verification step advances a request is resolved
// from four layers, lowest to highest -- the registry's profile, the workload's own value, the
// scenario's pin, the command line -- by the engine (build.ResolveAcceptance, called from the
// build site), and recorded in the results file. This file holds the flags and the record.

// The acceptance flags' names: the engine's refusals name them, so engine_build.go takes
// them from it.
const (
	acceptanceRateFlag     = acceptanceRateFlagName
	acceptanceLengthFlag   = acceptanceLengthFlagName
	acceptanceThinkingFlag = acceptanceThinkingFlagName
	acceptanceCategoryFlag = acceptanceCategoryFlagName
	acceptanceFallbackFlag = acceptanceFallbackFlagName
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
