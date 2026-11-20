package cmd

import (
	"fmt"
	"strings"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/acceptance"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/result"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim"
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
	speculativeRealisation      string  // --speculative-acceptance-realisation: natural|synthetic

	// scenarioShape is the scenario's deployment shape and its acceptance and workload terms,
	// read once by adoptKernelDeployment.
	scenarioShape scenarioShapeT

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
	cmd.Flags().StringVar(&speculativeRealisation, "speculative-acceptance-realisation", "", "Speculative decoding: how the acceptance that --"+acceptanceRateFlag+" or --"+acceptanceLengthFlag+" states was realised ("+strings.Join(acceptance.AllRealisations(), "|")+"). natural: the rate was measured with real verification, so it is spread over the draft positions with the registry's fitted decay for the speculative method (an estimate). synthetic, or unset: the engine forced the acceptance, so it is spread with the engine's minimum-variance schedule. Only with --"+acceptanceRateFlag+" or --"+acceptanceLengthFlag+".")
	cmd.Flags().StringVar(&speculativeFallbackCategory, "speculative-fallback-category", "", "Speculative decoding: a category whose profile to use, recorded as a fallback, when the workload's category has none. Without it an unmatched category is refused. Unlike the acceptance value, where the CLI is highest, a scenario pin's own fallback_category wins over this flag, as blis-schemas' acceptance.Resolve applies them.")
}

// acceptanceRecordOf is the results-file record of a resolution as the run simulates it: the
// length and rate are the config's own, so the record is what advanced requests.
func acceptanceRecordOf(r acceptance.Resolution, c speculativeConfig) *sim.AcceptanceRecord {
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
	// The provenance the substrate records in the Result (result.AcceptanceProperty), so the
	// two files agree on the step, what differed, the K rule, the profiles read and the schedule.
	prop := result.AcceptanceProperty(r)
	rec.Entry.Step, rec.Entry.Differed, rec.Entry.KRule, rec.Entry.Profiles = prop.Step, prop.Differed, prop.KRule, prop.Profiles
	rec.Entry.RatesSchedule = prop.RatesSchedule
	if s := prop.RatesShape; s != nil {
		rec.Entry.RatesShape = &sim.AcceptanceRatesShape{Set: s.Set, SpeculativeMethod: s.SpeculativeMethod,
			DecayExponent: s.DecayExponent, Method: string(s.Method)}
	}
	if p := r.Profile; p != nil {
		rec.Entry.Key = p.Key().String()
		rec.Entry.Method = string(p.Method)
		rec.Entry.CopiedFrom = p.TransferredFrom
		if p.Provenance != nil {
			rec.Entry.Run = p.Provenance.Run
		}
	}
	return rec
}

// borrowedAcceptance reports whether the acceptance is a registry profile borrowed from a run
// whose configuration differs from this one (the resolver's steps other than the exact key). A
// stated value, and an exact profile, are not borrowed.
func borrowedAcceptance(r *sim.AcceptanceRecord) bool {
	return r.Entry.Step != "" && r.Entry.Step != string(acceptance.StepExact)
}

// borrowedAcceptanceWarning is the one stderr line a run with a borrowed acceptance prints: the
// step that found the value, what differed, how the value was brought to the run's K, and the
// profile it was read from. The same terms are in the Result (acceptance.entry.step, differed,
// k_rule), so the line and the record agree.
func borrowedAcceptanceWarning(r *sim.AcceptanceRecord, k int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "speculative acceptance: borrowed, not measured for this configuration: AL %.4f at K=%d is the %s step", r.Length, k, r.Entry.Step)
	if len(r.Entry.Differed) > 0 {
		fmt.Fprintf(&b, ", differing in %s", strings.Join(r.Entry.Differed, ", "))
	}
	if r.Entry.KRule != "" {
		fmt.Fprintf(&b, ", brought to K by the %s rule", r.Entry.KRule)
	}
	src := r.Entry.Profiles
	if len(src) == 0 && r.Entry.Key != "" {
		src = []string{r.Entry.Set + " " + r.Entry.Key}
	}
	fmt.Fprintf(&b, "; read from profile %s", strings.Join(src, ", "))
	return b.String()
}

// logAcceptance reports the resolution on stderr, and warns where it substitutes: a fallback
// category, a borrowed profile, or a profile measured on other hardware.
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
	if borrowedAcceptance(r) {
		logrus.Warn(borrowedAcceptanceWarning(r, k))
	}
	if r.Entry.MeasuredOn != "" {
		logrus.Warnf("speculative acceptance: the profile was measured on %s, not %s", r.Entry.MeasuredOn, gpu)
	}
	if r.Entry.RatesDerived {
		how := "the engine's minimum-variance schedule, which forced acceptance follows; a rate measured from natural acceptance decays instead (--speculative-acceptance-realisation natural)"
		if r.Entry.RatesSchedule == string(acceptance.ScheduleNaturalDecay) {
			if sh := r.Entry.RatesShape; sh != nil {
				how = fmt.Sprintf("the fitted natural decay (an estimate) of shape %s %q decay_exponent %v method %s",
					sh.Set, sh.SpeculativeMethod, sh.DecayExponent, sh.Method)
			}
		}
		logrus.Warnf("speculative acceptance: no per-position rates were stated, so they were derived with %s; recorded in the results file as rates_schedule %s",
			how, r.Entry.RatesSchedule)
	}
}

// checkRealisationFlag refuses --speculative-acceptance-realisation without the rate or length
// it qualifies, and a value that is not a realisation.
func checkRealisationFlag(fl interface{ Changed(string) bool }) error {
	if !fl.Changed("speculative-acceptance-realisation") {
		return nil
	}
	if !acceptance.Realisation(speculativeRealisation).Valid() {
		return fmt.Errorf("--speculative-acceptance-realisation %q is not one of %v", speculativeRealisation, acceptance.AllRealisations())
	}
	if !fl.Changed(acceptanceRateFlag) && !fl.Changed(acceptanceLengthFlag) {
		return fmt.Errorf("--speculative-acceptance-realisation qualifies a value; state --%s or --%s with it", acceptanceRateFlag, acceptanceLengthFlag)
	}
	return nil
}
