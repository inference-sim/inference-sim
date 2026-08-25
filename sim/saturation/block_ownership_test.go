package saturation

import (
	"math"
	"strings"
	"testing"
)

// blockSetter names one config block and installs it into a SaturationConfig, so
// the tables below can enumerate (block, selection) pairs exhaustively.
type blockSetter struct {
	name  string // the YAML key, as it must appear in the error message
	owner string // the detector allowed to carry it
	set   func(*SaturationConfig)
}

func allBlockSetters() []blockSetter {
	one := 1.0
	ms := 5000.0
	k := 3.0
	return []blockSetter{
		{"composite", "composite", func(c *SaturationConfig) { c.Composite = &CompositeBlock{Sensitivity: &one} }},
		{"threshold", "threshold", func(c *SaturationConfig) { c.Threshold = &ThresholdBlock{ThresholdMs: &ms} }},
		{"backlog_drift", "backlog-drift", func(c *SaturationConfig) { c.BacklogDrift = &BacklogDriftBlock{SlopeK: &k} }},
		{"peak_rate", "peak-rate", func(c *SaturationConfig) { c.PeakRate = &PeakRateBlock{Threshold: &one} }},
	}
}

// BC-8 + BC-9: on the SINGLE-detector path a block is accepted exactly when its
// owner is the selected detector, and rejected (naming the block) otherwise --
// never silently dropped (R1). Covers composite in BOTH directions, which is new
// in #1614: composite previously owned no block at all.
func TestBlockOwnership_SingleDetectorPath(t *testing.T) {
	for _, bs := range allBlockSetters() {
		for _, selected := range AllDetectorNames() {
			t.Run(bs.name+"/"+selected, func(t *testing.T) {
				cfg := SaturationConfig{}
				bs.set(&cfg)
				_, err := BuildDetector(selected, cfg)

				if selected == bs.owner {
					if err != nil {
						t.Fatalf("block %q with its owner %q selected: unexpected error %v", bs.name, selected, err)
					}
					return
				}
				if err == nil {
					t.Fatalf("block %q with %q selected: expected an error, got none (block would be silently dropped)", bs.name, selected)
				}
				if !strings.Contains(err.Error(), bs.name) {
					t.Errorf("error should name the offending block %q, got: %v", bs.name, err)
				}
			})
		}
	}
}

// BC-8: on the BANK path a block is valid iff its owner is among the selected
// names. This is where a composite: block would previously have been silently
// dropped -- checkBlockOwnershipSet had no composite case at all.
func TestBlockOwnership_BankPath(t *testing.T) {
	for _, bs := range allBlockSetters() {
		t.Run(bs.name+"/owner selected", func(t *testing.T) {
			cfg := SaturationConfig{}
			bs.set(&cfg)
			if _, err := NewBank([]string{bs.owner}, cfg, NewInMemoryCollector()); err != nil {
				t.Fatalf("block %q with owner %q selected: unexpected error %v", bs.name, bs.owner, err)
			}
		})

		t.Run(bs.name+"/owner omitted", func(t *testing.T) {
			// Every detector EXCEPT this block's owner.
			var others []string
			for _, n := range AllDetectorNames() {
				if n != bs.owner {
					others = append(others, n)
				}
			}
			cfg := SaturationConfig{}
			bs.set(&cfg)
			_, err := NewBank(others, cfg, NewInMemoryCollector())
			if err == nil {
				t.Fatalf("block %q with owner %q NOT selected: expected an error, got none (silent drop)", bs.name, bs.owner)
			}
			if !strings.Contains(err.Error(), bs.name) {
				t.Errorf("error should name the offending block %q, got: %v", bs.name, err)
			}
		})

		t.Run(bs.name+"/all selected", func(t *testing.T) {
			cfg := SaturationConfig{}
			bs.set(&cfg)
			if _, err := NewBank(AllDetectorNames(), cfg, NewInMemoryCollector()); err != nil {
				t.Fatalf("block %q under --detectors all: unexpected error %v", bs.name, err)
			}
		})
	}
}

// The two ownership paths must agree on every (block, selection) pair. They are
// separate functions, so without a shared source of truth they can silently
// drift; this pins the equivalence rather than the implementation.
func TestBlockOwnership_SingleAndBankAgree(t *testing.T) {
	for _, bs := range allBlockSetters() {
		for _, selected := range AllDetectorNames() {
			cfg := SaturationConfig{}
			bs.set(&cfg)

			_, singleErr := BuildDetector(selected, cfg)
			_, bankErr := NewBank([]string{selected}, cfg, NewInMemoryCollector())

			if (singleErr == nil) != (bankErr == nil) {
				t.Errorf("block %q with %q selected: single path err=%v but bank path err=%v (the two ownership checks disagree)",
					bs.name, selected, singleErr, bankErr)
			}
		}
	}
}

// The unknown-name error must list every valid name, derived from the roster so a
// future detector cannot desync it.
func TestBuildDetector_UnknownNameListsEveryValidName(t *testing.T) {
	_, err := BuildDetector("no-such-detector", SaturationConfig{})
	if err == nil {
		t.Fatal("expected an error for an unknown detector name")
	}
	for _, n := range AllDetectorNames() {
		if !strings.Contains(err.Error(), n) {
			t.Errorf("error should list the valid name %q, got: %v", n, err)
		}
	}
}

// BC-7: every new knob is validated, and the error names the offending YAML
// field so the user can find it. Nothing panics (R6) -- the library boundary
// returns errors.
//
// The backlog_drift.slope_k cases matter especially: effectiveSlopeK() coerces a
// zero or non-finite SlopeK to the default for in-process struct literals, so
// without validation at the YAML layer a user's "slope_k: 0" would be silently
// accepted as 3.0 instead of reported (R1).
func TestBuildDetector_RejectsInvalidKnobs(t *testing.T) {
	nan := math.NaN()
	posInf := math.Inf(1)
	negInf := math.Inf(-1)
	zero := 0.0
	neg := -1.0
	zeroInt := 0

	tests := []struct {
		name      string
		detector  string
		cfg       SaturationConfig
		wantField string
	}{
		{"composite sensitivity zero", "composite",
			SaturationConfig{Composite: &CompositeBlock{Sensitivity: &zero}}, "composite.sensitivity"},
		{"composite sensitivity negative", "composite",
			SaturationConfig{Composite: &CompositeBlock{Sensitivity: &neg}}, "composite.sensitivity"},
		{"composite sensitivity NaN", "composite",
			SaturationConfig{Composite: &CompositeBlock{Sensitivity: &nan}}, "composite.sensitivity"},
		{"composite sensitivity +Inf", "composite",
			SaturationConfig{Composite: &CompositeBlock{Sensitivity: &posInf}}, "composite.sensitivity"},
		{"composite sensitivity -Inf", "composite",
			SaturationConfig{Composite: &CompositeBlock{Sensitivity: &negInf}}, "composite.sensitivity"},
		{"backlog_drift slope_k zero", "backlog-drift",
			SaturationConfig{BacklogDrift: &BacklogDriftBlock{SlopeK: &zero}}, "backlog_drift.slope_k"},
		{"backlog_drift slope_k negative", "backlog-drift",
			SaturationConfig{BacklogDrift: &BacklogDriftBlock{SlopeK: &neg}}, "backlog_drift.slope_k"},
		{"backlog_drift slope_k NaN", "backlog-drift",
			SaturationConfig{BacklogDrift: &BacklogDriftBlock{SlopeK: &nan}}, "backlog_drift.slope_k"},
		{"backlog_drift slope_k +Inf", "backlog-drift",
			SaturationConfig{BacklogDrift: &BacklogDriftBlock{SlopeK: &posInf}}, "backlog_drift.slope_k"},
		{"backlog_drift slope_k -Inf", "backlog-drift",
			SaturationConfig{BacklogDrift: &BacklogDriftBlock{SlopeK: &negInf}}, "backlog_drift.slope_k"},
		{"peak_rate threshold zero", "peak-rate",
			SaturationConfig{PeakRate: &PeakRateBlock{Threshold: &zero}}, "peak_rate.threshold"},
		{"peak_rate threshold negative", "peak-rate",
			SaturationConfig{PeakRate: &PeakRateBlock{Threshold: &neg}}, "peak_rate.threshold"},
		{"peak_rate threshold NaN", "peak-rate",
			SaturationConfig{PeakRate: &PeakRateBlock{Threshold: &nan}}, "peak_rate.threshold"},
		{"peak_rate threshold +Inf", "peak-rate",
			SaturationConfig{PeakRate: &PeakRateBlock{Threshold: &posInf}}, "peak_rate.threshold"},
		{"peak_rate min_observations zero", "peak-rate",
			SaturationConfig{PeakRate: &PeakRateBlock{MinObservations: &zeroInt}}, "peak_rate.min_observations"},
		{"peak_rate consecutive_k zero", "peak-rate",
			SaturationConfig{PeakRate: &PeakRateBlock{ConsecutiveK: &zeroInt}}, "peak_rate.consecutive_k"},
		{"peak_rate overload_multiple zero", "peak-rate",
			SaturationConfig{PeakRate: &PeakRateBlock{OverloadMultiple: &zero}}, "peak_rate.overload_multiple"},
		{"peak_rate overload_multiple negative", "peak-rate",
			SaturationConfig{PeakRate: &PeakRateBlock{OverloadMultiple: &neg}}, "peak_rate.overload_multiple"},
		{"peak_rate overload_multiple NaN", "peak-rate",
			SaturationConfig{PeakRate: &PeakRateBlock{OverloadMultiple: &nan}}, "peak_rate.overload_multiple"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			det, err := BuildDetector(tc.detector, tc.cfg)
			if err == nil {
				t.Fatalf("expected an error, got a detector: %v", det)
			}
			if det != nil {
				t.Errorf("no detector should be constructed on a validation error, got %v", det)
			}
			if !strings.Contains(err.Error(), tc.wantField) {
				t.Errorf("error should name the field %q, got: %v", tc.wantField, err)
			}
		})

		// The bank path must reject the same values (a selected detector's own
		// out-of-range parameter still errors -- R6).
		t.Run(tc.name+"/bank", func(t *testing.T) {
			if _, err := NewBank([]string{tc.detector}, tc.cfg, NewInMemoryCollector()); err == nil {
				t.Errorf("bank path accepted an invalid %s", tc.wantField)
			}
		})
	}
}

// A valid knob must be ACCEPTED on both paths -- the counterpart that keeps the
// rejection table above from passing vacuously.
func TestBuildDetector_AcceptsValidKnobs(t *testing.T) {
	two := 2.0
	half := 0.5
	for _, tc := range []struct {
		detector string
		cfg      SaturationConfig
	}{
		{"composite", SaturationConfig{Composite: &CompositeBlock{Sensitivity: &two}}},
		{"backlog-drift", SaturationConfig{BacklogDrift: &BacklogDriftBlock{SlopeK: &two}}},
		{"peak-rate", SaturationConfig{PeakRate: &PeakRateBlock{Threshold: &two}}},
		// A sub-1 overload_multiple is a legitimate "maximally severe" setting (every
		// fired event is OVERLOADED), matching what backlog_drift.slope_k allows.
		{"peak-rate", SaturationConfig{PeakRate: &PeakRateBlock{OverloadMultiple: &half}}},
	} {
		if _, err := BuildDetector(tc.detector, tc.cfg); err != nil {
			t.Errorf("%s: valid knob rejected on the single path: %v", tc.detector, err)
		}
		if _, err := NewBank([]string{tc.detector}, tc.cfg, NewInMemoryCollector()); err != nil {
			t.Errorf("%s: valid knob rejected on the bank path: %v", tc.detector, err)
		}
	}
}

// INV-6 for the bank with a TUNED knob: selection filters WHICH detectors run,
// never HOW they see traffic. A tuned detector's records under --detectors all
// must be byte-identical to its records under a single selection, and the other
// detectors must be unperturbed by the tuning of a peer.
func TestBank_TunedDetectorRecordsMatchUnderAll(t *testing.T) {
	events := makeCompositeStream(60)
	sens := 4.0
	cfg := SaturationConfig{Composite: &CompositeBlock{Sensitivity: &sens}}

	collect := func(names []string) map[string][]Result {
		sink := NewInMemoryCollector()
		bank, err := NewBank(names, cfg, sink)
		if err != nil {
			t.Fatalf("NewBank(%v): %v", names, err)
		}
		out := map[string][]Result{}
		for _, e := range events {
			bank.fanout(e)
		}
		for _, rec := range sink.Records() {
			out[rec.Detector] = append(out[rec.Detector], rec.Result)
		}
		return out
	}

	alone := collect([]string{"composite"})
	all := collect(AllDetectorNames())

	if len(alone["composite"]) == 0 {
		t.Fatal("no composite records collected; the comparison would be vacuous")
	}
	if len(alone["composite"]) != len(all["composite"]) {
		t.Fatalf("composite record count differs: %d alone vs %d under all",
			len(alone["composite"]), len(all["composite"]))
	}
	for i := range alone["composite"] {
		a, b := alone["composite"][i], all["composite"][i]
		if a.Level != b.Level || a.Score != b.Score || a.Confidence != b.Confidence {
			t.Errorf("record %d differs between a single selection and --detectors all: %+v vs %+v", i, a, b)
		}
		for k, av := range a.Signals {
			if bv := b.Signals[k]; bv != av {
				t.Errorf("record %d signal %q differs: %v vs %v", i, k, av, bv)
			}
		}
	}

	// The peer detectors must be untouched by composite's tuning: compare against
	// a bank where composite carries no override.
	untuned := func() map[string][]Result {
		sink := NewInMemoryCollector()
		bank, err := NewBank(AllDetectorNames(), SaturationConfig{}, sink)
		if err != nil {
			t.Fatalf("NewBank untuned: %v", err)
		}
		for _, e := range events {
			bank.fanout(e)
		}
		out := map[string][]Result{}
		for _, rec := range sink.Records() {
			out[rec.Detector] = append(out[rec.Detector], rec.Result)
		}
		return out
	}()

	for _, peer := range []string{"threshold", "backlog-drift"} {
		if len(all[peer]) != len(untuned[peer]) {
			t.Fatalf("%s record count changed when composite was tuned", peer)
		}
		for i := range all[peer] {
			if all[peer][i].Level != untuned[peer][i].Level {
				t.Errorf("%s record %d level changed when a PEER detector was tuned: %v vs %v",
					peer, i, all[peer][i].Level, untuned[peer][i].Level)
			}
		}
	}
}

// An unknown detector name must be reported as such, even when a VALID config
// block is also present. Ownership is meaningless for a name that is not a
// detector, so complaining about the block hides the actual mistake (the typo) and
// omits the valid names the user needs.
//
// This is a behavioral contract about which error a user receives, and it is the
// error that tells them what to fix.
func TestBuildDetector_UnknownNameIsReportedBeforeOwnership(t *testing.T) {
	ms := 5000.0
	one := 1.0
	for _, cfg := range []SaturationConfig{
		{},
		{Threshold: &ThresholdBlock{ThresholdMs: &ms}},
		{Composite: &CompositeBlock{Sensitivity: &one}},
	} {
		_, err := BuildDetector("bogus-detector", cfg)
		if err == nil {
			t.Fatal("expected an error for an unknown detector name")
		}
		if !strings.Contains(err.Error(), "bogus-detector") {
			t.Errorf("error should name the unknown detector, got: %v", err)
		}
		for _, valid := range AllDetectorNames() {
			if !strings.Contains(err.Error(), valid) {
				t.Errorf("error should list the valid name %q so the user can correct the typo, got: %v", valid, err)
			}
		}
	}
}
