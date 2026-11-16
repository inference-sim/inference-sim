package cmd

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
	"github.com/sirupsen/logrus"
	logtest "github.com/sirupsen/logrus/hooks/test"
	"github.com/spf13/cobra"
)

// inference-sim/inference-sim#2136: a run's derived per-position rates name the schedule that
// derived them, in a warning and in the results file.
func TestDerivedRatesNameTheirSchedule(t *testing.T) {
	al := 3.1
	rates, err := acceptance.NaturalRates(al, 7, 1.17)
	if err != nil {
		t.Fatal(err)
	}
	natural := acceptance.Resolution{AcceptanceLength: al, PerPositionRates: rates, RatesDerived: true,
		Schedule: acceptance.ScheduleNaturalDecay, ShapeSet: "inferencex-golden-al",
		Shape: &acceptance.Shape{SpeculativeMethod: "dspark", DecayExponent: 1.17, Method: vocab.MethodCalibrated}, Layer: acceptance.LayerCLI}
	step, _ := acceptance.MinimumVarianceRates(al, 7)
	forced := acceptance.Resolution{AcceptanceLength: al, PerPositionRates: step, RatesDerived: true,
		Schedule: acceptance.ScheduleMinimumVariance, Layer: acceptance.LayerCLI}
	stated := acceptance.Resolution{AcceptanceLength: 1.7, PerPositionRates: []float64{0.5, 0.2},
		Schedule: acceptance.ScheduleStated, Layer: acceptance.LayerCLI}
	for _, c := range []struct {
		name, schedule, shape, warn string
		r                           acceptance.Resolution
	}{
		{"natural", "natural_decay", `inferencex-golden-al "dspark" decay_exponent 1.17 method calibrated`, "fitted natural decay", natural},
		{"forced", "minimum_variance", "", "minimum-variance schedule", forced},
		{"stated", "", "", "", stated},
	} {
		rec := acceptanceRecordOf(c.r, speculativeConfig{})
		if rec.Entry.RatesSchedule != c.schedule || rec.Entry.RatesShape != c.shape {
			t.Errorf("%s: recorded schedule %q shape %q", c.name, rec.Entry.RatesSchedule, rec.Entry.RatesShape)
		}
		hook := logtest.NewGlobal()
		logAcceptance(rec, 7)
		var warned string
		for _, e := range hook.AllEntries() {
			if e.Level == logrus.WarnLevel && strings.Contains(e.Message, "per-position rates") {
				warned = e.Message
			}
		}
		hook.Reset()
		switch {
		case c.warn == "" && warned != "":
			t.Errorf("%s: stated rates warned %q", c.name, warned)
		case c.warn != "" && (!strings.Contains(warned, c.warn) || !strings.Contains(warned, c.schedule)):
			t.Errorf("%s: warning %q does not name %q and %q", c.name, warned, c.warn, c.schedule)
		}
	}
}

func TestTheRealisationFlagQualifiesAValue(t *testing.T) {
	run := func(args ...string) error {
		cmd := &cobra.Command{Use: "x"}
		registerAcceptanceFlags(cmd)
		if err := cmd.Flags().Parse(args); err != nil {
			t.Fatal(err)
		}
		return checkRealisationFlag(cmd.Flags())
	}
	defer func() { speculativeRealisation = "" }()
	if err := run("--speculative-acceptance-rate", "0.3", "--speculative-acceptance-realisation", "natural"); err != nil {
		t.Errorf("a natural rate was refused: %v", err)
	}
	if err := run("--speculative-acceptance-realisation", "natural"); err == nil || !strings.Contains(err.Error(), "qualifies a value") {
		t.Errorf("a realisation with no value: %v", err)
	}
	if err := run("--speculative-acceptance-length", "3", "--speculative-acceptance-realisation", "forced"); err == nil || !strings.Contains(err.Error(), "not one of") {
		t.Errorf("an unknown realisation: %v", err)
	}
	if err := run("--speculative-acceptance-rate", "0.3"); err != nil {
		t.Errorf("no realisation: %v", err)
	}
}

// The kernel's acceptance override (the adaptive-verification budget) takes the realisation too.
func TestTheKernelOverrideCarriesTheRealisation(t *testing.T) {
	cmd := &cobra.Command{Use: "x"}
	registerAcceptanceFlags(cmd)
	if err := cmd.Flags().Parse([]string{"--speculative-acceptance-rate", "0.3", "--speculative-acceptance-realisation", "natural"}); err != nil {
		t.Fatal(err)
	}
	defer func() { speculativeRealisation, speculativeAcceptance = "", 0 }()
	o := acceptanceOverridesOf(cmd.Flags(), 7)
	if o.CLI == nil || o.CLI.Realisation != acceptance.RealisationNatural {
		t.Fatalf("override %+v", o.CLI)
	}
}
