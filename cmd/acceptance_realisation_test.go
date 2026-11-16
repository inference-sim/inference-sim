package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
	"github.com/sirupsen/logrus"
	logtest "github.com/sirupsen/logrus/hooks/test"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim"
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
	naturalShape := &sim.AcceptanceRatesShape{Set: "inferencex-golden-al", SpeculativeMethod: "dspark", DecayExponent: 1.17, Method: "calibrated"}
	for _, c := range []struct {
		name, schedule, warn string
		shape                *sim.AcceptanceRatesShape
		r                    acceptance.Resolution
	}{
		{"natural", "natural_decay", "fitted natural decay", naturalShape, natural},
		{"forced", "minimum_variance", "minimum-variance schedule", nil, forced},
		{"stated", "", "", nil, stated},
	} {
		rec := acceptanceRecordOf(c.r, speculativeConfig{})
		if rec.Entry.RatesSchedule != c.schedule || !reflect.DeepEqual(rec.Entry.RatesShape, c.shape) {
			t.Errorf("%s: recorded schedule %q shape %+v, want %q %+v", c.name, rec.Entry.RatesSchedule, rec.Entry.RatesShape, c.schedule, c.shape)
		}
		if c.shape != nil {
			out, err := json.Marshal(rec.Entry)
			var back struct {
				RatesShape map[string]any `json:"rates_shape"`
			}
			if err != nil || json.Unmarshal(out, &back) != nil || back.RatesShape["set"] != "inferencex-golden-al" ||
				back.RatesShape["speculative_method"] != "dspark" || back.RatesShape["decay_exponent"] != 1.17 || back.RatesShape["method"] != "calibrated" {
				t.Errorf("%s: results file rates_shape %s is not the typed shape", c.name, out)
			}
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

// writeMiniMaxEagle3Scenario is the MiniMax-M3 b300 deployment with its pool drafting three
// EAGLE3 tokens, the drafter named by speculative_config.model, plus the given engine lines.
func writeMiniMaxEagle3Scenario(t *testing.T, extra string) string {
	t.Helper()
	body := readFile(t, "../testdata/kernel/direct/minimax-m3-b300-fp4-vllm-tp8-mtp-cfg4.yaml")
	const at = "      enable_prefix_caching: false\n"
	if !strings.Contains(body, at) {
		t.Fatalf("fixture has no %q to edit", at)
	}
	body = strings.Replace(body, at, at+"      speculative_config:\n        method: eagle3\n        num_speculative_tokens: 3\n"+
		"        model: minimax-m3-eagle3\n"+extra, 1)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "variant.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// resolveScenarioAcceptance adopts the scenario in dir and resolves its acceptance as a run does.
func resolveScenarioAcceptance(t *testing.T, dir string, extra ...string) acceptance.Resolution {
	t.Helper()
	_, catalog, registry := kernelRepos(t)
	setupKernelTestFixtures(t)
	savedModel, savedGPU := model, gpu
	t.Cleanup(func() {
		model, gpu = savedModel, savedGPU
		speculativeThinking, speculativeCategory = "", ""
	})
	cmd := &cobra.Command{}
	registerSimConfigFlags(cmd)
	if err := cmd.ParseFlags(append([]string{"--scenario", "variant.yaml", "--scenarios", dir,
		"--registry", registry, "--catalog", catalog,
		"--speculative-thinking", "on", "--speculative-category", "coding"}, extra...)); err != nil {
		t.Fatal(err)
	}
	adoptKernelDeployment(cmd)
	a, ok := resolveAcceptance(cmd, numSpeculativeTokens, speculativeMethod)
	if !ok {
		t.Fatalf("the scenario does not speculate (K %d, method %q)", numSpeculativeTokens, speculativeMethod)
	}
	return a.Res
}

// inference-sim/inference-sim#2142: a pool that names its drafter (speculative.draft_model) is
// looked up by that drafter, and at vLLM's resolved draft sampler and verifier, so MiniMax-M3 with
// the minimax-m3-eagle3 drafter matches that drafter's own profile exactly rather than borrowing
// another drafter's.
func TestAPoolsDrafterSelectsItsOwnProfile(t *testing.T) {
	res := resolveScenarioAcceptance(t, writeMiniMaxEagle3Scenario(t, ""))
	if res.Step != acceptance.StepExact || len(res.Differed) != 0 {
		t.Fatalf("step %q differed %v, want the drafter's own profile exactly", res.Step, res.Differed)
	}
	if res.Profile == nil || res.Profile.Drafter != "minimax-m3-eagle3" || res.Profile.NumSpecTokens != 3 {
		t.Fatalf("resolved profile %+v, want minimax-m3-eagle3 at K 3", res.Profile)
	}
}

// The results file records how a profile lookup found its value: a borrowed one names the step
// and the key fields that differed, as the substrate's Result property does.
func TestTheRecordNamesABorrowedStep(t *testing.T) {
	res := resolveScenarioAcceptance(t, writeMiniMaxEagle3Scenario(t, "        rejection_sample_method: block\n"))
	rec := acceptanceRecordOf(res, speculativeConfig{})
	want := result.AcceptanceProperty(res)
	if !res.Borrowed() || rec.Entry.Step != string(res.Step) || rec.Entry.Step == "" || len(rec.Entry.Differed) == 0 {
		t.Fatalf("recorded step %q differed %v for resolution step %q", rec.Entry.Step, rec.Entry.Differed, res.Step)
	}
	if !slices.Contains(rec.Entry.Differed, "verifier") || !slices.Equal(rec.Entry.Differed, want.Differed) ||
		!slices.Equal(rec.Entry.Profiles, want.Profiles) || rec.Entry.KRule != want.KRule {
		t.Errorf("recorded differed %v profiles %v k_rule %q, want %v %v %q", rec.Entry.Differed, rec.Entry.Profiles,
			rec.Entry.KRule, want.Differed, want.Profiles, want.KRule)
	}
	out, err := json.Marshal(rec)
	if err != nil {
		t.Fatal(err)
	}
	var back struct {
		Entry struct {
			Step     string   `json:"step"`
			Differed []string `json:"differed"`
		} `json:"entry"`
	}
	if err := json.Unmarshal(out, &back); err != nil || back.Entry.Step != rec.Entry.Step || !slices.Equal(back.Entry.Differed, rec.Entry.Differed) {
		t.Errorf("results file %s does not carry step and differed (%v)", out, err)
	}
}
