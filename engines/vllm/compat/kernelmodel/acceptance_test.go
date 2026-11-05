package kernelmodel

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	latencykernel "github.com/inference-sim/blis-substrate/kernel"
	"github.com/inference-sim/blis-substrate/load"

	"github.com/inference-sim/blis-vllm/internal/substratedata"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// museAdaptiveInputs are the Muse recipe default's Inputs, opened from vllmengine's testdata.
func museAdaptiveInputs(t *testing.T) (latencykernel.Inputs, engineconfig.Engine) {
	t.Helper()
	r := DefaultRepos()
	r.Scenarios, _ = filepath.Abs(filepath.Join("..", "..", "vllmengine", "testdata"))
	in, err := latencykernel.OpenInputs("muse-gb200-nvl-tep8-dspark-adaptive.yaml", r, 0)
	if err != nil {
		t.Fatal(err)
	}
	e, err := decodeEngine(in.Deployment.Pools[0])
	if err != nil {
		t.Fatal(err)
	}
	return in, e
}

// Under adaptive verification the kernel gets the run's acceptance, resolved from the
// scenario's pin (acceptance.value: num_spec_tokens 7, acceptance_length 2.84); without it the
// kernel gets none, so a pool that does not trim opens exactly as before. A resolution from a
// registry profile is recorded in the data's selected-entry report (VAL-6); a pinned value
// names no profile, so it records nothing.
func TestAcceptanceReachesTheKernelOnlyUnderAdaptiveVerification(t *testing.T) {
	in, e := museAdaptiveInputs(t)
	got, err := resolveAcceptance(in, e, repos().Data, AcceptanceOverrides{})
	if err != nil {
		t.Fatal(err)
	}
	if got == nil || got.AcceptanceLength != 2.84 || len(got.PerPositionRates) != 7 || got.Layer != "scenario" {
		t.Fatalf("resolved %+v, want the scenario's AL 2.84 over 7 positions", got)
	}

	// The registry layer: no pin, and a model, method and K the embedded registry has a profile
	// for (inferencex-golden-al: deepseek-v4.1-flash, dspark, K 5, coding, thinking off).
	data, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	reg, re := in, e
	reg.Scenario.Acceptance = nil
	reg.Scenario.Model = "deepseek-v4.1-flash"
	sp := *e.Speculative
	sp.NumSpecTokens = 5
	re.Speculative = &sp
	terms := acceptance.Workload{Category: "coding", Thinking: acceptance.ThinkingOff}
	got, err = resolveAcceptance(reg, re, data, AcceptanceOverrides{Workload: &terms})
	if err != nil {
		t.Fatal(err)
	}
	if got == nil || got.Layer != "registry" || got.Set != "inferencex-golden-al" {
		t.Fatalf("resolved %+v, want the registry's inferencex-golden-al profile", got)
	}
	var recorded []load.Selected
	for _, sel := range data.Report().Selected {
		if sel.Kind == load.KindAcceptance {
			recorded = append(recorded, sel)
		}
	}
	if len(recorded) != 1 || recorded[0].Set != "inferencex-golden-al" || recorded[0].Entry != got.Profile.Key().String() {
		t.Errorf("selected-entry report's acceptance entries %+v, want the one profile resolved (%s)",
			recorded, got.Profile.Key())
	}
	off := false
	e.Speculative.EnableAdaptiveVerification = &off
	if got, err := resolveAcceptance(in, e, repos().Data, AcceptanceOverrides{}); got != nil || err != nil {
		t.Errorf("flag off: resolved %+v (%v), want none", got, err)
	}
	e.Speculative.EnableAdaptiveVerification = nil
	if got, err := resolveAcceptance(in, e, repos().Data, AcceptanceOverrides{}); got != nil || err != nil {
		t.Errorf("flag unstated: resolved %+v (%v), want none", got, err)
	}
}

// A run's own acceptance layer outranks the scenario's pin, as in the simulator's resolution.
func TestAcceptanceOverridesOutrankThePin(t *testing.T) {
	in, e := museAdaptiveInputs(t)
	l := 4.0
	got, err := resolveAcceptance(in, e, repos().Data, AcceptanceOverrides{CLI: cliLength(7, &l)})
	if err != nil {
		t.Fatal(err)
	}
	if got.AcceptanceLength != 4 {
		t.Errorf("AL %v, want the override's 4", got.AcceptanceLength)
	}
}

// Adaptive verification with an acceptance that does not resolve is refused, naming the flag:
// the kernel cannot size the verify budget without it.
func TestAdaptiveVerificationWithoutAcceptanceIsRefused(t *testing.T) {
	in, e := museAdaptiveInputs(t)
	in.Scenario.Acceptance = nil
	_, err := resolveAcceptance(in, e, emptyRegistry(t), AcceptanceOverrides{})
	if err == nil || !strings.Contains(err.Error(), "enable_adaptive_verification") {
		t.Errorf("err %v, want a refusal naming enable_adaptive_verification", err)
	}
}

func cliLength(k int, l *float64) *acceptance.Value {
	return &acceptance.Value{NumSpecTokens: k, AcceptanceLength: l, Source: "test"}
}

// emptyRegistry is data whose registry holds no acceptance set: the embedded catalog beside a
// registry directory with the embedded coefficient sets only. A registry overlay cannot remove
// a set, so the registry is a directory of its own.
func emptyRegistry(t *testing.T) *load.Data {
	t.Helper()
	src, err := substratedata.Registry()
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.CopyFS(filepath.Join(dir, "coefficients"), os.DirFS(filepath.Join(src, "coefficients"))); err != nil {
		t.Fatal(err)
	}
	d, err := load.Open(load.Options{Registry: dir})
	if err != nil {
		t.Fatal(err)
	}
	return d
}
