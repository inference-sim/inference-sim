package kernelmodel

import (
	"path/filepath"
	"strings"
	"testing"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-schemas/spec/acceptance"

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
// kernel gets none, so a pool that does not trim opens exactly as before.
func TestAcceptanceReachesTheKernelOnlyUnderAdaptiveVerification(t *testing.T) {
	in, e := museAdaptiveInputs(t)
	got, err := resolveAcceptance(in, e, DefaultRegistry(), AcceptanceOverrides{})
	if err != nil {
		t.Fatal(err)
	}
	if got == nil || got.AcceptanceLength != 2.84 || len(got.PerPositionRates) != 7 || got.Layer != "scenario" {
		t.Fatalf("resolved %+v, want the scenario's AL 2.84 over 7 positions", got)
	}
	off := false
	e.Speculative.EnableAdaptiveVerification = &off
	if got, err := resolveAcceptance(in, e, DefaultRegistry(), AcceptanceOverrides{}); got != nil || err != nil {
		t.Errorf("flag off: resolved %+v (%v), want none", got, err)
	}
	e.Speculative.EnableAdaptiveVerification = nil
	if got, err := resolveAcceptance(in, e, DefaultRegistry(), AcceptanceOverrides{}); got != nil || err != nil {
		t.Errorf("flag unstated: resolved %+v (%v), want none", got, err)
	}
}

// A run's own acceptance layer outranks the scenario's pin, as in the simulator's resolution.
func TestAcceptanceOverridesOutrankThePin(t *testing.T) {
	in, e := museAdaptiveInputs(t)
	l := 4.0
	got, err := resolveAcceptance(in, e, DefaultRegistry(), AcceptanceOverrides{CLI: cliLength(7, &l)})
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
	_, err := resolveAcceptance(in, e, t.TempDir(), AcceptanceOverrides{})
	if err == nil || !strings.Contains(err.Error(), "enable_adaptive_verification") {
		t.Errorf("err %v, want a refusal naming enable_adaptive_verification", err)
	}
}

func cliLength(k int, l *float64) *acceptance.Value {
	return &acceptance.Value{NumSpecTokens: k, AcceptanceLength: l, Source: "test"}
}
