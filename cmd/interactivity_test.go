package cmd

import (
	"math"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/metrics"

	"github.com/inference-sim/inference-sim/sim"
)

// The Result states the registry's interactivity_of_mean_tpot, InferenceX's mean_intvty: 1000 over
// the mean time per output token. It is not interactivity_mean, the mean over requests of
// 1000/TPOT, which is higher whenever the per-request TPOT varies (Jensen). Two requests with a
// TPOT of 10 ms and 40 ms: the mean TPOT is 25 ms, so 40 tokens/s/user; the mean of 100 and 25
// is 62.5.
func TestStandardMetrics_InteractivityOfMeanTPOTIsTheReciprocalOfTheMean(t *testing.T) {
	if _, ok := metrics.Lookup("interactivity_of_mean_tpot"); !ok {
		t.Fatal("the pinned registry does not define interactivity_of_mean_tpot")
	}
	sum := sim.MeasuredSummary{RequestITLCount: 2, PerUserCount: 2, WindowStartUs: 0, WindowEndUs: 1e6}
	sum.RequestITLMeanMs = 25
	sum.OutputTokPerUserMean = (100.0 + 25.0) / 2
	sum.MeasuredWindowS = 1
	got := standardMetrics(sum, nil, 8, 0, 1)
	f, ok := got["interactivity_of_mean_tpot"]
	if !ok {
		t.Fatal("the Result states no interactivity_of_mean_tpot")
	}
	if f.Value != 40 || f.Samples != 2 || f.Scope != metrics.ScopeCluster {
		t.Errorf("interactivity_of_mean_tpot = %+v, want 40 over 2 requests at the cluster scope", f)
	}
	if m := got["interactivity_mean"]; m.Value != 62.5 {
		t.Errorf("interactivity_mean = %v, want it kept as the mean of the reciprocals, 62.5", m.Value)
	}
	if err := (metrics.Figure{Definition: mustDef(t, "interactivity_of_mean_tpot"), Value: f.Value, Scope: f.Scope}).Validate(); err != nil {
		t.Errorf("the figure is not valid at its scope: %v", err)
	}
	// No request with two tokens, no TPOT, no figure.
	if _, ok := standardMetrics(sim.MeasuredSummary{}, nil, 8, 0, 1)["interactivity_of_mean_tpot"]; ok {
		t.Error("a run with no TPOT sample states interactivity_of_mean_tpot")
	}
}

func mustDef(t *testing.T, id string) metrics.Definition {
	t.Helper()
	d, ok := metrics.Lookup(id)
	if !ok {
		t.Fatalf("no %s in the registry", id)
	}
	return d
}

// On a real run with requests of different TPOT, the Result's interactivity_of_mean_tpot is 1000
// over its tpot_mean_ms, and interactivity_mean, the mean of the reciprocals, is the higher one.
func TestResult_InteractivityOfMeanTPOTIsOneThousandOverMeanTPOT(t *testing.T) {
	stdout, stderr, err := runKernelCLI(t, runArgs(append([]string{"--scenarios", pdScenarios, "--scenario", closenessScenario,
		"--num-instances", "1", "--seed", "3"}, burst(40, 20, 256, 48)...))...)
	if err != nil {
		t.Fatalf("%v\n%s", err, lastLines(stderr, 5))
	}
	figs, _ := resultDoc(t, stdout)["metrics"].(map[string]any)
	val := func(id string) float64 {
		f, ok := figs[id].(map[string]any)
		if !ok {
			t.Fatalf("the Result states no %s", id)
		}
		v, _ := f["value"].(float64)
		return v
	}
	tpot, of, mean := val("tpot_mean_ms"), val("interactivity_of_mean_tpot"), val("interactivity_mean")
	if math.Abs(of-1000/tpot) > 1e-9*of {
		t.Errorf("interactivity_of_mean_tpot = %v, want 1000 / tpot_mean_ms = %v", of, 1000/tpot)
	}
	if mean <= of {
		t.Errorf("non-vacuity: interactivity_mean %v is not above interactivity_of_mean_tpot %v, so the two are not told apart by this run", mean, of)
	}
}
