package cmd

import (
	"path/filepath"
	"testing"
)

// With BLIS_ROUTING_IMBALANCE=balls-in-bins, an expert-parallel group that has no routing profile
// prices its busiest rank at the expectation of uniform routing (inference-sim#2309), so the
// expert-parallel run is slower; the same model at tensor parallel (no expert ranks) is unchanged,
// and with the switch off the expert-parallel run is the release's.
func TestBallsInBinsImbalanceSlowsExpertParallelOnly(t *testing.T) {
	tput := func(scenario string, on bool) float64 {
		if on {
			t.Setenv("BLIS_ROUTING_IMBALANCE", "balls-in-bins")
		} else {
			t.Setenv("BLIS_ROUTING_IMBALANCE", "")
		}
		dir, _ := filepath.Abs(filepath.Join("..", "testdata", "kernel"))
		stdout, stderr, err := runKernelCLI(t, "run", "--scenarios", dir, "--scenario", scenario, "--rate", "40", "--num-requests", "40",
			"--prompt-tokens", "256", "--output-tokens", "64", "--seed", "3", "--horizon", dpParityHorizon)
		if err != nil {
			t.Fatalf("%s: %v\n%s", scenario, err, lastLines(stderr, 3))
		}
		figs, _ := resultDoc(t, stdout)["metrics"].(map[string]any)
		f, _ := figs["output_tok_s"].(map[string]any)
		v, _ := f["value"].(float64)
		if v <= 0 {
			t.Fatalf("%s: no throughput", scenario)
		}
		return v
	}
	const ep, tp = "minimax-m25-h200-ep8.yaml", "minimax-m25-h200-tp8.yaml"
	epOff, epOn := tput(ep, false), tput(ep, true)
	tpOff, tpOn := tput(tp, false), tput(tp, true)
	if !(epOn < epOff) {
		t.Errorf("expert-parallel throughput with the imbalance %.2f, without %.2f: want lower", epOn, epOff)
	}
	if tpOn != tpOff {
		t.Errorf("tensor-parallel throughput changed with the switch: %.4f vs %.4f", tpOn, tpOff)
	}
}
