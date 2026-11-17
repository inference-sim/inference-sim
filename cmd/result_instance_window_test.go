package cmd

// NOTE: mutates package-level CLI vars through runAgentXReplay; no t.Parallel().

import (
	"testing"
)

// Each engine instance's block states its own figures over the run's measured window
// beside its whole-run ones (inference-sim#2087): the window is the Result's, the instances'
// measured completions add up to the cluster's, and the figures are present for every instance.
func TestResultInstancesStateTheMeasuredWindow(t *testing.T) {
	stdout, _ := runAgentXReplay(t, "--concurrency", "4", "--scenarios", "../examples", "--scenario", "agentx-dsv41-flash-h100-2x-tp8-dspark5.yaml", "--num-instances", "2")
	r := parseResult(t, string(stdout))
	insts := instanceFigures(t, string(stdout))
	if len(insts) != 2 {
		t.Fatalf("%d instance blocks, want 2", len(insts))
	}
	sum := 0.0
	for id, b := range insts {
		mw, ok := b["measured_window"].(map[string]any)
		if !ok {
			t.Fatalf("instance %s has no measured_window: %v", id, b)
		}
		if int64(mw["window_start_us"].(float64)) != r.Window.StartUs || int64(mw["window_end_us"].(float64)) != r.Window.EndUs {
			t.Errorf("instance %s window %v-%v, want the Result's %d-%d", id, mw["window_start_us"], mw["window_end_us"], r.Window.StartUs, r.Window.EndUs)
		}
		sum += mw["measured_completed_requests"].(float64)
		if _, ok := b["completed_requests"]; !ok {
			t.Errorf("instance %s lost its whole-run figures", id)
		}
	}
	if int(sum) != r.Window.Requests {
		t.Errorf("instances' measured completions add to %v, the Result's window holds %d", sum, r.Window.Requests)
	}
}
