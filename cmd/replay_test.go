package cmd

import "testing"

// TestReplayCmd_SimConfigFlags_Registered verifies BC-4:
// all sim config flags registered on replayCmd.
func TestReplayCmd_SimConfigFlags_Registered(t *testing.T) {
	flags := []string{
		"seed", "log", "defaults-filepath", "model", "hardware", "tp",
		"latency-model", "total-kv-blocks", "block-size-in-tokens",
		"max-num-running-reqs", "max-num-scheduled-tokens",
		"beta-coeffs", "alpha-coeffs", "num-instances",
		"routing-policy", "scheduler", "priority-policy",
		"results-path", "kv-cpu-blocks", "snapshot-refresh-interval",
		"horizon", "trace-header", "trace-data",
	}
	for _, name := range flags {
		f := replayCmd.Flags().Lookup(name)
		if f == nil {
			t.Errorf("replayCmd missing flag --%s", name)
		}
	}
}
