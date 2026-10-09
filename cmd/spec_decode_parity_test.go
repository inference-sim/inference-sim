package cmd

import (
	"path/filepath"
	"testing"
)

// mtpScenario drafts 3 tokens per step (MTP K=3); speculation comes from the scenario, and
// only the acceptance rate -- a property of the workload -- is a flag.
const mtpScenario = "glm-5-h200-tp8-mtp3.yaml"

// TestINV13_RunReplayParity_SpecDecode pins INV-13 for speculative decoding: a run of a drafting
// scenario and the replay of its exported trace -- same scenario, same acceptance rate -- are
// byte-identical. And whatever the acceptance rate, speculation changes how many steps a request
// takes, never how many tokens it produces: every request completes and the output-token total
// is the same at every rate (INV-1).
func TestINV13_RunReplayParity_SpecDecode(t *testing.T) {
	common := []string{"--scenarios", pdScenarios, "--scenario", mtpScenario, "--seed", "99",
		"--horizon", "600000000"}
	const numRequests = 20
	outputTokens := -1
	for _, acc := range []string{"0.0", "0.6", "1.0"} {
		t.Run("acceptance "+acc, func(t *testing.T) {
			prefix := filepath.Join(t.TempDir(), "trace")
			args := append([]string{"run", "--num-requests", "20", "--rate", "6",
				"--speculative-acceptance-rate", acc, "--trace-output", prefix}, common...)
			runOut, stderr, err := runKernelCLI(t, args...)
			if err != nil {
				t.Fatalf("run: %v\n%s", err, lastLines(stderr, 3))
			}
			if got := clusterMetricInt(t, runOut, "completed_requests"); got != numRequests {
				t.Fatalf("INV-11: %d of %d requests completed", got, numRequests)
			}
			tokens := clusterMetricInt(t, runOut, "total_output_tokens")
			if outputTokens >= 0 && tokens != outputTokens {
				t.Errorf("INV-1: acceptance %s produced %d output tokens, another rate %d -- speculation "+
					"must not change how many tokens a request emits", acc, tokens, outputTokens)
			}
			outputTokens = tokens

			replayOut, stderr, err := runKernelCLI(t, append([]string{"replay", "--trace-header", prefix + ".yaml",
				"--trace-data", prefix + ".csv", "--speculative-acceptance-rate", acc}, common...)...)
			if err != nil {
				t.Fatalf("replay: %v\n%s", err, lastLines(stderr, 3))
			}
			if replayOut != runOut {
				t.Errorf("INV-13: the replay of a drafting run differs from the run\n--- run\n%s\n--- replay\n%s",
					runOut, replayOut)
			}
		})
	}
}
