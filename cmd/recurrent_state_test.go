package cmd

import (
	"path/filepath"
	"testing"
)

// A model with recurrent state (Kimi-K3's KDA layers, Nemotron-3-Ultra's Mamba-2 layers) is served,
// not refused: the engine maps its recurrent layers to vLLM's Mamba cache spec and completes the
// block size, the cache mode and the KV layers itself, so the host changes nothing but the pin of
// the engine. The scenarios are copies of the engine's test inputs for the mapping (they are not
// published arms), and the figures of a run are not claims about accuracy: the test holds only that
// the request set is served to completion.
func TestRecurrentStateModelsAreServed(t *testing.T) {
	dir, err := filepath.Abs("../testdata/recurrent")
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		scenario string
		extra    []string
	}{
		{"recurrent-kimi-k3-h200-dep16.yaml", nil},
		// MTP needs an acceptance length; the scenario states none, so the run is given one.
		{"recurrent-nemotron-3-ultra-h100-tp8.yaml", []string{"--speculative-acceptance-length", "1.5"}},
	} {
		t.Run(tt.scenario, func(t *testing.T) {
			args := append([]string{"run", "--scenarios", dir, "--scenario", tt.scenario, "--seed", "42",
				"--num-requests", "16", "--concurrency", "4"}, tt.extra...)
			stdout, stderr, err := runKernelCLI(t, args...)
			if err != nil {
				t.Fatalf("a recurrent-state model was refused: %v\n%s", err, lastLines(stderr, 3))
			}
			if !completedIs(stdout, 16) {
				t.Errorf("the run did not complete its 16 requests:\n%s", lastLines(stderr, 3))
			}
		})
	}
}
