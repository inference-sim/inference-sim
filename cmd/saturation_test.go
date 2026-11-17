package cmd

import (
	"strings"
	"testing"
)

// A detector of the retired roster is refused by name, and the refusal names the one that
// replaced them (inference-sim#1618): a script that still passes one learns why it stopped.
func TestRetiredSaturationDetectorsAreRefused(t *testing.T) {
	for _, old := range []string{"composite", "threshold", "backlog-drift", "peak-rate", "all"} {
		_, stderr, err := runKernelCLI(t, resultRunArgs("--detectors", old)...)
		if err == nil || !strings.Contains(stderr, "settle") {
			t.Errorf("--detectors %s: err %v, stderr %q; want a refusal naming settle", old, err, lastLines(stderr, 2))
		}
	}
}
