package cmd

import (
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

// engine.long_prefill_token_threshold comes from the scenario only (#2021 B6): run and replay
// have no flag for it, so the CLI cannot run an engine its scenario does not describe.
func TestRunCmd_LongPrefillThresholdHasNoFlag(t *testing.T) {
	for _, c := range []*cobra.Command{runCmd, replayCmd} {
		if f := c.Flags().Lookup("long-prefill-token-threshold"); f != nil {
			t.Errorf("%s still registers --long-prefill-token-threshold", c.Name())
		}
	}
	_, stderr, err := runKernelCLI(t, "run", "--scenario", colocatedFixture, "--num-requests", "4", "--rate", "4",
		"--long-prefill-token-threshold", "256")
	if err == nil || !strings.Contains(stderr, "unknown flag: --long-prefill-token-threshold") {
		t.Errorf("the flag must be unknown: %v\n%s", err, lastLines(stderr, 3))
	}
}
