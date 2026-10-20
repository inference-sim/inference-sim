package cmd

import (
	"testing"

	"github.com/spf13/cobra"
)

// The command line's routing value is handed to the pools as the highest layer, and an
// expert ratio without a token ratio is refused rather than silently ignored.
func TestRoutingFlagsBecomeTheCLILayer(t *testing.T) {
	cmd := &cobra.Command{}
	registerRoutingFlags(cmd)
	if err := cmd.Flags().Set("routing-token-imbalance", "1.6"); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Flags().Set("routing-expert-imbalance", "0.8"); err != nil {
		t.Fatal(err)
	}
	applyRoutingOverrides(cmd)
	got := currentRoutingOverrides()
	if got.CLI == nil || got.CLI.TokenImbalance != 1.6 || got.CLI.Expert() != 0.8 {
		t.Fatalf("the flags did not become the cli layer: %+v", got.CLI)
	}

	cmd = &cobra.Command{}
	registerRoutingFlags(cmd)
	applyRoutingOverrides(cmd)
	if currentRoutingOverrides().CLI != nil {
		t.Error("an unset flag stated a cli value")
	}
}
