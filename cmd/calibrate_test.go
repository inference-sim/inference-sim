package cmd

import (
	"testing"
)

func TestCalibrateCmd_Flags_Registered(t *testing.T) {
	// GIVEN the calibrate command
	// WHEN we inspect its registered flags
	// THEN all 7 flags must be present
	flags := []string{
		"trace-header",
		"trace-data",
		"sim-results",
		"report",
		"warmup-requests",
		"network-rtt-us",
		"network-bandwidth-mbps",
	}
	for _, name := range flags {
		f := calibrateCmd.Flags().Lookup(name)
		if f == nil {
			t.Errorf("calibrateCmd missing flag --%s", name)
		}
	}
}
