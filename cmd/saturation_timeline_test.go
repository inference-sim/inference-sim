package cmd

import (
	"testing"
	"time"

	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"
)

// TestSaturationTimelineFlags_RegisteredOnAllCommands verifies the three timeline
// flags are present with identical defaults on run, replay, and observe — the
// run/replay/observe parity contract for post-hoc saturation flags (they are
// registered per-command via registerSaturationTimelineFlags, mirroring
// --post-hoc-detector).
func TestSaturationTimelineFlags_RegisteredOnAllCommands(t *testing.T) {
	flags := []struct {
		name     string
		defValue string
	}{
		{"saturation-interval", "0s"},
		{"saturation-unsure-min-requests", "20"},
		{"saturation-unsure-min-confidence", "0.5"},
	}
	cmds := map[string]*cobra.Command{"run": runCmd, "replay": replayCmd, "observe": observeCmd}
	for _, f := range flags {
		for cmdName, cmd := range cmds {
			fl := cmd.Flags().Lookup(f.name)
			assert.NotNilf(t, fl, "%s must register --%s", cmdName, f.name)
			if fl != nil {
				assert.Equalf(t, f.defValue, fl.DefValue,
					"%s --%s default = %q, want %q", cmdName, f.name, fl.DefValue, f.defValue)
			}
		}
	}
}

// TestSaturationTimelineConfig_FromFlags verifies the flag→config translation,
// especially the Duration→microseconds conversion (sim clock is µs).
func TestSaturationTimelineConfig_FromFlags(t *testing.T) {
	// GIVEN flag values
	saturationInterval = 2 * time.Second
	saturationUnsureMinRequests = 15
	saturationUnsureMinConf = 0.7
	defer func() { // restore defaults for other tests
		saturationInterval = 0
		saturationUnsureMinRequests = 20
		saturationUnsureMinConf = 0.5
	}()

	// WHEN we build the config
	cfg := saturationTimelineConfig()

	// THEN interval is in microseconds and thresholds pass through
	assert.Equal(t, int64(2_000_000), cfg.IntervalUs, "2s must be 2_000_000 µs")
	assert.Equal(t, 15, cfg.MinRequests)
	assert.InDelta(t, 0.7, cfg.MinConfidence, 1e-9)
}
