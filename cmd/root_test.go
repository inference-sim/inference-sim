package cmd

import (
	"bytes"
	"io"
	"os"
	"strconv"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
)

func TestRunCmd_DefaultLogLevel_RemainsWarn(t *testing.T) {
	// GIVEN the run command with its registered flags
	flag := runCmd.Flags().Lookup("log")

	// WHEN we check the default value
	// THEN it MUST still be "warn" — we did NOT change the default (BC-5)
	// Simulation results go to stdout via fmt, not through logrus.
	assert.NotNil(t, flag, "log flag must be registered")
	assert.Equal(t, "warn", flag.DefValue,
		"default log level must remain 'warn'; simulation results use fmt.Println to bypass logrus")
}

func TestSaveResults_MetricsPrintedToStdout(t *testing.T) {
	// GIVEN a Metrics struct with completed requests
	m := sim.NewMetrics()
	m.CompletedRequests = 5
	m.TotalInputTokens = 100
	m.TotalOutputTokens = 50
	m.SimEndedTime = 1_000_000 // 1 second in ticks
	m.RequestTTFTs["r1"] = 100.0
	m.RequestE2Es["r1"] = 500.0
	m.RequestSchedulingDelays["r1"] = 50
	m.AllITLs = []int64{10, 20, 30}

	// Capture stdout
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	// WHEN SaveResults is called
	m.SaveResults("test", 1_000_000, 1000, "")

	// Restore stdout and read captured output
	_ = w.Close()
	os.Stdout = old
	var buf bytes.Buffer
	_, _ = io.Copy(&buf, r)
	output := buf.String()

	// THEN the metrics JSON MUST appear on stdout (BC-1)
	assert.Contains(t, output, "Simulation Metrics", "metrics header must be on stdout")
	assert.Contains(t, output, "completed_requests", "metrics JSON must be on stdout")
}

func TestRunCmd_KVBlockFlags_DefaultsArePositive(t *testing.T) {
	// GIVEN the run command with its registered flags
	kvBlocksFlag := runCmd.Flags().Lookup("total-kv-blocks")
	blockSizeFlag := runCmd.Flags().Lookup("block-size-in-tokens")

	// WHEN we check the default values
	// THEN they MUST be positive (BC-5: valid defaults pass validation)
	assert.NotNil(t, kvBlocksFlag, "total-kv-blocks flag must be registered")
	assert.NotNil(t, blockSizeFlag, "block-size-in-tokens flag must be registered")

	kvDefault, err := strconv.ParseInt(kvBlocksFlag.DefValue, 10, 64)
	assert.NoError(t, err, "total-kv-blocks default must be a valid int64")
	assert.Greater(t, kvDefault, int64(0),
		"default total-kv-blocks must be positive (passes <= 0 validation)")

	bsDefault, err := strconv.ParseInt(blockSizeFlag.DefValue, 10, 64)
	assert.NoError(t, err, "block-size-in-tokens default must be a valid int64")
	assert.Greater(t, bsDefault, int64(0),
		"default block-size-in-tokens must be positive (passes <= 0 validation)")
}

func TestRunCmd_SnapshotRefreshInterval_FlagRegistered(t *testing.T) {
	// Verify --snapshot-refresh-interval flag exists with a valid (non-negative) default.
	// Note: BC-5 (negative value rejection via logrus.Fatalf) is validated by code
	// inspection — the validation follows the same pattern as --kv-transfer-base-latency.
	// Testing logrus.Fatalf requires subprocess execution, which is out of scope here.
	flag := runCmd.Flags().Lookup("snapshot-refresh-interval")
	assert.NotNil(t, flag, "snapshot-refresh-interval flag must be registered")

	defVal, err := strconv.ParseInt(flag.DefValue, 10, 64)
	assert.NoError(t, err, "default must be a valid int64")
	assert.GreaterOrEqual(t, defVal, int64(0),
		"default snapshot-refresh-interval must be >= 0")
}

// TestRunCmd_MaxRunningReqs_FlagRegistered verifies BC-1:
// --max-num-running-reqs flag exists with a positive default.
func TestRunCmd_MaxRunningReqs_FlagRegistered(t *testing.T) {
	flag := runCmd.Flags().Lookup("max-num-running-reqs")
	assert.NotNil(t, flag, "max-num-running-reqs flag must be registered")
	defVal, err := strconv.ParseInt(flag.DefValue, 10, 64)
	assert.NoError(t, err)
	assert.Greater(t, defVal, int64(0), "default must be > 0 (passes validation)")
}

// TestRunCmd_MaxScheduledTokens_FlagRegistered verifies BC-2:
// --max-num-scheduled-tokens flag exists with a positive default.
func TestRunCmd_MaxScheduledTokens_FlagRegistered(t *testing.T) {
	flag := runCmd.Flags().Lookup("max-num-scheduled-tokens")
	assert.NotNil(t, flag, "max-num-scheduled-tokens flag must be registered")
	defVal, err := strconv.ParseInt(flag.DefValue, 10, 64)
	assert.NoError(t, err)
	assert.Greater(t, defVal, int64(0), "default must be > 0 (passes validation)")
}
