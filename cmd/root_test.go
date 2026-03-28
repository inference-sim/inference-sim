package cmd

import (
	"bytes"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
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
	if err := m.SaveResults("test", 1_000_000, 1000, ""); err != nil {
		t.Fatalf("SaveResults returned error: %v", err)
	}

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

// TestApplyRopeScaling validates the pure function extraction of rope_scaling logic.
// Covers BC-1 (mrope), BC-2 (blacklist), BC-3 (gemma3), BC-4 (yarn), BC-8 (invalid input), BC-9 (never panics).
func TestApplyRopeScaling(t *testing.T) {
	tests := []struct {
		name        string
		maxPosEmb   int
		modelType   string
		ropeScaling any
		wantScaled  int
		wantApplied bool
	}{
		// Basic cases
		{name: "nil rope_scaling", maxPosEmb: 8192, modelType: "", ropeScaling: nil, wantScaled: 8192, wantApplied: false},
		{name: "linear factor 4", maxPosEmb: 8192, modelType: "", ropeScaling: map[string]any{"type": "linear", "factor": 4.0}, wantScaled: 32768, wantApplied: true},
		{name: "dynamic factor 2", maxPosEmb: 4096, modelType: "", ropeScaling: map[string]any{"type": "dynamic", "factor": 2.0}, wantScaled: 8192, wantApplied: true},
		{name: "default factor 2", maxPosEmb: 4096, modelType: "", ropeScaling: map[string]any{"type": "default", "factor": 2.0}, wantScaled: 8192, wantApplied: true},

		// BC-1: mrope — intentionally not excluded (vLLM normalizes mrope → "default" and applies factor)
		{name: "mrope factor 8", maxPosEmb: 8192, modelType: "", ropeScaling: map[string]any{"type": "mrope", "factor": 8.0}, wantScaled: 65536, wantApplied: true},

		// BC-2: Blacklist — su, longrope, llama3 excluded
		{name: "su excluded", maxPosEmb: 8192, modelType: "", ropeScaling: map[string]any{"type": "su", "factor": 4.0}, wantScaled: 8192, wantApplied: false},
		{name: "longrope excluded", maxPosEmb: 8192, modelType: "", ropeScaling: map[string]any{"type": "longrope", "factor": 4.0}, wantScaled: 8192, wantApplied: false},
		{name: "llama3 excluded", maxPosEmb: 8192, modelType: "", ropeScaling: map[string]any{"type": "llama3", "factor": 4.0}, wantScaled: 8192, wantApplied: false},

		// BC-3: gemma3 model_type exclusion (substring match covers text_config pivot)
		{name: "gemma3 skips rope_scaling", maxPosEmb: 8192, modelType: "gemma3", ropeScaling: map[string]any{"type": "linear", "factor": 4.0}, wantScaled: 8192, wantApplied: false},
		{name: "gemma3_text skips rope_scaling", maxPosEmb: 8192, modelType: "gemma3_text", ropeScaling: map[string]any{"type": "linear", "factor": 4.0}, wantScaled: 8192, wantApplied: false},

		// BC-4: yarn uses original_max_position_embeddings
		{name: "yarn with original", maxPosEmb: 8192, modelType: "", ropeScaling: map[string]any{"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 2048.0}, wantScaled: 8192, wantApplied: true},
		{name: "yarn without original", maxPosEmb: 4096, modelType: "", ropeScaling: map[string]any{"type": "yarn", "factor": 2.0}, wantScaled: 8192, wantApplied: true},

		// BC-8: Invalid inputs — warn and ignore
		{name: "non-object rope_scaling string", maxPosEmb: 8192, modelType: "", ropeScaling: "not-a-map", wantScaled: 8192, wantApplied: false},
		{name: "non-object rope_scaling array", maxPosEmb: 8192, modelType: "", ropeScaling: []any{1.0, 2.0}, wantScaled: 8192, wantApplied: false},
		{name: "factor not float64", maxPosEmb: 8192, modelType: "", ropeScaling: map[string]any{"type": "linear", "factor": "four"}, wantScaled: 8192, wantApplied: false},
		{name: "factor lte 1", maxPosEmb: 8192, modelType: "", ropeScaling: map[string]any{"type": "linear", "factor": 1.0}, wantScaled: 8192, wantApplied: false},
		{name: "no factor key", maxPosEmb: 8192, modelType: "", ropeScaling: map[string]any{"type": "linear"}, wantScaled: 8192, wantApplied: false},
		{name: "null type with factor", maxPosEmb: 4096, modelType: "", ropeScaling: map[string]any{"type": nil, "factor": 2.0}, wantScaled: 8192, wantApplied: true},
		{name: "empty type with factor", maxPosEmb: 4096, modelType: "", ropeScaling: map[string]any{"factor": 2.0}, wantScaled: 8192, wantApplied: true},

		// rope_type fallback key
		{name: "rope_type fallback", maxPosEmb: 4096, modelType: "", ropeScaling: map[string]any{"rope_type": "linear", "factor": 3.0}, wantScaled: 12288, wantApplied: true},

		// NaN/Inf defense-in-depth
		{name: "NaN factor", maxPosEmb: 8192, modelType: "", ropeScaling: map[string]any{"type": "linear", "factor": math.NaN()}, wantScaled: 8192, wantApplied: false},
		{name: "Inf factor", maxPosEmb: 8192, modelType: "", ropeScaling: map[string]any{"type": "linear", "factor": math.Inf(1)}, wantScaled: 8192, wantApplied: false},

		// Overflow guards
		{name: "overflow guard fires", maxPosEmb: math.MaxInt / 2, modelType: "", ropeScaling: map[string]any{"type": "linear", "factor": 4.0}, wantScaled: math.MaxInt / 2, wantApplied: false},
		{name: "yarn orig overflow", maxPosEmb: 4096, modelType: "", ropeScaling: map[string]any{"type": "yarn", "factor": 2.0, "original_max_position_embeddings": float64(math.MaxInt)}, wantScaled: 4096, wantApplied: false},

		// Degenerate base guards (R3)
		{name: "maxPosEmb zero", maxPosEmb: 0, modelType: "", ropeScaling: map[string]any{"type": "linear", "factor": 4.0}, wantScaled: 0, wantApplied: false},
		{name: "maxPosEmb negative", maxPosEmb: -1, modelType: "", ropeScaling: map[string]any{"type": "linear", "factor": 4.0}, wantScaled: -1, wantApplied: false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			scaled, applied := applyRopeScaling(tc.maxPosEmb, tc.modelType, tc.ropeScaling)
			assert.Equal(t, tc.wantScaled, scaled, "scaled value")
			assert.Equal(t, tc.wantApplied, applied, "applied flag")
		})
	}
}

func TestConvertCmd_NoCSVTraceSubcommand(t *testing.T) {
	// GIVEN the convert cobra command
	// WHEN listing its subcommands
	for _, sub := range convertCmd.Commands() {
		if sub.Name() == "csv-trace" {
			// THEN csv-trace must not be present
			t.Error("csv-trace subcommand should not exist after removal")
			return
		}
	}
}

// Regression: yarn with original uses original as base, not maxPosEmb
func TestApplyRopeScaling_YarnOriginal_UsesOriginalAsBase(t *testing.T) {
	scaled, applied := applyRopeScaling(8192, "", map[string]any{
		"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 2048.0,
	})
	// 2048 * 4 = 8192, NOT 8192 * 4 = 32768
	assert.Equal(t, 8192, scaled)
	assert.True(t, applied)
}

func TestRunCmd_NoWorkloadTracesFlag(t *testing.T) {
	// GIVEN the run cobra command
	// WHEN looking up the --workload-traces-filepath flag
	f := runCmd.Flags().Lookup("workload-traces-filepath")
	// THEN the flag does not exist
	if f != nil {
		t.Error("--workload-traces-filepath flag should not exist after removal")
	}
}

func TestRunCmd_WorkloadFlagDescriptionExcludesTraces(t *testing.T) {
	// GIVEN the run cobra command
	// WHEN inspecting the --workload flag description
	f := runCmd.Flags().Lookup("workload")
	if f == nil {
		t.Fatal("--workload flag must exist")
	}
	// THEN "traces" is not in the usage string
	if strings.Contains(f.Usage, "traces") {
		t.Errorf("--workload flag description must not contain 'traces', got: %q", f.Usage)
	}
}

func TestRunCmd_PDDirectDecodeThreshold_FlagRegistered(t *testing.T) {
	// GIVEN the run cobra command
	// WHEN looking up the --pd-direct-decode-threshold flag
	flag := runCmd.Flags().Lookup("pd-direct-decode-threshold")
	// THEN the flag is registered with the correct default
	assert.NotNil(t, flag, "pd-direct-decode-threshold flag must be registered")
	defVal, err := strconv.Atoi(flag.DefValue)
	assert.NoError(t, err, "default value must be a valid integer")
	assert.Equal(t, 256, defVal, "default pd-direct-decode-threshold must be 256")
}

func TestRunCmd_PDTransferContention_FlagRegistered(t *testing.T) {
	// GIVEN the run cobra command
	// WHEN looking up the --pd-transfer-contention flag
	flag := runCmd.Flags().Lookup("pd-transfer-contention")
	// THEN the flag is registered and defaults to false (off by default for backward compatibility)
	assert.NotNil(t, flag, "pd-transfer-contention flag must be registered")
	assert.Equal(t, "false", flag.DefValue, "pd-transfer-contention must default to false for backward compatibility")
}

// TestPrintPDMetrics_ContentionEnabled verifies that when contentionEnabled=true,
// printPDMetrics emits Peak Concurrent Transfers and Mean Transfer Queue Depth lines.
func TestPrintPDMetrics_ContentionEnabled(t *testing.T) {
	pd := &cluster.PDMetrics{
		DisaggregatedCount:      3,
		PeakConcurrentTransfers: 2,
		MeanTransferQueueDepth:  1.5,
		LoadImbalanceRatio:      1.0,
	}

	var buf bytes.Buffer

	// WHEN contention is enabled
	printPDMetrics(&buf, pd, true)
	out := buf.String()

	// THEN contention metrics must appear
	assert.Contains(t, out, "Peak Concurrent Transfers: 2", "Peak Concurrent Transfers must be printed when contentionEnabled=true")
	assert.Contains(t, out, "Mean Transfer Queue Depth: 1.5000", "Mean Transfer Queue Depth must be printed when contentionEnabled=true")
}

// TestPrintPDMetrics_ContentionDisabled verifies that when contentionEnabled=false,
// printPDMetrics does NOT emit contention-specific lines.
func TestPrintPDMetrics_ContentionDisabled(t *testing.T) {
	pd := &cluster.PDMetrics{
		DisaggregatedCount:      3,
		PeakConcurrentTransfers: 0,
		MeanTransferQueueDepth:  0,
		LoadImbalanceRatio:      1.0,
	}

	var buf bytes.Buffer

	// WHEN contention is disabled
	printPDMetrics(&buf, pd, false)
	out := buf.String()

	// THEN contention metrics must NOT appear
	assert.NotContains(t, out, "Peak Concurrent Transfers", "Peak Concurrent Transfers must not be printed when contentionEnabled=false")
	assert.NotContains(t, out, "Mean Transfer Queue Depth", "Mean Transfer Queue Depth must not be printed when contentionEnabled=false")
	// But the header and standard PD fields must still appear
	assert.Contains(t, out, "=== PD Metrics ===", "PD Metrics header must always appear")
	assert.Contains(t, out, "Disaggregated Requests: 3", "Disaggregated Requests must always appear")
}

// TestPrintPDMetrics_NilPD_ProducesNoOutput verifies the nil-pd guard:
// when pd is nil, printPDMetrics must return without writing any output.
func TestPrintPDMetrics_NilPD_ProducesNoOutput(t *testing.T) {
	var buf bytes.Buffer
	printPDMetrics(&buf, nil, true)
	assert.Empty(t, buf.String(), "printPDMetrics with nil pd must produce no output")
}

// TestRunCmdDistributionDefaults_NoHardcodedLiterals verifies that none of the distDefaults
// constant values appear as hardcoded literals in root.go's distribution flag IntVar calls
// (BC-2: single source of truth).
//
// Companion to TestObserveDistributionDefaults_NoHardcodedLiterals in observe_cmd_test.go.
// Together they ensure neither command can silently bypass the shared constants.
func TestRunCmdDistributionDefaults_NoHardcodedLiterals(t *testing.T) {
	data, err := os.ReadFile("root.go")
	if err != nil {
		t.Fatalf("cannot read root.go: %v", err)
	}
	content := string(data)

	// These patterns are the constant values that must not appear as inline literals
	// in the distribution flag IntVar calls.
	// If someone writes IntVar(&promptTokensMean, "prompt-tokens", 512, ...) instead
	// of IntVar(&promptTokensMean, "prompt-tokens", defaultPromptMean, ...), this fails.
	forbidden := []string{
		`"prompt-tokens", 512`,
		`"prompt-tokens-stdev", 256`,
		`"prompt-tokens-min", 2`,
		`"prompt-tokens-max", 7000`,
		`"output-tokens", 512`,
		`"output-tokens-stdev", 256`,
		`"output-tokens-min", 2`,
		`"output-tokens-max", 7000`,
	}
	for _, pattern := range forbidden {
		if strings.Contains(content, pattern) {
			t.Errorf("hardcoded literal found in root.go: %q\n"+
				"Use the distDefaults constants instead (BC-2).", pattern)
		}
	}
}

// TestRunCmdNumRequestsDefault_Is100 verifies that runCmd's --num-requests defaults to 100.
// This value is referenced in observe_cmd.go's --num-requests help text ("differs from blis run
// default of 100"). If this default ever changes, the help text must be updated too.
func TestRunCmdNumRequestsDefault_Is100(t *testing.T) {
	f := runCmd.Flags().Lookup("num-requests")
	if f == nil {
		t.Fatal("flag --num-requests not found on runCmd")
	}
	if f.DefValue != "100" {
		t.Errorf("--num-requests default: got %q, want \"100\" (referenced in observe --help text)", f.DefValue)
	}
}

// TestRunCmdDistributionDefaults_UseSharedConstants verifies that runCmd's eight distribution
// flag defaults equal the package-level constants (BC-1, BC-2: single source of truth).
//
// What this test catches: if someone changes a constant value, both commands change
// together and the test still passes. If someone bypasses the constants with a different
// hardcoded literal, the test fails.
func TestRunCmdDistributionDefaults_UseSharedConstants(t *testing.T) {
	tests := []struct {
		flag string
		want int
	}{
		{"prompt-tokens", defaultPromptMean},
		{"prompt-tokens-stdev", defaultPromptStdev},
		{"prompt-tokens-min", defaultPromptMin},
		{"prompt-tokens-max", defaultPromptMax},
		{"output-tokens", defaultOutputMean},
		{"output-tokens-stdev", defaultOutputStdev},
		{"output-tokens-min", defaultOutputMin},
		{"output-tokens-max", defaultOutputMax},
	}
	for _, tt := range tests {
		f := runCmd.Flags().Lookup(tt.flag)
		if f == nil {
			t.Fatalf("flag --%s not found on runCmd", tt.flag)
		}
		got, err := strconv.Atoi(f.DefValue)
		if err != nil {
			t.Fatalf("--%s DefValue %q is not an int: %v", tt.flag, f.DefValue, err)
		}
		if got != tt.want {
			t.Errorf("--%s default: got %d, want %d", tt.flag, got, tt.want)
		}
	}
}
