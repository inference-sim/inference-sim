package cmd

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/latency"
	"github.com/inference-sim/inference-sim/sim/workload"
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

// TestValidateDistributionParams verifies the behavioral contract of the extracted
// distribution-parameter validation helper (R3, R14).
//
// Contract: validateDistributionParams returns an empty string for valid inputs and
// a non-empty string describing the violation for any invalid input.
//
// GIVEN distribution token parameters
// WHEN validateDistributionParams is called
// THEN it returns empty string iff all parameters satisfy the bounds invariants
func TestValidateDistributionParams(t *testing.T) {
	// valid baseline — all defaults from the shared constants
	const (
		validMin    = defaultPromptMin
		validMax    = defaultPromptMax
		validMean   = defaultPromptMean
		validStdev  = defaultPromptStdev
		validOMin   = defaultOutputMin
		validOMax   = defaultOutputMax
		validOMean  = defaultOutputMean
		validOStdev = defaultOutputStdev
	)

	tests := []struct {
		name      string
		promptMin int
		promptMax int
		outputMin int
		outputMax int
		promptStdev int
		outputStdev int
		promptMean  int
		outputMean  int
		wantErr     bool
	}{
		{
			name:        "valid defaults produce no error",
			promptMin: validMin, promptMax: validMax,
			outputMin: validOMin, outputMax: validOMax,
			promptStdev: validStdev, outputStdev: validOStdev,
			promptMean: validMean, outputMean: validOMean,
			wantErr: false,
		},
		{
			name:        "prompt-tokens-min zero is rejected",
			promptMin: 0, promptMax: validMax,
			outputMin: validOMin, outputMax: validOMax,
			promptStdev: validStdev, outputStdev: validOStdev,
			promptMean: validMean, outputMean: validOMean,
			wantErr: true,
		},
		{
			name:        "prompt-tokens-min negative is rejected",
			promptMin: -5, promptMax: validMax,
			outputMin: validOMin, outputMax: validOMax,
			promptStdev: validStdev, outputStdev: validOStdev,
			promptMean: validMean, outputMean: validOMean,
			wantErr: true,
		},
		{
			name:        "prompt-tokens-max zero is rejected",
			promptMin: validMin, promptMax: 0,
			outputMin: validOMin, outputMax: validOMax,
			promptStdev: validStdev, outputStdev: validOStdev,
			promptMean: validMean, outputMean: validOMean,
			wantErr: true,
		},
		{
			name:        "output-tokens-min zero is rejected",
			promptMin: validMin, promptMax: validMax,
			outputMin: 0, outputMax: validOMax,
			promptStdev: validStdev, outputStdev: validOStdev,
			promptMean: validMean, outputMean: validOMean,
			wantErr: true,
		},
		{
			name:        "output-tokens-max zero is rejected",
			promptMin: validMin, promptMax: validMax,
			outputMin: validOMin, outputMax: 0,
			promptStdev: validStdev, outputStdev: validOStdev,
			promptMean: validMean, outputMean: validOMean,
			wantErr: true,
		},
		{
			name:        "negative prompt stdev is rejected",
			promptMin: validMin, promptMax: validMax,
			outputMin: validOMin, outputMax: validOMax,
			promptStdev: -1, outputStdev: validOStdev,
			promptMean: validMean, outputMean: validOMean,
			wantErr: true,
		},
		{
			name:        "negative output stdev is rejected",
			promptMin: validMin, promptMax: validMax,
			outputMin: validOMin, outputMax: validOMax,
			promptStdev: validStdev, outputStdev: -1,
			promptMean: validMean, outputMean: validOMean,
			wantErr: true,
		},
		{
			name:        "prompt min greater than max is rejected",
			promptMin: 500, promptMax: 100,
			outputMin: validOMin, outputMax: validOMax,
			promptStdev: 50, outputStdev: validOStdev,
			promptMean: 100, outputMean: validOMean,
			wantErr: true,
		},
		{
			name:        "output min greater than max is rejected",
			promptMin: validMin, promptMax: validMax,
			outputMin: 500, outputMax: 100,
			promptStdev: validStdev, outputStdev: 50,
			promptMean: validMean, outputMean: 100,
			wantErr: true,
		},
		{
			name:        "prompt mean above max is rejected",
			promptMin: 10, promptMax: 100,
			outputMin: validOMin, outputMax: validOMax,
			promptStdev: 10, outputStdev: validOStdev,
			promptMean: 200, outputMean: validOMean,
			wantErr: true,
		},
		{
			name:        "output mean below min is rejected",
			promptMin: validMin, promptMax: validMax,
			outputMin: 100, outputMax: 500,
			promptStdev: validStdev, outputStdev: 50,
			promptMean: validMean, outputMean: 50,
			wantErr: true,
		},
		// stdev=0 is a valid deterministic distribution; lower-bound check must be skipped
		{
			name:        "prompt stdev 0 with min 1 is accepted",
			promptMin: 1, promptMax: validMax,
			outputMin: validOMin, outputMax: validOMax,
			promptStdev: 0, outputStdev: validOStdev,
			promptMean: validMean, outputMean: validOMean,
			wantErr: false,
		},
		{
			name:        "output stdev 0 with min 1 is accepted",
			promptMin: validMin, promptMax: validMax,
			outputMin: 1, outputMax: validOMax,
			promptStdev: validStdev, outputStdev: 0,
			promptMean: validMean, outputMean: validOMean,
			wantErr: false,
		},
		{
			name:        "both stddevs 0 with large mins are accepted",
			promptMin: 100, promptMax: 1024,
			outputMin: 50, outputMax: 512,
			promptStdev: 0, outputStdev: 0,
			promptMean: 512, outputMean: 128,
			wantErr: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := validateDistributionParams(
				tc.promptMin, tc.promptMax,
				tc.outputMin, tc.outputMax,
				tc.promptStdev, tc.outputStdev,
				tc.promptMean, tc.outputMean,
			)
			if tc.wantErr && got == "" {
				t.Errorf("expected a non-empty error string, got empty")
			}
			if !tc.wantErr && got != "" {
				t.Errorf("expected no error, got %q", got)
			}
		})
	}
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

// TestAutoCalcKVBlocks_RespectsBlockSizeAndGPUMemUtil verifies the behavioral contract
// from issue #1035: when --total-kv-blocks is NOT provided, auto-calculation MUST
// respect --block-size and --gpu-memory-utilization flags.
//
// GIVEN: Different block-size or gpu-memory-utilization values
// WHEN: KV blocks are auto-calculated (--total-kv-blocks NOT set)
// THEN: The calculated block counts MUST differ accordingly
func TestAutoCalcKVBlocks_RespectsBlockSizeAndGPUMemUtil(t *testing.T) {
	// Representative model config (similar to Llama-3-8B)
	mc := sim.ModelConfig{
		NumLayers:       32,
		NumHeads:        32,
		NumKVHeads:      8,  // GQA
		HiddenDim:       4096,
		IntermediateDim: 14336,
		VocabSize:       128256,
		BytesPerParam:   2.0, // FP16
	}

	// Representative GPU hardware config (H100-like)
	hc := sim.HardwareCalib{
		MemoryGiB:  80.0,
		TFlopsPeak: 1000.0,
		BwPeakTBs:  3.35,
	}

	// KV capacity params for dense SwiGLU model
	params := latency.NewKVCapacityParams(
		false, // isMoE
		0,     // numLocalExperts
		false, // tieWordEmbeddings
		"silu", // hiddenAct
		0,     // moeExpertFFNDim
		0,     // sharedExpertFFNDim
	)

	const tp = 1

	// Test BC-1: Different block-size values produce different block counts
	blocks64, err := latency.CalculateKVBlocks(mc, hc, tp, 64, 0.9, params)
	if err != nil {
		t.Fatalf("CalculateKVBlocks with block-size=64: %v", err)
	}

	blocks128, err := latency.CalculateKVBlocks(mc, hc, tp, 128, 0.9, params)
	if err != nil {
		t.Fatalf("CalculateKVBlocks with block-size=128: %v", err)
	}

	// Larger block size → fewer blocks fit in memory
	if blocks64 <= blocks128 {
		t.Errorf("block-size sensitivity: blocks64=%d must be > blocks128=%d (larger blocks → fewer fit)",
			blocks64, blocks128)
	}

	// Test BC-2: Different gpu-memory-utilization values produce different block counts
	blocks85pct, err := latency.CalculateKVBlocks(mc, hc, tp, 64, 0.85, params)
	if err != nil {
		t.Fatalf("CalculateKVBlocks with gpu-util=0.85: %v", err)
	}

	blocks95pct, err := latency.CalculateKVBlocks(mc, hc, tp, 64, 0.95, params)
	if err != nil {
		t.Fatalf("CalculateKVBlocks with gpu-util=0.95: %v", err)
	}

	// Higher utilization → more blocks fit in memory
	if blocks95pct <= blocks85pct {
		t.Errorf("gpu-memory-utilization sensitivity: blocks95pct=%d must be > blocks85pct=%d (more memory → more blocks)",
			blocks95pct, blocks85pct)
	}
}

// TestRunCmd_HasMetricsPathFlag verifies BC-1: blis run exposes --metrics-path,
// not --results-path.
func TestRunCmd_HasMetricsPathFlag(t *testing.T) {
	if runCmd.Flags().Lookup("metrics-path") == nil {
		t.Error("BC-1: runCmd missing --metrics-path flag")
	}
	if runCmd.Flags().Lookup("results-path") != nil {
		t.Error("BC-1: runCmd must NOT have --results-path flag (schema footgun)")
	}
}

// TestReplayCmd_HasResultsPathFlag verifies BC-2: blis replay exposes --results-path,
// not --metrics-path.
func TestReplayCmd_HasResultsPathFlag(t *testing.T) {
	if replayCmd.Flags().Lookup("results-path") == nil {
		t.Error("BC-2: replayCmd missing --results-path flag")
	}
	if replayCmd.Flags().Lookup("metrics-path") != nil {
		t.Error("BC-2: replayCmd must NOT have --metrics-path flag")
	}
}

// TestRunCmd_TimeoutFlag_RegisteredWithDefault300s verifies that --timeout is registered
// on runCmd with a default of 300s. Default 300s matches the session-client default in
// computeDeadline (DefaultTimeoutUs), preserving termination for UnlimitedRounds sessions
// and providing consistent behavior for synthesized workloads (#1127).
func TestRunCmd_TimeoutFlag_RegisteredWithDefault300s(t *testing.T) {
	f := runCmd.Flags().Lookup("timeout")
	if f == nil {
		t.Fatal("flag --timeout not found on runCmd")
	}
	if f.DefValue != "300" {
		t.Errorf("--timeout default: got %q, want \"300\" (matches 300s session-client default in computeDeadline)", f.DefValue)
	}
}

// TestRunCmd_TimeoutFlag_NotOnReplay verifies that --timeout is NOT registered on replayCmd.
// blis replay replays a pre-captured trace and has no HTTP client; timeouts for replay
// are expressed via the workload spec's deadline_us field (#1127).
func TestRunCmd_TimeoutFlag_NotOnReplay(t *testing.T) {
	if replayCmd.Flags().Lookup("timeout") != nil {
		t.Error("replayCmd must NOT have --timeout flag; replay has no HTTP client and uses workload spec deadlines")
	}
}

// TestApplyTimeoutToSpec_SynthesizedSpec verifies the core behavioral contract of
// applyTimeoutToSpec: all ClientSpec.Timeout fields are set to the specified value,
// and a zero value sets an explicit no-timeout pointer (not nil).
//
// GIVEN a synthesized spec with clients having nil Timeout
// WHEN applyTimeoutToSpec is called with a non-zero timeout
// THEN all clients receive a Timeout pointer with the converted microsecond value
func TestApplyTimeoutToSpec_SynthesizedSpec(t *testing.T) {
	spec := buildSynthesizedSpec()

	applyTimeoutToSpec(spec, 60)

	for i, c := range spec.Clients {
		if c.Timeout == nil {
			t.Errorf("client[%d] Timeout is nil after applyTimeoutToSpec; want non-nil", i)
			continue
		}
		want := int64(60_000_000)
		if *c.Timeout != want {
			t.Errorf("client[%d] Timeout = %d, want %d (60s in µs)", i, *c.Timeout, want)
		}
	}
}

// TestApplyTimeoutToSpec_NonPositiveMeansDisabled verifies that timeout<=0 results in an
// explicit no-timeout pointer (*int64 pointing to 0), not nil. Both 0 and negative values
// disable the deadline. The CLI rejects 0 before calling this function (matching blis observe),
// but the function itself treats all non-positive values identically: explicit *0 disables
// the deadline and suppresses the 300s session default in computeDeadline (#1127).
//
// GIVEN a synthesized spec
// WHEN applyTimeoutToSpec is called with timeout=0 or timeout=-1
// THEN all clients have a non-nil Timeout pointer pointing to 0
func TestApplyTimeoutToSpec_NonPositiveMeansDisabled(t *testing.T) {
	for _, secs := range []int{0, -1, -300} {
		spec := buildSynthesizedSpec()
		applyTimeoutToSpec(spec, secs)
		for i, c := range spec.Clients {
			if c.Timeout == nil {
				t.Errorf("secs=%d client[%d] Timeout is nil; want explicit *0 to suppress 300s default", secs, i)
				continue
			}
			if *c.Timeout != 0 {
				t.Errorf("secs=%d client[%d] Timeout = %d, want 0 (disabled)", secs, i, *c.Timeout)
			}
		}
	}
}

// TestApplyTimeoutToSpec_CohortSpec verifies that applyTimeoutToSpec also sets
// Timeout on CohortSpec entries (cohorts are expanded to clients during generation).
func TestApplyTimeoutToSpec_CohortSpec(t *testing.T) {
	spec := buildSynthesizedSpec()
	// Add a cohort to simulate a spec with cohort-based clients.
	spec.Cohorts = append(spec.Cohorts, buildTestCohort())

	applyTimeoutToSpec(spec, 120)

	wantUs := int64(120_000_000)
	for i, coh := range spec.Cohorts {
		if coh.Timeout == nil {
			t.Errorf("cohort[%d] Timeout is nil; want non-nil", i)
			continue
		}
		if *coh.Timeout != wantUs {
			t.Errorf("cohort[%d] Timeout = %d, want %d", i, *coh.Timeout, wantUs)
		}
	}
}

// TestApplyTimeoutToSpec_NegativeMeansDisabled verifies that a negative timeoutSecs (the
// way to disable the timeout via --timeout -1) produces an explicit *int64(0), not a
// negative microsecond value. A negative µs deadline would violate INV-5 (causality:
// deadline < arrival). Using negative as the "disabled" sentinel while mapping to *0
// ensures backward-compatible behavior with computeDeadline (#1127).
//
// GIVEN a timeout of -1 seconds
// WHEN applyTimeoutToSpec is called
// THEN all clients have a non-nil Timeout pointer pointing to 0 (not negative µs)
func TestApplyTimeoutToSpec_NegativeMeansDisabled(t *testing.T) {
	spec := buildSynthesizedSpec()

	applyTimeoutToSpec(spec, -1)

	for i, c := range spec.Clients {
		if c.Timeout == nil {
			t.Errorf("client[%d] Timeout is nil; want explicit *0", i)
			continue
		}
		if *c.Timeout != 0 {
			t.Errorf("client[%d] Timeout = %d; want 0 (negative input maps to disabled, not negative µs)", i, *c.Timeout)
		}
	}
}

// TestApplyTimeoutToSpec_NotCalledForSpecFile verifies the dispatch guard:
// when a workload spec is loaded from a file and --timeout is not explicitly set,
// applyTimeoutToSpec must NOT be called (client-defined timeouts in the spec are preserved).
//
// GIVEN the package-level workloadSpecPath is non-empty and --timeout was not explicitly set
// WHEN the real guard condition (workloadSpecPath == "" || cmd.Flags().Changed("timeout"))
//      is evaluated
// THEN the client Timeout is unchanged (applyTimeoutToSpec was not called)
func TestApplyTimeoutToSpec_NotCalledForSpecFile(t *testing.T) {
	origPath := workloadSpecPath
	defer func() { workloadSpecPath = origPath }()
	workloadSpecPath = "/path/to/workload.yaml"

	spec := buildSynthesizedSpec()
	want := int64(120_000_000) // 120s, already set in the spec
	for i := range spec.Clients {
		v := want
		spec.Clients[i].Timeout = &v
	}

	// Exercise the actual guard condition — mirrors runCmd dispatch path.
	// workloadSpecPath != "" and --timeout not set → guard is false → skip.
	if workloadSpecPath == "" || runCmd.Flags().Changed("timeout") {
		applyTimeoutToSpec(spec, requestTimeoutSecs)
	}

	for i, c := range spec.Clients {
		if c.Timeout == nil || *c.Timeout != want {
			t.Errorf("client[%d] Timeout = %v; want %d (spec timeout preserved)", i, c.Timeout, want)
		}
	}
}

// TestApplyTimeoutToRequests_ZeroSetsDeadlineZero verifies that applyTimeoutToRequests
// with timeoutSecs=0 sets Deadline=0 on all requests regardless of ArrivalTime.
// Zero disables the deadline; a nil Timeout for session clients would trigger the 300s
// default in computeDeadline, so the explicit-zero post-pass is required (#1127).
func TestApplyTimeoutToRequests_ZeroSetsDeadlineZero(t *testing.T) {
	wl := &workload.GeneratedWorkload{
		Requests: []*sim.Request{
			{ID: "r0", ArrivalTime: 0, Deadline: 300_000_000},
			{ID: "r1", ArrivalTime: 1_000_000, Deadline: 301_000_000},
		},
	}

	applyTimeoutToRequests(wl, 0)

	for i, req := range wl.Requests {
		if req.Deadline != 0 {
			t.Errorf("request[%d] Deadline = %d; want 0 (no timeout)", i, req.Deadline)
		}
	}
}

// TestApplyTimeoutToRequests_NonZeroSetsDeadlineFromArrival verifies that a positive
// timeout sets Deadline = ArrivalTime + timeoutUs on each request.
func TestApplyTimeoutToRequests_NonZeroSetsDeadlineFromArrival(t *testing.T) {
	wl := &workload.GeneratedWorkload{
		Requests: []*sim.Request{
			{ID: "r0", ArrivalTime: 0},
			{ID: "r1", ArrivalTime: 500_000},
		},
	}

	applyTimeoutToRequests(wl, 60)

	for i, req := range wl.Requests {
		want := req.ArrivalTime + 60_000_000
		if req.Deadline != want {
			t.Errorf("request[%d] Deadline = %d; want %d (ArrivalTime + 60s)", i, req.Deadline, want)
		}
	}
}

// TestApplyTimeoutToRequests_ZeroSetsSessionBlueprintExplicitZero verifies that
// applyTimeoutToRequests with timeoutSecs=0 sets session blueprint Timeout to an
// explicit *int64(0), not nil. nil would trigger the 300s default for follow-up rounds.
func TestApplyTimeoutToRequests_ZeroSetsSessionBlueprintExplicitZero(t *testing.T) {
	wl := &workload.GeneratedWorkload{
		Sessions: []workload.SessionBlueprint{
			{SessionID: "s0"},
			{SessionID: "s1"},
		},
	}

	applyTimeoutToRequests(wl, 0)

	for i, bp := range wl.Sessions {
		if bp.Timeout == nil {
			t.Errorf("session[%d] Timeout is nil; want explicit *0 to suppress 300s default", i)
			continue
		}
		if *bp.Timeout != 0 {
			t.Errorf("session[%d] Timeout = %d; want 0", i, *bp.Timeout)
		}
	}
}

// TestApplyTimeoutToRequests_NonZeroSetsSessionBlueprintTimeout verifies that a positive
// timeout is propagated to session blueprints so follow-up round deadlines match.
func TestApplyTimeoutToRequests_NonZeroSetsSessionBlueprintTimeout(t *testing.T) {
	wl := &workload.GeneratedWorkload{
		Sessions: []workload.SessionBlueprint{
			{SessionID: "s0"},
		},
	}

	applyTimeoutToRequests(wl, 120)

	for i, bp := range wl.Sessions {
		if bp.Timeout == nil {
			t.Errorf("session[%d] Timeout is nil; want 120s in µs", i)
			continue
		}
		want := int64(120_000_000)
		if *bp.Timeout != want {
			t.Errorf("session[%d] Timeout = %d; want %d (120s in µs)", i, *bp.Timeout, want)
		}
	}
}

// TestApplyTimeoutToRequests_NegativeSetsDeadlineZero verifies that a negative timeoutSecs
// (the "disabled" sentinel, e.g. default -1) sets req.Deadline=0 on all requests.
// Negative must not produce a negative deadline (INV-5 violation: deadline < arrival).
func TestApplyTimeoutToRequests_NegativeSetsDeadlineZero(t *testing.T) {
	wl := &workload.GeneratedWorkload{
		Requests: []*sim.Request{
			{ID: "r0", ArrivalTime: 0, Deadline: 300_000_000},
			{ID: "r1", ArrivalTime: 1_000_000, Deadline: 301_000_000},
		},
	}

	applyTimeoutToRequests(wl, -1)

	for i, req := range wl.Requests {
		if req.Deadline != 0 {
			t.Errorf("request[%d] Deadline = %d; want 0 (negative input means disabled, not negative deadline)", i, req.Deadline)
		}
	}
}

// TestApplyTimeoutToRequests_NegativeSetsSessionBlueprintExplicitZero verifies that a
// negative timeoutSecs sets session blueprint Timeout to *int64(0), not nil and not negative.
// nil would trigger the 300s default for follow-up rounds; negative would violate INV-5.
func TestApplyTimeoutToRequests_NegativeSetsSessionBlueprintExplicitZero(t *testing.T) {
	wl := &workload.GeneratedWorkload{
		Sessions: []workload.SessionBlueprint{
			{SessionID: "s0"},
		},
	}

	applyTimeoutToRequests(wl, -1)

	for i, bp := range wl.Sessions {
		if bp.Timeout == nil {
			t.Errorf("session[%d] Timeout is nil; want explicit *0 (negative = disabled, not nil)", i)
			continue
		}
		if *bp.Timeout != 0 {
			t.Errorf("session[%d] Timeout = %d; want 0 (negative maps to disabled)", i, *bp.Timeout)
		}
	}
}

// TestTryAutoCalcKVBlocksBlackbox_ErrorPaths tests all error paths in
// tryAutoCalcKVBlocksBlackbox. Each error path should return (0, false) and
// emit a warning (not tested here — warnings go to logrus stderr).
func TestTryAutoCalcKVBlocksBlackbox_ErrorPaths(t *testing.T) {
	// Create temporary test fixtures
	tmpDir := t.TempDir()

	// Valid hardware config file (JSON format)
	hwConfigPath := filepath.Join(tmpDir, "hw.json")
	hwJSON := `{
  "H100": {
    "MemoryGiB": 80.0,
    "TFlopsPeak": 1000.0,
    "BwPeakTBs": 3.35
  }
}`
	if err := os.WriteFile(hwConfigPath, []byte(hwJSON), 0644); err != nil {
		t.Fatalf("write hw config: %v", err)
	}

	// Valid model config.json file (minimal Llama-like)
	validConfigJSON := `{
  "architectures": ["LlamaForCausalLM"],
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "vocab_size": 128256,
  "torch_dtype": "float16",
  "hidden_act": "silu"
}`

	validModelDir := filepath.Join(tmpDir, "valid-model")
	if err := os.MkdirAll(validModelDir, 0755); err != nil {
		t.Fatalf("create valid model dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(validModelDir, "config.json"), []byte(validConfigJSON), 0644); err != nil {
		t.Fatalf("write valid config.json: %v", err)
	}

	// Invalid model config.json (malformed JSON)
	invalidModelDir := filepath.Join(tmpDir, "invalid-model")
	if err := os.MkdirAll(invalidModelDir, 0755); err != nil {
		t.Fatalf("create invalid model dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(invalidModelDir, "config.json"), []byte("{invalid json}"), 0644); err != nil {
		t.Fatalf("write invalid config.json: %v", err)
	}

	// Incomplete model config.json (missing required fields)
	incompleteModelDir := filepath.Join(tmpDir, "incomplete-model")
	if err := os.MkdirAll(incompleteModelDir, 0755); err != nil {
		t.Fatalf("create incomplete model dir: %v", err)
	}
	incompleteJSON := `{"architectures": ["LlamaForCausalLM"]}`
	if err := os.WriteFile(filepath.Join(incompleteModelDir, "config.json"), []byte(incompleteJSON), 0644); err != nil {
		t.Fatalf("write incomplete config.json: %v", err)
	}

	// MoE model without num_local_experts (triggers ExtractKVCapacityParams error)
	moeModelDir := filepath.Join(tmpDir, "moe-model")
	if err := os.MkdirAll(moeModelDir, 0755); err != nil {
		t.Fatalf("create moe model dir: %v", err)
	}
	moeJSON := `{
  "architectures": ["MixtralForCausalLM"],
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "vocab_size": 128256,
  "torch_dtype": "float16",
  "hidden_act": "silu",
  "num_experts_per_tok": 2
}`
	if err := os.WriteFile(filepath.Join(moeModelDir, "config.json"), []byte(moeJSON), 0644); err != nil {
		t.Fatalf("write moe config.json: %v", err)
	}

	// Hardware config with zero memory (JSON format)
	zeroMemHWPath := filepath.Join(tmpDir, "zero-mem-hw.json")
	zeroMemJSON := `{
  "H100": {
    "MemoryGiB": 0,
    "TFlopsPeak": 1000.0,
    "BwPeakTBs": 3.35
  }
}`
	if err := os.WriteFile(zeroMemHWPath, []byte(zeroMemJSON), 0644); err != nil {
		t.Fatalf("write zero-mem hw config: %v", err)
	}

	// Empty defaults.yaml for tests that don't need it
	emptyDefaults := filepath.Join(tmpDir, "defaults.yaml")
	if err := os.WriteFile(emptyDefaults, []byte("version: \"1.0\"\n"), 0644); err != nil {
		t.Fatalf("write empty defaults: %v", err)
	}

	tests := []struct {
		name                string
		modelConfigFolder   string
		defaultsFilePath    string
		hwConfigPath        string
		gpu                 string
		expectSuccess       bool
		description         string
	}{
		{
			name:              "nonexistent model path",
			modelConfigFolder: filepath.Join(tmpDir, "nonexistent"),
			defaultsFilePath:  emptyDefaults,
			hwConfigPath:      hwConfigPath,
			gpu:               "H100",
			expectSuccess:     false,
			description:       "resolveModelConfig fails when path doesn't exist",
		},
		{
			name:              "malformed config.json",
			modelConfigFolder: invalidModelDir,
			defaultsFilePath:  emptyDefaults,
			hwConfigPath:      hwConfigPath,
			gpu:               "H100",
			expectSuccess:     false,
			description:       "ParseHFConfig fails on invalid JSON",
		},
		{
			name:              "incomplete config.json",
			modelConfigFolder: incompleteModelDir,
			defaultsFilePath:  emptyDefaults,
			hwConfigPath:      hwConfigPath,
			gpu:               "H100",
			expectSuccess:     false,
			description:       "GetModelConfigFromHF fails when required fields missing",
		},
		{
			name:              "missing hardware config",
			modelConfigFolder: validModelDir,
			defaultsFilePath:  emptyDefaults,
			hwConfigPath:      filepath.Join(tmpDir, "nonexistent-hw.json"),
			gpu:               "H100",
			expectSuccess:     false,
			description:       "resolveHardwareConfig fails when file doesn't exist",
		},
		{
			name:              "unknown GPU type",
			modelConfigFolder: validModelDir,
			defaultsFilePath:  emptyDefaults,
			hwConfigPath:      hwConfigPath,
			gpu:               "UnknownGPU",
			expectSuccess:     false,
			description:       "GetHWConfig fails when GPU not in hardware config",
		},
		{
			name:              "zero GPU memory",
			modelConfigFolder: validModelDir,
			defaultsFilePath:  emptyDefaults,
			hwConfigPath:      zeroMemHWPath,
			gpu:               "H100",
			expectSuccess:     false,
			description:       "MemoryGiB <= 0 check fails",
		},
		{
			name:              "MoE without num_local_experts",
			modelConfigFolder: moeModelDir,
			defaultsFilePath:  emptyDefaults,
			hwConfigPath:      hwConfigPath,
			gpu:               "H100",
			expectSuccess:     false,
			description:       "ExtractKVCapacityParams fails for MoE with num_experts_per_tok but no total expert count",
		},
		{
			name:              "happy path",
			modelConfigFolder: validModelDir,
			defaultsFilePath:  emptyDefaults,
			hwConfigPath:      hwConfigPath,
			gpu:               "H100",
			expectSuccess:     true,
			description:       "All steps succeed, returns calculated blocks",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			blocks, ok := tryAutoCalcKVBlocksBlackbox(
				"test-model",
				tc.modelConfigFolder,
				tc.defaultsFilePath,
				tc.hwConfigPath,
				tc.gpu,
				1,    // tp
				16,   // blockSize
				0.9,  // gpuMemUtil
				1000, // currentBlocks (fallback value)
			)

			if tc.expectSuccess {
				if !ok {
					t.Errorf("%s: expected success but got failure", tc.description)
				}
				if blocks <= 0 {
					t.Errorf("%s: expected positive block count, got %d", tc.description, blocks)
				}
			} else {
				if ok {
					t.Errorf("%s: expected failure but got success with blocks=%d", tc.description, blocks)
				}
				if blocks != 0 {
					t.Errorf("%s: expected blocks=0 on failure, got %d", tc.description, blocks)
				}
			}
		})
	}
}

// TestAutoCalcKVBlocks_SuppressedByExplicitFlag verifies that when --total-kv-blocks
// is explicitly set by the user, auto-calculation is suppressed (guard condition).
//
// GIVEN: A command with --total-kv-blocks explicitly set
// WHEN: We check if auto-calculation should run
// THEN: cmd.Flags().Changed("total-kv-blocks") returns true, suppressing auto-calc
func TestAutoCalcKVBlocks_SuppressedByExplicitFlag(t *testing.T) {
	// Create a cobra command with the total-kv-blocks flag
	testCmd := &cobra.Command{}
	var totalKV int64
	testCmd.Flags().Int64Var(&totalKV, "total-kv-blocks", 1000000, "")

	// Test case 1: Flag NOT set by user (default value)
	if err := testCmd.ParseFlags([]string{}); err != nil {
		t.Fatalf("ParseFlags with no args: %v", err)
	}
	if testCmd.Flags().Changed("total-kv-blocks") {
		t.Errorf("Flag not set by user: Changed() should be false (auto-calc allowed)")
	}

	// Test case 2: Flag explicitly set by user
	testCmd2 := &cobra.Command{}
	var totalKV2 int64
	testCmd2.Flags().Int64Var(&totalKV2, "total-kv-blocks", 1000000, "")
	if err := testCmd2.ParseFlags([]string{"--total-kv-blocks", "5000"}); err != nil {
		t.Fatalf("ParseFlags with --total-kv-blocks: %v", err)
	}
	if !testCmd2.Flags().Changed("total-kv-blocks") {
		t.Errorf("Flag set by user: Changed() should be true (auto-calc suppressed)")
	}
	if totalKV2 != 5000 {
		t.Errorf("Flag value: got %d, want 5000", totalKV2)
	}
}

// TestRunCmd_MetricsPath_WritesMetricsOutput verifies BC-3: --metrics-path on
// blis run produces MetricsOutput JSON (instance_id string ≠ SimResult request_id int).
// NOTE: Do NOT use t.Parallel() — mutates package-level vars.
func TestRunCmd_MetricsPath_WritesMetricsOutput(t *testing.T) {
	outFile := filepath.Join(t.TempDir(), "metrics.json")

	// Save and restore all package-level flag vars mutated by runCmd.Run.
	// Base list copied from TestReplayCmd_EndToEnd_BlackboxMode:377-434;
	// run-only workload vars and metricsPath added on top.
	origMetrics := metricsPath
	origModel := model
	origBackend := latencyModelBackend
	origBeta := betaCoeffs
	origAlpha := alphaCoeffs
	origTotalKV := totalKVBlocks
	origBlockSize := blockSizeTokens
	origMaxRunning := maxRunningReqs
	origMaxSched := maxScheduledTokens
	origInstances := numInstances
	origSeed := seed
	origResults := resultsPath
	origThreshold := longPrefillTokenThreshold
	origKVCPU := kvCPUBlocks
	origOffload := kvOffloadThreshold
	origBandwidth := kvTransferBandwidth
	origBaseLatency := kvTransferBaseLatency
	origSnapRefresh := snapshotRefreshInterval
	origAdmission := admissionPolicy
	origRouting := routingPolicy
	origPriority := priorityPolicy
	origScheduler := scheduler
	origPolicyConfig := policyConfigPath
	origMaxModelLen := maxModelLen
	origTraceLevel := traceLevel
	origCounterfactualK := counterfactualK
	origSimHorizon := simulationHorizon
	// run-only workload vars
	origWorkloadType := workloadType
	origRate := rate
	origNumReqs := numRequests
	origConcurrency := concurrency
	origThinkTime := thinkTimeMs
	origPrefix := prefixTokens
	origPromptMean := promptTokensMean
	origPromptStdev := promptTokensStdev
	origPromptMin := promptTokensMin
	origPromptMax := promptTokensMax
	origOutputMean := outputTokensMean
	origOutputStdev := outputTokensStdev
	origOutputMin := outputTokensMin
	origOutputMax := outputTokensMax
	origWorkloadSpec := workloadSpecPath
	origRequestTimeout := requestTimeoutSecs
	origTraceOut := traceOutput
	origLogLevel := logLevel
	defer func() {
		metricsPath = origMetrics
		model = origModel
		latencyModelBackend = origBackend
		betaCoeffs = origBeta
		alphaCoeffs = origAlpha
		totalKVBlocks = origTotalKV
		blockSizeTokens = origBlockSize
		maxRunningReqs = origMaxRunning
		maxScheduledTokens = origMaxSched
		numInstances = origInstances
		seed = origSeed
		resultsPath = origResults
		longPrefillTokenThreshold = origThreshold
		kvCPUBlocks = origKVCPU
		kvOffloadThreshold = origOffload
		kvTransferBandwidth = origBandwidth
		kvTransferBaseLatency = origBaseLatency
		snapshotRefreshInterval = origSnapRefresh
		admissionPolicy = origAdmission
		routingPolicy = origRouting
		priorityPolicy = origPriority
		scheduler = origScheduler
		policyConfigPath = origPolicyConfig
		maxModelLen = origMaxModelLen
		traceLevel = origTraceLevel
		counterfactualK = origCounterfactualK
		simulationHorizon = origSimHorizon
		workloadType = origWorkloadType
		rate = origRate
		numRequests = origNumReqs
		concurrency = origConcurrency
		thinkTimeMs = origThinkTime
		prefixTokens = origPrefix
		promptTokensMean = origPromptMean
		promptTokensStdev = origPromptStdev
		promptTokensMin = origPromptMin
		promptTokensMax = origPromptMax
		outputTokensMean = origOutputMean
		outputTokensStdev = origOutputStdev
		outputTokensMin = origOutputMin
		outputTokensMax = origOutputMax
		workloadSpecPath = origWorkloadSpec
		requestTimeoutSecs = origRequestTimeout
		traceOutput = origTraceOut
		logLevel = origLogLevel
	}()

	// Set required run vars — runCmd.Run has logrus.Fatalf guards on zero/invalid values.
	metricsPath = outFile
	workloadType = "distribution" // avoids preset-path Fatalf("Undefined workload")
	rate = 1.0                    // required by distribution rate-mode path
	numRequests = 1               // minimal run
	promptTokensMean = 512
	promptTokensStdev = 256
	promptTokensMin = 2
	promptTokensMax = 7000
	outputTokensMean = 512
	outputTokensStdev = 256
	outputTokensMin = 2
	outputTokensMax = 7000

	// Build testCmd with Changed() tracking so resolveLatencyConfig works.
	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	// Register run-only workload flags that runCmd.Run checks via Changed().
	testCmd.Flags().IntVar(&numRequests, "num-requests", 0, "")
	testCmd.Flags().Float64Var(&rate, "rate", 0, "")
	testCmd.Flags().StringVar(&workloadType, "workload", "", "")
	if err := testCmd.ParseFlags([]string{
		"--model", "qwen/qwen3-14b",
		"--latency-model", "blackbox", // avoids roofline HF config fetch
		"--beta-coeffs", "10000.0,1.0,1.0",
		"--alpha-coeffs", "0.0,0.0,0.0",
		"--total-kv-blocks", "1000",
		"--num-requests", "1",
		"--seed", "42",
		"--rate", "1.0",
		"--workload", "distribution",
	}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}

	runCmd.Run(testCmd, nil)

	data, err := os.ReadFile(outFile)
	if err != nil {
		t.Fatalf("BC-3: metrics file not written: %v", err)
	}
	var out sim.MetricsOutput
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("BC-3: not MetricsOutput JSON: %v\nraw: %s", err, data)
	}
	// MetricsOutput.InstanceID is "cluster"; SimResult.RequestID is an int — schemas are distinct.
	if out.InstanceID == "" {
		t.Error("BC-3: InstanceID empty — wrong schema or SaveResults not called")
	}
}

func TestRunCmd_ModelAutoscalerIntervalUs_FlagRegistered(t *testing.T) {
	flag := runCmd.Flags().Lookup("model-autoscaler-interval-us")
	assert.NotNil(t, flag, "model-autoscaler-interval-us flag must be registered")
	defVal, err := strconv.ParseFloat(flag.DefValue, 64)
	assert.NoError(t, err, "default must be a valid float64")
	assert.Equal(t, 0.0, defVal, "default must be 0 (disabled)")
}

// buildSynthesizedSpec returns a minimal WorkloadSpec as produced by SynthesizeFromDistribution,
// with two clients and no cohorts, for use in applyTimeoutToSpec tests.
func buildSynthesizedSpec() *workload.WorkloadSpec {
	return &workload.WorkloadSpec{
		Version: "2",
		Clients: []workload.ClientSpec{
			{ID: "client-0"},
			{ID: "client-1"},
		},
	}
}

// buildTestCohort returns a minimal CohortSpec with no Timeout set.
func buildTestCohort() workload.CohortSpec {
	return workload.CohortSpec{ID: "cohort-0"}
}

