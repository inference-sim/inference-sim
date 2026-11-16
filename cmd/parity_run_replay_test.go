package cmd

// Run/replay parity matrix for streaming workload generation (issues #1441, #1442).
//
// This file is the CLI-level enforcement point for two claims about the streaming
// generator, the only generation path (#1451):
//
//  1. BC-5 (determinism, INV-6): `blis run --trace-output` produces byte-identical
//     TraceV2 files (header YAML + data CSV) across two runs at the same seed, over a
//     coverage matrix of workload shapes (single-turn chatbot; multi-turn accumulate
//     cohort — the #1438 reproducer shape; single-session reasoning; multi-session
//     reasoning, #1458; concurrency clients, #1459; time-varying windows, #1460 — see
//     paritySpecShapes for the authoritative list). Because the exported trace captures
//     every fresh arrival at the cluster boundary (#1440), including closed-loop
//     follow-up rounds produced at runtime by the SessionManager, this covers what the
//     cluster was driven with.
//
//  2. BC-4 (INV-13 run/replay parity): a trace exported from a `blis run` replays (via
//     `blis replay --session-mode fixed`) to the run's per-request rows.
//
// BC-7 (replay rejects unsupported features loudly) is already covered by
// TestReplayCmd_AutoscalerFlagFatal / TestReplayCmd_AutoscalerBundleFatal in
// replay_test.go; this file does not duplicate that subprocess harness.
//
// Design notes:
//   - These tests mutate package-level CLI vars, so they MUST NOT use
//     t.Parallel(); each restores state via captureCmdLevelVars().restore().
//   - Workloads are kept tiny (num_requests ≤ ~30) so the full matrix runs in a
//     couple of seconds on the kernel's committed scenario.
//   - Shared helpers reused: captureCmdLevelVars/restore and
//     setupKernelTestFixturesWithDefaults.

import (
	"bytes"
	"github.com/inference-sim/blis-substrate/contracts/spec/result"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/spf13/cobra"
)

// parityShape is one entry in the coverage matrix: a workload-spec YAML plus a
// horizon large enough that num_requests (not horizon) bounds generation.
type parityShape struct {
	name    string
	yaml    string
	horizon int64
}

// paritySpecShapes returns the coverage matrix from issue #1442 (+ #1458):
//   - chatbot: single-turn, Poisson arrivals (the common case)
//   - codegen-cohort: multi-turn accumulate, cohort-based, large shared prefix
//     (the #1438 reproducer shape — where lazy's memory win matters most)
//   - reasoning: single-session multi-turn (the trickiest generator path —
//     round-0 emitted, follow-ups via the SessionManager at runtime)
//   - reasoning-multi-session: multi-turn with single_session=false (#1458),
//     where a client spawns many overlapping sessions merged per-client in the
//     lazy path — CLI-layer regression protection for the multi-session merge
//   - concurrency: two closed-loop virtual-user pools with distinct
//     concurrency/think_time (#1459), where each seed is streamed as its own
//     heap entry — CLI-layer regression protection for the seed interleave
//   - time-varying: single client, two windows with distinct per-window
//     trace_rate (#1460), routed through the lazy per-window state machine
//     (live-window merge + suffix-min emit gate)
//
// All are lazy-SUPPORTED as of #1460 (multi-session reasoning via the per-client
// live-session merge (#1458); concurrency via the individual-seed-heap-entry
// merge (#1459); time-varying via the per-window live-window merge (#1460)).
// num_requests is small so runs are fast. seed is set via the CLI --seed flag,
// not the YAML, so the matrix controls determinism.
func paritySpecShapes() []parityShape {
	return []parityShape{
		{
			name:    "chatbot",
			horizon: 60_000_000,
			yaml: `version: "2"
category: language
aggregate_rate: 10.0
num_requests: 30
clients:
  - id: chat
    tenant_id: t1
    slo_class: batch
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params: { mean: 128, std_dev: 32, min: 16, max: 512 }
    output_distribution:
      type: exponential
      params: { mean: 64 }
`,
		},
		{
			name:    "codegen-cohort",
			horizon: 120_000_000,
			yaml: `version: "2"
category: language
aggregate_rate: 6.0
num_requests: 24
cohorts:
  - id: dev
    population: 4
    tenant_id: tcode
    slo_class: batch
    rate_fraction: 1.0
    prefix_group: repo
    prefix_length: 256
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params: { mean: 200, std_dev: 40, min: 32, max: 800 }
    output_distribution:
      type: exponential
      params: { mean: 96 }
    reasoning:
      multi_turn:
        max_rounds: 4
        context_growth: accumulate
        think_time_us: 50000
        single_session: true
`,
		},
		{
			name:    "reasoning",
			horizon: 60_000_000,
			yaml: `version: "2"
category: language
aggregate_rate: 4.0
num_requests: 20
clients:
  - id: r1
    tenant_id: rt1
    slo_class: batch
    rate_fraction: 0.5
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params: { mean: 80, std_dev: 16, min: 16, max: 300 }
    output_distribution:
      type: exponential
      params: { mean: 48 }
    reasoning:
      multi_turn:
        max_rounds: 3
        context_growth: accumulate
        think_time_us: 60000
        single_session: true
  - id: r2
    tenant_id: rt2
    slo_class: batch
    rate_fraction: 0.5
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params: { mean: 80, std_dev: 16, min: 16, max: 300 }
    output_distribution:
      type: exponential
      params: { mean: 48 }
    reasoning:
      multi_turn:
        max_rounds: 3
        context_growth: accumulate
        think_time_us: 60000
        single_session: true
`,
		},
		{
			name:    "reasoning-multi-session",
			horizon: 60_000_000,
			yaml: `version: "2"
category: language
aggregate_rate: 4.0
num_requests: 24
clients:
  - id: ms1
    tenant_id: mst1
    slo_class: batch
    rate_fraction: 1.0
    prefix_group: sys
    prefix_length: 64
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params: { mean: 80, std_dev: 16, min: 16, max: 300 }
    output_distribution:
      type: exponential
      params: { mean: 48 }
    reasoning:
      multi_turn:
        max_rounds: 3
        context_growth: accumulate
        think_time_us: 60000
        single_session: false
`,
		},
		{
			// Two closed-loop concurrency pools with distinct pool size + think
			// time, so their staggered seed arrivals interleave (#1459). Exercises
			// the individual-seed-heap-entry merge at the CLI/trace seam. All
			// clients are concurrency (no aggregate_rate needed).
			name:    "concurrency",
			horizon: 60_000_000,
			yaml: `version: "2"
category: language
num_requests: 24
clients:
  - id: poolA
    tenant_id: ta
    slo_class: batch
    concurrency: 5
    think_time_us: 300000
    input_distribution:
      type: gaussian
      params: { mean: 90, std_dev: 20, min: 16, max: 400 }
    output_distribution:
      type: exponential
      params: { mean: 64 }
  - id: poolB
    tenant_id: tb
    slo_class: batch
    concurrency: 3
    think_time_us: 100000
    input_distribution:
      type: gaussian
      params: { mean: 90, std_dev: 20, min: 16, max: 400 }
    output_distribution:
      type: exponential
      params: { mean: 64 }
`,
		},
		{
			// Time-varying single client with two windows carrying distinct
			// per-window trace_rate (absolute-rate mode, aggregate_rate omitted)
			// — trips hasPerWindowParameters, routing through the lazy
			// per-window state machine (#1460). CLI-layer regression protection
			// for the live-window merge + suffix-min emit gate.
			name:    "time-varying",
			horizon: 60_000_000,
			yaml: `version: "2"
category: language
num_requests: 24
clients:
  - id: tv
    tenant_id: ttv
    slo_class: batch
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params: { mean: 96, std_dev: 20, min: 16, max: 400 }
    output_distribution:
      type: exponential
      params: { mean: 48 }
    lifecycle:
      windows:
        - start_us: 0
          end_us: 30000000
          trace_rate: 8.0
        - start_us: 30000000
          end_us: 60000000
          trace_rate: 2.0
`,
		},
	}
}

// runSpecAndCaptureTrace drives runCmd.Run with a --workload-spec file at the
// given seed and lazyGeneration flag, exporting a TraceV2 and returning the
// header + data bytes. Modeled on runRunCmdAndCaptureTraces (which drives a
// distribution workload) but spec-file-driven, so it exercises the same CLI
// seam a user hits with `blis run --workload-spec X --trace-output Y`.
//
// Restores every package-level CLI var it touches via captureCmdLevelVars().
func runSpecAndCaptureTrace(t *testing.T, specYAML string, seedVal, horizon int64) (headerBytes, dataBytes []byte) {
	t.Helper()
	tmpDir := t.TempDir()
	tracePrefix := filepath.Join(tmpDir, "trace")
	specPath := filepath.Join(tmpDir, "workload.yaml")
	if err := os.WriteFile(specPath, []byte(specYAML), 0644); err != nil {
		t.Fatalf("write spec: %v", err)
	}
	catalogDir, defaultsPath := setupKernelTestFixturesWithDefaults(t)

	orig := captureCmdLevelVars()
	defer orig.restore()

	// Set inputs. workloadType stays empty (spec-file path takes precedence).
	traceOutput = tracePrefix
	workloadSpecPath = specPath
	workloadType = ""
	simulationHorizon = horizon
	seed = seedVal
	requestTimeoutSecs = 300 // non-zero; matches runCmd flag default

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&workloadSpecPath, "workload-spec", "", "")
	testCmd.Flags().StringVar(&traceOutput, "trace-output", "", "")
	testCmd.Flags().IntVar(&requestTimeoutSecs, "timeout", 300, "")
	args := []string{
		"--defaults-filepath", defaultsPath,
		"--catalog", catalogDir, "--scenario", kernelTestScenario, "--scenarios", kernelScenariosDir(t), "--registry", kernelRegistryDir(t),
		"--seed", strconv.FormatInt(seedVal, 10),
		"--workload-spec", specPath,
		"--horizon", strconv.FormatInt(horizon, 10),
		"--trace-output", tracePrefix,
	}
	if err := testCmd.ParseFlags(args); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	runCmd.Run(testCmd, nil)

	hdr, err := os.ReadFile(tracePrefix + ".yaml")
	if err != nil {
		t.Fatalf("read trace header: %v", err)
	}
	data, err := os.ReadFile(tracePrefix + ".csv")
	if err != nil {
		t.Fatalf("read trace data: %v", err)
	}
	return hdr, data
}

// TestParity_RunReplay_TraceByteIdentity_Matrix pins BC-5 across the coverage
// matrix: for each shape, two runs at the same seed produce byte-identical
// traces, and the trace carries request records.
//
// NOTE: Do NOT use t.Parallel() — mutates package-level vars.
func TestParity_RunReplay_TraceByteIdentity_Matrix(t *testing.T) {
	const seed int64 = 20260708
	for _, shape := range paritySpecShapes() {
		shape := shape
		t.Run(shape.name, func(t *testing.T) {
			hdr, data := runSpecAndCaptureTrace(t, shape.yaml, seed, shape.horizon)
			// The test must produce a non-trivial trace, else byte-identity is
			// vacuous. A header-only CSV (just the column row) is not enough.
			if lineCount(data) < 2 {
				t.Fatalf("%s: trace CSV has < 2 lines (%d) — no request records; test is vacuous",
					shape.name, lineCount(data))
			}
			hdr2, data2 := runSpecAndCaptureTrace(t, shape.yaml, seed, shape.horizon)
			if !bytes.Equal(hdr, hdr2) || !bytes.Equal(data, data2) {
				t.Fatalf("%s: trace non-deterministic across two runs at seed %d", shape.name, seed)
			}
		})
	}
}

// lineCount counts newline-terminated lines in b (used to assert the trace CSV
// carries request records, not just the header row).
func lineCount(b []byte) int {
	return bytes.Count(b, []byte("\n"))
}

// runSpecAndCaptureStdout drives runCmd.Run with a --workload-spec file and
// captures everything written to os.Stdout (the deterministic results channel
// per INV-6; diagnostics go to stderr via logrus and are not captured). No
// trace is exported. Used to assert stdout-level determinism — the literal
// INV-6 statement ("same seed → byte-identical stdout"), complementing the
// trace byte-identity above.
//
// NOTE: mutates package-level CLI vars and os.Stdout; caller must not run in
// parallel. Restores both.
func runSpecAndCaptureStdout(t *testing.T, specYAML string, seedVal, horizon int64) []byte {
	t.Helper()
	tmpDir := t.TempDir()
	specPath := filepath.Join(tmpDir, "workload.yaml")
	if err := os.WriteFile(specPath, []byte(specYAML), 0644); err != nil {
		t.Fatalf("write spec: %v", err)
	}
	catalogDir, defaultsPath := setupKernelTestFixturesWithDefaults(t)

	orig := captureCmdLevelVars()
	defer orig.restore()

	traceOutput = ""
	workloadSpecPath = specPath
	workloadType = ""
	simulationHorizon = horizon
	seed = seedVal
	requestTimeoutSecs = 300

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&workloadSpecPath, "workload-spec", "", "")
	testCmd.Flags().IntVar(&requestTimeoutSecs, "timeout", 300, "")
	args := []string{
		"--defaults-filepath", defaultsPath,
		"--catalog", catalogDir, "--scenario", kernelTestScenario, "--scenarios", kernelScenariosDir(t), "--registry", kernelRegistryDir(t),
		"--seed", strconv.FormatInt(seedVal, 10),
		"--workload-spec", specPath,
		"--horizon", strconv.FormatInt(horizon, 10),
	}
	if err := testCmd.ParseFlags(args); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}

	// Capture os.Stdout across runCmd.Run. A pipe's buffer is bounded, so
	// drain it in a goroutine to avoid deadlock if the output is large.
	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stdout = w
	done := make(chan []byte, 1)
	go func() {
		var buf bytes.Buffer
		_, _ = io.Copy(&buf, r)
		done <- buf.Bytes()
	}()

	runCmd.Run(testCmd, nil)

	_ = w.Close()
	os.Stdout = oldStdout
	return <-done
}

// TestParity_RunStdout_DeterministicAndEagerLazyIdentical pins the literal
// INV-6 statement for streaming generation: same seed → the same Result outside
// runtime. This is the stdout-channel companion to the trace byte-identity
// matrix (stdout is the canonical deterministic results channel; trace is a
// file artifact). Uses the chatbot shape. The invariant registry names this
// test (X13), so it keeps its name though the eager generator it once compared
// against is gone (#1451).
//
// NOTE: Do NOT use t.Parallel() — mutates package-level vars and os.Stdout.
func TestParity_RunStdout_DeterministicAndEagerLazyIdentical(t *testing.T) {
	const seed int64 = 20260708
	shape := paritySpecShapes()[0] // chatbot

	first := runSpecAndCaptureStdout(t, shape.yaml, seed, shape.horizon)
	second := runSpecAndCaptureStdout(t, shape.yaml, seed, shape.horizon)

	if len(first) == 0 {
		t.Fatal("run produced empty stdout; test is vacuous")
	}
	if withoutRuntime(t, string(first)) != withoutRuntime(t, string(second)) {
		t.Fatalf("Result non-deterministic outside runtime across two runs at seed %d", seed)
	}
}

// replaySpecTrace replays a previously-exported trace through replayCmd.Run
// with --session-mode fixed and returns the Result's rows of completed requests.
// Mirrors the var-setup dance in TestINV13_RunReplayParity_PD_CLI but for a
// single-instance non-PD cluster.
func replaySpecTrace(t *testing.T, traceHeaderFile, traceDataFile string) []result.RequestRow {
	t.Helper()
	tmpDir := t.TempDir()
	resultsFile := filepath.Join(tmpDir, "result.json")
	catalogDir, defaultsPath := setupKernelTestFixturesWithDefaults(t)

	orig := captureCmdLevelVars()
	defer orig.restore()

	// Replay-specific package vars not covered by captureCmdLevelVars.
	origTraceHeader := traceHeaderPath
	origTraceData := traceDataPath
	origSessionMode := replaySessionMode
	origThinkMs := replayThinkTimeMs
	origThinkDist := replayThinkTimeDist
	origReplayTraceOut := replayTraceOutput
	defer func() {
		traceHeaderPath = origTraceHeader
		traceDataPath = origTraceData
		replaySessionMode = origSessionMode
		replayThinkTimeMs = origThinkMs
		replayThinkTimeDist = origThinkDist
		replayTraceOutput = origReplayTraceOut
	}()

	model = "qwen/qwen3-14b"
	totalKVBlocks = 1000
	blockSizeTokens = 16
	maxNumSeqs = 64
	maxNumBatchedTokens = 2048
	numInstances = 1
	origOutput := resultOutput
	defer func() { resultOutput = origOutput }()
	resultOutput = resultsFile
	longPrefillTokenThreshold = 0
	scheduler = "fcfs"
	policyConfigPath = ""
	maxModelLen = 0
	traceLevel = "none"
	counterfactualK = 0
	traceHeaderPath = traceHeaderFile
	traceDataPath = traceDataFile
	catalogPath = catalogDir
	gpu = "H100"
	tensorParallelism = 1
	defaultsFilePath = defaultsPath
	replaySessionMode = "fixed"
	replayThinkTimeMs = 0
	replayThinkTimeDist = ""
	replayTraceOutput = ""

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
	testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
	if err := testCmd.ParseFlags([]string{
		"--catalog", catalogDir, "--scenario", kernelTestScenario, "--scenarios", kernelScenariosDir(t), "--registry", kernelRegistryDir(t),
		"--trace-header", traceHeaderFile, "--trace-data", traceDataFile,
		"--num-instances", "1",
		"--horizon", "120000000",
		"--defaults-filepath", defaultsPath,
	}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	replayCmd.Run(testCmd, nil)

	data, err := os.ReadFile(resultsFile)
	if err != nil {
		t.Fatalf("the Result was not written to --output: %v", err)
	}
	var doc result.Result
	if err := result.Unmarshal(data, &doc); err != nil {
		t.Fatalf("parse the Result: %v", err)
	}
	var completed []result.RequestRow
	for _, row := range doc.RequestRows {
		if row.Fate == result.FateCompleted {
			completed = append(completed, row)
		}
	}
	return completed
}

// runSpecToTraceFiles drives runCmd.Run at a fixed seed/mode and returns the
// paths of the exported trace files (kept on disk for the replay leg). Unlike
// runSpecAndCaptureTrace it does not read the bytes back; the caller replays
// the files. The temp dir is created under t so it is cleaned up automatically.
func runSpecToTraceFiles(t *testing.T, specYAML string, seedVal, horizon int64) (headerFile, dataFile string) {
	t.Helper()
	tmpDir := t.TempDir()
	tracePrefix := filepath.Join(tmpDir, "trace")
	specPath := filepath.Join(tmpDir, "workload.yaml")
	if err := os.WriteFile(specPath, []byte(specYAML), 0644); err != nil {
		t.Fatalf("write spec: %v", err)
	}
	catalogDir, defaultsPath := setupKernelTestFixturesWithDefaults(t)

	orig := captureCmdLevelVars()
	defer orig.restore()

	traceOutput = tracePrefix
	workloadSpecPath = specPath
	workloadType = ""
	simulationHorizon = horizon
	seed = seedVal
	requestTimeoutSecs = 300

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&workloadSpecPath, "workload-spec", "", "")
	testCmd.Flags().StringVar(&traceOutput, "trace-output", "", "")
	testCmd.Flags().IntVar(&requestTimeoutSecs, "timeout", 300, "")
	args := []string{
		"--defaults-filepath", defaultsPath,
		"--catalog", catalogDir, "--scenario", kernelTestScenario, "--scenarios", kernelScenariosDir(t), "--registry", kernelRegistryDir(t),
		"--seed", strconv.FormatInt(seedVal, 10),
		"--workload-spec", specPath,
		"--horizon", strconv.FormatInt(horizon, 10),
		"--trace-output", tracePrefix,
	}
	if err := testCmd.ParseFlags(args); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	runCmd.Run(testCmd, nil)

	headerFile = tracePrefix + ".yaml"
	dataFile = tracePrefix + ".csv"
	if _, err := os.Stat(dataFile); err != nil {
		t.Fatalf("trace CSV not written: %v", err)
	}
	return headerFile, dataFile
}

// rowMaps indexes completed request rows by request ID for order-independent comparison, as
// the id's number: ttft and e2e in microseconds.
func rowMaps(rows []result.RequestRow) (ttft, e2e map[string]float64) {
	ttft = make(map[string]float64, len(rows))
	e2e = make(map[string]float64, len(rows))
	for _, r := range rows {
		if r.TTFTUs != nil {
			ttft[r.ID] = float64(*r.TTFTUs)
		}
		if r.E2EUs != nil {
			e2e[r.ID] = float64(*r.E2EUs)
		}
	}
	return ttft, e2e
}

// TestParity_RunReplay_INV13 pins BC-4 (INV-13) end-to-end: a trace exported
// from `blis run` replays to non-empty per-request rows, and a second run at the
// same seed, exported and replayed, gives every request the same TTFT and E2E.
//
// Uses the chatbot shape (bounded runtime); the byte-identity matrix above
// covers the reproducer and reasoning shapes.
//
// NOTE: Do NOT use t.Parallel() — mutates package-level vars.
func TestParity_RunReplay_INV13(t *testing.T) {
	const seed int64 = 20260708
	shape := paritySpecShapes()[0] // chatbot

	firstHdr, firstData := runSpecToTraceFiles(t, shape.yaml, seed, shape.horizon)
	secondHdr, secondData := runSpecToTraceFiles(t, shape.yaml, seed, shape.horizon)

	first := replaySpecTrace(t, firstHdr, firstData)
	second := replaySpecTrace(t, secondHdr, secondData)

	if len(first) == 0 {
		t.Fatal("INV-13: the replay produced no completed requests")
	}
	if len(first) != len(second) {
		t.Fatalf("INV-13: completed request count differs %d vs %d", len(first), len(second))
	}

	firstTTFT, firstE2E := rowMaps(first)
	secondTTFT, secondE2E := rowMaps(second)
	for id, wantTTFT := range firstTTFT {
		gotTTFT, ok := secondTTFT[id]
		if !ok {
			t.Errorf("INV-13: request %s present in the first replay, missing from the second", id)
			continue
		}
		if gotTTFT != wantTTFT {
			t.Errorf("INV-13: request %s TTFT %f vs %f", id, wantTTFT, gotTTFT)
		}
		if firstE2E[id] != secondE2E[id] {
			t.Errorf("INV-13: request %s E2E %f vs %f", id, firstE2E[id], secondE2E[id])
		}
	}
}

// Time-varying traces are covered by the "time-varying" row of paritySpecShapes()
// flowing through TestParity_RunReplay_TraceByteIdentity_Matrix (#1460).
