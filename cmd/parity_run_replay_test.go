package cmd

// Run/replay parity matrix for the lazy-generation feature (issue #1442, A4).
//
// This file is the authoritative CLI-level enforcement point for two claims
// that make --lazy-generation (A3, #1441) safe:
//
//  1. BC-3 (trace byte-identity): `blis run --trace-output` produces
//     byte-identical TraceV2 files (header YAML + data CSV) whether run in
//     eager or lazy mode, across a coverage matrix of workload shapes
//     (single-turn chatbot, multi-turn accumulate cohort — the #1438
//     reproducer shape, and single-session reasoning). Because the exported
//     trace captures every fresh arrival at the cluster boundary (#1440),
//     byte-identity here proves the two generation modes drive the cluster
//     identically — including closed-loop follow-up rounds produced at runtime
//     by the SessionManager.
//
//  2. BC-4 (INV-13 run/replay parity): a trace exported from a `blis run`
//     replays (via `blis replay --session-mode fixed`) to identical per-request
//     SimResults, and the eager-sourced and lazy-sourced replays agree. This is
//     INV-13 under both generation modes.
//
// It also pins BC-5 (lazy determinism, INV-6: two lazy runs at the same seed
// yield identical traces) and BC-6 (an unsupported lazy shape falls back to the
// eager generator transparently — same trace bytes, not merely non-fatal).
//
// BC-7 (replay rejects unsupported features loudly) is already covered by
// TestReplayCmd_AutoscalerFlagFatal / TestReplayCmd_AutoscalerBundleFatal in
// replay_test.go; this file does not duplicate that subprocess harness.
//
// Design notes:
//   - These tests mutate package-level CLI vars, so they MUST NOT use
//     t.Parallel(); each restores state via captureCmdLevelVars().restore().
//   - Workloads are kept tiny (num_requests ≤ ~30, a 2-layer model fixture) so
//     the full matrix runs in a couple of seconds.
//   - Shared helpers reused: captureCmdLevelVars/restore and
//     setupTrainedPhysicsTestFixturesWithDefaults.

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim/workload"
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
//
// All are lazy-SUPPORTED (no per-window params, no concurrency; multi-session
// reasoning is streamed via the per-client live-session merge as of #1458).
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
	}
}

// runSpecAndCaptureTrace drives runCmd.Run with a --workload-spec file at the
// given seed and lazyGeneration flag, exporting a TraceV2 and returning the
// header + data bytes. Modeled on runRunCmdAndCaptureTraces (which drives a
// distribution workload) but spec-file-driven, so it exercises the same CLI
// seam a user hits with `blis run --workload-spec X --trace-output Y`.
//
// Restores every package-level CLI var it touches via captureCmdLevelVars().
func runSpecAndCaptureTrace(t *testing.T, specYAML string, seedVal, horizon int64, lazyFlag bool) (headerBytes, dataBytes []byte) {
	t.Helper()
	tmpDir := t.TempDir()
	tracePrefix := filepath.Join(tmpDir, "trace")
	specPath := filepath.Join(tmpDir, "workload.yaml")
	if err := os.WriteFile(specPath, []byte(specYAML), 0644); err != nil {
		t.Fatalf("write spec: %v", err)
	}
	mcFolder, hwPath, defaultsPath := setupTrainedPhysicsTestFixturesWithDefaults(t)

	orig := captureCmdLevelVars()
	defer orig.restore()

	// Set inputs. workloadType stays empty (spec-file path takes precedence).
	traceOutput = tracePrefix
	workloadSpecPath = specPath
	workloadType = ""
	simulationHorizon = horizon
	seed = seedVal
	lazyGeneration = lazyFlag
	requestTimeoutSecs = 300 // non-zero; matches runCmd flag default

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&workloadSpecPath, "workload-spec", "", "")
	testCmd.Flags().StringVar(&traceOutput, "trace-output", "", "")
	testCmd.Flags().BoolVar(&lazyGeneration, "lazy-generation", false, "")
	testCmd.Flags().IntVar(&requestTimeoutSecs, "timeout", 300, "")
	args := []string{
		"--model", "qwen/qwen3-14b",
		"--latency-model", "trained-physics",
		"--defaults-filepath", defaultsPath,
		"--model-config-folder", mcFolder,
		"--hardware-config", hwPath,
		"--hardware", "H100",
		"--tp", "1",
		"--total-kv-blocks", "1000",
		"--seed", strconv.FormatInt(seedVal, 10),
		"--workload-spec", specPath,
		"--horizon", strconv.FormatInt(horizon, 10),
		"--trace-output", tracePrefix,
	}
	if lazyFlag {
		args = append(args, "--lazy-generation=true")
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

// TestParity_RunReplay_TraceByteIdentity_Matrix pins BC-3 and BC-5 across the
// coverage matrix: for each shape, the eager and lazy runs produce
// byte-identical traces, and two lazy runs at the same seed are also identical.
//
// NOTE: Do NOT use t.Parallel() — mutates package-level vars.
func TestParity_RunReplay_TraceByteIdentity_Matrix(t *testing.T) {
	const seed int64 = 20260708
	for _, shape := range paritySpecShapes() {
		shape := shape
		t.Run(shape.name, func(t *testing.T) {
			hdrEager, dataEager := runSpecAndCaptureTrace(t, shape.yaml, seed, shape.horizon, false)
			hdrLazy, dataLazy := runSpecAndCaptureTrace(t, shape.yaml, seed, shape.horizon, true)

			// BC-3: eager and lazy traces are byte-identical.
			if !bytes.Equal(hdrEager, hdrLazy) {
				t.Fatalf("%s: trace header diverged eager vs lazy\nEAGER:\n%s\nLAZY:\n%s",
					shape.name, hdrEager, hdrLazy)
			}
			if !bytes.Equal(dataEager, dataLazy) {
				t.Fatalf("%s: trace data diverged eager vs lazy\nEAGER:\n%s\nLAZY:\n%s",
					shape.name, dataEager, dataLazy)
			}

			// The test must produce a non-trivial trace, else byte-identity is
			// vacuous. A header-only CSV (just the column row) is not enough.
			if lineCount(dataEager) < 2 {
				t.Fatalf("%s: trace CSV has < 2 lines (%d) — no request records; test is vacuous",
					shape.name, lineCount(dataEager))
			}

			// BC-5 (INV-6): a second lazy run at the same seed is identical.
			hdrLazy2, dataLazy2 := runSpecAndCaptureTrace(t, shape.yaml, seed, shape.horizon, true)
			if !bytes.Equal(hdrLazy, hdrLazy2) || !bytes.Equal(dataLazy, dataLazy2) {
				t.Fatalf("%s: lazy trace non-deterministic across two runs at seed %d", shape.name, seed)
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
// trace is exported. Used to assert stdout-level determinism and eager≡lazy
// stdout parity — the literal INV-6 statement ("same seed → byte-identical
// stdout"), complementing the trace byte-identity above.
//
// NOTE: mutates package-level CLI vars and os.Stdout; caller must not run in
// parallel. Restores both.
func runSpecAndCaptureStdout(t *testing.T, specYAML string, seedVal, horizon int64, lazyFlag bool) []byte {
	t.Helper()
	tmpDir := t.TempDir()
	specPath := filepath.Join(tmpDir, "workload.yaml")
	if err := os.WriteFile(specPath, []byte(specYAML), 0644); err != nil {
		t.Fatalf("write spec: %v", err)
	}
	mcFolder, hwPath, defaultsPath := setupTrainedPhysicsTestFixturesWithDefaults(t)

	orig := captureCmdLevelVars()
	defer orig.restore()

	traceOutput = ""
	workloadSpecPath = specPath
	workloadType = ""
	simulationHorizon = horizon
	seed = seedVal
	lazyGeneration = lazyFlag
	requestTimeoutSecs = 300

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&workloadSpecPath, "workload-spec", "", "")
	testCmd.Flags().BoolVar(&lazyGeneration, "lazy-generation", false, "")
	testCmd.Flags().IntVar(&requestTimeoutSecs, "timeout", 300, "")
	args := []string{
		"--model", "qwen/qwen3-14b",
		"--latency-model", "trained-physics",
		"--defaults-filepath", defaultsPath,
		"--model-config-folder", mcFolder,
		"--hardware-config", hwPath,
		"--hardware", "H100",
		"--tp", "1",
		"--total-kv-blocks", "1000",
		"--seed", strconv.FormatInt(seedVal, 10),
		"--workload-spec", specPath,
		"--horizon", strconv.FormatInt(horizon, 10),
	}
	if lazyFlag {
		args = append(args, "--lazy-generation=true")
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
// INV-6 statement for lazy mode: same seed → byte-identical stdout, and eager
// and lazy modes produce identical stdout. This is the stdout-channel companion
// to the trace byte-identity matrix (stdout is the canonical deterministic
// results channel; trace is a file artifact). Uses the chatbot shape.
//
// NOTE: Do NOT use t.Parallel() — mutates package-level vars and os.Stdout.
func TestParity_RunStdout_DeterministicAndEagerLazyIdentical(t *testing.T) {
	const seed int64 = 20260708
	shape := paritySpecShapes()[0] // chatbot

	eager := runSpecAndCaptureStdout(t, shape.yaml, seed, shape.horizon, false)
	lazy := runSpecAndCaptureStdout(t, shape.yaml, seed, shape.horizon, true)
	lazy2 := runSpecAndCaptureStdout(t, shape.yaml, seed, shape.horizon, true)

	if len(eager) == 0 {
		t.Fatal("run produced empty stdout; test is vacuous")
	}
	// INV-6 (determinism): two lazy runs at the same seed match.
	if !bytes.Equal(lazy, lazy2) {
		t.Fatalf("lazy stdout non-deterministic across two runs at seed %d", seed)
	}
	// eager ≡ lazy at the stdout level.
	if !bytes.Equal(eager, lazy) {
		t.Fatalf("stdout diverged between eager and lazy modes\nEAGER:\n%s\nLAZY:\n%s", eager, lazy)
	}
}

// replaySpecTrace replays a previously-exported trace through replayCmd.Run
// with --session-mode fixed and returns the parsed per-request SimResults.
// Mirrors the var-setup dance in TestINV13_RunReplayParity_PD_CLI but for a
// single-instance non-PD cluster.
func replaySpecTrace(t *testing.T, traceHeaderFile, traceDataFile string) []workload.SimResult {
	t.Helper()
	tmpDir := t.TempDir()
	resultsFile := filepath.Join(tmpDir, "results.json")
	mcFolder, hwPath, defaultsPath := setupTrainedPhysicsTestFixturesWithDefaults(t)

	orig := captureCmdLevelVars()
	defer orig.restore()

	// Replay-specific package vars not covered by captureCmdLevelVars.
	origTraceHeader := traceHeaderPath
	origTraceData := traceDataPath
	origSessionMode := replaySessionMode
	origThinkMs := replayThinkTimeMs
	origThinkDist := replayThinkTimeDist
	origReplayTraceOut := replayTraceOutput
	origCacheSignalDelay := cacheSignalDelay
	origFlowControlEnabled := flowControlEnabled
	defer func() {
		traceHeaderPath = origTraceHeader
		traceDataPath = origTraceData
		replaySessionMode = origSessionMode
		replayThinkTimeMs = origThinkMs
		replayThinkTimeDist = origThinkDist
		replayTraceOutput = origReplayTraceOut
		cacheSignalDelay = origCacheSignalDelay
		flowControlEnabled = origFlowControlEnabled
	}()

	model = "qwen/qwen3-14b"
	latencyModelBackend = "trained-physics"
	totalKVBlocks = 1000
	blockSizeTokens = 16
	maxRunningReqs = 64
	maxScheduledTokens = 2048
	numInstances = 1
	resultsPath = resultsFile
	longPrefillTokenThreshold = 0
	kvCPUBlocks = 0
	kvOffloadThreshold = 0.9
	kvTransferBandwidth = 100.0
	kvTransferBaseLatency = 0
	snapshotRefreshInterval = 0
	admissionPolicy = "always-admit"
	routingPolicy = "round-robin"
	scheduler = "fcfs"
	policyConfigPath = ""
	maxModelLen = 0
	traceLevel = "none"
	counterfactualK = 0
	traceHeaderPath = traceHeaderFile
	traceDataPath = traceDataFile
	modelConfigFolder = mcFolder
	hwConfigPath = hwPath
	gpu = "H100"
	tensorParallelism = 1
	defaultsFilePath = defaultsPath
	replaySessionMode = "fixed"
	replayThinkTimeMs = 0
	replayThinkTimeDist = ""
	replayTraceOutput = ""
	cacheSignalDelay = 0
	flowControlEnabled = false

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
	testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
	testCmd.Flags().StringVar(&resultsPath, "results-path", "", "")
	if err := testCmd.ParseFlags([]string{
		"--model", "qwen/qwen3-14b", "--latency-model", "trained-physics",
		"--total-kv-blocks", "1000", "--hardware", "H100", "--tp", "1",
		"--model-config-folder", mcFolder, "--hardware-config", hwPath,
		"--trace-header", traceHeaderFile, "--trace-data", traceDataFile,
		"--results-path", resultsFile,
		"--num-instances", "1",
		"--horizon", "120000000",
		"--defaults-filepath", defaultsPath,
	}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}
	replayCmd.Run(testCmd, nil)

	data, err := os.ReadFile(resultsFile)
	if err != nil {
		t.Fatalf("results file not written: %v", err)
	}
	var results []workload.SimResult
	if err := json.Unmarshal(data, &results); err != nil {
		t.Fatalf("parse SimResult JSON: %v", err)
	}
	return results
}

// runSpecToTraceFiles drives runCmd.Run at a fixed seed/mode and returns the
// paths of the exported trace files (kept on disk for the replay leg). Unlike
// runSpecAndCaptureTrace it does not read the bytes back; the caller replays
// the files. The temp dir is created under t so it is cleaned up automatically.
func runSpecToTraceFiles(t *testing.T, specYAML string, seedVal, horizon int64, lazyFlag bool) (headerFile, dataFile string) {
	t.Helper()
	tmpDir := t.TempDir()
	tracePrefix := filepath.Join(tmpDir, "trace")
	specPath := filepath.Join(tmpDir, "workload.yaml")
	if err := os.WriteFile(specPath, []byte(specYAML), 0644); err != nil {
		t.Fatalf("write spec: %v", err)
	}
	mcFolder, hwPath, defaultsPath := setupTrainedPhysicsTestFixturesWithDefaults(t)

	orig := captureCmdLevelVars()
	defer orig.restore()

	traceOutput = tracePrefix
	workloadSpecPath = specPath
	workloadType = ""
	simulationHorizon = horizon
	seed = seedVal
	lazyGeneration = lazyFlag
	requestTimeoutSecs = 300

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&workloadSpecPath, "workload-spec", "", "")
	testCmd.Flags().StringVar(&traceOutput, "trace-output", "", "")
	testCmd.Flags().BoolVar(&lazyGeneration, "lazy-generation", false, "")
	testCmd.Flags().IntVar(&requestTimeoutSecs, "timeout", 300, "")
	args := []string{
		"--model", "qwen/qwen3-14b",
		"--latency-model", "trained-physics",
		"--defaults-filepath", defaultsPath,
		"--model-config-folder", mcFolder,
		"--hardware-config", hwPath,
		"--hardware", "H100",
		"--tp", "1",
		"--total-kv-blocks", "1000",
		"--seed", strconv.FormatInt(seedVal, 10),
		"--workload-spec", specPath,
		"--horizon", strconv.FormatInt(horizon, 10),
		"--trace-output", tracePrefix,
	}
	if lazyFlag {
		args = append(args, "--lazy-generation=true")
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

// simResultMaps indexes SimResults by request ID for order-independent
// comparison (extractSimResults iterates a map, so slice order is not stable).
func simResultMaps(results []workload.SimResult) (ttft, e2e map[int]float64) {
	ttft = make(map[int]float64, len(results))
	e2e = make(map[int]float64, len(results))
	for _, r := range results {
		ttft[r.RequestID] = r.TTFT
		e2e[r.RequestID] = r.E2E
	}
	return ttft, e2e
}

// TestParity_RunReplay_INV13_BothModes pins BC-4 (INV-13) end-to-end: a trace
// exported from `blis run` replays to non-empty per-request SimResults, and the
// eager-sourced and lazy-sourced replays produce identical TTFT/E2E per request.
//
// Uses the chatbot shape (bounded runtime); the byte-identity matrix above
// covers the reproducer and reasoning shapes. Because the eager and lazy traces
// are byte-identical (proven above), their replays are necessarily identical —
// this test confirms the full run→export→replay pipeline works under both modes
// and that INV-13 holds, not just that the traces match.
//
// NOTE: Do NOT use t.Parallel() — mutates package-level vars.
func TestParity_RunReplay_INV13_BothModes(t *testing.T) {
	const seed int64 = 20260708
	shape := paritySpecShapes()[0] // chatbot

	eagerHdr, eagerData := runSpecToTraceFiles(t, shape.yaml, seed, shape.horizon, false)
	lazyHdr, lazyData := runSpecToTraceFiles(t, shape.yaml, seed, shape.horizon, true)

	eagerResults := replaySpecTrace(t, eagerHdr, eagerData)
	lazyResults := replaySpecTrace(t, lazyHdr, lazyData)

	if len(eagerResults) == 0 {
		t.Fatal("INV-13: eager-sourced replay produced no completed requests")
	}
	if len(eagerResults) != len(lazyResults) {
		t.Fatalf("INV-13: completed request count differs eager=%d lazy=%d",
			len(eagerResults), len(lazyResults))
	}

	eagerTTFT, eagerE2E := simResultMaps(eagerResults)
	lazyTTFT, lazyE2E := simResultMaps(lazyResults)
	for id, wantTTFT := range eagerTTFT {
		gotTTFT, ok := lazyTTFT[id]
		if !ok {
			t.Errorf("INV-13: request %d present in eager replay, missing from lazy replay", id)
			continue
		}
		if gotTTFT != wantTTFT {
			t.Errorf("INV-13: request %d TTFT eager=%f lazy=%f", id, wantTTFT, gotTTFT)
		}
		if eagerE2E[id] != lazyE2E[id] {
			t.Errorf("INV-13: request %d E2E eager=%f lazy=%f", id, eagerE2E[id], lazyE2E[id])
		}
	}
}

// TestParity_LazyTimeVaryingFallback_MatchesEager pins BC-6: when
// --lazy-generation is combined with a time-varying (per-window) spec — a shape
// the lazy source cannot handle — cmd/root.go logs a warning and falls back to
// the eager generator. This test proves the fallback is TRANSPARENT: the lazy
// invocation produces a trace byte-identical to a plain eager run of the same
// spec (not merely that it doesn't fatal — that weaker property is covered by
// TestRunCmd_LazyGeneration_ConcurrencyFallback_DoesNotFatal).
//
// NOTE: Do NOT use t.Parallel() — mutates package-level vars.
func TestParity_LazyTimeVaryingFallback_MatchesEager(t *testing.T) {
	const seed int64 = 4242
	// A per-window trace_rate makes hasPerWindowParameters true → lazy falls
	// back to eager (ErrLazyUnsupportedTimeVarying).
	specYAML := `version: "2"
category: language
aggregate_rate: 8.0
num_requests: 20
clients:
  - id: tv
    tenant_id: t1
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
`
	const horizon int64 = 60_000_000
	hdrEager, dataEager := runSpecAndCaptureTrace(t, specYAML, seed, horizon, false)
	hdrLazy, dataLazy := runSpecAndCaptureTrace(t, specYAML, seed, horizon, true)

	if !bytes.Equal(hdrEager, hdrLazy) {
		t.Fatalf("time-varying fallback: header diverged\nEAGER:\n%s\nLAZY:\n%s", hdrEager, hdrLazy)
	}
	if !bytes.Equal(dataEager, dataLazy) {
		t.Fatalf("time-varying fallback: data diverged (lazy fallback is not transparent)\nEAGER:\n%s\nLAZY:\n%s", hdrEager, dataLazy)
	}
	if lineCount(dataEager) < 2 {
		t.Fatalf("time-varying fallback: trace CSV has < 2 lines — test is vacuous")
	}
}
