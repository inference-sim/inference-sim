package cmd

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	sim "github.com/inference-sim/inference-sim/sim"
)

// End-to-end catalog provenance tests (#1732, R1/S5) at the CLI seam: `blis run` and
// `blis replay` both record the catalog path, revision and dirty flag into their
// --metrics-path results file (AC-1, INV-13), and their stdout is byte-identical
// whether the catalog is clean or dirty (AC-4, INV-6).
//
// NOTE: these tests mutate package-level CLI vars and os.Stdout — do NOT use
// t.Parallel(). Every mutated var is restored.

// provenanceTestModel is the model the runs below simulate. Its catalog entry short name
// is the last path segment, which writeTestCatalog creates.
const provenanceTestModel = "qwen/qwen3-14b"

// Workload sizing for the runs below. These are constants rather than reads of the
// package-level CLI vars because registerSimConfigFlags RESETS every bound var to its
// flag default at registration time — a value read after registration would be the
// default, not what the test set.
const (
	provenanceNumRequests = 8
	provenanceHorizonUs   = int64(120_000_000)
	provenanceSeed        = int64(4242)
)

// setupGitCatalogFixtures builds the fixture set the CLI provenance tests need: a model
// catalog that is its OWN git repository (so the dirty flag is scoped to the catalog and
// nothing else), plus the hardware config and defaults.yaml the trained-physics backend
// needs — deliberately placed OUTSIDE the catalog so they cannot make it dirty.
func setupGitCatalogFixtures(t *testing.T) (catalogRoot, hwPath, defaultsPath, headRevision string) {
	t.Helper()
	requireGit(t)
	tmp := t.TempDir()

	catalogRoot = filepath.Join(tmp, "catalog")
	if err := os.MkdirAll(catalogRoot, 0o755); err != nil {
		t.Fatalf("mkdir catalog root: %v", err)
	}
	// 2-layer Llama-like config, matching setupTrainedPhysicsTestFixtures, so the
	// simulation is fast.
	configJSON := `{
  "architectures": ["LlamaForCausalLM"],
  "num_attention_heads": 4,
  "num_hidden_layers": 2,
  "hidden_size": 64,
  "intermediate_size": 128,
  "num_key_value_heads": 4,
  "torch_dtype": "float16",
  "max_position_embeddings": 4096
}`
	if _, err := writeTestCatalog(catalogRoot, configJSON); err != nil {
		t.Fatalf("write test catalog: %v", err)
	}

	hwPath = filepath.Join(tmp, "hw.json")
	hwJSON := `{
  "H100": {
    "MemoryGiB": 80.0,
    "TFlopsPeak": 1.0,
    "BwPeakTBs": 0.001
  }
}`
	if err := os.WriteFile(hwPath, []byte(hwJSON), 0o644); err != nil {
		t.Fatalf("write hw config: %v", err)
	}
	defaultsPath = filepath.Join(tmp, "defaults.yaml")
	defaultsYAML := `trained_physics_coefficients:
  alpha_coeffs: [100.0, 1.0, 100.0]
  beta_coeffs: [0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
`
	if err := os.WriteFile(defaultsPath, []byte(defaultsYAML), 0o644); err != nil {
		t.Fatalf("write defaults.yaml: %v", err)
	}

	gitInRepo(t, catalogRoot, "init", "--quiet")
	gitInRepo(t, catalogRoot, "add", ".")
	gitInRepo(t, catalogRoot, "commit", "--quiet", "-m", "catalog: initial entries")
	headRevision = strings.TrimSpace(gitInRepo(t, catalogRoot, "rev-parse", "HEAD"))
	return catalogRoot, hwPath, defaultsPath, headRevision
}

// dirtyCatalogEntry makes the catalog entry for provenanceTestModel differ from its
// committed state WITHOUT changing the model config it parses to: it appends a newline,
// which JSON ignores. This is what makes the INV-6 assertion non-vacuous — the catalog is
// genuinely dirty while the simulation's inputs, and therefore its stdout, are unchanged.
func dirtyCatalogEntry(t *testing.T, catalogRoot string) {
	t.Helper()
	shortName := provenanceTestModel[strings.LastIndexByte(provenanceTestModel, '/')+1:]
	cfg := filepath.Join(catalogRoot, shortName, "config.json")
	data, err := os.ReadFile(cfg)
	if err != nil {
		t.Fatalf("read catalog entry %s: %v", cfg, err)
	}
	if err := os.WriteFile(cfg, append(data, '\n'), 0o644); err != nil {
		t.Fatalf("dirty catalog entry %s: %v", cfg, err)
	}
	if out := gitInRepo(t, catalogRoot, "status", "--porcelain", "--", "."); strings.TrimSpace(out) == "" {
		t.Fatalf("fixture failed to dirty the catalog: git reports a clean subtree")
	}
}

// readMetricsFile parses a --metrics-path results file.
func readMetricsFile(t *testing.T, path string) sim.MetricsOutput {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read metrics file %s: %v", path, err)
	}
	var out sim.MetricsOutput
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("parse metrics file %s: %v\n%s", path, err, data)
	}
	return out
}

// captureRunStdout redirects os.Stdout across fn and returns what was written. logrus
// diagnostics go to stderr and are deliberately not captured (INV-6 governs stdout).
func captureRunStdout(t *testing.T, fn func()) []byte {
	t.Helper()
	old := os.Stdout
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
	fn()
	_ = w.Close()
	os.Stdout = old
	return <-done
}

// setProvenanceRunVars sets the package-level CLI vars shared by the run and replay
// drivers below to a small deterministic single-instance configuration.
func setProvenanceRunVars(catalogDir, hwPath, defaultsPath string) {
	model = provenanceTestModel
	latencyModelBackend = "trained-physics"
	catalogPath = catalogDir
	hwConfigPath = hwPath
	defaultsFilePath = defaultsPath
	gpu = "H100"
	tensorParallelism = 1
	totalKVBlocks = 1000
	blockSizeTokens = 16
	maxNumSeqs = 64
	maxNumBatchedTokens = 2048
	numInstances = 1
	seed = provenanceSeed
	simulationHorizon = provenanceHorizonUs
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
	resultsPath = ""
	workloadSpecPath = ""
	lazyGeneration = false
	requestTimeoutSecs = 300
}

// runWithProvenance drives `blis run --metrics-path` in-process over a tiny distribution
// workload, optionally exporting a trace, and returns stdout plus the parsed results file.
func runWithProvenance(t *testing.T, catalogDir, hwPath, defaultsPath, tracePrefix string) ([]byte, sim.MetricsOutput) {
	t.Helper()
	metricsFile := filepath.Join(t.TempDir(), "metrics.json")

	orig := captureCmdLevelVars()
	origResolvedCatalog := resolvedCatalogRoot
	defer func() {
		orig.restore()
		resolvedCatalogRoot = origResolvedCatalog
	}()

	setProvenanceRunVars(catalogDir, hwPath, defaultsPath)
	resolvedCatalogRoot = ""
	metricsPath = metricsFile
	traceOutput = tracePrefix
	workloadType = "distribution"
	rate = 1.0
	numRequests = provenanceNumRequests
	promptTokensMean, promptTokensStdev, promptTokensMin, promptTokensMax = 64, 8, 2, 256
	outputTokensMean, outputTokensStdev, outputTokensMin, outputTokensMax = 16, 4, 2, 64

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().IntVar(&numRequests, "num-requests", 0, "")
	testCmd.Flags().Float64Var(&rate, "rate", 0, "")
	testCmd.Flags().StringVar(&workloadType, "workload", "", "")
	testCmd.Flags().StringVar(&traceOutput, "trace-output", "", "")
	testCmd.Flags().StringVar(&metricsPath, "metrics-path", "", "")
	testCmd.Flags().IntVar(&requestTimeoutSecs, "timeout", 300, "")
	args := []string{
		"--model", provenanceTestModel,
		"--latency-model", "trained-physics",
		"--defaults-filepath", defaultsPath,
		"--catalog", catalogDir,
		"--hardware-config", hwPath,
		"--hardware", "H100",
		"--tp", "1",
		"--total-kv-blocks", "1000",
		"--num-requests", strconv.Itoa(provenanceNumRequests),
		"--seed", strconv.FormatInt(provenanceSeed, 10),
		"--rate", "1.0",
		"--workload", "distribution",
		"--horizon", strconv.FormatInt(provenanceHorizonUs, 10),
		"--metrics-path", metricsFile,
	}
	if tracePrefix != "" {
		args = append(args, "--trace-output", tracePrefix)
	}
	if err := testCmd.ParseFlags(args); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}

	stdout := captureRunStdout(t, func() { runCmd.Run(testCmd, nil) })
	return stdout, readMetricsFile(t, metricsFile)
}

// replayWithProvenance drives `blis replay --metrics-path` in-process over a trace
// exported by runWithProvenance, returning stdout plus the parsed results file.
func replayWithProvenance(t *testing.T, catalogDir, hwPath, defaultsPath, tracePrefix string) ([]byte, sim.MetricsOutput) {
	t.Helper()
	metricsFile := filepath.Join(t.TempDir(), "metrics.json")

	orig := captureCmdLevelVars()
	origTraceHeader, origTraceData := traceHeaderPath, traceDataPath
	origSessionMode, origThinkMs, origThinkDist := replaySessionMode, replayThinkTimeMs, replayThinkTimeDist
	origReplayTraceOut, origReplayMetrics := replayTraceOutput, replayMetricsPath
	origCacheSignalDelay, origFlowControl := cacheSignalDelay, flowControlEnabled
	origResolvedCatalog := resolvedCatalogRoot
	defer func() {
		orig.restore()
		traceHeaderPath, traceDataPath = origTraceHeader, origTraceData
		replaySessionMode, replayThinkTimeMs, replayThinkTimeDist = origSessionMode, origThinkMs, origThinkDist
		replayTraceOutput, replayMetricsPath = origReplayTraceOut, origReplayMetrics
		cacheSignalDelay, flowControlEnabled = origCacheSignalDelay, origFlowControl
		resolvedCatalogRoot = origResolvedCatalog
	}()

	headerFile, dataFile := tracePrefix+".yaml", tracePrefix+".csv"
	setProvenanceRunVars(catalogDir, hwPath, defaultsPath)
	resolvedCatalogRoot = ""
	traceOutput = ""
	traceHeaderPath = headerFile
	traceDataPath = dataFile
	replaySessionMode = "fixed"
	replayThinkTimeMs = 0
	replayThinkTimeDist = ""
	replayTraceOutput = ""
	replayMetricsPath = metricsFile
	cacheSignalDelay = 0
	flowControlEnabled = false

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
	testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
	testCmd.Flags().StringVar(&replayMetricsPath, "metrics-path", "", "")
	if err := testCmd.ParseFlags([]string{
		"--model", provenanceTestModel,
		"--latency-model", "trained-physics",
		"--defaults-filepath", defaultsPath,
		"--catalog", catalogDir,
		"--hardware-config", hwPath,
		"--hardware", "H100",
		"--tp", "1",
		"--total-kv-blocks", "1000",
		"--num-instances", "1",
		"--horizon", strconv.FormatInt(provenanceHorizonUs, 10),
		"--trace-header", headerFile,
		"--trace-data", dataFile,
		"--metrics-path", metricsFile,
	}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}

	stdout := captureRunStdout(t, func() { replayCmd.Run(testCmd, nil) })
	return stdout, readMetricsFile(t, metricsFile)
}

// assertProvenance checks a results file's catalog block against the expected catalog
// root, revision and dirty verdict.
func assertProvenance(t *testing.T, label string, out sim.MetricsOutput, wantPath, wantRevision string, wantDirty bool) {
	t.Helper()
	if out.Catalog == nil {
		t.Fatalf("%s: results file records no catalog provenance (AC-1)", label)
	}
	if out.Catalog.Path != wantPath {
		t.Errorf("%s: catalog path = %q, want %q", label, out.Catalog.Path, wantPath)
	}
	if out.Catalog.Revision != wantRevision {
		t.Errorf("%s: catalog revision = %q, want %q", label, out.Catalog.Revision, wantRevision)
	}
	if out.Catalog.Dirty != wantDirty {
		t.Errorf("%s: catalog dirty = %v, want %v", label, out.Catalog.Dirty, wantDirty)
	}
}

// TestCatalogProvenance_RunCLI_RecordsCatalogAndKeepsStdoutIdentical is the AC-1/AC-2/
// AC-3/AC-4 end-to-end contract for `blis run`: the results file names the catalog, its
// revision and its dirty state, the dirty flag flips when a catalog config.json is edited
// without committing, and stdout is byte-identical across the two runs (INV-6).
//
// NOTE: Do NOT use t.Parallel() — mutates package-level vars and os.Stdout.
func TestCatalogProvenance_RunCLI_RecordsCatalogAndKeepsStdoutIdentical(t *testing.T) {
	catalogRoot, hwPath, defaultsPath, head := setupGitCatalogFixtures(t)

	cleanStdout, cleanFile := runWithProvenance(t, catalogRoot, hwPath, defaultsPath, "")
	assertProvenance(t, "clean run", cleanFile, catalogRoot, head, false)

	dirtyCatalogEntry(t, catalogRoot)
	dirtyStdout, dirtyFile := runWithProvenance(t, catalogRoot, hwPath, defaultsPath, "")
	assertProvenance(t, "dirty run", dirtyFile, catalogRoot, head, true)

	// AC-4 / INV-6: the only difference between the two runs is catalog git state, which
	// is file-only metadata — stdout must not move. The fixture edit is whitespace-only,
	// so the simulation's inputs are identical and any stdout difference is the
	// provenance leaking.
	if !bytes.Equal(cleanStdout, dirtyStdout) {
		t.Errorf("INV-6 VIOLATION: `blis run` stdout differs between a clean and a dirty "+
			"catalog.\n--- clean ---\n%s\n--- dirty ---\n%s", cleanStdout, dirtyStdout)
	}
	if !bytes.Contains(cleanStdout, []byte("=== Simulation Metrics ===")) {
		t.Fatalf("non-vacuity: stdout carries no metrics block:\n%s", cleanStdout)
	}
	if bytes.Contains(cleanStdout, []byte(catalogRoot)) || bytes.Contains(cleanStdout, []byte(head)) {
		t.Errorf("INV-6 VIOLATION: provenance appears on stdout:\n%s", cleanStdout)
	}
	// Non-vacuity for the run itself: a run that completed nothing would make the
	// stdout comparison trivial.
	if cleanFile.CompletedRequests == 0 {
		t.Fatalf("non-vacuity: the run completed no requests, so stdout parity is trivial")
	}
}

// TestCatalogProvenance_ReplayCLI_RecordsCatalog is the INV-13 half: `blis replay`
// records the same provenance block through the same shared helper, including the dirty
// verdict, and its stdout is likewise unaffected.
//
// NOTE: Do NOT use t.Parallel() — mutates package-level vars and os.Stdout.
func TestCatalogProvenance_ReplayCLI_RecordsCatalog(t *testing.T) {
	catalogRoot, hwPath, defaultsPath, head := setupGitCatalogFixtures(t)
	tracePrefix := filepath.Join(t.TempDir(), "trace")

	// Export a trace from a clean-catalog run, then replay it.
	if _, runFile := runWithProvenance(t, catalogRoot, hwPath, defaultsPath, tracePrefix); runFile.Catalog == nil {
		t.Fatal("the exporting run must itself record provenance")
	}
	cleanStdout, cleanFile := replayWithProvenance(t, catalogRoot, hwPath, defaultsPath, tracePrefix)
	assertProvenance(t, "clean replay", cleanFile, catalogRoot, head, false)

	dirtyCatalogEntry(t, catalogRoot)
	dirtyStdout, dirtyFile := replayWithProvenance(t, catalogRoot, hwPath, defaultsPath, tracePrefix)
	assertProvenance(t, "dirty replay", dirtyFile, catalogRoot, head, true)

	if !bytes.Equal(cleanStdout, dirtyStdout) {
		t.Errorf("INV-6 VIOLATION: `blis replay` stdout differs between a clean and a dirty "+
			"catalog.\n--- clean ---\n%s\n--- dirty ---\n%s", cleanStdout, dirtyStdout)
	}
	if bytes.Contains(cleanStdout, []byte(catalogRoot)) || bytes.Contains(cleanStdout, []byte(head)) {
		t.Errorf("INV-6 VIOLATION: provenance appears on replay stdout:\n%s", cleanStdout)
	}
	if cleanFile.CompletedRequests == 0 {
		t.Fatalf("non-vacuity: the replay completed no requests")
	}
}

// TestCatalogProvenance_StdoutOnlyRun_NoProvenanceAnywhere pins that a run WITHOUT
// --metrics-path is byte-identical to a pre-#1732 build: no provenance on stdout, and no
// file written. The shared helper short-circuits before consulting git at all.
//
// NOTE: Do NOT use t.Parallel() — mutates package-level vars and os.Stdout.
func TestCatalogProvenance_StdoutOnlyRun_NoProvenanceAnywhere(t *testing.T) {
	catalogRoot, hwPath, defaultsPath, head := setupGitCatalogFixtures(t)

	orig := captureCmdLevelVars()
	origResolvedCatalog := resolvedCatalogRoot
	defer func() {
		orig.restore()
		resolvedCatalogRoot = origResolvedCatalog
	}()

	setProvenanceRunVars(catalogRoot, hwPath, defaultsPath)
	resolvedCatalogRoot = ""
	metricsPath = ""
	traceOutput = ""
	workloadType = "distribution"
	rate = 1.0
	numRequests = provenanceNumRequests
	promptTokensMean, promptTokensStdev, promptTokensMin, promptTokensMax = 64, 8, 2, 256
	outputTokensMean, outputTokensStdev, outputTokensMin, outputTokensMax = 16, 4, 2, 64

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().IntVar(&numRequests, "num-requests", 0, "")
	testCmd.Flags().Float64Var(&rate, "rate", 0, "")
	testCmd.Flags().StringVar(&workloadType, "workload", "", "")
	testCmd.Flags().StringVar(&metricsPath, "metrics-path", "", "")
	testCmd.Flags().IntVar(&requestTimeoutSecs, "timeout", 300, "")
	if err := testCmd.ParseFlags([]string{
		"--model", provenanceTestModel,
		"--latency-model", "trained-physics",
		"--defaults-filepath", defaultsPath,
		"--catalog", catalogRoot,
		"--hardware-config", hwPath,
		"--hardware", "H100",
		"--tp", "1",
		"--total-kv-blocks", "1000",
		"--num-requests", strconv.Itoa(provenanceNumRequests),
		"--seed", strconv.FormatInt(provenanceSeed, 10),
		"--rate", "1.0",
		"--workload", "distribution",
		"--horizon", strconv.FormatInt(provenanceHorizonUs, 10),
	}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}

	stdout := captureRunStdout(t, func() { runCmd.Run(testCmd, nil) })
	for _, forbidden := range []string{catalogRoot, head, `"catalog"`, `"revision"`, `"dirty"`} {
		if strings.Contains(string(stdout), forbidden) {
			t.Errorf("INV-6 VIOLATION: stdout-only run leaked provenance fragment %q:\n%s",
				forbidden, stdout)
		}
	}
}
