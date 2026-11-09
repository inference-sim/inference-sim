package cmd

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/inference-sim/blis-substrate/contracts/spec/result"
)

// End-to-end data-record tests at the CLI seam (design §3.3, X8): `blis run` and `blis replay`
// record the catalog's source and content hash in the Result's data record and in the
// --metrics-path file's catalog block, the same values through one helper (INV-13), and a catalog
// whose bytes changed but whose content did not (a trailing newline) changes the content hash and
// nothing else in the Result (INV-6).
//
// NOTE: these tests mutate package-level CLI vars and os.Stdout — do NOT use
// t.Parallel(). Every mutated var is restored.

// Workload sizing for the runs below. These are constants rather than reads of the
// package-level CLI vars because registerSimConfigFlags RESETS every bound var to its
// flag default at registration time — a value read after registration would be the
// default, not what the test set.
const (
	provenanceNumRequests = 8
	provenanceHorizonUs   = int64(120_000_000)
	provenanceSeed        = int64(4242)
)

// setupCatalogFixtures builds the fixture set the data-record tests need: a copy of the test
// catalog in a temporary directory, read as a path, plus the repository's defaults.yaml.
func setupCatalogFixtures(t *testing.T) (catalogRoot, defaultsPath string) {
	t.Helper()
	catalogRoot = copyKernelCatalog(t)
	defaultsPath, err := filepath.Abs("../defaults.yaml")
	if err != nil {
		t.Fatal(err)
	}
	return catalogRoot, defaultsPath
}

// editCatalogEntry changes the bytes of the model graph the kernel scenario reads WITHOUT
// changing what it parses to: it appends a newline, which YAML ignores. The content hash covers
// bytes, so it changes while the simulation's inputs, and therefore every simulated figure, do
// not. That is what makes the INV-6 assertion non-vacuous.
func editCatalogEntry(t *testing.T, catalogRoot string) {
	t.Helper()
	graph := filepath.Join(catalogRoot, catalogModelsSubdir, "gpt-oss-120b", "graph.yaml")
	data, err := os.ReadFile(graph)
	if err != nil {
		t.Fatalf("read catalog entry %s: %v", graph, err)
	}
	if err := os.WriteFile(graph, append(data, '\n'), 0o644); err != nil {
		t.Fatalf("edit catalog entry %s: %v", graph, err)
	}
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
func setProvenanceRunVars(catalogDir, defaultsPath string) {
	catalogPath = catalogDir
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
	scheduler = "fcfs"
	policyConfigPath = ""
	maxModelLen = 0
	traceLevel = "none"
	counterfactualK = 0
	workloadSpecPath = ""
	lazyGeneration = false
	requestTimeoutSecs = 300
}

// runWithProvenance drives `blis run` in-process over a tiny distribution workload, optionally
// exporting a trace, and returns stdout.
func runWithProvenance(t *testing.T, catalogDir, defaultsPath, tracePrefix string) []byte {
	t.Helper()

	orig := captureCmdLevelVars()
	origData := openedData
	defer func() {
		orig.restore()
		openedData = origData
	}()

	setProvenanceRunVars(catalogDir, defaultsPath)
	openedData = nil
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
	testCmd.Flags().IntVar(&requestTimeoutSecs, "timeout", 300, "")
	args := []string{
		"--defaults-filepath", defaultsPath,
		"--catalog", catalogDir, "--scenario", kernelTestScenario, "--scenarios", kernelScenariosDir(t), "--registry", kernelRegistryDir(t),
		"--num-requests", strconv.Itoa(provenanceNumRequests),
		"--seed", strconv.FormatInt(provenanceSeed, 10),
		"--rate", "1.0",
		"--workload", "distribution",
		"--horizon", strconv.FormatInt(provenanceHorizonUs, 10),
	}
	if tracePrefix != "" {
		args = append(args, "--trace-output", tracePrefix)
	}
	if err := testCmd.ParseFlags(args); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}

	stdout := captureRunStdout(t, func() { runCmd.Run(testCmd, nil) })
	return stdout
}

// replayWithProvenance drives `blis replay` in-process over a trace exported by
// runWithProvenance, returning stdout.
func replayWithProvenance(t *testing.T, catalogDir, defaultsPath, tracePrefix string) []byte {
	t.Helper()

	orig := captureCmdLevelVars()
	origTraceHeader, origTraceData := traceHeaderPath, traceDataPath
	origSessionMode, origThinkMs, origThinkDist := replaySessionMode, replayThinkTimeMs, replayThinkTimeDist
	origReplayTraceOut := replayTraceOutput
	origData := openedData
	defer func() {
		orig.restore()
		traceHeaderPath, traceDataPath = origTraceHeader, origTraceData
		replaySessionMode, replayThinkTimeMs, replayThinkTimeDist = origSessionMode, origThinkMs, origThinkDist
		replayTraceOutput = origReplayTraceOut
		openedData = origData
	}()

	headerFile, dataFile := tracePrefix+".yaml", tracePrefix+".csv"
	setProvenanceRunVars(catalogDir, defaultsPath)
	openedData = nil
	traceOutput = ""
	traceHeaderPath = headerFile
	traceDataPath = dataFile
	replaySessionMode = "fixed"
	replayThinkTimeMs = 0
	replayThinkTimeDist = ""
	replayTraceOutput = ""

	testCmd := &cobra.Command{}
	registerSimConfigFlags(testCmd)
	testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
	testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
	if err := testCmd.ParseFlags([]string{
		"--defaults-filepath", defaultsPath,
		"--catalog", catalogDir, "--scenario", kernelTestScenario, "--scenarios", kernelScenariosDir(t), "--registry", kernelRegistryDir(t),
		"--num-instances", "1",
		"--horizon", strconv.FormatInt(provenanceHorizonUs, 10),
		"--trace-header", headerFile,
		"--trace-data", dataFile,
	}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}

	stdout := captureRunStdout(t, func() { replayCmd.Run(testCmd, nil) })
	return stdout
}

// assertProvenance checks that a run recorded its catalog in the Result's data record: a path
// source and the combined content hash. It returns the hash.
func assertProvenance(t *testing.T, label string, stdout []byte) string {
	t.Helper()
	d := parseResult(t, string(stdout)).Data.Catalog
	if d.Source != result.SourcePath || d.Hash == "" || d.CombinedHash != d.Hash {
		t.Errorf("%s: data.catalog = %+v, want a path source with its hash", label, d)
	}
	return d.CombinedHash
}

// TestCatalogProvenance_RunCLI_RecordsCatalogAndKeepsStdoutIdentical: `blis run` records the
// catalog's source and content hash, an edit to a catalog file's bytes changes the hash, and
// the Result is otherwise byte-identical (INV-6).
//
// NOTE: Do NOT use t.Parallel() — mutates package-level vars and os.Stdout.
func TestCatalogProvenance_RunCLI_RecordsCatalogAndKeepsStdoutIdentical(t *testing.T) {
	catalogRoot, defaultsPath := setupCatalogFixtures(t)

	beforeStdout := runWithProvenance(t, catalogRoot, defaultsPath, "")
	before := assertProvenance(t, "run", beforeStdout)

	// One process opens each catalog path once (kernelmodel.OpenData caches by path), as a CLI
	// invocation does, so the edited catalog is a second directory with the same content.
	edited := filepath.Join(t.TempDir(), "catalog")
	if err := os.CopyFS(edited, os.DirFS(catalogRoot)); err != nil {
		t.Fatalf("copy the catalog: %v", err)
	}
	editCatalogEntry(t, edited)
	afterStdout := runWithProvenance(t, edited, defaultsPath, "")
	after := assertProvenance(t, "run after the edit", afterStdout)
	if before == after {
		t.Errorf("the catalog's bytes changed but its content hash did not: %s", before)
	}
	if a, b := withoutRevisionsAndRuntime(t, string(beforeStdout)), withoutRevisionsAndRuntime(t, string(afterStdout)); a != b {
		t.Errorf("INV-6 VIOLATION: `blis run` Result differs after a content-neutral catalog edit "+
			"outside the data record and runtime:\n%s", firstDiff(a, b))
	}
	if bytes.Contains(beforeStdout, []byte(catalogRoot)) || bytes.Contains(afterStdout, []byte(edited)) {
		t.Errorf("the Result names the catalog's location, a fact of the host:\n%s", beforeStdout)
	}
	if parseResult(t, string(beforeStdout)).Requests.Completed == 0 {
		t.Fatalf("non-vacuity: the run completed no requests, so stdout parity is trivial")
	}
}

// TestCatalogProvenance_ReplayCLI_RecordsCatalog is the INV-13 half: `blis replay` records the
// same data record through the same helper.
//
// NOTE: Do NOT use t.Parallel() — mutates package-level vars and os.Stdout.
func TestCatalogProvenance_ReplayCLI_RecordsCatalog(t *testing.T) {
	catalogRoot, defaultsPath := setupCatalogFixtures(t)
	tracePrefix := filepath.Join(t.TempDir(), "trace")

	runStdout := runWithProvenance(t, catalogRoot, defaultsPath, tracePrefix)
	want := assertProvenance(t, "exporting run", runStdout)
	stdout := replayWithProvenance(t, catalogRoot, defaultsPath, tracePrefix)
	if got := assertProvenance(t, "replay", stdout); got != want {
		t.Errorf("the replay records catalog hash %s, the run %s: the same data", got, want)
	}
	if bytes.Contains(stdout, []byte(catalogRoot)) {
		t.Errorf("the replay's Result names the catalog's location, a fact of the host:\n%s", stdout)
	}
	if parseResult(t, string(stdout)).Requests.Completed == 0 {
		t.Fatalf("non-vacuity: the replay completed no requests")
	}
}

// TestCatalogProvenance_StdoutOnlyRun_NoProvenanceAnywhere pins that a run's Result never names the catalog's
// location: the catalog is recorded by its content hash (data.catalog), never by the path it
// was found at, a fact of the host.
//
// NOTE: Do NOT use t.Parallel() — mutates package-level vars and os.Stdout.
func TestCatalogProvenance_StdoutOnlyRun_NoProvenanceAnywhere(t *testing.T) {
	catalogRoot, defaultsPath := setupCatalogFixtures(t)

	orig := captureCmdLevelVars()
	origData := openedData
	defer func() {
		orig.restore()
		openedData = origData
	}()

	setProvenanceRunVars(catalogRoot, defaultsPath)
	openedData = nil
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
	testCmd.Flags().IntVar(&requestTimeoutSecs, "timeout", 300, "")
	if err := testCmd.ParseFlags([]string{
		"--defaults-filepath", defaultsPath,
		"--catalog", catalogRoot, "--scenario", kernelTestScenario, "--scenarios", kernelScenariosDir(t), "--registry", kernelRegistryDir(t),
		"--num-requests", strconv.Itoa(provenanceNumRequests),
		"--seed", strconv.FormatInt(provenanceSeed, 10),
		"--rate", "1.0",
		"--workload", "distribution",
		"--horizon", strconv.FormatInt(provenanceHorizonUs, 10),
	}); err != nil {
		t.Fatalf("ParseFlags: %v", err)
	}

	stdout := captureRunStdout(t, func() { runCmd.Run(testCmd, nil) })
	if strings.Contains(string(stdout), catalogRoot) {
		t.Errorf("the Result names the catalog's location %q, a fact of the host:\n%s", catalogRoot, stdout)
	}
	if r := parseResult(t, string(stdout)); len(r.Artifacts) != 0 {
		t.Errorf("a run without --metrics-path references artifacts %v", r.Artifacts)
	}
}
