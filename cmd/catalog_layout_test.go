package cmd

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// completedRequestsRe extracts the completed_requests count from a metrics-JSON stdout.
var completedRequestsRe = regexp.MustCompile(`"completed_requests":\s*(\d+)`)

// requireCompletedRequests asserts the run's stdout is a metrics JSON whose
// completed_requests is > 0 — a non-vacuity gate stronger than a bare substring check
// (which `"completed_requests": 0` would satisfy), so a leg that resolved the model but
// simulated nothing cannot pass as a successful run.
func requireCompletedRequests(t *testing.T, label, stdout string) {
	t.Helper()
	m := completedRequestsRe.FindStringSubmatch(stdout)
	if m == nil {
		t.Fatalf("non-vacuity: %s produced no completed_requests metric:\n%s", label, stdout)
	}
	if n, err := strconv.Atoi(m[1]); err != nil || n <= 0 {
		t.Fatalf("non-vacuity: %s completed %s requests (want > 0):\n%s", label, m[1], stdout)
	}
}

// CLI-level contract tests for the catalog CLONE ROOT layout: --catalog / BLIS_CATALOG
// names the clone root, so a real `blis run` and `blis replay` read the model config from
// <catalog>/models/<short-name>/config.json.
//
// #1774 introduced this layout alongside a transition fallback that also accepted the flat
// <catalog>/<short-name> layout (the then-bundled model_configs/ tree). #1771 deleted that
// bundled tree and the fallback, so there is now exactly ONE layout — the models/ namespace
// — and the "layout is not an input to the simulation" law #1774 pinned across the two
// layouts is no longer expressible (there is nothing to compare against). What remains, and
// what these tests pin, is that the canonical layout is WIRED INTO the commands and that the
// removed fallback did not become a way IN for an uncatalogued model:
//
//	INV-6  — the clone-root layout resolves and runs deterministically (byte-identical
//	         stdout across repeated runs of the same catalog).
//	INV-13 — run and replay share resolveModelConfig, so both resolve the clone-root layout.
//	NS-6   — a model absent from the (sole) layout is refused, naming the canonical path.
//
// The unit-level layout laws (candidate derivation, malformed-entry boundary, relative vs
// absolute paths, uncatalogued refusal) live on catalogModelDirs / resolveModelConfigInCatalog
// in hfconfig_test.go; these tests prove the contract is wired into the commands.

// Environment variables driving the re-exec subprocess legs. A leg runs the real cobra
// tree so a logrus.Fatalf surfaces as a non-zero exit status.
const (
	catalogLayoutLegEnv     = "BLIS_CATALOG_LAYOUT_LEG"
	catalogLayoutCatalogEnv = "BLIS_CATALOG_LAYOUT_CATALOG"
	catalogLayoutTraceEnv   = "BLIS_CATALOG_LAYOUT_TRACE"
)

// catalogLayoutModel is a model catalogued in the committed test catalog
// testdata/catalog/models/, so newCloneRootCatalog can copy its entry.
const catalogLayoutModel = "qwen/qwen3-14b"

// newCloneRootCatalog builds a catalog CLONE ROOT whose models/ namespace holds a
// byte-for-byte copy of the test-catalog entry for catalogLayoutModel, and returns its
// root. The copy (rather than a symlink) makes the "same config.json bytes" premise of the
// determinism comparison explicit and independent of symlink support.
func newCloneRootCatalog(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	shortName := catalogLayoutModel[strings.Index(catalogLayoutModel, "/")+1:]

	src := filepath.Join("..", "testdata", "catalog", "models", shortName, hfConfigFile)
	content, err := os.ReadFile(src)
	if err != nil {
		t.Fatalf("read test catalog entry %s: %v", src, err)
	}
	entryDir := filepath.Join(root, catalogModelsSubdir, shortName)
	if err := os.MkdirAll(entryDir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", entryDir, err)
	}
	if err := os.WriteFile(filepath.Join(entryDir, hfConfigFile), content, 0o644); err != nil {
		t.Fatalf("write clone-root catalog entry: %v", err)
	}
	return root
}

// runCatalogLayoutLeg re-execs this test binary as the named leg with the given catalog
// root and trace prefix, returning its stdout. Failure is fatal to the calling test: an
// empty or partial stdout must never be compared as if it were a result.
func runCatalogLayoutLeg(t *testing.T, testName, leg, catalog, tracePrefix string) string {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^"+testName+"$")
	cmd.Env = append(os.Environ(),
		catalogLayoutLegEnv+"="+leg,
		catalogLayoutCatalogEnv+"="+catalog,
		catalogLayoutTraceEnv+"="+tracePrefix,
		// Neutralize any ambient BLIS_CATALOG so a leg cannot pass via the environment
		// instead of the catalog it was handed.
		catalogEnvVar+"=",
	)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("leg %q (catalog %s) failed: %v\nstdout:\n%s\nstderr:\n%s",
			leg, catalog, err, stdout.String(), stderr.String())
	}
	return stdout.String()
}

// catalogLayoutSubprocess executes the leg named by the environment, if any, and reports
// whether it did (in which case the parent test body must not run). Legs terminate the
// process, so this never returns true.
func catalogLayoutSubprocess() bool {
	leg := os.Getenv(catalogLayoutLegEnv)
	if leg == "" {
		return false
	}
	catalog := os.Getenv(catalogLayoutCatalogEnv)
	tracePrefix := os.Getenv(catalogLayoutTraceEnv)

	var args []string
	switch leg {
	case "run", "run-export":
		args = []string{
			"run", "--model", catalogLayoutModel,
			"--hardware", "H100", "--tp", "1",
			"--seed", "42", "--num-requests", "20",
			"--defaults-filepath", "../defaults.yaml",
			"--catalog", catalog,
		}
		if leg == "run-export" {
			args = append(args, "--trace-output", tracePrefix)
		}
	case "replay":
		args = []string{
			"replay",
			"--trace-header", tracePrefix + ".yaml",
			"--trace-data", tracePrefix + ".csv",
			"--model", catalogLayoutModel,
			"--hardware", "H100", "--tp", "1",
			"--seed", "42",
			"--defaults-filepath", "../defaults.yaml",
			"--catalog", catalog,
		}
	default:
		os.Exit(2)
	}

	rootCmd.SetArgs(args)
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
	os.Exit(0)
	return true
}

// TestRunCmd_CatalogCloneRootLayout_ResolvesAndRuns is the #1771 INV-6 contract at the
// `blis run` boundary: a model config stored the way the authoritative blis-catalog
// repository stores it — <catalog>/models/<name>/config.json — resolves and runs, and the
// same catalog produces byte-identical stdout across runs (the layout is not a source of
// nondeterminism). With the flat transition fallback gone, this is now the only layout.
func TestRunCmd_CatalogCloneRootLayout_ResolvesAndRuns(t *testing.T) {
	if catalogLayoutSubprocess() {
		return
	}
	const name = "TestRunCmd_CatalogCloneRootLayout_ResolvesAndRuns"

	catalog := newCloneRootCatalog(t)
	first := runCatalogLayoutLeg(t, name, "run", catalog, "")
	second := runCatalogLayoutLeg(t, name, "run", catalog, "")

	// Non-vacuity: the leg must have actually simulated the 20 requests, not merely emitted
	// a metrics JSON — a run that resolved the model but completed nothing would otherwise
	// pass on the substring alone.
	requireCompletedRequests(t, "clone-root run leg", first)
	if first != second {
		t.Errorf("stdout must be byte-identical across runs of the clone-root layout (INV-6)\nfirst:\n%s\nsecond:\n%s",
			first, second)
	}
}

// TestReplayCmd_CatalogCloneRootLayout_ResolvesAndRuns is the INV-13 half: `blis replay`
// shares resolveModelConfig with `blis run`, so it resolves the clone-root layout too. A
// trace exported through the clone-root catalog replays deterministically through it.
func TestReplayCmd_CatalogCloneRootLayout_ResolvesAndRuns(t *testing.T) {
	if catalogLayoutSubprocess() {
		return
	}
	const name = "TestReplayCmd_CatalogCloneRootLayout_ResolvesAndRuns"

	catalog := newCloneRootCatalog(t)
	tracePrefix := filepath.Join(t.TempDir(), "layout")
	runCatalogLayoutLeg(t, name, "run-export", catalog, tracePrefix)
	if _, err := os.Stat(tracePrefix + ".csv"); err != nil {
		t.Fatalf("trace export produced no data file: %v", err)
	}

	first := runCatalogLayoutLeg(t, name, "replay", catalog, tracePrefix)
	second := runCatalogLayoutLeg(t, name, "replay", catalog, tracePrefix)

	requireCompletedRequests(t, "clone-root replay leg", first)
	if first != second {
		t.Errorf("replay stdout must be byte-identical across runs of the clone-root layout (INV-13)\nfirst:\n%s\nsecond:\n%s",
			first, second)
	}
}

// TestRunCmd_CatalogCloneRootLayout_UncataloguedModelStillRefused guards the sole layout
// from becoming a way IN: a model absent from <catalog>/models/ must not resolve. A run
// naming such a model is refused, and the refusal names the canonical clone-root path the
// entry belongs at (NS-6, #1733). #1771 removed the flat fallback, so the only path an
// entry can live at is the one the refusal names.
func TestRunCmd_CatalogCloneRootLayout_UncataloguedModelStillRefused(t *testing.T) {
	root := t.TempDir()
	if _, err := resolveModelConfigInCatalog("test-org/not-catalogued", root); err == nil {
		t.Fatal("expected refusal for a model absent from the catalog")
	} else if want := filepath.Join(root, catalogModelsSubdir, "not-catalogued", hfConfigFile); !strings.Contains(err.Error(), want) {
		t.Errorf("refusal must name the canonical clone-root path (%s), got: %v", want, err)
	}

	// And nothing may be created under the catalog root by the attempt.
	entries, err := os.ReadDir(root)
	if err != nil {
		t.Fatalf("read catalog root: %v", err)
	}
	if len(entries) != 0 {
		t.Errorf("resolution must not create anything under the catalog root; found %v", entries)
	}
}
