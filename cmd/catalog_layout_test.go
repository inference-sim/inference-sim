package cmd

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// CLI-level contract tests for #1774: --catalog / BLIS_CATALOG names the catalog CLONE
// ROOT, so a real `blis run` and `blis replay` read the model config from
// <catalog>/models/<short-name>/config.json — and the transition fallback keeps the flat
// <catalog>/<short-name> layout (the bundled model_configs/ tree) working unchanged.
//
// The law these pin is that the LAYOUT is not an input to the simulation:
//
//	INV-6  — the same config.json bytes reached through either layout produce
//	         byte-identical stdout.
//	INV-13 — run and replay share resolveModelConfig, so both resolve the clone-root
//	         layout and both land on the same numbers as the flat layout.
//
// The unit-level layout laws (candidate order, malformed-entry boundary, relative vs
// absolute paths) live on catalogModelDirs / resolveModelConfigInCatalog in
// hfconfig_test.go; these tests prove the contract is WIRED INTO the commands.

// Environment variables driving the re-exec subprocess legs. A leg runs the real cobra
// tree so a logrus.Fatalf surfaces as a non-zero exit status.
const (
	catalogLayoutLegEnv     = "BLIS_CATALOG_LAYOUT_LEG"
	catalogLayoutCatalogEnv = "BLIS_CATALOG_LAYOUT_CATALOG"
	catalogLayoutTraceEnv   = "BLIS_CATALOG_LAYOUT_TRACE"
)

// catalogLayoutModel is a model catalogued in the repository's bundled model_configs/
// tree, which is FLAT — so it is reachable directly (fallback layout) and, once mirrored
// by newCloneRootCatalog, under a models/ namespace (canonical layout).
const catalogLayoutModel = "qwen/qwen3-14b"

// newCloneRootCatalog builds a catalog CLONE ROOT whose models/ namespace holds a
// byte-for-byte copy of the bundled entry for catalogLayoutModel, and returns its root.
// The copy (rather than a symlink) makes the "same config.json bytes" premise of the
// byte-identity comparison explicit and independent of symlink support.
func newCloneRootCatalog(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	shortName := catalogLayoutModel[strings.Index(catalogLayoutModel, "/")+1:]

	src := filepath.Join("..", "model_configs", shortName, hfConfigFile)
	content, err := os.ReadFile(src)
	if err != nil {
		t.Fatalf("read bundled catalog entry %s: %v", src, err)
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

// TestRunCmd_CatalogCloneRootLayout_ByteIdenticalToFlat is the #1774 INV-6 contract at
// the `blis run` boundary: the SAME config.json reached through the canonical clone-root
// layout (<catalog>/models/<name>) and through the flat transition layout
// (<catalog>/<name>, the bundled model_configs/ tree) must produce byte-identical stdout.
// Where the entry sits inside the catalog is not an input to the simulation.
func TestRunCmd_CatalogCloneRootLayout_ByteIdenticalToFlat(t *testing.T) {
	if catalogLayoutSubprocess() {
		return
	}
	const name = "TestRunCmd_CatalogCloneRootLayout_ByteIdenticalToFlat"

	viaClone := runCatalogLayoutLeg(t, name, "run", newCloneRootCatalog(t), "")
	viaFlat := runCatalogLayoutLeg(t, name, "run", filepath.Join("..", "model_configs"), "")

	// Non-vacuity: an empty or metric-less stdout would make the comparison trivial, and
	// would also hide a clone-root leg that silently simulated nothing.
	if !strings.Contains(viaClone, "completed_requests") {
		t.Fatalf("non-vacuity: clone-root leg produced no metrics:\n%s", viaClone)
	}
	if viaClone != viaFlat {
		t.Errorf("stdout must be byte-identical whichever catalog layout holds the entry (INV-6)\nclone-root:\n%s\nflat:\n%s",
			viaClone, viaFlat)
	}
}

// TestReplayCmd_CatalogCloneRootLayout_ByteIdenticalToFlat is the INV-13 half: `blis
// replay` shares resolveModelConfig with `blis run`, so it resolves the clone-root layout
// too and lands on the same numbers as the flat layout over the same trace.
//
// The comparison is replay-vs-replay on one trace, so no horizon has to be pinned across
// commands — both legs differ only in the catalog layout.
func TestReplayCmd_CatalogCloneRootLayout_ByteIdenticalToFlat(t *testing.T) {
	if catalogLayoutSubprocess() {
		return
	}
	const name = "TestReplayCmd_CatalogCloneRootLayout_ByteIdenticalToFlat"

	// Export a trace once, through the flat layout, so both replay legs read identical
	// input and the only difference between them is where the model config was found.
	tracePrefix := filepath.Join(t.TempDir(), "layout")
	runCatalogLayoutLeg(t, name, "run-export", filepath.Join("..", "model_configs"), tracePrefix)
	if _, err := os.Stat(tracePrefix + ".csv"); err != nil {
		t.Fatalf("trace export produced no data file: %v", err)
	}

	viaClone := runCatalogLayoutLeg(t, name, "replay", newCloneRootCatalog(t), tracePrefix)
	viaFlat := runCatalogLayoutLeg(t, name, "replay", filepath.Join("..", "model_configs"), tracePrefix)

	if !strings.Contains(viaClone, "completed_requests") {
		t.Fatalf("non-vacuity: clone-root replay leg produced no metrics:\n%s", viaClone)
	}
	if viaClone != viaFlat {
		t.Errorf("replay stdout must be byte-identical whichever catalog layout holds the entry (INV-13)\nclone-root:\n%s\nflat:\n%s",
			viaClone, viaFlat)
	}
}

// TestRunCmd_CatalogCloneRootLayout_UncataloguedModelStillRefused guards the fallback
// from becoming a way IN: adding the models/ candidate must not make an uncatalogued
// model resolve. A run naming a model absent from both layouts is still refused, and the
// refusal names the canonical clone-root path the entry belongs at (NS-6, #1733).
func TestRunCmd_CatalogCloneRootLayout_UncataloguedModelStillRefused(t *testing.T) {
	root := t.TempDir()
	if _, err := resolveModelConfigInCatalog("test-org/not-catalogued", root); err == nil {
		t.Fatal("expected refusal for a model absent from both layouts")
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
