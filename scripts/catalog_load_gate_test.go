package scripts_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// catalog-load-gate.sh is the R1/C6 acceptance gate (#1750): it loads every entry in the
// authoritative catalog through the production loader and fails on any rejection. Its failure
// mode is the reason it is a script rather than an inline workflow step — the test it runs
// SKIPS when BLIS_CATALOG never reaches it, and `go test` reports a skip as a PASS, so a
// broken checkout would otherwise turn the gate green having loaded nothing.

const catalogGateScript = "catalog-load-gate.sh"

// TestCatalogLoadGate_PinsTheCatalogRevision pins the properties a reader of the script cannot
// get from its exit code: the catalog is a PINNED 40-hex revision of the authoritative
// repository, and the run is guarded against a silent skip.
//
// A floating ref (a branch or tag) would make a past green run unreproducible and let an
// unrelated catalog commit redden an unrelated PR, which is the opposite of what a gate is for.
func TestCatalogLoadGate_PinsTheCatalogRevision(t *testing.T) {
	body, err := os.ReadFile(catalogGateScript)
	if err != nil {
		t.Fatalf("read %s: %v", catalogGateScript, err)
	}
	source := string(body)

	for _, want := range []struct{ fragment, why string }{
		{"inference-sim/blis-catalog", "the gate must load the authoritative catalog"},
		{"TestCatalogStrictLoad_RealCatalog", "the gate must run the whole-catalog load"},
		// Both guards are ANCHORED to the top-level test name. Unanchored, `--- SKIP` also
		// matches a skipped SUBTEST (go test indents those) or any line quoting the string, and
		// would fail the gate on a load that ran fine. The PASS half catches the opposite
		// failure: a `-run` pattern that matched no test exits 0 with no SKIP line at all.
		{`'^--- SKIP: TestCatalogStrictLoad_RealCatalog`, "a skipped load must fail the gate; go test reports a skip as a pass"},
		{`'^--- PASS: TestCatalogStrictLoad_RealCatalog`, "the load must be proven to have RUN, not merely not-failed"},
		{"set -euo pipefail", "a failing step must not be swallowed"},
	} {
		if !strings.Contains(source, want.fragment) {
			t.Errorf("%s no longer contains %q — %s", catalogGateScript, want.fragment, want.why)
		}
	}

	revision := defaultShellAssignment(t, source, "CATALOG_REVISION")
	if len(revision) != 40 || strings.Trim(revision, "0123456789abcdef") != "" {
		t.Errorf("%s: CATALOG_REVISION must default to a 40-hex commit SHA, got %q — a floating ref "+
			"makes a green run unreproducible", catalogGateScript, revision)
	}
}

// defaultShellAssignment extracts the default of a `NAME="${NAME:-default}"` assignment.
func defaultShellAssignment(t *testing.T, source, name string) string {
	t.Helper()
	marker := name + `="${` + name + `:-`
	idx := strings.Index(source, marker)
	if idx < 0 {
		t.Fatalf("%s: no overridable %s assignment found", catalogGateScript, name)
	}
	rest := source[idx+len(marker):]
	end := strings.Index(rest, `}"`)
	if end < 0 {
		t.Fatalf("%s: malformed %s assignment", catalogGateScript, name)
	}
	return rest[:end]
}

// TestCatalogLoadGate_PassesOnACleanCatalogAndFailsOnABrokenOne is the behavioural half: the
// script must actually clone the revision it is pointed at, load it, and let the load's verdict
// decide its exit code. Both directions are asserted — a gate that always passed and a gate
// that always failed would each satisfy one of these alone.
//
// The catalog is a throwaway local git repository rather than the network, so the test is
// offline and its revision is a real SHA the script must check out.
func TestCatalogLoadGate_PassesOnACleanCatalogAndFailsOnABrokenOne(t *testing.T) {
	requireGit(t)

	t.Run("clean catalog passes", func(t *testing.T) {
		repo, revision := newCatalogFixtureRepo(t, false)
		out, err := runCatalogGate(t, repo, revision)
		if err != nil {
			t.Fatalf("the gate rejected a clean catalog: %v\n%s", err, out)
		}
		if !strings.Contains(out, "every catalog entry loaded") {
			t.Errorf("expected the success line; got:\n%s", out)
		}
	})

	t.Run("catalog stating a deployment fact fails", func(t *testing.T) {
		repo, revision := newCatalogFixtureRepo(t, true)
		out, err := runCatalogGate(t, repo, revision)
		if err == nil {
			t.Fatalf("the gate accepted a catalog that states a deployment fact:\n%s", out)
		}
		if !strings.Contains(out, "model.yaml") {
			t.Errorf("the failure must name the offending file; got:\n%s", out)
		}
	})
}

// TestCatalogLoadGate_UsesAnExistingCheckout pins that BLIS_CATALOG short-circuits the clone —
// the local-development path, and what a workflow that already checked the catalog out would
// use. Proven by pointing CATALOG_REPO at a path that cannot be cloned: if the script tried, it
// would fail there instead of loading.
func TestCatalogLoadGate_UsesAnExistingCheckout(t *testing.T) {
	requireGit(t)
	repo, _ := newCatalogFixtureRepo(t, false)

	cmd := exec.Command("bash", scriptPath(t, catalogGateScript))
	cmd.Env = append(os.Environ(),
		"BLIS_CATALOG="+repo,
		"CATALOG_REPO="+filepath.Join(t.TempDir(), "definitely-not-a-git-repo"),
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("the gate must load an already-checked-out catalog without cloning: %v\n%s", err, out)
	}
	if strings.Contains(string(out), "cloning") {
		t.Errorf("BLIS_CATALOG was set, so nothing should have been cloned; got:\n%s", out)
	}
}

// runCatalogGate runs the script against a local catalog repository at the given revision.
func runCatalogGate(t *testing.T, repo, revision string) (string, error) {
	t.Helper()
	cmd := exec.Command("bash", scriptPath(t, catalogGateScript))
	cmd.Env = append(os.Environ(),
		"CATALOG_REPO="+repo,
		"CATALOG_REVISION="+revision,
		"CATALOG_CHECKOUT="+filepath.Join(t.TempDir(), "checkout"),
		// BLIS_CATALOG must not leak in from the environment, or the clone path under test
		// would be skipped.
		"BLIS_CATALOG=",
	)
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// newCatalogFixtureRepo commits a minimal clone-root-shaped catalog into a throwaway git
// repository and returns its path and HEAD revision. With stateDeploymentFact set, the model
// entry states a GPU — the rule whose violation the gate must report.
func newCatalogFixtureRepo(t *testing.T, stateDeploymentFact bool) (string, string) {
	t.Helper()
	repo := t.TempDir()

	// A real vendor config.json from the committed fixture catalog: the gate parses it through
	// the production model-config path, so a hand-written stub would not survive.
	config, err := os.ReadFile(filepath.Join("..", "testdata", "catalog", "models", "qwen3-14b", "config.json"))
	if err != nil {
		t.Fatalf("read fixture config.json: %v", err)
	}
	modelEntry := `name: qwen3-14b
source:
  provider: huggingface
  repo: Qwen/Qwen3-14B
  revision: 0000000000000000000000000000000000000000
  retrieved: 2026-09-16
`
	if stateDeploymentFact {
		modelEntry += "gpu: H100\n"
	}
	writeFixtureFile(t, filepath.Join(repo, "models", "qwen3-14b", "config.json"), string(config))
	writeFixtureFile(t, filepath.Join(repo, "models", "qwen3-14b", "model.yaml"), modelEntry)
	writeFixtureFile(t, filepath.Join(repo, "workloads", "chatbot.yaml"), `prefix_tokens: 0
prompt_tokens: 256
prompt_tokens_stdev: 100
prompt_tokens_min: 2
prompt_tokens_max: 800
output_tokens: 256
output_tokens_stdev: 100
output_tokens_min: 1
output_tokens_max: 1024
`)
	writeFixtureFile(t, filepath.Join(repo, "devices", "storage.yaml"),
		"cpu_dram: {read_bandwidth: 2.0e4, write_bandwidth: 2.0e4, base_latency: 1.0}\n")
	writeFixtureFile(t, filepath.Join(repo, "hardware", "h100.yaml"), `TFlopsPeak: 989.5
TFlopsFP8: 1979.0
BwPeakTBs: 3.35
mfuPrefill: 0.45
mfuDecode: 0.30
MemoryGiB: 80.0
IntraNodeBwGBps: 450
InterNodeBwGBps: 50
`)

	for _, args := range [][]string{
		{"init", "--quiet"},
		{"config", "user.email", "gate@example.com"},
		{"config", "user.name", "catalog gate test"},
		{"add", "-A"},
		{"commit", "--quiet", "-m", "catalog fixture"},
	} {
		cmd := exec.Command("git", args...)
		cmd.Dir = repo
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v in fixture repo: %v\n%s", args, err, out)
		}
	}
	revCmd := exec.Command("git", "rev-parse", "HEAD")
	revCmd.Dir = repo
	rev, err := revCmd.Output()
	if err != nil {
		t.Fatalf("git rev-parse in fixture repo: %v", err)
	}
	return repo, strings.TrimSpace(string(rev))
}

// writeFixtureFile writes one catalog fixture file, creating its directory.
func writeFixtureFile(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}
