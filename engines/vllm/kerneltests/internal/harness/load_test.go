// Moved from blis-latency-kernel/internal/harness/load_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package harness

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts"
	"github.com/inference-sim/blis-substrate/load"
)

// TestDefaultDataIsTheEmbeddedCopy is the hermeticity check. With neither BLIS_CATALOG nor
// BLIS_REGISTRY set, the commands read the catalog and registry the pinned blis-substrate
// release embeds. They used to read absolute paths in one developer's home directory and SKIP
// when that failed, which is how a real incompatibility stayed hidden for the life of a
// dependency pin.
func TestDefaultDataIsTheEmbeddedCopy(t *testing.T) {
	t.Setenv("BLIS_CATALOG", "")
	t.Setenv("BLIS_REGISTRY", "")
	d, err := Data("", "")
	if err != nil {
		t.Fatal(err)
	}
	if d.Catalog.Source != load.SourceEmbedded || d.Registry.Source != load.SourceEmbedded {
		t.Errorf("catalog %s, registry %s; want both embedded", d.Catalog.Source, d.Registry.Source)
	}
	if _, err := d.Chip("h200"); err != nil {
		t.Errorf("the embedded catalog lacks a chip the fixtures name: %v", err)
	}
}

// TestAFlagOrTheEnvironmentReplacesTheEmbeddedCopy keeps the escape hatch working, so a
// working copy can be tested against a candidate: a path replaces the embedded copy, and a
// path that holds no catalog is refused, not ignored.
func TestAFlagOrTheEnvironmentReplacesTheEmbeddedCopy(t *testing.T) {
	empty := t.TempDir()
	t.Setenv("BLIS_CATALOG", empty)
	if _, err := Data("", ""); err == nil {
		t.Errorf("BLIS_CATALOG=%s, an empty directory, was accepted as the catalog", empty)
	}
	t.Setenv("BLIS_CATALOG", "")
	if _, err := Data(empty, ""); err == nil {
		t.Errorf("-catalog %s, an empty directory, was accepted as the catalog", empty)
	}
}

// A misspelled key inside a pool's engine block is refused when the kernel is opened, at its line
// and naming the file: blis-schemas holds the block unopened, and the engine settings are decoded
// strictly by the engine model (Repos.EngineSettings, vllmsettings.OfPool for vLLM).
func TestOpenRefusesAMisspelledEngineKey(t *testing.T) {
	src, err := os.ReadFile(filepath.Join(repoRoot(), "testdata", "minimax-m25-h200-tp8.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	const key = "      cache_dtype: fp8\n"
	if strings.Count(string(src), key) != 1 {
		t.Fatalf("the fixture states cache_dtype %d times, want once", strings.Count(string(src), key))
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "typo.yaml"),
		[]byte(strings.Replace(string(src), key, "      cache_dytpe: fp8\n", 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := contracts.LoadBundle(filepath.Join(dir, "typo.yaml")); err != nil {
		t.Fatalf("LoadBundle refused the engine block it holds unopened: %v", err)
	}
	_, err = Open("typo.yaml", Repos{Scenarios: dir, Data: DefaultRepos().Data})
	if err == nil || !strings.Contains(err.Error(), "cache_dytpe") || !strings.Contains(err.Error(), "typo.yaml") {
		t.Errorf("Open: %v; want a refusal naming the misspelled key and the file", err)
	}
}
