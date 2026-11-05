// Moved from blis-latency-kernel/internal/harness/load_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package harness

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts"
)

// TestDefaultRootsAreThePinnedSubstratesData is the hermeticity check.
//
// The suite reads the catalog and registry of the blis-substrate release go.mod pins. It used
// to read absolute paths in one developer's home directory and SKIP when that failed, which is
// how a real incompatibility stayed hidden for the life of a dependency pin. If these
// defaults ever resolve somewhere else, that silence comes back.
func TestDefaultRootsAreThePinnedSubstratesData(t *testing.T) {
	t.Setenv("BLIS_CATALOG", "")
	t.Setenv("BLIS_REGISTRY", "")

	for _, c := range []struct{ what, got, base, probe string }{
		{"catalog", DefaultCatalog(), "catalog", filepath.Join("devices", "storage.yaml")},
		{"registry", DefaultRegistry(), "registry",
			filepath.Join("coefficients", "cost-model-primitives.yaml")},
	} {
		t.Run(c.what, func(t *testing.T) {
			if !strings.Contains(c.got, "github.com/inference-sim/blis-substrate@") ||
				filepath.Base(c.got) != c.base {
				t.Errorf("%s root is %q, want the pinned blis-substrate module's %s/", c.what,
					c.got, c.base)
			}
			// Resolving to a path is not enough; the artifacts have to be there.
			if _, err := os.Stat(filepath.Join(c.got, c.probe)); err != nil {
				t.Errorf("%s root %q does not hold %s: %v", c.what, c.got, c.probe, err)
			}
		})
	}
}

// TestDefaultRootsHonourTheEnvironment keeps the escape hatch working, so a working copy
// can be scored against a live upstream checkout.
func TestDefaultRootsHonourTheEnvironment(t *testing.T) {
	t.Setenv("BLIS_CATALOG", "/tmp/some-catalog")
	t.Setenv("BLIS_REGISTRY", "/tmp/some-registry")
	if got := DefaultCatalog(); got != "/tmp/some-catalog" {
		t.Errorf("DefaultCatalog() = %q; BLIS_CATALOG must win", got)
	}
	if got := DefaultRegistry(); got != "/tmp/some-registry" {
		t.Errorf("DefaultRegistry() = %q; BLIS_REGISTRY must win", got)
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
	_, err = Open("typo.yaml", Repos{Scenarios: dir, Catalog: DefaultCatalog(), Registry: DefaultRegistry()})
	if err == nil || !strings.Contains(err.Error(), "cache_dytpe") || !strings.Contains(err.Error(), "typo.yaml") {
		t.Errorf("Open: %v; want a refusal naming the misspelled key and the file", err)
	}
}
