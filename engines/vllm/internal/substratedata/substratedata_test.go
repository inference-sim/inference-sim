// Original to blis-vllm.

package substratedata

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// With no override, each root is the pinned substrate's own directory in the module cache,
// and holds the data the tests read.
func TestTheRootsAreThePinnedSubstratesData(t *testing.T) {
	t.Setenv(catalogEnv, "")
	t.Setenv(registryEnv, "")
	for _, c := range []struct {
		name  string
		get   func() (string, error)
		probe string
	}{
		{"catalog", Catalog, filepath.Join("hardware", "h200.yaml")},
		{"registry", Registry, filepath.Join("coefficients", "cost-model-primitives.yaml")},
	} {
		got, err := c.get()
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(got, Module+"@") || filepath.Base(got) != c.name {
			t.Errorf("%s root %q is not <module cache>/%s@<version>/%s", c.name, got, Module, c.name)
		}
		if _, err := os.Stat(filepath.Join(got, c.probe)); err != nil {
			t.Errorf("%s root %q lacks %s: %v", c.name, got, c.probe, err)
		}
	}
}

// Each root is overridable on its own.
func TestEachRootIsIndependentlyOverridable(t *testing.T) {
	t.Setenv(catalogEnv, "/elsewhere/catalog")
	t.Setenv(registryEnv, "")
	if got, _ := Catalog(); got != "/elsewhere/catalog" {
		t.Errorf("Catalog() = %q with %s set", got, catalogEnv)
	}
	if got, _ := Registry(); !strings.Contains(got, Module+"@") {
		t.Errorf("Registry() = %q moved when only %s was set", got, catalogEnv)
	}
}

// A path under the module is the released data; an override elsewhere is not.
func TestInModuleRecognisesOnlyTheModulesOwnData(t *testing.T) {
	t.Setenv(catalogEnv, "")
	cat, err := Catalog()
	if err != nil {
		t.Fatal(err)
	}
	if !InModule(cat) {
		t.Errorf("InModule(%q) = false for the module's own catalog", cat)
	}
	if InModule(t.TempDir()) {
		t.Error("InModule is true for a directory outside the module")
	}
}
