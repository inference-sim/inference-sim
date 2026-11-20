// Original to blis-vllm.

package substratedata

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// With no override, each root is the repository's own substrate directory, and holds the data
// the tests read.
func TestTheRootsAreTheRepositorysSubstrateData(t *testing.T) {
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
		if !strings.HasSuffix(filepath.ToSlash(got), "/substrate/"+c.name) {
			t.Errorf("%s root %q is not <repository>/substrate/%s", c.name, got, c.name)
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
	if got, _ := Registry(); !strings.HasSuffix(filepath.ToSlash(got), "/substrate/registry") {
		t.Errorf("Registry() = %q moved when only %s was set", got, catalogEnv)
	}
}

// A path under the substrate directory is the repository's data; an override elsewhere is not.
func TestInModuleRecognisesOnlyTheRepositorysOwnData(t *testing.T) {
	t.Setenv(catalogEnv, "")
	cat, err := Catalog()
	if err != nil {
		t.Fatal(err)
	}
	if !InModule(cat) {
		t.Errorf("InModule(%q) = false for the repository's own catalog", cat)
	}
	if InModule(t.TempDir()) {
		t.Error("InModule is true for a directory outside the substrate")
	}
}

// The directory is found from a go.mod that declares the module, not from a nearer go.mod of
// another module or from the environment.
func TestLocateWalksToTheModulesGoMod(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte(simulatorModule+"\n\ngo 1.27.2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	deep := filepath.Join(root, "a", "b")
	if err := os.MkdirAll(deep, 0o755); err != nil {
		t.Fatal(err)
	}
	if got, err := locate(deep); err != nil || got != filepath.Join(root, "substrate") {
		t.Errorf("locate(%s) = %q, %v", deep, got, err)
	}
	other := t.TempDir()
	if err := os.WriteFile(filepath.Join(other, "go.mod"), []byte("module example.com/other\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := locate(other); err == nil {
		t.Errorf("locate in another module's tree = %q, want an error", got)
	}
}
