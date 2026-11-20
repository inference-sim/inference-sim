// Original to blis-vllm.

package substratedata

import (
	"os"
	"os/exec"
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

// A go.work (or GOWORK) that names a sibling blis-substrate checkout does not change the
// directory reported: the pinned release in the module cache stays the answer (X25).
func TestAWorkspaceDoesNotMoveThePinnedSubstrate(t *testing.T) {
	t.Setenv(catalogEnv, "")
	t.Setenv(registryEnv, "")
	pinned, err := locate("")
	if err != nil {
		t.Fatal(err)
	}
	own, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	parent := t.TempDir()
	sibling := filepath.Join(parent, "blis-substrate")
	if err := os.MkdirAll(sibling, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sibling, "go.mod"), []byte("module "+Module+"\n\ngo 1.27.2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	work := filepath.Join(parent, "go.work")
	if err := os.WriteFile(work, []byte("go 1.27.2\n\nuse (\n\t"+own+"\n\t"+sibling+"\n)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Non-vacuity: the workspace really does move an unguarded go list.
	cmd := exec.Command("go", "list", "-m", "-f", "{{.Dir}}", Module)
	cmd.Dir = own
	cmd.Env = append(os.Environ(), "GOWORK="+work)
	raw, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v: %s", err, raw)
	}
	if got := strings.TrimSpace(string(raw)); got != sibling {
		t.Fatalf("the workspace reports %q for an unguarded go list, want the sibling %q: the test proves nothing", got, sibling)
	}
	t.Setenv("GOWORK", work)
	t.Setenv("GOFLAGS", "-mod=readonly")
	got, err := locate("")
	if err != nil || got != pinned {
		t.Errorf("under GOWORK=%s the substrate is %q (%v), want the pinned %q", work, got, err, pinned)
	}
}
