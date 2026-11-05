package substratedata

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Each root is the selected substrate module's own directory in the module cache, and holds
// the data the tests read.
func TestTheRootsAreTheSelectedSubstratesData(t *testing.T) {
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
