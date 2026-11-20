package substratedata

import (
	"os"
	"path/filepath"
	"testing"
)

// Each root is the repository's own substrate directory, and holds the data the tests read.
func TestTheRootsAreTheRepositorysSubstrateData(t *testing.T) {
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
		if filepath.Base(got) != c.name || filepath.Base(filepath.Dir(got)) != "substrate" {
			t.Errorf("%s root %q is not <repository>/substrate/%s", c.name, got, c.name)
		}
		if _, err := os.Stat(filepath.Join(got, c.probe)); err != nil {
			t.Errorf("%s root %q lacks %s: %v", c.name, got, c.probe, err)
		}
	}
}
