package substrate

import (
	"archive/zip"
	"bytes"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"golang.org/x/mod/module"
	modzip "golang.org/x/mod/zip"
)

// The module asks for the Go its own code needs (go 1.24), not one a dependency chose: the go
// command raises a module's directive to its highest requirement's, so a pin on a module that
// declares a newer Go would silently move every consumer's toolchain (blis-latency-kernel#87).
func TestTheModuleAsksForGo124(t *testing.T) {
	raw, err := os.ReadFile("go.mod")
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(string(raw), "\n") {
		if v, ok := strings.CutPrefix(strings.TrimSpace(line), "go "); ok {
			if !strings.HasPrefix(v, "1.24") {
				t.Errorf("go.mod declares go %s; a dependency raised it. Keep go 1.24, and lower "+
					"the dependency's directive instead", v)
			}
			return
		}
	}
	t.Fatal("go.mod has no go directive")
}

// SUB-13: blis-substrate is one Go module. A go.mod below the root would make its directory a
// separate module, which is the split this repository exists to remove.
func TestNoNestedModule(t *testing.T) {
	err := filepath.WalkDir(".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && (d.Name() == ".git" || d.Name() == "node_modules" || d.Name() == ".venv") {
			return filepath.SkipDir
		}
		if !d.IsDir() && path != "go.mod" && (d.Name() == "go.mod" || d.Name() == "go.work") {
			t.Errorf("%s: blis-substrate is one module; remove it (SUB-13)", path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// SUB-13: the files in the module zip that Go fetches total under 100 MB uncompressed, well
// below Go's own limit of 500 MB. The zip is built by golang.org/x/mod/zip, the code the go
// command uses, so the files counted are exactly those a consumer downloads.
func TestModuleZipIsUnder100MB(t *testing.T) {
	var buf bytes.Buffer
	if err := modzip.CreateFromDir(&buf, module.Version{Path: "github.com/inference-sim/blis-substrate", Version: "v0.0.0"}, "."); err != nil {
		t.Fatalf("building the module zip: %v", err)
	}
	r, err := zip.NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	var total uint64
	for _, f := range r.File {
		total += f.UncompressedSize64
	}
	const limit = 100 << 20
	if total >= limit {
		t.Errorf("the module zip holds %d bytes uncompressed; it must stay under %d (SUB-13)", total, limit)
	}
	t.Logf("module zip: %d files, %d bytes uncompressed, %d compressed", len(r.File), total, buf.Len())
}

// SUB-9: the go:embed directive must list literal patterns, so it repeats CatalogPatterns
// and RegistryPatterns. The loader reads a path with those two lists; if the directive
// differed, an embedded load and a path load would read different files.
func TestEmbedPatternsEqualLoaderPatterns(t *testing.T) {
	raw, err := os.ReadFile("data.go")
	if err != nil {
		t.Fatal(err)
	}
	var embedded []string
	for _, line := range strings.Split(string(raw), "\n") {
		if rest, ok := strings.CutPrefix(line, "//go:embed "); ok {
			embedded = append(embedded, strings.Fields(rest)...)
		}
	}
	var want []string
	for _, p := range CatalogPatterns {
		want = append(want, "catalog/"+p)
	}
	for _, p := range RegistryPatterns {
		want = append(want, "registry/"+p)
	}
	sort.Strings(embedded)
	sort.Strings(want)
	if strings.Join(embedded, "\n") != strings.Join(want, "\n") {
		t.Errorf("data.go's go:embed patterns differ from the loader's.\n go:embed: %v\n loader:   %v",
			embedded, want)
	}
}

// D7, COV-2: the three published role lists are runtime data and ship in the release. go:embed
// refuses a pattern that matches no file, so the pattern is added in the change that commits the
// lists; this test fails the other order, lists committed and pattern missing, which would build
// a release with no anchor list for the dashboard to score. Before the lists are committed there
// is nothing to embed and the test checks only that the pattern is not there early.
func TestPublishedListsAreEmbeddedWhenPresent(t *testing.T) {
	const pattern = "ledger/published/*.yaml"
	files, err := filepath.Glob("catalog/ledger/published/*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	listed := false
	for _, p := range CatalogPatterns {
		listed = listed || p == pattern
	}
	if len(files) == 0 {
		if listed {
			t.Errorf("CatalogPatterns lists %q but catalog/ledger/published/ holds no list; go:embed would not build", pattern)
		}
		return
	}
	if !listed {
		t.Fatalf("catalog/ledger/published/ holds %d lists but CatalogPatterns lacks %q, so the release would carry none (COV-2)", len(files), pattern)
	}
	for _, role := range []string{"fit", "anchor", "report"} {
		raw, err := fs.ReadFile(Catalog(), "ledger/published/"+role+".yaml")
		if err != nil {
			t.Errorf("the embedded catalog lacks the %s list: %v", role, err)
		} else if len(raw) == 0 {
			t.Errorf("the embedded %s list is empty", role)
		}
	}
}
