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
	if err := modzip.CreateFromDir(&buf, module.Version{Path: "github.com/inference-sim/inference-sim/substrate", Version: "v0.0.0"}, "."); err != nil {
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

// Design §3.3, §10.2: nothing under catalog/ledger/ is embedded, so the simulator's data carries
// no ledger file and no pattern can name one.
func TestNothingUnderTheLedgerIsEmbedded(t *testing.T) {
	for _, p := range CatalogPatterns {
		if strings.HasPrefix(p, "ledger/") {
			t.Errorf("CatalogPatterns lists %q; the ledger is not data", p)
		}
	}
	err := fs.WalkDir(Catalog(), ".", func(p string, _ fs.DirEntry, err error) error {
		if err == nil && (p == "ledger" || strings.HasPrefix(p, "ledger/")) {
			t.Errorf("the embedded catalog holds %s", p)
		}
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
}
