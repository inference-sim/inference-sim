package cmd

import (
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"testing"
	"testing/fstest"

	substrate "github.com/inference-sim/blis-substrate"

	"github.com/inference-sim/inference-sim/internal/substratedata"
)

// unembedded returns the data files of a checkout that the binary's embedded tree lacks: every
// file of checkout that one of patterns (path.Match syntax) selects, which the loader reads from
// a checkout, must also be in embedded, or a run would see less offline than from a checkout
// (SUB-9).
func unembedded(checkout, embedded fs.FS, patterns []string) []string {
	var missing []string
	_ = fs.WalkDir(checkout, ".", func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		for _, pat := range patterns {
			if ok, _ := path.Match(pat, p); ok {
				if _, err := fs.Stat(embedded, p); err != nil {
					missing = append(missing, p)
				}
				break
			}
		}
		return nil
	})
	sort.Strings(missing)
	return missing
}

// The release pinned here embeds every data file its loader reads (SUB-9): the checkout of
// the pinned module and the binary's embedded tree agree on the catalog and on the registry.
// The known-bad input is an embedded tree that leaves a data file out.
func TestEmbeddedTreeHoldsEveryFileTheLoaderReads(t *testing.T) {
	checkout := fstest.MapFS{
		"devices/h100.yaml":    {Data: []byte("x")},
		"workloads/chat.yaml":  {Data: []byte("x")},
		"models/m/config.json": {Data: []byte("not data")},
	}
	leftOut := fstest.MapFS{"devices/h100.yaml": {Data: []byte("x")}}
	if got := unembedded(checkout, leftOut, substrate.CatalogPatterns); len(got) != 1 || got[0] != "workloads/chat.yaml" {
		t.Fatalf("an embedded tree without a workload: missing %v, want exactly workloads/chat.yaml", got)
	}

	dir, err := substratedata.Dir()
	if err != nil {
		t.Fatal(err)
	}
	if got := unembedded(os.DirFS(filepath.Join(dir, "catalog")), substrate.Catalog(), substrate.CatalogPatterns); len(got) != 0 {
		t.Errorf("the pinned release's catalog files the loader reads but the binary does not embed: %v", got)
	}
	if got := unembedded(os.DirFS(filepath.Join(dir, "registry")), substrate.Registry(), substrate.RegistryPatterns); len(got) != 0 {
		t.Errorf("the pinned release's registry files the loader reads but the binary does not embed: %v", got)
	}
}
