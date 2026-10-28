// Moved from blis-latency-kernel/internal/docgen/docgen_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// TestGeneratedDocsAreCurrent fails while a committed file in docs/generated/ differs from what
// `go run ./internal/docgen` would write, or while one exists that it would not write at all.
// The step figure is priced by this kernel, so a change to the cost model that moves it fails
// here until the figure is regenerated, and the page cannot show a model the code no longer is.
func TestGeneratedDocsAreCurrent(t *testing.T) {
	root := repoRoot()
	want, err := generate(root)
	if err != nil {
		t.Fatalf("generating: %v (the figure is priced from testdata/catalog and "+
			"testdata/registry; run scripts/fetch-testdata.sh)", err)
	}
	dir := filepath.Join(root, "docs", "generated")
	for name, body := range want {
		got, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil || !bytes.Equal(got, body) {
			t.Errorf("docs/generated/%s is not current: run `go run ./internal/docgen` and "+
				"commit the result", name)
		}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if _, ok := want[e.Name()]; !ok {
			t.Errorf("docs/generated/%s is not written by internal/docgen; delete it", e.Name())
		}
	}
}

// TestSourcesNamesEveryInferenceSimPin: the pins table lists each module go.mod requires from
// this project, so a reader can see which rules packs priced the documented figures as well as
// which document types.
func TestSourcesNamesEveryInferenceSimPin(t *testing.T) {
	kernelDir, err := kernelSource()
	if err != nil {
		t.Fatal(err)
	}
	body, err := sources(kernelDir, repoRoot())
	if err != nil {
		t.Fatal(err)
	}
	// The kernel requires blis-schemas alone: it imports nothing from blis-vllm.
	for _, mod := range []string{"blis-schemas"} {
		if !bytes.Contains(body, []byte("["+mod+"]")) {
			t.Errorf("docs/generated/sources.md names no %s pin:\n%s", mod, body)
		}
	}
}
