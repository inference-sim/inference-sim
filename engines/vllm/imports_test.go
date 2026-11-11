// Original to blis-vllm.

package blisvllm

import (
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// X7: an engine repository imports blis-substrate only through contracts/. blis-vllm is declared an
// exception until the boundary transition's equivalence step replaces its direct cost-model imports:
// the kernel and the load package are imported by the packages below, and by no other. The list is
// exact in both directions: a new package importing either, or any other substrate package outside
// contracts/, fails, and so does an entry no package uses any more (the list shrinks with the
// transition).
//
// blis-substrate/load is here for one reason: core/vllm's Engine.RecordSelected(*load.Data), the
// optional method by which the host registers each engine's pricing kernel with the run's Data. It
// is retired together with the other cost-model imports.
var declaredSubstrateExceptions = []string{
	"github.com/inference-sim/blis-substrate/kernel compat/kernelmodel",
	"github.com/inference-sim/blis-substrate/kernel kerneltests/cmd/worked-table",
	"github.com/inference-sim/blis-substrate/kernel kerneltests/internal/docgen",
	"github.com/inference-sim/blis-substrate/kernel kerneltests/internal/harness",
	"github.com/inference-sim/blis-substrate/load compat/build",
	"github.com/inference-sim/blis-substrate/load compat/kernelmodel",
	"github.com/inference-sim/blis-substrate/load core/vllm",
	"github.com/inference-sim/blis-substrate/load kerneltests/internal/harness",
	"github.com/inference-sim/blis-substrate/load vllmengine",
}

// substrateImportsOutsideContracts lists "<import path> <package dir>" for every non-test Go file under root
// that imports a blis-substrate package other than those under contracts/.
func substrateImportsOutsideContracts(t *testing.T, root string) []string {
	t.Helper()
	seen := map[string]bool{}
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && p != root && strings.HasPrefix(d.Name(), ".") {
			return filepath.SkipDir
		}
		if d.IsDir() || !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") {
			return nil
		}
		f, err := parser.ParseFile(token.NewFileSet(), p, nil, parser.ImportsOnly)
		if err != nil {
			return err
		}
		for _, im := range f.Imports {
			v := strings.Trim(im.Path.Value, `"`)
			if strings.HasPrefix(v, "github.com/inference-sim/blis-substrate/") && !strings.Contains(v, "/contracts") {
				rel, _ := filepath.Rel(root, filepath.Dir(p))
				seen[v+" "+filepath.ToSlash(rel)] = true
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for k := range seen {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func TestSubstrateImportsOutsideContractsAreTheDeclaredExceptions(t *testing.T) {
	got := substrateImportsOutsideContracts(t, ".")
	want := append([]string(nil), declaredSubstrateExceptions...)
	sort.Strings(want)
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Errorf("imports of blis-substrate outside contracts/ differ from the declared exceptions:\n got  %q\n want %q", got, want)
	}
}

func TestAnUndeclaredSubstrateImportIsFound(t *testing.T) {
	dir := t.TempDir()
	write := func(rel, body string) {
		t.Helper()
		p := filepath.Join(dir, rel)
		if err := mkdirWrite(p, body); err != nil {
			t.Fatal(err)
		}
	}
	write("a/a.go", "package a\nimport _ \"github.com/inference-sim/blis-substrate/ledger\"\n")
	write("b/b.go", "package b\nimport _ \"github.com/inference-sim/blis-substrate/contracts/engine\"\n")
	write("c/c_test.go", "package c\nimport _ \"github.com/inference-sim/blis-substrate/registry\"\n")
	got := substrateImportsOutsideContracts(t, dir)
	if len(got) != 1 || got[0] != "github.com/inference-sim/blis-substrate/ledger a" {
		t.Errorf("found %q, want only the ledger import of package a (contracts/ and test files are not exceptions)", got)
	}
}

func mkdirWrite(p, body string) error {
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	return os.WriteFile(p, []byte(body), 0o644)
}
