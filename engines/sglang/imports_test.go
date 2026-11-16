// Original to blis-sglang.

package blissglang

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

// X7 and design §5: an engine repository imports blis-substrate only through contracts/, and
// imports nothing from another engine repository. blis-sglang has no declared exception. Test
// files are held to the same rule, since a test that imports blis-vllm would make blis-vllm a
// dependency of this module.

const substrateContracts = "github.com/inference-sim/blis-substrate/contracts"

// isContracts reports whether an import path is the substrate's contracts package or one under
// it. A path that merely contains "/contracts", such as .../kernel/contracts, is not.
func isContracts(path string) bool {
	return path == substrateContracts || strings.HasPrefix(path, substrateContracts+"/")
}

// forbidden says why an import is not allowed, or "" when it is.
func forbidden(path string) string {
	switch {
	case path == "github.com/inference-sim/blis-vllm" || strings.HasPrefix(path, "github.com/inference-sim/blis-vllm/"):
		return "an engine repository imports no other engine repository"
	case strings.HasPrefix(path, "github.com/inference-sim/blis-substrate/") && !isContracts(path):
		return "blis-substrate is imported only through contracts/"
	}
	return ""
}

// forbiddenImports lists "<file>: <import>: <why>" for every Go file under root.
func forbiddenImports(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && p != root && strings.HasPrefix(d.Name(), ".") {
			return filepath.SkipDir
		}
		if d.IsDir() || !strings.HasSuffix(p, ".go") {
			return nil
		}
		f, err := parser.ParseFile(token.NewFileSet(), p, nil, parser.ImportsOnly)
		if err != nil {
			return err
		}
		for _, im := range f.Imports {
			v := strings.Trim(im.Path.Value, `"`)
			if why := forbidden(v); why != "" {
				rel, _ := filepath.Rel(root, p)
				out = append(out, filepath.ToSlash(rel)+": "+v+": "+why)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(out)
	return out
}

func TestImportsAreSubstrateContractsOnly(t *testing.T) {
	for _, b := range forbiddenImports(t, ".") {
		t.Error(b)
	}
}

func TestAForbiddenImportIsFound(t *testing.T) {
	dir := t.TempDir()
	for rel, body := range map[string]string{
		"a/a.go":      "package a\nimport _ \"github.com/inference-sim/blis-substrate/ledger\"\n",
		"b/b.go":      "package b\nimport _ \"github.com/inference-sim/blis-substrate/contracts/engine\"\n",
		"c/c_test.go": "package c\nimport _ \"github.com/inference-sim/blis-vllm/core/sched\"\n",
		"d/d.go":      "package d\nimport _ \"github.com/inference-sim/blis-substrate/contracts\"\n",
		"e/e.go":      "package e\nimport _ \"github.com/inference-sim/blis-substrate/kernel/contracts\"\n",
		"f/f.go":      "package f\nimport _ \"github.com/inference-sim/blis-vllm\"\n",
		"g/g.go":      "package g\nimport _ \"github.com/inference-sim/blis-vllmx\"\n",
	} {
		p := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got := forbiddenImports(t, dir)
	want := []string{
		"a/a.go: github.com/inference-sim/blis-substrate/ledger: blis-substrate is imported only through contracts/",
		"c/c_test.go: github.com/inference-sim/blis-vllm/core/sched: an engine repository imports no other engine repository",
		"e/e.go: github.com/inference-sim/blis-substrate/kernel/contracts: blis-substrate is imported only through contracts/",
		"f/f.go: github.com/inference-sim/blis-vllm: an engine repository imports no other engine repository",
	}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Errorf("found\n%s\nwant\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
}
