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

// X7 and design §5: an engine directory imports the substrate only through contracts/, imports
// nothing from another engine directory, and nothing of the simulator above it. The SGLang engine
// has no declared exception. Test files are held to the same rule, since a test that imports the
// vLLM engine would make it a dependency of this one.

const (
	module             = "github.com/inference-sim/inference-sim"
	ownTree            = module + "/engines/sglang"
	substrateContracts = module + "/substrate/contracts"
)

func under(path, dir string) bool { return path == dir || strings.HasPrefix(path, dir+"/") }

// isContracts reports whether an import path is the substrate's contracts package or one under
// it. A path that merely contains "/contracts", such as .../kernel/contracts, is not.
func isContracts(path string) bool {
	return path == substrateContracts || strings.HasPrefix(path, substrateContracts+"/")
}

// forbidden says why an import is not allowed, or "" when it is. Packages outside this module
// (the standard library, third-party modules) are not this rule's concern.
func forbidden(path string) string {
	switch {
	case under(path, ownTree):
		return ""
	case under(path, module+"/engines"):
		return "an engine directory imports no other engine directory"
	case under(path, module+"/substrate") && !isContracts(path):
		return "the substrate is imported only through contracts/"
	case under(path, module+"/substrate"):
		return ""
	case under(path, module):
		return "an engine directory imports nothing from the simulator above it"
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
		"a/a.go":      "package a\nimport _ \"github.com/inference-sim/inference-sim/substrate/ledger\"\n",
		"b/b.go":      "package b\nimport _ \"github.com/inference-sim/inference-sim/substrate/contracts/engine\"\n",
		"c/c_test.go": "package c\nimport _ \"github.com/inference-sim/inference-sim/engines/vllm/core/sched\"\n",
		"d/d.go":      "package d\nimport _ \"github.com/inference-sim/inference-sim/substrate/contracts\"\n",
		"e/e.go":      "package e\nimport _ \"github.com/inference-sim/inference-sim/substrate/kernel/contracts\"\n",
		"f/f.go":      "package f\nimport _ \"github.com/inference-sim/inference-sim/engines/vllm\"\n",
		"g/g.go":      "package g\nimport _ \"github.com/inference-sim/inference-sim/engines/sglang/core\"\n",
		"j/j.go":      "package j\nimport _ \"github.com/inference-sim/inference-sim/substrate/kernel\"\nimport _ \"github.com/spf13/cobra\"\n",
		"h/h.go":      "package h\nimport _ \"github.com/inference-sim/inference-sim/substrate\"\n",
		"i/i.go":      "package i\nimport _ \"github.com/inference-sim/inference-sim/sim\"\n",
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
		"a/a.go: github.com/inference-sim/inference-sim/substrate/ledger: the substrate is imported only through contracts/",
		"c/c_test.go: github.com/inference-sim/inference-sim/engines/vllm/core/sched: an engine directory imports no other engine directory",
		"e/e.go: github.com/inference-sim/inference-sim/substrate/kernel/contracts: the substrate is imported only through contracts/",
		"f/f.go: github.com/inference-sim/inference-sim/engines/vllm: an engine directory imports no other engine directory",
		"h/h.go: github.com/inference-sim/inference-sim/substrate: the substrate is imported only through contracts/",
		"i/i.go: github.com/inference-sim/inference-sim/sim: an engine directory imports nothing from the simulator above it",
		"j/j.go: github.com/inference-sim/inference-sim/substrate/kernel: the substrate is imported only through contracts/",
	}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Errorf("found\n%s\nwant\n%s", strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
}
