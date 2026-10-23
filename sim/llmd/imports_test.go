package llmd

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

const (
	routerModule = "github.com/llm-d/llm-d-router"
	// routerConstructor is the one fork package BLIS may import, and New the one identifier it
	// may use from it: the fork's implementation of blis-schemas embed.Constructor.
	routerConstructor = routerModule + "/pkg/embed"
	forkGuide         = "docs/contributing/llm-d-router-fork.md"
)

// TestRouterImportsOnlyTheEmbedConstructor guards the router contract: BLIS programs against
// blis-schemas embed, and takes only the constructor pkg/embed.New from the llm-d-router fork. A
// router-internal import would tie BLIS to code upstream semver does not cover, so a same-major
// router release could no longer reach BLIS as a pin bump alone.
func TestRouterImportsOnlyTheEmbedConstructor(t *testing.T) {
	root := moduleRoot(t)
	problems := routerImportProblems(t, root)
	if len(problems) > 0 {
		t.Fatalf("BLIS may take only %s.New from the llm-d-router fork; everything else comes from "+
			"blis-schemas embed (see %s):\n  %s", routerConstructor, forkGuide, strings.Join(problems, "\n  "))
	}
}

// TestRouterImportGuardDetects mutation-checks the guard: a forbidden import and a forbidden
// identifier of the constructor package are both reported, and New alone is not.
func TestRouterImportGuardDetects(t *testing.T) {
	for _, tc := range []struct {
		name, src string
		want      string // a substring of the single problem, or "" for none
	}{
		{"constructor only", "package p\nimport fork " + q(routerConstructor) + "\nvar _ = fork.New", ""},
		{"internal import", "package p\nimport _ " + q(routerModule+"/pkg/epp/handlers"), "imports " + q(routerModule+"/pkg/epp/handlers")},
		{"constructor subpackage", "package p\nimport _ " + q(routerConstructor+"/internal"), "imports " + q(routerConstructor+"/internal")},
		{"other identifier", "package p\nimport " + q(routerConstructor) + "\nvar _ = embed.Adapter", "uses embed.Adapter"},
		{"renamed, other identifier", "package p\nimport r " + q(routerConstructor) + "\nvar _ = r.Options{}", "uses r.Options"},
		{"dot import", "package p\nimport . " + q(routerConstructor), "dot-imports"},
		// A blank import of the constructor package uses no identifier of it: allowed, as New alone is.
		{"blank constructor import", "package p\nimport _ " + q(routerConstructor), ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "x.go"), []byte(tc.src), 0o600); err != nil {
				t.Fatal(err)
			}
			got := routerImportProblems(t, dir)
			switch {
			case tc.want == "" && len(got) != 0:
				t.Fatalf("want no problem, got %q", got)
			case tc.want != "" && (len(got) != 1 || !strings.Contains(got[0], tc.want)):
				t.Fatalf("want one problem containing %q, got %q", tc.want, got)
			}
		})
	}
}

var q = strconv.Quote

// routerImportProblems parses every .go file under root (tests included, whatever their build tags;
// nested modules, vendor, testdata and hidden directories excluded) and reports each llm-d-router
// import other than the constructor package, each dot import of it, and each use of an identifier
// of it other than New. A blank import of the constructor package uses none, so it is allowed.
func routerImportProblems(t *testing.T, root string) []string {
	t.Helper()
	var problems []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			name := d.Name()
			if path != root && (strings.HasPrefix(name, ".") || name == "vendor" || name == "testdata") {
				return filepath.SkipDir
			}
			if path != root {
				if _, err := os.Stat(filepath.Join(path, "go.mod")); err == nil {
					return filepath.SkipDir // another module
				}
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		rel, _ := filepath.Rel(root, path)
		f, err := parser.ParseFile(token.NewFileSet(), path, nil, parser.SkipObjectResolution)
		if err != nil {
			return err
		}
		problems = append(problems, fileProblems(rel, f)...)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(problems)
	return problems
}

func fileProblems(rel string, f *ast.File) []string {
	var problems []string
	var local string // the constructor package's name in this file
	for _, imp := range f.Imports {
		path, _ := strconv.Unquote(imp.Path.Value)
		if path != routerModule && !strings.HasPrefix(path, routerModule+"/") {
			continue
		}
		if path != routerConstructor {
			problems = append(problems, rel+": imports "+strconv.Quote(path))
			continue
		}
		local = "embed"
		if imp.Name != nil {
			local = imp.Name.Name
		}
		if local == "." {
			problems = append(problems, rel+": dot-imports "+strconv.Quote(path))
			local = ""
		}
	}
	if local == "" || local == "_" {
		return problems
	}
	ast.Inspect(f, func(n ast.Node) bool {
		sel, ok := n.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		if x, ok := sel.X.(*ast.Ident); ok && x.Name == local && sel.Sel.Name != "New" {
			problems = append(problems, rel+": uses "+local+"."+sel.Sel.Name)
		}
		return true
	})
	return problems
}

// moduleRoot is the directory of the go.mod enclosing this package.
func moduleRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("no go.mod above the test's directory")
		}
		dir = parent
	}
}
