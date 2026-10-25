package cmd

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The engine that cmd/engine_build.go builds reads a release's rules only through blis-schemas'
// rules.EngineRules (#2021 B4; PR BC-3): the pinned blis-vllm code this simulator calls --
// compat/build and the kernel adapter compat/kernelmodel -- neither type-asserts to the
// concrete *rules.Pack nor looks packs up through the global rules.Apply or rules.Lookup. The
// test reads the pinned module's own source, so the claim is checked against the version
// go.mod names rather than taken on trust.
func TestPinnedEngineReadsRulesOnlyThroughEngineRules(t *testing.T) {
	out, err := exec.Command("go", "list", "-m", "-f", "{{.Dir}}", "github.com/inference-sim/blis-vllm").Output()
	if err != nil {
		t.Fatalf("locating the pinned blis-vllm module: %v", err)
	}
	root := strings.TrimSpace(string(out))
	checked := 0
	for _, pkg := range []string{"compat/build", "compat/kernelmodel"} {
		files, err := filepath.Glob(filepath.Join(root, pkg, "*.go"))
		if err != nil || len(files) == 0 {
			t.Fatalf("%s: no source under %s (%v)", pkg, root, err)
		}
		for _, f := range files {
			if strings.HasSuffix(f, "_test.go") {
				continue
			}
			src, err := os.ReadFile(f)
			if err != nil {
				t.Fatal(err)
			}
			fset := token.NewFileSet()
			file, err := parser.ParseFile(fset, f, src, parser.SkipObjectResolution)
			if err != nil {
				t.Fatal(err)
			}
			rulesName := ""
			for _, imp := range file.Imports {
				if strings.Trim(imp.Path.Value, `"`) == "github.com/inference-sim/blis-schemas/rules" {
					rulesName = "rules"
					if imp.Name != nil {
						rulesName = imp.Name.Name
					}
				}
			}
			checked++
			if rulesName == "" {
				continue
			}
			ast.Inspect(file, func(n ast.Node) bool {
				switch x := n.(type) {
				case *ast.TypeAssertExpr:
					if star, ok := x.Type.(*ast.StarExpr); ok && isSel(star.X, rulesName, "Pack") {
						t.Errorf("%s: %s type-asserts to *%s.Pack", pkg, fset.Position(x.Pos()), rulesName)
					}
				case *ast.CallExpr:
					for _, fn := range []string{"Apply", "Lookup"} {
						if isSel(x.Fun, rulesName, fn) {
							t.Errorf("%s: %s calls the global %s.%s", pkg, fset.Position(x.Pos()), rulesName, fn)
						}
					}
				}
				return true
			})
		}
	}
	if checked == 0 {
		t.Fatal("no source checked")
	}
}

func isSel(e ast.Expr, pkg, name string) bool {
	sel, ok := e.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	id, ok := sel.X.(*ast.Ident)
	return ok && id.Name == pkg && sel.Sel.Name == name
}
