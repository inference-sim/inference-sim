package harness

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strconv"
)

// clockCalls are the functions of package time that read the clock, wait on it or start a timer.
var clockCalls = map[string]bool{
	"Now": true, "Since": true, "Until": true, "Sleep": true, "After": true,
	"AfterFunc": true, "Tick": true, "NewTicker": true, "NewTimer": true,
}

// ClockAndGoroutineUses returns, for each file (relative to root), every go statement and every
// call of a clock-reading function of package time, under whatever name the file imports it (X6).
// Engine code takes the simulated instant as a value and starts no goroutine; that is what makes a
// run reproducible and lets the oracle drive the upstream's own decision clocks on the same virtual
// time. A repository passes the files of its engine packages, leaving out offline tools that may
// time themselves.
func ClockAndGoroutineUses(root string, files []string) ([]string, error) {
	var bad []string
	fset := token.NewFileSet()
	for _, p := range files {
		f, err := parser.ParseFile(fset, filepath.Join(root, filepath.FromSlash(p)), nil, 0)
		if err != nil {
			return nil, err
		}
		timeName := ""
		for _, imp := range f.Imports {
			if path, _ := strconv.Unquote(imp.Path.Value); path == "time" {
				timeName = "time"
				if imp.Name != nil {
					timeName = imp.Name.Name
				}
			}
		}
		ast.Inspect(f, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.GoStmt:
				bad = append(bad, fmt.Sprintf("%s: starts a goroutine", rel(fset, root, x.Pos())))
			case *ast.SelectorExpr:
				if id, ok := x.X.(*ast.Ident); ok && timeName != "" && id.Name == timeName && clockCalls[x.Sel.Name] {
					bad = append(bad, fmt.Sprintf("%s: calls time.%s; engine code takes the instant as a value", rel(fset, root, x.Pos()), x.Sel.Name))
				}
			}
			return true
		})
	}
	return bad, nil
}

// rel is a position with its file named relative to root.
func rel(fset *token.FileSet, root string, pos token.Pos) string {
	p := fset.Position(pos)
	if r, err := filepath.Rel(root, p.Filename); err == nil {
		p.Filename = filepath.ToSlash(r)
	}
	return p.String()
}
