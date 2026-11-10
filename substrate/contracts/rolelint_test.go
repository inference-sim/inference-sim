package contracts

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// SUB-20: no price, size or engine decision depends on a free-text label. A graph node's Role
// is a name kept for people to read; a consumer that branches on it has made the label part of
// the model. This is the syntax-tree lint of SUB-20: it finds every comparison of a `.Role`
// selector, and every switch on one, in the cost model and in the model-graph package.
//
// It cannot tell a Role of a graph node from the typed Role of a pool or a routing section, so
// it reports comparisons whose other side is a string literal, a string variable or a field,
// and leaves the typed enums (an identifier that is a constant of a named type) alone by
// reporting only what it cannot show to be one. The existing sites are in
// testdata/role_branches.txt. The list may only shrink: a new branch on a label fails, and a
// listed branch that no longer exists fails, so removing one means deleting its line.
//
// The fix is the typed node roles of blis-schemas#67, a contract-change review set (SUB-1).
func findRoleBranches(t *testing.T, roots ...string) []string {
	t.Helper()
	var found []string
	for _, root := range roots {
		err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return err
			}
			fset := token.NewFileSet()
			f, err := parser.ParseFile(fset, path, nil, 0)
			if err != nil {
				return err
			}
			render := func(n ast.Node) string {
				var b bytes.Buffer
				_ = printer.Fprint(&b, fset, n)
				return strings.Join(strings.Fields(b.String()), " ")
			}
			isRole := func(e ast.Expr) bool {
				s, ok := e.(*ast.SelectorExpr)
				return ok && s.Sel.Name == "Role"
			}
			// typedConst: an identifier that begins with Role or is a package-qualified Role
			// constant, such as RoleDefault or deployment.RoleDecode: a typed enum, not a label.
			typedConst := func(e ast.Expr) bool {
				switch v := e.(type) {
				case *ast.Ident:
					return strings.HasPrefix(v.Name, "Role")
				case *ast.SelectorExpr:
					return strings.HasPrefix(v.Sel.Name, "Role")
				}
				return false
			}
			ast.Inspect(f, func(n ast.Node) bool {
				switch v := n.(type) {
				case *ast.BinaryExpr:
					if v.Op != token.EQL && v.Op != token.NEQ {
						return true
					}
					switch {
					case isRole(v.X) && !typedConst(v.Y) && !isRole(v.Y):
						found = append(found, filepath.ToSlash(path)+": "+render(v))
					case isRole(v.Y) && !typedConst(v.X) && !isRole(v.X):
						found = append(found, filepath.ToSlash(path)+": "+render(v))
					}
				case *ast.SwitchStmt:
					if v.Tag != nil && isRole(v.Tag) {
						typed := false
						for _, c := range v.Body.List {
							for _, e := range c.(*ast.CaseClause).List {
								typed = typed || typedConst(e)
							}
						}
						if !typed {
							found = append(found, filepath.ToSlash(path)+": switch "+render(v.Tag))
						}
					}
				}
				return true
			})
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	sort.Strings(found)
	return found
}

func TestNoPriceDependsOnAFreeTextLabel(t *testing.T) {
	got := findRoleBranches(t, "../kernel", "spec/model")
	raw, err := os.ReadFile("testdata/role_branches.txt")
	if err != nil {
		t.Fatal(err)
	}
	var want []string
	for _, l := range strings.Split(string(raw), "\n") {
		if l != "" && !strings.HasPrefix(l, "#") {
			want = append(want, l)
		}
	}
	sort.Strings(want)
	if os.Getenv("UPDATE_COMPLETENESS") != "" {
		var head []string
		for _, l := range strings.Split(string(raw), "\n") {
			if strings.HasPrefix(l, "#") {
				head = append(head, l)
			}
		}
		out := strings.Join(head, "\n") + "\n" + strings.Join(got, "\n") + "\n"
		if err := os.WriteFile("testdata/role_branches.txt", []byte(out), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Errorf("the comparisons on a node's Role differ from testdata/role_branches.txt.\nwant:\n%s\ngot:\n%s",
			strings.Join(want, "\n"), strings.Join(got, "\n"))
	}
}

// X18 for SUB-20: the lint finds a comparison against a literal and against a variable, in
// either order, and a switch; it leaves a typed constant alone.
func TestRoleLintFindsItsKnownBadInputs(t *testing.T) {
	dir := t.TempDir()
	src := `package p
type N struct{ Role string }
type P struct{ Role K }
type K string
const RoleDecode K = "decode"
func f(n N, p P, r string) bool {
	_ = n.Role == "experts"
	_ = "x" != n.Role
	_ = n.Role == r
	switch n.Role { case "a": }
	_ = p.Role == RoleDecode
	switch p.Role { case RoleDecode: }
	_ = n.Role == n.Role
	return true
}
`
	if err := os.WriteFile(filepath.Join(dir, "p.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	got := findRoleBranches(t, dir)
	if len(got) != 4 {
		t.Fatalf("want 4 findings (literal, reversed literal, variable, switch), got %d: %v", len(got), got)
	}
}
