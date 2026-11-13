// Original to blis-vllm.

package blisvllm

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// SUB-20: no price, size or engine decision depends on a free-text label. A string field kept
// for people to read (the reason a rule gives, a note, a description) changes no output; code
// that compares one, or tests it with a string function, has made the words part of the model,
// and rewording a sentence would change a decision.
//
// This is the syntax-tree lint SUB-20 asks of each engine repository: it finds, in every
// non-test Go file, a comparison (==, !=), a switch, or a call of a strings function on a
// selector whose field is one of documentationOnly. The sites that exist are in
// testdata/label_branches.txt. The list may only shrink: a new branch on a label fails, and a
// listed branch that is gone fails, so removing one means deleting its line.
//
// The set of fields is a judgement, by name. A field named for what it holds (Because, Reason,
// Note, Why, What, Description, Rationale, Evidence, Message, Citation) is prose in this
// module; a new prose field is added to the set in the change that adds it.
var documentationOnly = map[string]bool{
	"Because": true, "Reason": true, "Note": true, "Why": true, "What": true, "Description": true,
	"Rationale": true, "Evidence": true, "Message": true, "Citation": true,
}

// stringFuncs are the strings package functions that read a string's content.
var stringFuncs = map[string]bool{
	"Contains": true, "ContainsAny": true, "ContainsRune": true, "HasPrefix": true, "HasSuffix": true,
	"EqualFold": true, "Index": true, "IndexAny": true, "IndexByte": true, "LastIndex": true,
	"Count": true, "Compare": true, "Cut": true, "CutPrefix": true, "CutSuffix": true, "Fields": true,
	"Split": true, "SplitN": true, "TrimPrefix": true, "TrimSuffix": true, "Trim": true, "TrimSpace": true,
}

// findLabelBranches returns, sorted, each place under roots (a directory, walked, or a file)
// where a documentation-only field is compared or searched, as "path: expression".
func findLabelBranches(t *testing.T, skipDirs map[string]bool, roots ...string) []string {
	t.Helper()
	var found []string
	for _, root := range roots {
		err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				name := d.Name()
				if path != root && (strings.HasPrefix(name, ".") || name == "testdata" || skipDirs[filepath.ToSlash(path)]) {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			fset := token.NewFileSet()
			f, perr := parser.ParseFile(fset, path, nil, 0)
			if perr != nil {
				return perr
			}
			render := func(n ast.Node) string {
				var b bytes.Buffer
				_ = printer.Fprint(&b, fset, n)
				return strings.Join(strings.Fields(b.String()), " ")
			}
			isLabel := func(e ast.Expr) bool {
				s, ok := e.(*ast.SelectorExpr)
				return ok && documentationOnly[s.Sel.Name]
			}
			// mentions: the expression is, or contains, a label selector (a call on its value, such as
			// strings.ToLower(r.Because), still reads the words).
			mentions := func(e ast.Expr) bool {
				hit := false
				ast.Inspect(e, func(n ast.Node) bool {
					if x, ok := n.(ast.Expr); ok && isLabel(x) {
						hit = true
					}
					return !hit
				})
				return hit
			}
			rel := filepath.ToSlash(path)
			ast.Inspect(f, func(n ast.Node) bool {
				switch v := n.(type) {
				case *ast.BinaryExpr:
					if (v.Op == token.EQL || v.Op == token.NEQ) && (mentions(v.X) || mentions(v.Y)) {
						found = append(found, rel+": "+render(v))
					}
				case *ast.SwitchStmt:
					if v.Tag != nil && mentions(v.Tag) {
						found = append(found, rel+": switch "+render(v.Tag))
					}
				case *ast.CallExpr:
					if s, ok := v.Fun.(*ast.SelectorExpr); ok {
						if p, ok := s.X.(*ast.Ident); ok && p.Name == "strings" && stringFuncs[s.Sel.Name] {
							for _, a := range v.Args {
								if mentions(a) {
									found = append(found, rel+": "+render(v))
									break
								}
							}
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

func readRatchet(t *testing.T, path string) (head []string, lines []string) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range strings.Split(string(raw), "\n") {
		switch {
		case strings.HasPrefix(l, "#"):
			head = append(head, l)
		case l != "":
			lines = append(lines, l)
		}
	}
	sort.Strings(lines)
	return head, lines
}

// TestNoDecisionDependsOnAFreeTextLabel runs the lint over the engine's own code. compat/ is the
// compatibility engine, deleted in M4 (plan, "Relaxing the deprecation rule until M4"); it keeps
// its sites on the list under its own prefix so the ratchet covers it too.
func TestNoDecisionDependsOnAFreeTextLabel(t *testing.T) {
	got := findLabelBranches(t, nil, ".")
	_, want := readRatchet(t, "testdata/label_branches.txt")
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		var added, gone []string
		have := map[string]bool{}
		for _, g := range got {
			have[g] = true
		}
		listed := map[string]bool{}
		for _, w := range want {
			listed[w] = true
			if !have[w] {
				gone = append(gone, w)
			}
		}
		for _, g := range got {
			if !listed[g] {
				added = append(added, g)
			}
		}
		t.Errorf("the comparisons on a documentation-only field differ from testdata/label_branches.txt.\n"+
			"new (a decision now depends on a label; use a typed field):\n  %s\n"+
			"listed but gone (delete the line):\n  %s",
			strings.Join(added, "\n  "), strings.Join(gone, "\n  "))
	}
}

// X18 for SUB-20: the lint finds a comparison, in either order, a switch, a strings function over
// the field and one over a call on it, and leaves a field that is not prose, a comparison of two
// plain strings and a use outside a condition alone.
func TestTheLabelLintFindsItsKnownBadInputs(t *testing.T) {
	dir := t.TempDir()
	src := `package p
import "strings"
type R struct{ Because, Name, Reason string }
func f(r R, s string) string {
	_ = r.Because == "x"
	_ = "x" != r.Reason
	switch r.Because { case "a": }
	_ = strings.Contains(r.Reason, "y")
	_ = strings.HasPrefix(strings.ToLower(r.Because), "z")
	_ = r.Name == "ok"
	_ = s == "ok"
	_ = strings.Contains(s, r.Name)
	return r.Because
}
`
	if err := os.WriteFile(filepath.Join(dir, "p.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	got := findLabelBranches(t, nil, dir)
	if len(got) != 5 {
		t.Fatalf("want 5 findings (== , reversed !=, switch, strings.Contains, strings.HasPrefix over a call), got %d:\n%s",
			len(got), strings.Join(got, "\n"))
	}
}

// The ratchet cannot grow quietly: a listed line that is not in the tree fails, and so does a
// site in the tree that is not listed. The comparison the real test makes is run on a list with a
// line added, and on one with a line removed.
func TestTheLabelRatchetFailsBothWays(t *testing.T) {
	got := findLabelBranches(t, nil, ".")
	_, want := readRatchet(t, "testdata/label_branches.txt")
	if len(want) == 0 {
		t.Fatal("the list is empty; the ratchet has nothing to remove")
	}
	more := append(append([]string{}, want...), "nowhere.go: r.Because == \"x\"")
	sort.Strings(more)
	if strings.Join(got, "\n") == strings.Join(more, "\n") {
		t.Error("a listed site that is not in the tree passed")
	}
	fewer := append([]string{}, want[1:]...)
	if strings.Join(got, "\n") == strings.Join(fewer, "\n") {
		t.Error("a site in the tree that is not listed passed")
	}
}
