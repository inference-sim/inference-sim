package sim

// The lint against unnamed randomness (design X6, DEV-6). Every random draw the simulator makes
// must come from a named substream (NewStream). A math/rand or math/rand/v2 function other than
// those that only name a type is an unnamed source: rand.New(rand.NewSource(seed)) seeded from
// whatever integer is at hand, or the process-global generator behind rand.Intn and its kind.
// This test walks every non-test Go file under sim/ and cmd/ and refuses any such call outside
// this package's rng.go, the one place a generator is built.

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// streamConstructorFile is the one file that may build a generator.
const streamConstructorFile = "sim/rng.go"

// unnamedRandomnessExempt lists files the lint does not read: the deprecated observe and
// calibrate commands, which M1 removes (inference-sim #1901), and nothing else.
var unnamedRandomnessExempt = map[string]bool{
	"cmd/observe_cmd.go": true,
}

// allowedRandIdents are math/rand names that draw nothing: types, and math/rand's NewZipf,
// which wraps a *rand.Rand the caller passes in (so its draws are that stream's).
var allowedRandIdents = map[string]bool{"Rand": true, "Source": true, "Source64": true, "Zipf": true, "NewZipf": true}

// unnamedRandomnessUses returns each use, in file, of a math/rand or math/rand/v2 identifier
// that is not in allowedRandIdents, as "line: rand.Name".
func unnamedRandomnessUses(fset *token.FileSet, file *ast.File) []string {
	local := map[string]bool{}
	for _, imp := range file.Imports {
		path, _ := strconv.Unquote(imp.Path.Value)
		if path != "math/rand" && path != "math/rand/v2" {
			continue
		}
		name := filepath.Base(path)
		if path == "math/rand/v2" {
			name = "rand"
		}
		if imp.Name != nil {
			name = imp.Name.Name
		}
		local[name] = true
	}
	if len(local) == 0 {
		return nil
	}
	var uses []string
	ast.Inspect(file, func(n ast.Node) bool {
		sel, ok := n.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		pkg, ok := sel.X.(*ast.Ident)
		if !ok || !local[pkg.Name] || pkg.Obj != nil {
			return true
		}
		if !allowedRandIdents[sel.Sel.Name] {
			uses = append(uses, strconv.Itoa(fset.Position(sel.Pos()).Line)+": "+pkg.Name+"."+sel.Sel.Name)
		}
		return true
	})
	return uses
}

func TestX6_NoUnnamedRandomness(t *testing.T) {
	root := filepath.Join("..")
	var bad []string
	scanned := 0
	for _, dir := range []string{"sim", "cmd"} {
		err := filepath.WalkDir(filepath.Join(root, dir), func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() || !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			rel, _ := filepath.Rel(root, path)
			rel = filepath.ToSlash(rel)
			if rel == streamConstructorFile || unnamedRandomnessExempt[rel] {
				return nil
			}
			fset := token.NewFileSet()
			f, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
			if err != nil {
				return err
			}
			scanned++
			for _, u := range unnamedRandomnessUses(fset, f) {
				bad = append(bad, rel+":"+u)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if scanned < 100 {
		t.Fatalf("scanned %d files under sim/ and cmd/; the walk is not reaching the tree", scanned)
	}
	sort.Strings(bad)
	for _, b := range bad {
		t.Errorf("unnamed randomness at %s: draw from a named stream (sim.NewStream) instead (DEV-6, X6)", b)
	}
}

// TestX6_LintRefusesUnnamedRandomness is the lint's known-bad input (X18): each shape of unnamed
// randomness is found, and the uses that draw nothing are not.
func TestX6_LintRefusesUnnamedRandomness(t *testing.T) {
	cases := []struct {
		name, src string
		want      []string
	}{
		{"seeded generator", `package p
import "math/rand"
func f(seed int64) *rand.Rand { return rand.New(rand.NewSource(seed)) }`, []string{"3: rand.New", "3: rand.NewSource"}},
		{"global generator", `package p
import "math/rand"
func f() int { return rand.Intn(3) }`, []string{"3: rand.Intn"}},
		{"renamed v2 import", `package p
import r2 "math/rand/v2"
func f() uint64 { return r2.NewPCG(1, 2).Uint64() }`, []string{"3: r2.NewPCG"}},
		{"types only", `package p
import "math/rand"
func f(r *rand.Rand) int { return r.Intn(3) }`, nil},
		{"zipf over a passed stream", `package p
import "math/rand"
func f(r *rand.Rand) *rand.Zipf { return rand.NewZipf(r, 1.1, 1, 5) }`, nil},
		{"no import", `package p
func f(rand struct{ Intn func(int) int }) int { return rand.Intn(3) }`, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fset := token.NewFileSet()
			f, err := parser.ParseFile(fset, "x.go", c.src, parser.SkipObjectResolution)
			if err != nil {
				t.Fatal(err)
			}
			got := unnamedRandomnessUses(fset, f)
			if strings.Join(got, ",") != strings.Join(c.want, ",") {
				t.Errorf("uses = %v, want %v", got, c.want)
			}
		})
	}
}
