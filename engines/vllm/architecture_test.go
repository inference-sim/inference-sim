package blisvllm

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// goFiles returns every non-test Go file in this module and its nested modules, by path
// relative to the repository root, skipping tools that are not engine code.
func goFiles(t *testing.T, skipDirs ...string) []string {
	t.Helper()
	var out []string
	err := filepath.WalkDir(".", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			name := d.Name()
			if p != "." && (strings.HasPrefix(name, ".") || name == "testdata") {
				return filepath.SkipDir
			}
			for _, s := range skipDirs {
				if p == s {
					return filepath.SkipDir
				}
			}
			return nil
		}
		if strings.HasSuffix(p, ".go") && !strings.HasSuffix(p, "_test.go") {
			out = append(out, p)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 {
		t.Fatal("no Go files found; the walk is broken")
	}
	return out
}

var (
	derivedFrom = regexp.MustCompile(`^// Derived from vllm/\S+ at vLLM v\d+\.\d+\.\d+ \([0-9a-f]{40}\)`)
	movedFrom   = regexp.MustCompile(`^// Moved from inference-sim/\S+ at [0-9a-f]{40}`)
	original    = "// Original to blis-vllm."
	spdx        = []string{
		"// SPDX-License-Identifier: Apache-2.0",
		"// SPDX-FileCopyrightText: Copyright contributors to the vLLM project",
	}
)

// TestEveryFileStatesItsProvenance: each Go file says where its logic comes from, in its
// first comment block. A file translated from vLLM names the upstream file and commit and
// keeps vLLM's licence (#2021 attribution rule), so the next vLLM release can be diffed
// function by function; a file moved from inference-sim names the commit it was taken from;
// anything else says it is original. A file that says nothing fails, which is what keeps the
// rule from decaying as files are added.
func TestEveryFileStatesItsProvenance(t *testing.T) {
	for _, p := range goFiles(t) {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(string(raw), "\n")
		var header []string
		for _, l := range lines {
			if !strings.HasPrefix(l, "//") {
				break
			}
			header = append(header, l)
		}
		var derived, moved, orig bool
		for _, l := range header {
			derived = derived || derivedFrom.MatchString(l)
			moved = moved || movedFrom.MatchString(l)
			orig = orig || strings.HasPrefix(l, original)
		}
		switch {
		case derived:
			if len(header) < 2 || header[0] != spdx[0] || header[1] != spdx[1] {
				t.Errorf("%s is derived from vLLM but does not start with vLLM's SPDX licence lines", p)
			}
		case moved, orig:
		default:
			t.Errorf("%s states no provenance: start it with a \"Derived from vllm/...\", \"Moved from inference-sim/...\" or %q line", p, original)
		}
	}
}

// TestNoClockNoGoroutines: engine code never reads a clock, sleeps or starts a goroutine
// (#2021 B3). Time reaches it only as a value the simulator passes in, which is what makes
// a run reproducible and lets the oracle run vLLM's own decision clocks on the same virtual
// time. Offline tools under cmd/ may time themselves; engine packages may not.
func TestNoClockNoGoroutines(t *testing.T) {
	banned := map[string]bool{
		"Now": true, "Since": true, "Until": true, "Sleep": true, "After": true,
		"AfterFunc": true, "Tick": true, "NewTicker": true, "NewTimer": true,
	}
	fset := token.NewFileSet()
	for _, p := range goFiles(t, "cmd") {
		f, err := parser.ParseFile(fset, p, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		timeName := ""
		for _, imp := range f.Imports {
			if imp.Path.Value == `"time"` {
				timeName = "time"
				if imp.Name != nil {
					timeName = imp.Name.Name
				}
			}
		}
		ast.Inspect(f, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.GoStmt:
				t.Errorf("%s: starts a goroutine", fset.Position(x.Pos()))
			case *ast.SelectorExpr:
				if id, ok := x.X.(*ast.Ident); ok && timeName != "" && id.Name == timeName && banned[x.Sel.Name] {
					t.Errorf("%s: calls time.%s; engine code takes the instant as a value", fset.Position(x.Pos()), x.Sel.Name)
				}
			}
			return true
		})
	}
}
