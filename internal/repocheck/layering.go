// Package repocheck holds the checks on the shape of the repository itself: the import graph over
// directories (design Table 1, X2, X7), the path-reference lint of X7, and the module checks of
// SUB-13. They read the tree and `go list`, never the code's behaviour, and they pass trivially
// while a directory they police does not exist, so they can land before the directory does.
package repocheck

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os/exec"
	"sort"
	"strings"
)

// Package is the part of `go list -json` the import-graph lint reads.
type Package struct {
	ImportPath   string
	Imports      []string
	TestImports  []string
	XTestImports []string
	Error        *struct{ Err string }
}

// Violation is one finding; Rule says which row of the design it breaks.
type Violation struct {
	Where string
	What  string
	Rule  string
}

func (v Violation) String() string { return fmt.Sprintf("%s: %s (%s)", v.Where, v.What, v.Rule) }

// Component is one directory component of design Table 1: the directories it owns and the
// directories of the module its packages may import besides its own. A pattern ending in "/..."
// is that directory and everything below it; any other pattern is that one package directory.
// A package imports anything outside the module (the standard library, third-party modules, the
// router fork) without a rule here: those edges have their own checks (SUB-1, RTR-3).
type Component struct {
	Name string
	Dirs []string
	May  []string
}

// Table1 is the import column of design Table 1 for the directories that exist or are planned.
// The simulator (every directory outside substrate/ and engines/) may import the whole substrate
// and the engine modules, so it has no row. A directory under substrate/ or engines/ that no row
// owns is a finding: a new directory costs a row, the way a new path costs a class in
// scripts/ci-changes.sh. Tests count as imports, because a check that needs both sides of a
// boundary runs in the simulator's tests (X7).
var Table1 = []Component{
	{"contracts", []string{"substrate/contracts/..."}, nil},
	{"catalog", []string{"substrate/catalog/..."}, []string{"substrate/contracts/..."}},
	{"coefficient registry", []string{"substrate/registry/..."}, []string{"substrate/contracts/..."}},
	{"cost model", []string{"substrate/kernel/..."}, []string{"substrate/contracts/...", "substrate/load/..."}},
	{"validators and loader", []string{"substrate", "substrate/validate/...", "substrate/load/..."}, []string{"substrate/contracts/..."}},
	{"engine module vllm", []string{"engines/vllm/..."}, []string{"substrate/contracts/..."}},
	{"engine module sglang", []string{"engines/sglang/..."}, []string{"substrate/contracts/..."}},
}

func matches(pattern, dir string) bool {
	if base, ok := strings.CutSuffix(pattern, "/..."); ok {
		return dir == base || strings.HasPrefix(dir, base+"/")
	}
	return dir == pattern
}

func matchesAny(patterns []string, dir string) bool {
	for _, p := range patterns {
		if matches(p, dir) {
			return true
		}
	}
	return false
}

func governed(dir string) bool {
	return matches("substrate/...", dir) || matches("engines/...", dir)
}

// ListPackages runs `go list -e -json ./...` in dir and returns the module path and its packages.
func ListPackages(dir string) (string, []Package, error) {
	mod := exec.Command("go", "list", "-m")
	mod.Dir = dir
	modOut, err := mod.Output()
	if err != nil {
		return "", nil, fmt.Errorf("go list -m: %w", err)
	}
	cmd := exec.Command("go", "list", "-e", "-json=ImportPath,Imports,TestImports,XTestImports,Error", "./...")
	cmd.Dir = dir
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return "", nil, fmt.Errorf("go list ./...: %w\n%s", err, stderr.String())
	}
	var pkgs []Package
	dec := json.NewDecoder(bytes.NewReader(out))
	for dec.More() {
		var p Package
		if err := dec.Decode(&p); err != nil {
			return "", nil, err
		}
		if p.Error != nil {
			return "", nil, fmt.Errorf("go list: %s: %s", p.ImportPath, p.Error.Err)
		}
		pkgs = append(pkgs, p)
	}
	return strings.TrimSpace(string(modOut)), pkgs, nil
}

// CheckImports returns the packages of the module whose imports break table, sorted.
func CheckImports(module string, pkgs []Package, table []Component) []Violation {
	rel := func(path string) (string, bool) {
		if path == module {
			return ".", true
		}
		return strings.CutPrefix(path, module+"/")
	}
	var out []Violation
	for _, p := range pkgs {
		dir, ok := rel(p.ImportPath)
		if !ok {
			continue
		}
		var comp *Component
		for i := range table {
			if matchesAny(table[i].Dirs, dir) {
				comp = &table[i]
				break
			}
		}
		if comp == nil {
			if governed(dir) {
				out = append(out, Violation{dir, "is in no component of design Table 1; add a row to repocheck.Table1", "X7, Table 1"})
			}
			continue
		}
		seen := map[string]bool{}
		for _, imp := range append(append(append([]string{}, p.Imports...), p.TestImports...), p.XTestImports...) {
			to, ok := rel(imp)
			if !ok || seen[to] || matchesAny(comp.Dirs, to) || matchesAny(comp.May, to) {
				continue
			}
			seen[to] = true
			out = append(out, Violation{dir, fmt.Sprintf("imports %s, which the %s component may not (it may import its own directories and %s)",
				to, comp.Name, mayList(comp)), ruleOf(dir, to)})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Where != out[j].Where {
			return out[i].Where < out[j].Where
		}
		return out[i].What < out[j].What
	})
	return out
}

func mayList(c *Component) string {
	if len(c.May) == 0 {
		return "nothing else of the module"
	}
	return strings.Join(c.May, ", ")
}

func ruleOf(from, to string) string {
	switch {
	case matches("substrate/...", from) && matches("engines/...", to):
		return "X2: the substrate knows no engine"
	case matches("engines/...", from) && matches("engines/...", to):
		return "Table 1: an engine module imports no other engine module"
	}
	return "X7, Table 1"
}
