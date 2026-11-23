package scripts_test

import (
	"os"
	"os/exec"
	"sort"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// ciWorkflow models only the fields TestEveryPackageWithTestsRunsInCI reads from ci.yml.
type ciWorkflow struct {
	Jobs map[string]struct {
		Strategy struct {
			Matrix struct {
				Include []struct {
					Name     string `yaml:"name"`
					Packages string `yaml:"packages"`
					Exclude  string `yaml:"exclude"`
				} `yaml:"include"`
			} `yaml:"matrix"`
		} `yaml:"strategy"`
	} `yaml:"jobs"`
}

// ciEntry is one matrix entry's package patterns and the packages it leaves out (its exclude),
// relative to the repository root ("./sim/cluster", "./tools/...").
type ciEntry struct{ patterns, exclude []string }

// ciTestPatterns returns the entries of ci.yml's test matrices that list packages.
func ciTestPatterns(t *testing.T, path string) []ciEntry {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var wf ciWorkflow
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	// The test matrix is split over several jobs (the sharded ones run one package as parallel
	// shards), so every job's matrix counts. A package is covered when some entry names it.
	var out []ciEntry
	for _, job := range wf.Jobs {
		for _, e := range job.Strategy.Matrix.Include {
			if e.Packages != "" {
				out = append(out, ciEntry{strings.Fields(e.Packages), strings.Fields(e.Exclude)})
			}
		}
	}
	if _, ok := wf.Jobs["test"]; !ok {
		t.Fatalf("job \"test\" missing from %s: the test matrix was restructured, so re-derive "+
			"this check rather than deleting it", path)
	}
	if len(out) == 0 {
		t.Fatalf("%s: no job's matrix lists any packages", path)
	}
	return out
}

// coveredBy reports whether a go test pattern, as `go test` reads it from the repository root,
// runs the package at the root-relative path pkg ("." or "./sim/cluster").
func coveredBy(pattern, pkg string) bool {
	if base, ok := strings.CutSuffix(pattern, "/..."); ok {
		return pkg == base || strings.HasPrefix(pkg, base+"/")
	}
	return pattern == pkg
}

// runs reports whether the entry runs pkg: a pattern covers it and no exclude does.
func (e ciEntry) runs(pkg string) bool {
	in := func(pats []string) bool {
		for _, pat := range pats {
			if coveredBy(pat, pkg) {
				return true
			}
		}
		return false
	}
	return in(e.patterns) && !in(e.exclude)
}

// uncovered returns the packages no entry runs, sorted.
func uncovered(entries []ciEntry, pkgs []string) []string {
	var out []string
	for _, p := range pkgs {
		hit := false
		for _, e := range entries {
			if e.runs(p) {
				hit = true
				break
			}
		}
		if !hit {
			out = append(out, p)
		}
	}
	sort.Strings(out)
	return out
}

// The test matrix lists packages explicitly rather than ./..., so a package whose tests no entry
// names never runs in CI, and nothing says so. sim/vendored's lock test was such a package for a
// time. Every package with tests must be run by some entry.
func TestEveryPackageWithTestsRunsInCI(t *testing.T) {
	if testing.Short() {
		t.Skip("lists the module's packages with go list")
	}
	cmd := exec.Command("go", "list", "-f", "{{if or .TestGoFiles .XTestGoFiles}}{{.ImportPath}}{{end}}", "./...")
	cmd.Dir = ".."
	raw, err := cmd.Output()
	if err != nil {
		t.Fatalf("go list: %v", err)
	}
	const module = "github.com/inference-sim/inference-sim"
	var pkgs []string
	for _, line := range strings.Fields(string(raw)) {
		rel := strings.TrimPrefix(line, module)
		pkgs = append(pkgs, "."+rel)
	}
	if len(pkgs) == 0 {
		t.Fatal("go list found no package with tests")
	}
	if missing := uncovered(ciTestPatterns(t, "../.github/workflows/ci.yml"), pkgs); len(missing) > 0 {
		t.Errorf("packages with tests that no entry of ci.yml's test matrix runs: %s. Add each to an "+
			"entry (a new entry, not a renamed one: the existing entries' names are required checks)",
			strings.Join(missing, ", "))
	}
}

// The coverage check reads a pattern the way go test does: an exact package, or a /... tree.
func TestCITestPatternCoverage(t *testing.T) {
	for _, c := range []struct {
		pattern, pkg string
		want         bool
	}{
		{"./sim/cluster", "./sim/cluster", true},
		{"./sim", "./sim/cluster", false},
		{"./tools/...", "./tools", true},
		{"./tools/...", "./tools/harness/internal/artifacts", true},
		{"./tools/...", "./toolsx", false},
		{".", ".", true},
		{".", "./cmd", false},
		{"./sim/internal/...", "./sim/internal/testutil", true},
	} {
		if got := coveredBy(c.pattern, c.pkg); got != c.want {
			t.Errorf("coveredBy(%q, %q) = %v, want %v", c.pattern, c.pkg, got, c.want)
		}
	}
	got := uncovered([]ciEntry{{patterns: []string{"./a", "./b/..."}}}, []string{"./a", "./b/c", "./d", "./c"})
	if strings.Join(got, ",") != "./c,./d" {
		t.Errorf("uncovered = %v, want [./c ./d]", got)
	}
	// An excluded package counts only when another entry runs it.
	tree := ciEntry{patterns: []string{"./e/..."}, exclude: []string{"./e/slow"}}
	if got := uncovered([]ciEntry{tree}, []string{"./e/fast", "./e/slow"}); strings.Join(got, ",") != "./e/slow" {
		t.Errorf("uncovered with an exclude = %v, want [./e/slow]", got)
	}
	if got := uncovered([]ciEntry{tree, {patterns: []string{"./e/slow"}}}, []string{"./e/fast", "./e/slow"}); len(got) != 0 {
		t.Errorf("uncovered with the excluded package run elsewhere = %v, want none", got)
	}
}
