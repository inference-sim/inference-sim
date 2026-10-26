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
				} `yaml:"include"`
			} `yaml:"matrix"`
		} `yaml:"strategy"`
	} `yaml:"jobs"`
}

// ciTestPatterns returns the package patterns ci.yml's test matrix runs, one per word of each
// entry's packages, relative to the repository root ("./sim/cluster", "./sim/kernelmodel/...").
func ciTestPatterns(t *testing.T, path string) []string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var wf ciWorkflow
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	job, ok := wf.Jobs["test"]
	if !ok {
		t.Fatalf("job \"test\" missing from %s: the test matrix was restructured, so re-derive "+
			"this check rather than deleting it", path)
	}
	var out []string
	for _, e := range job.Strategy.Matrix.Include {
		out = append(out, strings.Fields(e.Packages)...)
	}
	if len(out) == 0 {
		t.Fatalf("%s: the test job's matrix lists no packages", path)
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

// uncovered returns the packages no pattern runs, sorted.
func uncovered(patterns, pkgs []string) []string {
	var out []string
	for _, p := range pkgs {
		hit := false
		for _, pat := range patterns {
			if coveredBy(pat, p) {
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
// time (verification pass 8 of #2021, gap 1). Every package with tests must be run by some entry.
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
		{"./sim/kernelmodel/...", "./sim/kernelmodel", true},
		{"./sim/kernelmodel/...", "./sim/kernelmodel/harness", true},
		{"./sim/kernelmodel/...", "./sim/kernelmodelx", false},
		{".", ".", true},
		{".", "./cmd", false},
		{"./sim/internal/...", "./sim/internal/testutil/fakelatency", true},
	} {
		if got := coveredBy(c.pattern, c.pkg); got != c.want {
			t.Errorf("coveredBy(%q, %q) = %v, want %v", c.pattern, c.pkg, got, c.want)
		}
	}
	got := uncovered([]string{"./a", "./b/..."}, []string{"./a", "./b/c", "./d", "./c"})
	if strings.Join(got, ",") != "./c,./d" {
		t.Errorf("uncovered = %v, want [./c ./d]", got)
	}
}
