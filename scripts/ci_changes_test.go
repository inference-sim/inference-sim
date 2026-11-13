package scripts_test

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// scripts/ci-changes.sh decides which ci.yml jobs a change needs. A wrong "false" lets a broken
// change merge with a green run, so these tests pin both directions: the sample change lists a
// reviewer would reason about, the fail-closed fallback, and (against the Go import graph) that
// every Go file a job compiles turns that job on.

const ciModule = "github.com/inference-sim/inference-sim"

var ciGroups = strings.Fields("build lint core cluster kernel workload cmd root scripts llmd bytegate other stack selected actionlint claudemd")

// changesFor runs the filter on a list of paths and returns the groups it turns on, sorted.
func changesFor(t *testing.T, args []string, paths ...string) []string {
	t.Helper()
	script, err := filepath.Abs("ci-changes.sh")
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("bash", append([]string{script}, args...)...)
	cmd.Stdin = strings.NewReader(strings.Join(paths, "\n") + "\n")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("ci-changes.sh %v: %v\n%s", paths, err, stderr.String())
	}
	var on []string
	seen := map[string]bool{}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		name, val, ok := strings.Cut(line, "=")
		if !ok || (val != "true" && val != "false") {
			t.Fatalf("malformed output line %q", line)
		}
		seen[name] = true
		if val == "true" {
			on = append(on, name)
		}
	}
	for _, g := range ciGroups {
		if !seen[g] {
			t.Errorf("group %q missing from the output", g)
		}
	}
	sort.Strings(on)
	return on
}

func groupsOf(s string) []string {
	f := strings.Fields(s)
	sort.Strings(f)
	return f
}

func sameSet(a, b []string) bool { return strings.Join(a, " ") == strings.Join(b, " ") }

func TestCIChangesSampleChangeLists(t *testing.T) {
	everything := groupsOf(strings.Join(ciGroups, " "))
	for _, c := range []struct {
		name  string
		paths []string
		want  []string
	}{
		{"a documentation page", []string{"docs/guide/routing.md"},
			groupsOf("core cluster cmd scripts stack")},
		{"a test file of the cluster package", []string{"sim/cluster/instance_test.go"},
			groupsOf("lint cluster cmd")},
		{"production code of the cluster package", []string{"sim/cluster/instance.go"},
			groupsOf("build lint cluster cmd root llmd other bytegate selected")},
		{"production code of the core package", []string{"sim/metrics.go"},
			groupsOf("build lint core cluster kernel workload cmd root llmd other bytegate selected")},
		{"a byte gate golden file", []string{"testdata/bytegate/golden/a.json"},
			groupsOf("bytegate workload cmd")},
		{"a kernel fixture", []string{"testdata/kernel/glm5-h200-tp8.yaml"},
			groupsOf("cmd core cluster kernel other scripts")},
		{"a workload trace fixture", []string{"testdata/trace-v3/a.csv"}, groupsOf("workload")},
		{"the stack record", []string{"stack.json"}, groupsOf("cmd other stack selected root")},
		{"the router test directory only", []string{"sim/llmd/router_test.go"}, groupsOf("lint llmd cmd")},
		{"a workflow other than ci.yml", []string{".github/workflows/docs.yml"}, groupsOf("scripts actionlint")},
		{"an issue template (data under .github)", []string{".github/ISSUE_TEMPLATE/bug_report.md"}, nil},
		{"the license", []string{"LICENSE"}, nil},
		{"a spec document", []string{"specs/001-infra-nodes-gpus-instances/spec.md"}, nil},
		{"the Dockerfile", []string{"Dockerfile"}, groupsOf("scripts")},
		{"the size guard script", []string{"scripts/claude-md-size.sh"}, groupsOf("claudemd scripts")},
		{"a script no job but the scripts package reads", []string{"scripts/archon-build.sh"}, groupsOf("scripts")},
		{"the filter itself", []string{"scripts/ci-changes.sh"}, everything},
		{"the workflow itself", []string{".github/workflows/ci.yml"}, everything},
		{"go.sum", []string{"go.sum"}, everything},
		{"a path no class names", []string{"newdir/file.txt"}, everything},
		{"two changes add up", []string{"testdata/trace-v3/a.csv", "docs/index.md"},
			groupsOf("workload core cluster cmd scripts stack")},
		{"a move out of a package counts at both ends", []string{"sim/cluster/a.go", "sim/llmd/a.go"},
			groupsOf("build lint cluster cmd root llmd other bytegate selected")},
	} {
		got := changesFor(t, nil, c.paths...)
		if !sameSet(got, c.want) {
			t.Errorf("%s: %v turn on %v, want %v", c.name, c.paths, got, c.want)
		}
	}
}

// --all and an empty change list run everything: a dispatch is read as the whole suite, and an
// empty diff is not evidence that nothing changed.
func TestCIChangesAllAndEmpty(t *testing.T) {
	everything := groupsOf(strings.Join(ciGroups, " "))
	if got := changesFor(t, []string{"--all"}); !sameSet(got, everything) {
		t.Errorf("--all turned on %v", got)
	}
	if got := changesFor(t, nil, ""); !sameSet(got, everything) {
		t.Errorf("an empty change list turned on %v, want everything", got)
	}
}

// goListDirs runs go list with a format and returns one repository-relative directory per line.
func goList(t *testing.T, args ...string) []string {
	t.Helper()
	cmd := exec.Command("go", append([]string{"list"}, args...)...)
	cmd.Dir = ".."
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("go list %v: %v", args, err)
	}
	return strings.Fields(string(out))
}

func relDir(importPath string) string {
	rel := strings.TrimPrefix(importPath, ciModule)
	rel = strings.TrimPrefix(rel, "/")
	if rel == "" {
		return "."
	}
	return rel
}

func inDir(dir, file string) string {
	if dir == "." {
		return file
	}
	return dir + "/" + file
}

// ciMatrixLegs reads the groups and the packages ci.yml's matrices run for them.
func ciMatrixLegs(t *testing.T) map[string][]string {
	t.Helper()
	raw, err := os.ReadFile("../.github/workflows/ci.yml")
	if err != nil {
		t.Fatal(err)
	}
	var wf struct {
		Jobs map[string]struct {
			Strategy struct {
				Matrix struct {
					Include []struct {
						Packages string `yaml:"packages"`
						Flag     string `yaml:"flag"`
					} `yaml:"include"`
				} `yaml:"matrix"`
			} `yaml:"strategy"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		t.Fatal(err)
	}
	legs := map[string][]string{}
	for _, job := range wf.Jobs {
		for _, e := range job.Strategy.Matrix.Include {
			if e.Packages == "" {
				continue
			}
			if e.Flag == "" {
				t.Errorf("a matrix entry running %q has no flag, so no filter group turns it on", e.Packages)
			}
			for _, p := range strings.Fields(e.Packages) {
				if !contains(legs[e.Flag], p) {
					legs[e.Flag] = append(legs[e.Flag], p)
				}
			}
		}
	}
	return legs
}

func contains(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

// For each matrix entry, a Go file the entry's tests compile must turn the entry's group on: a
// _test.go file in the packages it tests, a non-test file in every package those tests import.
// The table in ci-changes.sh was written from `go list -deps -test`; this keeps it honest when
// an import is added.
func TestCIChangesCoverTheGoImportGraph(t *testing.T) {
	if testing.Short() {
		t.Skip("lists the module's packages with go list")
	}
	legs := ciMatrixLegs(t)
	if len(legs) == 0 {
		t.Fatal("ci.yml's matrices name no packages")
	}
	for flag, patterns := range legs {
		own := map[string]bool{}
		for _, p := range goList(t, append([]string{"-f", "{{.ImportPath}}"}, patterns...)...) {
			own[relDir(p)] = true
		}
		deps := goList(t, append([]string{"-deps", "-test", "-f", "{{.ImportPath}}"}, patterns...)...)
		closure := map[string]bool{}
		for _, d := range deps {
			if !strings.HasPrefix(d, ciModule) {
				continue
			}
			d = strings.TrimSuffix(d, ".test")
			if i := strings.Index(d, " ["); i >= 0 {
				d = d[:i]
			}
			d = strings.TrimSuffix(d, "_test")
			closure[relDir(d)] = true
		}
		for dir := range own {
			for _, f := range []string{"x.go", "x_test.go"} {
				if got := changesFor(t, nil, inDir(dir, f)); !contains(got, flag) {
					t.Errorf("group %s: %s is a file of a package it tests, yet turns on only %v", flag, inDir(dir, f), got)
				}
			}
		}
		for dir := range closure {
			if own[dir] {
				continue
			}
			if got := changesFor(t, nil, inDir(dir, "x.go")); !contains(got, flag) {
				t.Errorf("group %s: its tests import %s, yet x.go there turns on only %v", flag, dir, got)
			}
		}
	}
}

// The byte gate builds the blis binary and the selected-deps job links it, so a non-test Go file
// of any package the binary links, and every file it embeds, must turn both on. A _test.go file
// of those packages must not turn the byte gate on.
func TestCIChangesCoverTheBinary(t *testing.T) {
	if testing.Short() {
		t.Skip("lists the module's packages with go list")
	}
	pkgs := goList(t, "-deps", "-f", "{{if and (not .Standard) (not .DepOnly)}}{{end}}{{.ImportPath}}|{{.Dir}}|{{join .EmbedFiles \",\"}}", ".")
	root, err := filepath.Abs("..")
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for _, line := range pkgs {
		parts := strings.SplitN(line, "|", 3)
		if len(parts) != 3 || !strings.HasPrefix(parts[0], ciModule) {
			continue
		}
		dir := relDir(parts[0])
		checked++
		for _, g := range []string{"bytegate", "selected", "build"} {
			if got := changesFor(t, nil, inDir(dir, "x.go")); !contains(got, g) {
				t.Errorf("the binary links %s, yet x.go there leaves %s off (on: %v)", dir, g, got)
			}
		}
		if got := changesFor(t, nil, inDir(dir, "x_test.go")); contains(got, "bytegate") {
			t.Errorf("a _test.go file in %s turns the byte gate on, though the binary does not contain it", dir)
		}
		for _, e := range strings.Split(parts[2], ",") {
			if e == "" {
				continue
			}
			rel, err := filepath.Rel(root, filepath.Join(parts[1], e))
			if err != nil {
				t.Fatal(err)
			}
			if got := changesFor(t, nil, filepath.ToSlash(rel)); !contains(got, "bytegate") {
				t.Errorf("the binary embeds %s, yet it leaves the byte gate off (on: %v)", rel, got)
			}
		}
	}
	if checked < 5 {
		t.Fatalf("only %d packages of the binary found; the test is not looking at the build", checked)
	}
}

// Every group the filter prints is one ci.yml reads, and every group ci.yml reads is printed: a
// misspelled flag would otherwise leave a job permanently off.
func TestCIChangesGroupsMatchTheWorkflow(t *testing.T) {
	raw, err := os.ReadFile("../.github/workflows/ci.yml")
	if err != nil {
		t.Fatal(err)
	}
	var wf struct {
		Jobs map[string]struct {
			Outputs  map[string]string `yaml:"outputs"`
			Strategy struct {
				Matrix struct {
					Include []map[string]any `yaml:"include"`
				} `yaml:"matrix"`
			} `yaml:"strategy"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		t.Fatal(err)
	}
	outputs := wf.Jobs["changes"].Outputs
	for _, g := range ciGroups {
		if _, ok := outputs[g]; !ok {
			t.Errorf("group %s is printed by ci-changes.sh but is not an output of the changes job", g)
		}
	}
	for o := range outputs {
		if !contains(ciGroups, o) {
			t.Errorf("the changes job exports %s, which ci-changes.sh does not print", o)
		}
	}
	for name, job := range wf.Jobs {
		for _, e := range job.Strategy.Matrix.Include {
			if f, ok := e["flag"].(string); ok && !contains(ciGroups, f) {
				t.Errorf("job %s has a matrix entry with flag %q, which ci-changes.sh does not print", name, f)
			}
		}
	}
}
