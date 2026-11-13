package scripts_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// gofmtStep returns the run block of the gofmt workflow's check step.
func gofmtStep(t *testing.T) string {
	t.Helper()
	var wf struct {
		Jobs map[string]struct {
			Steps []struct {
				Name string `yaml:"name"`
				Run  string `yaml:"run"`
			} `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal([]byte(readRepoFile(t, ".github", "workflows", "gofmt.yml")), &wf); err != nil {
		t.Fatal(err)
	}
	for _, j := range wf.Jobs {
		for _, s := range j.Steps {
			if s.Run != "" && strings.Contains(s.Run, "gofmt") {
				return s.Run
			}
		}
	}
	t.Fatal("gofmt.yml has no step that runs gofmt")
	return ""
}

// The gofmt job's step fails on an unformatted Go file anywhere in the tree and names it, passes
// on a formatted tree, and fails on a tree with no Go file (X18). The step is run as the job runs
// it, under bash with -e, in a throwaway git repository.
func TestGofmtWorkflowStepFailsOnAnUnformattedFile(t *testing.T) {
	step := gofmtStep(t)
	run := func(files map[string]string) (string, error) {
		dir := t.TempDir()
		for name, body := range files {
			p := filepath.Join(dir, name)
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		for _, args := range [][]string{{"init", "-q"}, {"add", "-A"}} {
			c := exec.Command("git", args...)
			c.Dir = dir
			if out, err := c.CombinedOutput(); err != nil {
				t.Fatalf("git %v: %v\n%s", args, err, out)
			}
		}
		c := exec.Command("bash", "-e", "-c", step)
		c.Dir = dir
		out, err := c.CombinedOutput()
		return string(out), err
	}
	clean := "package a\n\nfunc F() {}\n"
	dirty := "package b\nfunc  G( ) {}\n"

	if out, err := run(map[string]string{"a.go": clean, "sub/dir/c.go": clean}); err != nil {
		t.Errorf("a formatted tree failed: %v\n%s", err, out)
	}
	out, err := run(map[string]string{"a.go": clean, "sub/dir/b.go": dirty})
	if err == nil || !strings.Contains(out, "sub/dir/b.go") {
		t.Errorf("a nested unformatted file: err %v, output %q; want a failure naming sub/dir/b.go", err, out)
	}
	if out, err := run(map[string]string{"README.md": "x"}); err == nil {
		t.Errorf("a tree with no Go file passed:\n%s", out)
	}
}

// docs/guide/workloads.md states what `blis compose` does with num_requests, horizon and seed,
// and no longer says it drops num_requests (sim/workload.ComposeSpecs sums, takes the largest
// and keeps the shared seed; TestComposeSpecs_KeepsNumRequestsHorizonAndSeed pins the code).
func TestComposeDocStatesWhatComposeKeeps(t *testing.T) {
	doc := readRepoFile(t, "docs", "guide", "workloads.md")
	if strings.Contains(doc, "**Drops** `num_requests`") {
		t.Error("workloads.md still says compose drops num_requests")
	}
	for _, want := range []string{"**Sums** `num_requests`", "**largest** `horizon`", "**shared** `seed`"} {
		if !strings.Contains(doc, want) {
			t.Errorf("workloads.md does not state %s", want)
		}
	}
}
