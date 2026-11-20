package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// selectedBy runs the `changes` job's filter step of .github/workflows/ci.yml, exactly as the
// workflow has it, on a change to the given files, and returns the job outputs it writes.
func selectedBy(t *testing.T, changed ...string) map[string]string {
	t.Helper()
	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(root, ".github/workflows/ci.yml"))
	if err != nil {
		t.Fatal(err)
	}
	var wf struct {
		Jobs map[string]struct {
			Steps []struct {
				ID  string `yaml:"id"`
				Run string `yaml:"run"`
			} `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		t.Fatal(err)
	}
	var script string
	for _, s := range wf.Jobs["changes"].Steps {
		if s.ID == "filter" {
			script = s.Run
		}
	}
	if script == "" {
		t.Fatal("ci.yml has no `changes` step with id filter")
	}
	out := filepath.Join(t.TempDir(), "output")
	cmd := exec.Command("bash", "-c", script)
	cmd.Dir = root
	// BASE is a commit that exists, so the step does not take its run-everything shortcut.
	cmd.Env = append(os.Environ(), "GOFLAGS=", "GOWORK=off", "GOTOOLCHAIN=local", "BASE=HEAD", "GITHUB_OUTPUT="+out, "CHANGED_FILES="+strings.Join(changed, "\n"))
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("the filter step failed: %v\n%s", err, b)
	}
	got := map[string]string{}
	b, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		k, v, _ := strings.Cut(l, "=")
		got[k] = v
	}
	return got
}

// A change under load/ reaches the ledger job, because ledger imports load. The hand-written
// list in ci.yml skipped the ledger, catalog, and registry jobs on such a change.
func TestAChangeUnderLoadSelectsTheLedgerJob(t *testing.T) {
	got := selectedBy(t, "load/load.go")
	for _, job := range []string{"ledger", "kernel", "contracts"} {
		if got[job] != "true" {
			t.Errorf("a change to load/load.go does not select the %s job (outputs %v)", job, got)
		}
	}
}

// The selection stays narrow where nothing reaches: a README change runs neither the ledger nor
// the cost model.
func TestAReadmeChangeSelectsNeitherLedgerNorKernel(t *testing.T) {
	got := selectedBy(t, "README.md")
	for _, job := range []string{"ledger", "kernel", "registry", "catalog"} {
		if got[job] != "false" {
			t.Errorf("a change to README.md selects the %s job (outputs %v)", job, got)
		}
	}
}

// A new importer is followed with no edit to the tool: a changed file in a package that only the
// ledger imports selects the ledger and not the cost model.
func TestAChangeInALedgerPackageSelectsTheLedgerNotTheKernel(t *testing.T) {
	got := selectedBy(t, "ledger/inferencex/anything.go")
	if got["ledger"] != "true" || got["kernel"] != "false" {
		t.Errorf("want ledger selected and kernel not, got %v", got)
	}
}
