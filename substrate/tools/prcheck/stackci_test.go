package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// pullRequestTrigger is the on.pull_request block of a workflow file.
type pullRequestTrigger struct {
	Branches       []string `yaml:"branches"`
	BranchesIgnore []string `yaml:"branches-ignore"`
	Types          []string `yaml:"types"`
	Paths          []string `yaml:"paths"`
	present        bool
}

type workflowFile struct {
	On          map[string]yaml.Node `yaml:"on"`
	Concurrency struct {
		Group            string `yaml:"group"`
		CancelInProgress any    `yaml:"cancel-in-progress"`
	} `yaml:"concurrency"`
	Jobs map[string]struct {
		Name string `yaml:"name"`
	} `yaml:"jobs"`
}

func readWorkflow(t *testing.T, text string) (workflowFile, pullRequestTrigger) {
	t.Helper()
	var w workflowFile
	if err := yaml.Unmarshal([]byte(text), &w); err != nil {
		t.Fatal(err)
	}
	var pr pullRequestTrigger
	if n, ok := w.On["pull_request"]; ok {
		if err := n.Decode(&pr); err != nil {
			t.Fatal(err)
		}
		pr.present = true
	}
	return w, pr
}

// stackCIFindings states how the pull-request workflows split by base (a stack's layers run
// ci-light.yml, a pull request to main runs the full set), and what keeps a retarget to main
// from starting no run. dir maps a workflow file name to its text.
func stackCIFindings(t *testing.T, files map[string]string) []string {
	t.Helper()
	var bad []string
	add := func(f, s string) { bad = append(bad, f+": "+s) }
	full := []string{"ci.yml", "rederive.yml", "ledger-candidate.yml"}
	for _, f := range full {
		_, pr := readWorkflow(t, files[f])
		if !pr.present {
			add(f, "has no pull_request trigger")
			continue
		}
		if !slices.Equal(pr.Branches, []string{"main"}) {
			add(f, "the full set must run for pull requests based on main only (branches: [main])")
		}
		// A retarget to main is reported as the activity type "edited", which is not a default
		// type: without it, retargeting a stacked layer to main starts no run.
		for _, ty := range []string{"edited", "synchronize", "opened", "reopened", "ready_for_review"} {
			if !slices.Contains(pr.Types, ty) {
				add(f, "pull_request types lack "+ty+", so a retarget or a push could start no run")
			}
		}
	}
	w, pr := readWorkflow(t, files["ci-light.yml"])
	if !slices.Equal(pr.BranchesIgnore, []string{"main"}) || len(pr.Branches) != 0 {
		add("ci-light.yml", "must run for every base except main (branches-ignore: [main])")
	}
	if !slices.Contains(pr.Types, "edited") {
		add("ci-light.yml", "pull_request types lack edited")
	}
	for id, j := range w.Jobs {
		for _, n := range []string{id, j.Name} {
			if n == "required" || n == "scan" || strings.HasPrefix(n, "pr / blis") {
				add("ci-light.yml", "job "+id+" is named like a required check, which a stacked layer must not carry")
			}
		}
	}
	// The required checks exist for a pull request to main and run whatever the base.
	ci, _ := readWorkflow(t, files["ci.yml"])
	if ci.Jobs["required"].Name != "required" {
		add("ci.yml", "has no job named required")
	}
	for _, f := range []string{"blis.yml", "private-names.yml"} {
		_, p := readWorkflow(t, files[f])
		if len(p.Branches) != 0 || len(p.BranchesIgnore) != 0 || !slices.Contains(p.Types, "edited") {
			add(f, "must run for every base and for edited pull requests: its check is required on main")
		}
	}
	// Two runs for one head must not both run, or a cancelled one must be harmless to the gate.
	for _, f := range []string{"ci.yml", "rederive.yml", "ledger-candidate.yml"} {
		c, _ := readWorkflow(t, files[f])
		if c.Concurrency.Group == "" || c.Concurrency.CancelInProgress == nil {
			add(f, "has no concurrency group that cancels the run a new one replaces")
		}
	}
	return bad
}

func stackCIFiles(t *testing.T) map[string]string {
	t.Helper()
	m := map[string]string{}
	for _, f := range []string{"ci.yml", "rederive.yml", "ledger-candidate.yml", "ci-light.yml", "blis.yml", "private-names.yml"} {
		b, err := os.ReadFile(filepath.Join("..", "..", ".github", "workflows", f))
		if err != nil {
			t.Fatal(err)
		}
		m[f] = string(b)
	}
	return m
}

// The full set runs for a base of main and the light set for any other base; both see a
// retarget; the names branch protection requires still run. Known-bad inputs: each edit below
// reverts one line of the change.
func TestStackAwareTriggers(t *testing.T) {
	files := stackCIFiles(t)
	none(t, stackCIFindings(t, files))
	mutate := func(file, old, new string) map[string]string {
		m := map[string]string{}
		for k, v := range files {
			m[k] = v
		}
		if !strings.Contains(m[file], old) {
			t.Fatalf("%s: %q not found", file, old)
		}
		m[file] = strings.Replace(m[file], old, new, 1)
		return m
	}
	t.Run("a full workflow with no branch filter runs on every layer", func(t *testing.T) {
		has(t, stackCIFindings(t, mutate("ci.yml", "branches: [main]\n    types", "types")), "branches: [main]")
	})
	t.Run("without edited a retarget to main starts no run", func(t *testing.T) {
		has(t, stackCIFindings(t, mutate("rederive.yml", "reopened, edited, ready_for_review", "reopened, ready_for_review")), "lack edited")
	})
	t.Run("a light workflow that also runs on main", func(t *testing.T) {
		has(t, stackCIFindings(t, mutate("ci-light.yml", "branches-ignore: [main]", "branches: [main]")), "branches-ignore")
	})
	t.Run("a layer job named like a required check", func(t *testing.T) {
		has(t, stackCIFindings(t, mutate("ci-light.yml", "  light-lint:", "  required:")), "named like a required check")
	})
	t.Run("rederive without a concurrency group", func(t *testing.T) {
		has(t, stackCIFindings(t, mutate("rederive.yml", "  group: rederive-", "  xgroup: rederive-")), "no concurrency group")
	})
	t.Run("the private-name scan loses edited", func(t *testing.T) {
		has(t, stackCIFindings(t, mutate("private-names.yml", "[opened, edited, synchronize, reopened]", "[opened, synchronize, reopened]")), "required on main")
	})
}
