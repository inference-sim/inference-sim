package main

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
)

// The nightly status page (X16, design §13.4). .github/gates.yaml lists each gate: a
// workflow whose failure blocks a merge or a release, the paths that trigger it, and the
// invariants it checks:
//
//	- name: landing gate
//	  workflow: gate-landing.yml
//	  paths: [registry/**, kernel/**]
//	  invariants: [VAL-6]
//
// The page lists each gate's last real run (completed, not skipped or cancelled) on the
// default branch, and shows a gate whose last real run is more than seven days old, or
// that has never run, as failing. It also lists every invariant this repository runs a
// check for, with the result of the gates that check it. The job fails while any gate is
// shown as failing.

// Gate is one entry of gates.yaml.
type Gate struct {
	Name       string   `yaml:"name"`
	Workflow   string   `yaml:"workflow"`
	Paths      []string `yaml:"paths"`
	Invariants []string `yaml:"invariants"`
}

// Run is a gate's last real run.
type Run struct {
	When       time.Time
	Conclusion string // success, failure
	URL        string
}

const staleAfter = 7 * 24 * time.Hour

// ParseGates reads gates.yaml strictly, and checks every cited invariant exists.
func ParseGates(src []byte, reg *invariants.Registry) ([]Gate, error) {
	dec := yaml.NewDecoder(bytes.NewReader(src))
	dec.KnownFields(true)
	var gs []Gate
	if err := dec.Decode(&gs); err != nil {
		return nil, fmt.Errorf("gates.yaml: %w", err)
	}
	seen := map[string]bool{}
	for i, g := range gs {
		switch {
		case g.Name == "" || g.Workflow == "":
			return nil, fmt.Errorf("gates.yaml: entry %d needs name and workflow", i+1)
		case seen[g.Name]:
			return nil, fmt.Errorf("gates.yaml: %q is listed twice", g.Name)
		case len(g.Paths) == 0:
			return nil, fmt.Errorf("gates.yaml: %s declares no trigger paths (X16)", g.Name)
		}
		seen[g.Name] = true
		for _, id := range g.Invariants {
			if _, ok := reg.Get(id); !ok {
				return nil, fmt.Errorf("gates.yaml: %s cites %s, which is not in the invariant registry", g.Name, id)
			}
		}
	}
	return gs, nil
}

// CheckGatePaths compares each gate's declared paths with its workflow's pull_request
// paths, so removing a trigger needs a reviewed change to gates.yaml as well (X16).
func CheckGatePaths(gs []Gate, workflowPaths map[string][]string) []string {
	var bad []string
	for _, g := range gs {
		got, ok := workflowPaths[g.Workflow]
		if !ok {
			bad = append(bad, fmt.Sprintf("%s: workflow %s does not exist or has no pull_request paths (X16)", g.Name, g.Workflow))
			continue
		}
		a, b := append([]string(nil), g.Paths...), append([]string(nil), got...)
		sort.Strings(a)
		sort.Strings(b)
		if strings.Join(a, ",") != strings.Join(b, ",") {
			bad = append(bad, fmt.Sprintf("%s: %s triggers on %v, but gates.yaml declares %v (X16)", g.Name, g.Workflow, b, a))
		}
	}
	return bad
}

// StatusPage renders the page and returns the gates shown as failing.
func StatusPage(repo string, now time.Time, gs []Gate, last map[string]*Run, reg *invariants.Registry) (string, []string) {
	var b strings.Builder
	var failing []string
	fmt.Fprintf(&b, "# Status of %s on %s\n\n", repo, now.UTC().Format("2006-01-02 15:04 UTC"))
	b.WriteString("Generated nightly by the shared nightly workflow (design §13.4). A gate whose last real\n")
	b.WriteString("run is more than seven days old, or that has never run, is failing (X16).\n\n")
	b.WriteString("## Gates\n\n| Gate | Workflow | Last real run | Result | Status |\n|---|---|---|---|---|\n")
	state := map[string]string{}
	for _, g := range gs {
		r := last[g.Name]
		status, when, result := "passing", "never", "none"
		switch {
		case r == nil:
			status = "failing: never run"
		default:
			when = fmt.Sprintf("[%s](%s)", r.When.UTC().Format("2006-01-02"), r.URL)
			result = r.Conclusion
			switch {
			case now.Sub(r.When) > staleAfter:
				status = fmt.Sprintf("failing: last run %d days ago", int(now.Sub(r.When).Hours()/24))
			case r.Conclusion != "success":
				status = "failing"
			}
		}
		if status != "passing" {
			failing = append(failing, g.Name+": "+status)
		}
		for _, id := range g.Invariants {
			if state[id] == "" || status != "passing" {
				state[id] = status
			}
		}
		fmt.Fprintf(&b, "| %s | `%s` | %s | %s | %s |\n", g.Name, g.Workflow, when, result, status)
	}
	b.WriteString("\n## Invariants with a check in this repository\n\n| Invariant | Registry status | Target | Gate result |\n|---|---|---|---|\n")
	for _, inv := range reg.Invariants {
		in := false
		for _, s := range inv.Scope {
			in = in || s == repo
		}
		if !in {
			continue
		}
		target := "reached"
		if inv.Target.Milestone != nil {
			target = fmt.Sprintf("%s by %s", inv.Target.Status, invariants.MilestoneLabel(inv.Target.Milestone))
		}
		gate := state[inv.ID]
		if gate == "" {
			gate = "no gate"
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %s |\n", inv.ID, inv.Status, target, gate)
	}
	return b.String(), failing
}

// workflowPRPaths returns each workflow file's pull_request paths.
func workflowPRPaths(dir string) (map[string][]string, error) {
	files, err := filepath.Glob(filepath.Join(dir, "*.y*ml"))
	if err != nil {
		return nil, err
	}
	out := map[string][]string{}
	for _, f := range files {
		src, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		var doc map[string]any
		if err := yaml.Unmarshal(src, &doc); err != nil {
			return nil, fmt.Errorf("%s: %w", f, err)
		}
		on, _ := doc["on"].(map[string]any)
		pr, _ := on["pull_request"].(map[string]any)
		paths, _ := pr["paths"].([]any)
		if len(paths) == 0 {
			continue
		}
		var ps []string
		for _, p := range paths {
			if s, ok := p.(string); ok {
				ps = append(ps, s)
			}
		}
		out[filepath.Base(f)] = ps
	}
	return out, nil
}

// ghLastRealRun returns the last completed, not skipped or cancelled, run of a workflow on
// the default branch.
func ghLastRealRun(repo, workflow string) (*Run, error) {
	var res struct {
		WorkflowRuns []struct {
			Conclusion string    `json:"conclusion"`
			UpdatedAt  time.Time `json:"updated_at"`
			HTMLURL    string    `json:"html_url"`
		} `json:"workflow_runs"`
	}
	if err := ghJSON(&res, "api", fmt.Sprintf("repos/%s/actions/workflows/%s/runs?status=completed&per_page=50&exclude_pull_requests=true", repo, workflow)); err != nil {
		return nil, err
	}
	for _, r := range res.WorkflowRuns {
		if r.Conclusion == "success" || r.Conclusion == "failure" || r.Conclusion == "timed_out" {
			return &Run{When: r.UpdatedAt, Conclusion: r.Conclusion, URL: r.HTMLURL}, nil
		}
	}
	return nil, nil
}
