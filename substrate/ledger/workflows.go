package ledger

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// SeedEnvironment is the GitHub environment that holds the seed. A ledger owner creates it,
// with a branch policy that allows only main (D7).
const SeedEnvironment = "ledger"

// SeedSecret is the secret's name inside SeedEnvironment.
const SeedSecret = "LEDGER_SEED"

// SeedJobs are the only jobs that may run in SeedEnvironment: the release, select, error and
// gate jobs (design §10.2, glossary "Seed-holding job").
var SeedJobs = []string{"release", "select", "error", "gate"}

type workflow struct {
	On          yaml.Node        `yaml:"on"`
	Permissions yaml.Node        `yaml:"permissions"`
	Jobs        map[string]wfJob `yaml:"jobs"`
	Rest        map[string]any   `yaml:",inline"`
}

type wfJob struct {
	If          string         `yaml:"if"`
	Environment yaml.Node      `yaml:"environment"`
	Permissions yaml.Node      `yaml:"permissions"`
	Steps       []wfStep       `yaml:"steps"`
	Rest        map[string]any `yaml:",inline"`
}

type wfStep struct {
	Uses string            `yaml:"uses"`
	Run  string            `yaml:"run"`
	With map[string]string `yaml:"with"`
	Env  map[string]string `yaml:"env"`
	Rest map[string]any    `yaml:",inline"`
}

var (
	seedRef    = regexp.MustCompile(`secrets\.` + SeedSecret + `\b`)
	anySecret  = regexp.MustCompile(`secrets\.[A-Za-z_]`)
	mainOnlyIf = regexp.MustCompile(`github\.ref\s*==\s*'refs/heads/main'`)
	// listsUnitReads matches a job that lists the artifacts of unit-reads records; such a job
	// must keep only those a run on main of this repository uploaded.
	listsUnitReads   = regexp.MustCompile(`unit-reads-`)
	unitReadsOnMain  = regexp.MustCompile(`workflow_run\.head_branch\s*==\s*"main"`)
	unitReadsOwnRepo = regexp.MustCompile(`workflow_run\.head_repository_id`)
	// takesRunID matches a job that downloads the run a person named.
	takesRunID     = regexp.MustCompile(`inputs\.run_id`)
	runVerifyAPI   = regexp.MustCompile(`actions/runs/\$RUN_ID`)
	runVerifyName  = regexp.MustCompile(`ledger candidate`)
	runVerifyState = regexp.MustCompile(`success`)
	runVerifyRepo  = regexp.MustCompile(`head_repository`)
	prControlled   = regexp.MustCompile(`github\.event\.(pull_request|workflow_run)\.(head|head_sha|head_branch|head_repository)|github\.head_ref`)
)

// LintWorkflows is D7's workflow lint. For every workflow under dir it checks:
//
//   - A job that runs in SeedEnvironment is one of SeedJobs, by job id or by the prefix
//     "<seed job>-"; it is guarded by `if: github.ref == 'refs/heads/main'`; its workflow is
//     started only by workflow_run, schedule or workflow_dispatch, never by a pull request or a
//     push; every checkout in it checks out ref main of this repository, so a local action it
//     runs after the first checkout is main's; it uses no expression a pull request controls
//     (the head ref or sha).
//   - A seed-holding job that lists unit-reads artifacts keeps only those uploaded by a run on
//     main of this repository (workflow_run.head_branch and head_repository_id), since a pull
//     request run can upload an artifact of that name.
//   - A seed-holding job that takes inputs.run_id checks, before downloading, that the run is a
//     successful run of "ledger candidate" whose head repository is this one.
//   - No other job reads the seed secret.
//   - A candidate job (job id "candidate" or "candidate-…") reads no secret, runs in no
//     environment and declares read-only permissions.
//
// It returns the number of seed-holding jobs found, and fails if there are none (X22).
func LintWorkflows(dir string) (int, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*.yml"))
	if err != nil {
		return 0, err
	}
	more, _ := filepath.Glob(filepath.Join(dir, "*.yaml"))
	paths = append(paths, more...)
	sort.Strings(paths)
	var errs []error
	seedJobs := 0
	for _, p := range paths {
		n, err := lintWorkflow(p)
		seedJobs += n
		if err != nil {
			errs = append(errs, err)
		}
	}
	if seedJobs == 0 {
		errs = append(errs, fmt.Errorf("%s: no job runs in environment %q; the release, select, error and gate jobs are missing", dir, SeedEnvironment))
	}
	return seedJobs, errors.Join(errs...)
}

func lintWorkflow(path string) (int, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	var wf workflow
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		return 0, fmt.Errorf("%s: %w", path, err)
	}
	name := filepath.Base(path)
	triggers := triggerNames(&wf.On)
	prStarted := triggers["pull_request"] || triggers["pull_request_target"] || triggers["push"] || triggers["merge_group"] || triggers["issue_comment"]
	var errs []error
	seed := 0
	for _, id := range sortedKeys(wf.Jobs) {
		j := wf.Jobs[id]
		at := fmt.Sprintf("%s: jobs.%s", name, id)
		text := jobText(&j)
		if environmentName(&j.Environment) == SeedEnvironment {
			seed++
			if !isSeedJob(id) {
				errs = append(errs, fmt.Errorf("%s: runs in environment %q but is not one of the seed-holding jobs %v", at, SeedEnvironment, SeedJobs))
			}
			for t := range triggers {
				if t != "workflow_run" && t != "schedule" && t != "workflow_dispatch" {
					errs = append(errs, fmt.Errorf("%s: holds the seed in a workflow started by %s; a seed-holding job is started only by workflow_run, schedule or workflow_dispatch (D7)", at, t))
				}
			}
			if !mainOnlyIf.MatchString(j.If) {
				errs = append(errs, fmt.Errorf("%s: holds the seed without `if: github.ref == 'refs/heads/main'` (D7)", at))
			}
			if prControlled.MatchString(text) {
				errs = append(errs, fmt.Errorf("%s: holds the seed and reads a ref or sha a pull request controls (D7)", at))
			}
			if listsUnitReads.MatchString(text) && (!unitReadsOnMain.MatchString(text) || !unitReadsOwnRepo.MatchString(text)) {
				errs = append(errs, fmt.Errorf("%s: lists unit-reads artifacts without keeping only those a run on main of this repository uploaded (workflow_run.head_branch == \"main\" and workflow_run.head_repository_id) (D2)", at))
			}
			if takesRunID.MatchString(text) && (!runVerifyAPI.MatchString(text) || !runVerifyName.MatchString(text) ||
				!runVerifyState.MatchString(text) || !runVerifyRepo.MatchString(text)) {
				errs = append(errs, fmt.Errorf("%s: downloads the run in inputs.run_id without checking, through actions/runs/$RUN_ID, that it is a successful ledger candidate run of this repository (D7)", at))
			}
			checkouts := 0
			for i, s := range j.Steps {
				if strings.HasPrefix(s.Uses, "./") && checkouts == 0 {
					errs = append(errs, fmt.Errorf("%s.steps[%d]: runs a local action before checking out main (D7)", at, i))
				}
				if strings.HasPrefix(s.Uses, "actions/checkout@") {
					checkouts++
					if ref := s.With["ref"]; ref != "main" && ref != "refs/heads/main" {
						errs = append(errs, fmt.Errorf("%s.steps[%d]: checks out ref %q; a seed-holding job checks out only main (D7)", at, i, ref))
					}
					if repo := s.With["repository"]; repo != "" && repo != "${{ github.repository }}" {
						errs = append(errs, fmt.Errorf("%s.steps[%d]: checks out %s; a seed-holding job checks out only this repository's main (D7)", at, i, repo))
					}
				}
			}
			continue
		}
		if seedRef.MatchString(text) {
			errs = append(errs, fmt.Errorf("%s: reads %s outside environment %q (D7)", at, SeedSecret, SeedEnvironment))
		}
		if strings.HasPrefix(id, "candidate") {
			if anySecret.MatchString(text) {
				errs = append(errs, fmt.Errorf("%s: a candidate job reads a secret (D7)", at))
			}
			if !j.Environment.IsZero() {
				errs = append(errs, fmt.Errorf("%s: a candidate job runs in no environment (D7)", at))
			}
			if !readOnly(&j.Permissions) && (!j.Permissions.IsZero() || !readOnly(&wf.Permissions)) {
				errs = append(errs, fmt.Errorf("%s: a candidate job declares read-only permissions (D7)", at))
			}
		}
	}
	if prStarted && seed > 0 {
		errs = append(errs, fmt.Errorf("%s: a workflow a pull request starts holds the seed (D7)", name))
	}
	return seed, errors.Join(errs...)
}

func isSeedJob(id string) bool {
	for _, s := range SeedJobs {
		if id == s || strings.HasPrefix(id, s+"-") {
			return true
		}
	}
	return false
}

func triggerNames(n *yaml.Node) map[string]bool {
	out := map[string]bool{}
	switch n.Kind {
	case yaml.ScalarNode:
		out[n.Value] = true
	case yaml.SequenceNode:
		for _, c := range n.Content {
			out[c.Value] = true
		}
	case yaml.MappingNode:
		for i := 0; i+1 < len(n.Content); i += 2 {
			out[n.Content[i].Value] = true
		}
	}
	return out
}

func environmentName(n *yaml.Node) string {
	switch n.Kind {
	case yaml.ScalarNode:
		return n.Value
	case yaml.MappingNode:
		for i := 0; i+1 < len(n.Content); i += 2 {
			if n.Content[i].Value == "name" {
				return n.Content[i+1].Value
			}
		}
	}
	return ""
}

// readOnly reports whether a permissions block grants no write: "read-all", "{}", or a map
// whose every value is read or none.
func readOnly(n *yaml.Node) bool {
	switch n.Kind {
	case yaml.ScalarNode:
		return n.Value == "read-all" || n.Value == "{}"
	case yaml.MappingNode:
		for i := 0; i+1 < len(n.Content); i += 2 {
			if v := n.Content[i+1].Value; v != "read" && v != "none" {
				return false
			}
		}
		return true
	}
	return false
}

func jobText(j *wfJob) string {
	b, _ := yaml.Marshal(j)
	return string(b)
}
