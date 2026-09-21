package scripts_test

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// The delivery phases' failure reporters must be guarded on `always() && !success()`.
//
// The SEMANTICS of that expression are exercised against real GitHub infrastructure by
// .github/workflows/deliver-guards-selftest.yml, which cancels a step via a job timeout and asserts
// the reporter is reached. That workflow cannot run on every pull request: cancelling a job is the
// condition under test, so it necessarily reports a non-green job and would leave a permanently red
// check on every PR.
//
// This test is the half that CAN run everywhere. It asserts the workflows still use the guard the
// self-test exercises, which is what stops a green self-test result from vouching for a file that
// has since been reverted. Cheap, and it runs in the existing `test (scripts)` CI group.
func TestDeliveryReportersUseTheExercisedGuard(t *testing.T) {
	phases := []string{
		"deliver-implement.yml",
		"deliver-verify.yml",
		"deliver-correct.yml",
	}

	// Matched as a step condition rather than anywhere in the file, so the explanatory prose in the
	// surrounding comments — which necessarily quotes the old form to explain why it was replaced —
	// does not trip this.
	oldGuard := regexp.MustCompile(`(?m)^\s*if:.*failure\(\)\s*\|\|\s*cancelled\(\)`)

	for _, phase := range phases {
		t.Run(phase, func(t *testing.T) {
			path := filepath.Join("..", ".github", "workflows", phase)
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("reading %s: %v", path, err)
			}
			body := string(raw)

			if !strings.Contains(body, "if: always() && !success()") {
				t.Errorf("%s has no reporter guarded on `always() && !success()`. "+
					"deliver-guards-selftest.yml exercises that guard; a phase using a different "+
					"one is not covered by it, and a cancelled step could go unreported as on #1685",
					phase)
			}

			if loc := oldGuard.FindString(body); loc != "" {
				t.Errorf("%s reintroduces `failure() || cancelled()` as a step condition (%q). "+
					"It is a strict subset of the exercised guard, so replacing it silently narrows "+
					"which failures get reported",
					phase, strings.TrimSpace(loc))
			}
		})
	}
}

// implementStep is the subset of a deliver-implement.yml step this file asserts on.
type implementStep struct {
	ID   string `yaml:"id"`
	Name string `yaml:"name"`
	Uses string `yaml:"uses"`
	If   string `yaml:"if"`
	With struct {
		Prompt string `yaml:"prompt"`
	} `yaml:"with"`
}

func loadImplementSteps(t *testing.T) []implementStep {
	t.Helper()
	path := filepath.Join("..", ".github", "workflows", "deliver-implement.yml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	var wf struct {
		Jobs map[string]struct {
			TimeoutMinutes int             `yaml:"timeout-minutes"`
			Steps          []implementStep `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		t.Fatalf("parsing %s: %v", path, err)
	}
	job, ok := wf.Jobs["deliver"]
	if !ok {
		t.Fatal("deliver-implement.yml has no `deliver` job")
	}
	if len(job.Steps) == 0 {
		t.Fatal("the `deliver` job has no steps")
	}
	return job.Steps
}

func indexOfStep(steps []implementStep, match func(implementStep) bool) int {
	for i, s := range steps {
		if match(s) {
			return i
		}
	}
	return -1
}

// The delivery BRANCH must be created and pushed BEFORE the agent runs (#1722).
//
// This is what makes an interrupted delivery recoverable. A runner that dies mid-job executes no
// step, not even `always()` ones, so an agent that pushes only at the end leaves nothing behind —
// that lost #1706 a finished 52-minute implementation.
//
// Asserted as step ORDER rather than mere presence: a seeding step placed after the agent would
// satisfy a `strings.Contains` check while restoring exactly the failure mode it removes.
func TestDeliverImplementSeedsTheBranchBeforeTheAgentRuns(t *testing.T) {
	steps := loadImplementSteps(t)

	seed := indexOfStep(steps, func(s implementStep) bool { return s.ID == "seed" })
	if seed < 0 {
		t.Fatal("deliver-implement.yml has no step with `id: seed`. The delivery branch must be " +
			"pushed by the workflow before the agent runs, so that a run interrupted part-way " +
			"leaves recoverable commits (#1722)")
	}

	agent := indexOfStep(steps, func(s implementStep) bool {
		return strings.HasPrefix(s.Uses, "anthropics/claude-code-action")
	})
	if agent < 0 {
		t.Fatal("deliver-implement.yml no longer runs anthropics/claude-code-action")
	}

	if seed > agent {
		t.Errorf("the `seed` step is at index %d, AFTER the agent step at index %d. Seeding after "+
			"the agent means a dead runner again leaves no branch — the #1706 failure",
			seed, agent)
	}

	// The hand-off must require BOTH a PR and real work. The PR because the agent opens it and may
	// not have; the work because a PR alone is not evidence anything was built.
	handoff := indexOfStep(steps, func(s implementStep) bool { return s.Name == "Hand off to verify" })
	if handoff < 0 {
		t.Fatal("deliver-implement.yml has no `Hand off to verify` step")
	}
	if !strings.Contains(steps[handoff].If, "steps.work.outputs.changed == 'true'") {
		t.Errorf("`Hand off to verify` is guarded on %q, which does not require "+
			"`steps.work.outputs.changed == 'true'`. Without it an agent that built nothing hands "+
			"an empty PR to a 120-minute verify plus a full CI dispatch",
			steps[handoff].If)
	}
	if !strings.Contains(steps[handoff].If, "steps.pr.outputs.number != ''") {
		t.Errorf("`Hand off to verify` is guarded on %q, which does not require a PR to exist. "+
			"Verify is dispatched with a PR number and cannot run without one",
			steps[handoff].If)
	}
}

// The workflow must NOT create the pull request itself, and this is a security property rather
// than a style preference.
//
// `gh pr create` from a workflow step uses GITHUB_TOKEN, which GitHub gates behind the repository
// setting "Allow GitHub Actions to create and approve pull requests". That is a SINGLE toggle
// (`can_approve_pull_request_reviews`) granting creation AND approval to every workflow in the
// repository — there is no way to take only the half this loop needs. The delivery loop must never
// approve anything: it labels, and a human merges. So PR creation stays with the agent, which uses
// an App installation token obtained via OIDC that the setting does not govern (every delivery PR
// to date — #1680, #1708, #1713 — is authored by `app/claude`).
//
// Reintroducing a workflow-side `gh pr create` would silently re-acquire that dependency and, to
// make deliveries work at all, pressure someone into enabling repo-wide PR approval for Actions.
func TestDeliverImplementDoesNotCreatePRsWithTheWorkflowToken(t *testing.T) {
	// GLOBBED, not a list of the three delivery phases: the setting this protects is repo-WIDE, so
	// a `gh pr create` added to any other workflow (archon.yml, claude.yml, …) acquires the same
	// approval capability. Break-tested on #1723: with a hardcoded list, the same line added to
	// archon.yml was missed.
	paths, err := filepath.Glob(filepath.Join("..", ".github", "workflows", "*.yml"))
	if err != nil {
		t.Fatalf("globbing workflows: %v", err)
	}
	// `.yaml` too: GitHub accepts both extensions, so matching only one leaves a way in.
	yamlPaths, err := filepath.Glob(filepath.Join("..", ".github", "workflows", "*.yaml"))
	if err != nil {
		t.Fatalf("globbing workflows: %v", err)
	}
	paths = append(paths, yamlPaths...)
	if len(paths) < 5 {
		t.Fatalf("found only %d workflow files; the glob is not matching the workflow directory", len(paths))
	}
	for _, path := range paths {
		phase := filepath.Base(path)
		t.Run(phase, func(t *testing.T) {
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("reading %s: %v", path, err)
			}
			var wf struct {
				Jobs map[string]struct {
					Steps []struct {
						Name string `yaml:"name"`
						Run  string `yaml:"run"`
						With struct {
							Script string `yaml:"script"`
						} `yaml:"with"`
					} `yaml:"steps"`
				} `yaml:"jobs"`
			}
			if err := yaml.Unmarshal(raw, &wf); err != nil {
				t.Fatalf("parsing %s: %v", path, err)
			}
			// Only `run:` scripts and github-script bodies — the agent's PROMPT legitimately tells
			// the agent to call `gh pr create`, and that runs with the App token, not GITHUB_TOKEN.
			//
			// Comment lines are stripped before matching, for the reason the reporter-guard test
			// above gives for itself: the explanatory comment on the seeding step has to NAME
			// `gh pr create` in order to explain why it deliberately does not call it, and matching
			// prose would make this test fail on the very code that satisfies it.
			for job, j := range wf.Jobs {
				for _, s := range j.Steps {
					for label, code := range map[string]string{"run": s.Run, "script": s.With.Script} {
						code = stripCommentLines(code)
						// Three spellings of the same call: the gh porcelain, the octokit binding
						// github-script exposes, and a raw REST POST. Missing the last one was
						// flagged in review — `gh api -X POST /repos/o/r/pulls` creates a PR just
						// as effectively.
						restPost := strings.Contains(code, "gh api") &&
							strings.Contains(code, "POST") &&
							strings.Contains(code, "/pulls")
						if strings.Contains(code, "gh pr create") || strings.Contains(code, "pulls.create") || restPost {
							t.Errorf("%s job %q step %q creates a pull request from a workflow %s. "+
								"That uses GITHUB_TOKEN, which requires the repo-wide \"Allow GitHub "+
								"Actions to create and approve pull requests\" setting — one toggle that "+
								"also grants APPROVAL to every workflow here. This loop must never "+
								"approve. Let the agent open the PR with its App token instead",
								phase, job, s.Name, label)
						}
					}
				}
			}
		})
	}
}

// The implement prompt has to say the things that stop the two #1706 failures recurring.
//
// Both were prompt-level, and neither is visible in the workflow's structure: the agent invoked
// `superpowers:brainstorming` and ended its turn awaiting a human, and it spent the run
// duplicating ci.yml's build/test/lint on the resource-constrained self-hosted runner.
func TestDeliverImplementPromptContract(t *testing.T) {
	steps := loadImplementSteps(t)
	agent := indexOfStep(steps, func(s implementStep) bool {
		return strings.HasPrefix(s.Uses, "anthropics/claude-code-action")
	})
	if agent < 0 {
		t.Fatal("deliver-implement.yml no longer runs anthropics/claude-code-action")
	}
	prompt := steps[agent].With.Prompt
	if strings.TrimSpace(prompt) == "" {
		t.Fatal("the agent step has no prompt")
	}

	required := []struct {
		needle string
		why    string
	}{
		{
			needle: "UNATTENDED",
			why: "the prompt must tell the agent nobody will reply. On #1706 it posted a design " +
				"proposal and ended its turn, and the phase reported success with zero commits",
		},
		{
			needle: "superpowers:brainstorming",
			why: "the prompt must name brainstorming as out of scope. The superpowers SessionStart " +
				"hook injects a `1% chance ⇒ you MUST invoke it` rule that reads a delivery prompt " +
				"as \"let's build X\", which is how #1706 ended its turn without implementing",
		},
		{
			needle: "PUSH AS YOU GO",
			why: "the prompt must require incremental pushes. #1706's second attempt finished the " +
				"work and lost all of it because the runner was evicted before the single push",
		},
		{
			needle: "ci.yml",
			why: "the prompt must point at ci.yml as the build/test/lint authority instead of " +
				"having the agent run them, which is what exhausted the runner on #1706",
		},
		{
			needle: "WITH THE THREE",
			why: "the three pr-workflow.md exceptions are the fix for this PR's headline failure — the " +
				"agent ending its turn at a BLOCKING GATE — and were the only load-bearing part of the " +
				"prompt with nothing pinning them",
		},
		{
			needle: "YOUR FIRST ACTION",
			why: "the agent — not the workflow — opens the PR now, so the prompt must demand it " +
				"before any code is written. Opened last, a dead runner leaves a branch with no PR, " +
				"which deliver-stall-sweep.yml cannot see because it selects from the PR side",
		},
		{
			needle: "gh pr create --draft",
			why: "the prompt must spell out the draft PR command, including --draft: a non-draft PR " +
				"opened before the work exists advertises itself as reviewable",
		},
		{
			needle: "RESUMING",
			why: "the prompt must tell the agent to continue an existing branch. Seeding keeps the " +
				"work, but a prompt that says only \"implement it\" makes a re-issued command " +
				"re-derive everything from scratch, which wastes the recovery",
		},
	}
	// The allowlist is stated positively, so a catalogue skill nobody has thought of yet is out
	// of scope by default. Banning brainstorming alone would leave every other
	// propose-then-await-the-human skill able to end a delivery the same way.
	for _, skill := range []string{
		"superpowers:using-git-worktrees",
		"superpowers:writing-plans",
		"superpowers:executing-plans",
		"superpowers:systematic-debugging",
		"superpowers:verification-before-completion",
		"superpowers:subagent-driven-development",
	} {
		required = append(required, struct {
			needle string
			why    string
		}{
			needle: skill,
			why: "the prompt names the skills that ARE in scope, so that anything else in the " +
				"injected catalogue is excluded by default; dropping one silently removes a " +
				"capability pr-workflow.md depends on",
		})
	}
	for _, r := range required {
		if !strings.Contains(prompt, r.needle) {
			t.Errorf("the implement prompt no longer mentions %q: %s", r.needle, r.why)
		}
	}

	// The prompt must not go back to commissioning the full suite or the linter. Matched on the
	// instruction shape, so the sentence that tells the agent NOT to run them does not trip this.
	banned := []struct {
		pattern *regexp.Regexp
		why     string
	}{
		{
			pattern: regexp.MustCompile(`run ` + "`" + `go build \./\.\.\.` + "`" + `, ` + "`" + `go test \./\.\.\.` + "`"),
			why: "ci.yml owns the full test suite and deliver-verify.yml dispatches it; running it " +
				"here duplicates a parity obligation and is part of what killed the #1706 runner",
		},
		{
			// Matched as the bare command, not as "run `golangci-lint …`": the wording this
			// replaced wrapped the line so that `golangci-lint run ./...` began a line with no
			// verb in front of it, and a verb-anchored pattern would have missed it entirely.
			// The surviving mention ("do NOT run or install `golangci-lint`") does not contain
			// the command form, so it does not trip this.
			pattern: regexp.MustCompile("`?golangci-lint run"),
			why: "ci.yml lints via golangci-lint-action with a cached prebuilt binary. The " +
				"self-hosted runner has no linter, so the agent built it from source — the #1706 " +
				"run was evicted 75 seconds into that build",
		},
	}
	for _, b := range banned {
		if loc := b.pattern.FindString(prompt); loc != "" {
			t.Errorf("the implement prompt reinstates %q: %s", loc, b.why)
		}
	}
}

// The no-work reporter must not name a cause it has not established.
//
// The version this replaced told every reader "the usual reason is an unmerged `Depends on:`
// blocker". On #1706 that was simply false — the issue declares no dependency and the real cause
// was an agent ending its turn on a design proposal — and the confident wrong cause sent the first
// hour of diagnosis the wrong way. A reporter is the one place in the loop a human trusts without
// checking, so it reports observations and enumerates possibilities instead of asserting one.
func TestDeliverImplementNoWorkReporterDoesNotGuessACause(t *testing.T) {
	path := filepath.Join("..", ".github", "workflows", "deliver-implement.yml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	body := string(raw)

	if strings.Contains(body, "The usual reason is an unmerged") {
		t.Error("deliver-implement.yml reinstates \"The usual reason is an unmerged `Depends on:` " +
			"blocker\". #1706 had no dependency, and asserting that cause cost an hour of " +
			"misdirected diagnosis. Report what the phase observed, not the most common cause")
	}

	// A blocked issue never reaches this reporter: check-permissions fails the whole job first,
	// so a blocker cannot be the explanation for a phase that ran and produced nothing.
	if strings.Contains(body, "Re-issue `/approve-issue-for-pr-delivery` once the blocker is resolved.") {
		t.Error("the no-work reporter still tells the reader to resolve a blocker. The " +
			"blocked-dependency guard lives in check-permissions and fails the job before the " +
			"deliver job starts, so this advice can never apply to a run that reached the reporter")
	}
}

// Every phase budget must stay inside deliver-stall-sweep.yml's quiet window. The sweep stands
// down while a phase run is newer than that window, and its own comment rests on the budgets
// sitting inside it — a phase allowed to run longer than the window could be flagged
// `needs-human` while still legitimately working, which halts a healthy delivery.
func TestDeliverPhaseBudgetsFitTheStallSweepWindow(t *testing.T) {
	sweepPath := filepath.Join("..", ".github", "workflows", "deliver-stall-sweep.yml")
	raw, err := os.ReadFile(sweepPath)
	if err != nil {
		t.Fatalf("reading %s: %v", sweepPath, err)
	}
	m := regexp.MustCompile(`QUIET_MINUTES:\s*'(\d+)'`).FindStringSubmatch(string(raw))
	if m == nil {
		t.Fatal("deliver-stall-sweep.yml no longer declares QUIET_MINUTES")
	}
	quiet := m[1]

	for _, phase := range []string{"deliver-implement.yml", "deliver-verify.yml", "deliver-correct.yml"} {
		t.Run(phase, func(t *testing.T) {
			path := filepath.Join("..", ".github", "workflows", phase)
			body, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("reading %s: %v", path, err)
			}
			var wf struct {
				Jobs map[string]struct {
					RunsOn         yaml.Node `yaml:"runs-on"`
					TimeoutMinutes int       `yaml:"timeout-minutes"`
				} `yaml:"jobs"`
			}
			if err := yaml.Unmarshal(body, &wf); err != nil {
				t.Fatalf("parsing %s: %v", path, err)
			}
			// Only the self-hosted agent jobs matter: they are the long ones, and the ones the
			// sweep's stand-down is reasoning about. `runs-on` is a scalar in some phases and a
			// sequence in others, so both shapes are decoded.
			checked := 0
			for name, job := range wf.Jobs {
				if job.TimeoutMinutes == 0 || !runsOnSelfHosted(job.RunsOn) {
					continue
				}
				checked++
				if got, want := job.TimeoutMinutes, mustAtoi(t, quiet); got >= want {
					t.Errorf("job %q has timeout-minutes %d, which is not inside "+
						"deliver-stall-sweep.yml's QUIET_MINUTES of %d. The sweep only stands down "+
						"for runs newer than that window, so a longer phase can be flagged "+
						"`needs-human` while it is still legitimately working",
						name, got, want)
				}
			}
			// A phase whose self-hosted job stopped being recognised would pass this test
			// vacuously, which is the failure mode that matters: it is the long jobs that can
			// outgrow the window.
			if checked == 0 {
				t.Errorf("no self-hosted job with a timeout was found in %s, so nothing was "+
					"checked against the quiet window", phase)
			}
		})
	}
}

// runsOnSelfHosted reports whether a job's `runs-on` names the self-hosted runner, accepting both
// the scalar (`runs-on: self-hosted`) and sequence (`runs-on: [self-hosted]`) forms.
func runsOnSelfHosted(node yaml.Node) bool {
	var scalar string
	if err := node.Decode(&scalar); err == nil {
		return scalar == "self-hosted"
	}
	var list []string
	if err := node.Decode(&list); err == nil {
		for _, l := range list {
			if l == "self-hosted" {
				return true
			}
		}
	}
	return false
}

func mustAtoi(t *testing.T, s string) int {
	t.Helper()
	n := 0
	for _, r := range s {
		if r < '0' || r > '9' {
			t.Fatalf("%q is not a number", s)
		}
		n = n*10 + int(r-'0')
	}
	return n
}

// stripCommentLines removes whole-line shell (`#`) and JS (`//`) comments, so a test that looks for
// a command in workflow code is not tripped by a comment explaining why that command is absent.
// Deliberately line-oriented and not a parser: it only needs to keep prose out of a substring match.
func stripCommentLines(code string) string {
	var kept []string
	for _, line := range strings.Split(code, "\n") {
		t := strings.TrimSpace(line)
		if strings.HasPrefix(t, "#") || strings.HasPrefix(t, "//") {
			continue
		}
		kept = append(kept, line)
	}
	return strings.Join(kept, "\n")
}

// No line of the agent prompt may begin with `#`.
//
// `prompt: |` is a YAML BLOCK SCALAR, where `#` is ordinary text rather than a comment. An earlier
// revision of #1723 put seven explanatory lines inside it at the surrounding indentation; they read
// as YAML comments but were handed to the agent as instructions — and one of them described the COST
// of pushing frequently, immediately above the bullet telling the agent to push frequently. Caught
// in review, not by any test, which is why this exists.
//
// Deliberately a dumb check. Anything that needs to say `#` at the start of a line (an issue
// reference, say) should be reworded rather than teaching this test to be clever.
func TestDeliverImplementPromptHasNoStrayYamlComments(t *testing.T) {
	steps := loadImplementSteps(t)
	agent := indexOfStep(steps, func(s implementStep) bool {
		return strings.HasPrefix(s.Uses, "anthropics/claude-code-action")
	})
	if agent < 0 {
		t.Fatal("deliver-implement.yml no longer runs anthropics/claude-code-action")
	}
	for n, line := range strings.Split(steps[agent].With.Prompt, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "#") {
			t.Errorf("prompt line %d begins with `#` and is therefore sent to the agent as "+
				"instruction text, not dropped as a comment: %q. Move it above `prompt:` if it is a "+
				"note for maintainers, or reword it if it is meant for the agent", n+1, strings.TrimSpace(line))
		}
	}
}

// An existing PR's base must win over the base recomputed from the issue body.
//
// Both are recomputed on every run, including a resume. If a sub-issue's `## Target branch` is edited
// between attempts, the branch and the open PR stay on the OLD base while the prompt and the work
// check would use the NEW one — producing a diff full of unrelated commits and a work check measured
// against the wrong ancestor, silently. Reported on #1723 (F1/G8). The PR is the authority once it
// exists: the branch is already based on it, and the agent is forbidden from retargeting it.
func TestDeliverImplementPrefersAnExistingPRsBase(t *testing.T) {
	path := filepath.Join("..", ".github", "workflows", "deliver-implement.yml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	body := string(raw)

	if !strings.Contains(body, "baseRefName") {
		t.Fatal("the seed step never reads the existing PR's baseRefName, so a resumed delivery can " +
			"use a base that disagrees with the branch it is resuming and with the PR it will hand off")
	}
	if !strings.Contains(body, `base="$pr_base"`) {
		t.Error("the seed step reads the PR's base but never adopts it; the recomputed value would " +
			"still reach the prompt and the work check")
	}

	// Ordering matters as much as presence: the base output must be written AFTER the PR lookup, or
	// the override cannot reach `steps.seed.outputs.base`.
	lookup := strings.Index(body, "pr=$(gh pr list")
	baseOut := strings.Index(body, `echo "base=$base" >> "$GITHUB_OUTPUT"`)
	if lookup < 0 || baseOut < 0 {
		t.Fatalf("could not locate the PR lookup (%d) and the base output (%d)", lookup, baseOut)
	}
	if baseOut < lookup {
		t.Errorf("`base` is written to GITHUB_OUTPUT at offset %d, BEFORE the PR lookup at %d, so an "+
			"existing PR's base cannot override the value computed from the issue body", baseOut, lookup)
	}
}

// deliver-verify.yml's `Hand off to correct` must dispatch deliver-correct.yml on the delivery
// BRANCH, not on `github.ref_name` (#1751).
//
// `github.ref_name` differs by the trigger that started the verify run: `deliver/issue-<N>` on a
// push, `main` on a workflow_dispatch (implement dispatches verify on its `issue_comment` ref, the
// default branch), and the PR MERGE ref `<pr>/merge` on `pull_request_review[_comment]`. Only the
// last is an invalid dispatch ref — `gh workflow run` (which dispatch-with-retry.sh forwards to)
// rejects it with `HTTP 422: No ref found`, which failed the job and sent every post-convergence
// review finding to `needs-human` on an infrastructure error rather than a verdict (observed on
// PRs #1742 and #1743).
//
// `steps.target.outputs.branch` is the resolved `deliver/issue-<N>` the CI dispatch already uses,
// validated in `Validate the target PR` and populated on every trigger, so it is a valid dispatch
// ref regardless of event. This test guards both ends of that dependency: the CONSUMER (the
// hand-off dispatches `--ref` on the branch expression, not `github.ref_name`) and the PRODUCER
// (`Validate the target PR` still writes `branch=` to `$GITHUB_OUTPUT` — drop that and the output
// is empty, `--ref ""` silently falls back to the default branch, and the consumer test alone would
// still pass).
func TestDeliverVerifyHandsOffOnTheDeliveryBranchNotTheEventRef(t *testing.T) {
	path := filepath.Join("..", ".github", "workflows", "deliver-verify.yml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	var wf struct {
		Jobs map[string]struct {
			Steps []struct {
				Name string `yaml:"name"`
				Run  string `yaml:"run"`
			} `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		t.Fatalf("parsing %s: %v", path, err)
	}
	job, ok := wf.Jobs["verify"]
	if !ok {
		t.Fatal("deliver-verify.yml has no `verify` job")
	}
	stepRun := func(name string) (string, bool) {
		for _, s := range job.Steps {
			if s.Name == name {
				return stripCommentLines(s.Run), true
			}
		}
		return "", false
	}

	// CONSUMER: the hand-off dispatch. Dispatched via dispatch-with-retry.sh (#1757), which forwards
	// to `gh workflow run`.
	run, ok := stepRun("Hand off to correct")
	if !ok {
		t.Fatal("deliver-verify.yml has no `Hand off to correct` step")
	}
	if !strings.Contains(run, "dispatch-with-retry.sh deliver-correct.yml") {
		t.Fatal("`Hand off to correct` no longer dispatches deliver-correct.yml; this test guards its ref")
	}
	if strings.Contains(run, "github.ref_name") {
		t.Errorf("`Hand off to correct` dispatches with `github.ref_name`. On a review-triggered run "+
			"that is `<pr>/merge`, which the dispatch rejects with HTTP 422, dead-ending the "+
			"finding at needs-human (#1751). Dispatch on the delivery branch instead. Step run:\n%s", run)
	}
	// Anchored to `--ref`, not a bare substring: a step that merely NAMES the expression in prose
	// while passing `--ref "$SOMETHING_ELSE"` would satisfy a `Contains` check without dispatching
	// on the branch.
	handoffRef := regexp.MustCompile(`--ref\s+"\$\{\{\s*steps\.target\.outputs\.branch\s*\}\}"`)
	if !handoffRef.MatchString(run) {
		t.Errorf("`Hand off to correct` does not pass `--ref \"${{ steps.target.outputs.branch }}\"` — "+
			"the resolved deliver/issue-<N> the CI dispatch already uses and the only ref valid on "+
			"every trigger. Step run:\n%s", run)
	}

	// PRODUCER: `Validate the target PR` must keep writing the branch output the consumer reads.
	prod, ok := stepRun("Validate the target PR")
	if !ok {
		t.Fatal("deliver-verify.yml has no `Validate the target PR` step to produce steps.target.outputs.branch")
	}
	// `branch=` with SOME non-empty value, not the exact shell var name: the contract is that the
	// output is written, so a benign rename (`branch=$head_ref`) must not false-alarm, while dropping
	// the line entirely — the real regression — still trips it.
	branchOut := regexp.MustCompile(`branch=\S`)
	if !branchOut.MatchString(prod) || !strings.Contains(prod, "GITHUB_OUTPUT") {
		t.Errorf("`Validate the target PR` no longer writes a `branch=` value to $GITHUB_OUTPUT, so "+
			"`steps.target.outputs.branch` would be empty and the hand-off's `--ref \"\"` would "+
			"silently fall back to the default branch. Step run:\n%s", prod)
	}
}

// deliver-verify.yml's `Review the PR` prompt must direct the review agent to fetch INLINE
// (line-level) PR review comments and weigh them as evidence (#1801).
//
// Inline review comments live at GET /repos/{owner}/{repo}/pulls/{n}/comments — a DIFFERENT endpoint
// from the conversation comments `gh pr view --comments` returns. The verify prompt pointed the
// agent at the diff, the sub-issue contracts and the refinements, but never at pulls/{n}/comments,
// so a reviewer's precise inline fix-request was invisible to the verdict reasoning; the correction
// round (whose work list is only the bot findings) therefore never fixed it. Observed on PR #1680,
// where two valid inline comments from a reviewer were dropped.
//
// The trust model is UNCHANGED and this test does not assert any new authority: inline comments are
// data to be assessed on the merits, exactly like conversation comments — a human comment still
// cannot command a verdict (the SECURITY block already covers that). The contract here is only that
// the endpoint is fetched and its output reaches the agent.
func TestDeliverVerifyReviewPromptFetchesInlineReviewComments(t *testing.T) {
	path := filepath.Join("..", ".github", "workflows", "deliver-verify.yml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	var wf struct {
		Jobs map[string]struct {
			Steps []implementStep `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		t.Fatalf("parsing %s: %v", path, err)
	}
	job, ok := wf.Jobs["verify"]
	if !ok {
		t.Fatal("deliver-verify.yml has no `verify` job")
	}
	agent := indexOfStep(job.Steps, func(s implementStep) bool {
		return strings.HasPrefix(s.Uses, "anthropics/claude-code-action")
	})
	if agent < 0 {
		t.Fatal("deliver-verify.yml's `verify` job no longer runs anthropics/claude-code-action")
	}
	prompt := job.Steps[agent].With.Prompt
	if strings.TrimSpace(prompt) == "" {
		t.Fatal("the review agent step has no prompt")
	}

	// Anchored on the pulls/{n}/comments endpoint, NOT a bare "comments" mention: the prompt already
	// discusses conversation comments, and the whole defect is that the INLINE endpoint — the one
	// `gh pr view --comments` does NOT return — was never fetched. `[^\n]*` tolerates the templated
	// PR-number expression (`pulls/${{ env.PR }}/comments`), which carries spaces inside its braces.
	inlineEndpoint := regexp.MustCompile(`pulls/[^\n]*/comments`)
	if !inlineEndpoint.MatchString(prompt) {
		t.Errorf("the review prompt does not direct the agent to fetch inline review comments from the "+
			"`pulls/{n}/comments` endpoint. Inline comments live at a different API than conversation "+
			"comments and are invisible to the verdict reasoning without it (#1801). Prompt:\n%s", prompt)
	}
	// The fetch must be a `gh api` call: `gh pr view --comments` returns only conversation comments,
	// so a prompt that named that command instead would reintroduce the gap.
	if !strings.Contains(prompt, "gh api") {
		t.Errorf("the review prompt references the inline-comments endpoint but not `gh api`, the only "+
			"gh call that returns inline review comments (`gh pr view --comments` does not). Prompt:\n%s", prompt)
	}
}

// The #1751 defect is a CLASS, not one step: any workflow reachable from an event whose GITHUB_REF
// is the PR merge ref `refs/pull/N/merge` must not dispatch another workflow on
// `${{ github.ref_name }}`, because that dispatch 422s. Those events are pull_request,
// pull_request_target, pull_request_review and pull_request_review_comment.
//
// GLOBBED, not a fixed list, for the reason TestDeliverImplementDoesNotCreatePRsWithTheWorkflowToken
// gives: on #1723 a hardcoded list missed the same offending line added to archon.yml. A
// self-modifying delivery loop can add a review trigger to any deliver-*.yml, so the guard must find
// the defect wherever it lands — including a NEW dispatch step in deliver-verify.yml, or a review
// trigger added to deliver-correct.yml / deliver-implement.yml (whose current dispatches on
// `github.ref_name` are safe only because their triggers are issue_comment / workflow_dispatch).
//
// Matches the dispatch in any of its three spellings — a bare `gh workflow run`, the
// `dispatch-with-retry.sh` wrapper (#1757) that forwards to it, and a raw REST POST to the
// workflow-dispatch endpoint (the spelling the sibling PR-creation guard had to add after review) —
// and looks for `github.ref_name` both in the run script AND in the step's own `env:`, since a ref
// can be staged as `env: REF: ${{ github.ref_name }}` and passed as `--ref "$REF"`. Job- and
// workflow-level `env:` indirection is a residual gap: the CONSUMER assertion above locks the actual
// hand-off regardless of spelling, so this class guard is defence-in-depth for the rest of the family.
func TestNoReviewReachableWorkflowDispatchesOnTheEventRef(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join("..", ".github", "workflows", "*.yml"))
	if err != nil {
		t.Fatalf("globbing workflows: %v", err)
	}
	yamlPaths, err := filepath.Glob(filepath.Join("..", ".github", "workflows", "*.yaml"))
	if err != nil {
		t.Fatalf("globbing workflows: %v", err)
	}
	paths = append(paths, yamlPaths...)
	if len(paths) < 5 {
		t.Fatalf("found only %d workflow files; the glob is not matching the workflow directory", len(paths))
	}

	reviewReachable := map[string]bool{
		"pull_request":                true,
		"pull_request_target":         true,
		"pull_request_review":         true,
		"pull_request_review_comment": true,
	}

	// Sanity: at least one workflow must be detected as review-reachable (deliver-verify.yml is).
	// Without this, a parsing regression that left every `on:` block unread would silently turn this
	// guard into a no-op that passes on the very defect it exists to catch.
	sawReachable := false

	for _, path := range paths {
		phase := filepath.Base(path)
		t.Run(phase, func(t *testing.T) {
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("reading %s: %v", path, err)
			}
			var wf struct {
				On   yaml.Node `yaml:"on"`
				Jobs map[string]struct {
					Steps []struct {
						Name string               `yaml:"name"`
						Run  string               `yaml:"run"`
						Env  map[string]yaml.Node `yaml:"env"`
					} `yaml:"steps"`
				} `yaml:"jobs"`
			}
			if err := yaml.Unmarshal(raw, &wf); err != nil {
				t.Fatalf("parsing %s: %v", path, err)
			}

			reachable := false
			for _, trig := range triggerNames(wf.On) {
				if reviewReachable[trig] {
					reachable = true
					break
				}
			}
			if !reachable {
				return
			}
			sawReachable = true

			for job, j := range wf.Jobs {
				for _, s := range j.Steps {
					code := stripCommentLines(s.Run)
					// A dispatch spelled any of three ways: the gh porcelain, the retry wrapper that
					// forwards to it (#1757), and a raw REST POST to the workflow-dispatch endpoint.
					restDispatch := strings.Contains(code, "gh api") &&
						strings.Contains(code, "POST") &&
						strings.Contains(code, "dispatches")
					dispatches := strings.Contains(code, "gh workflow run") ||
						strings.Contains(code, "dispatch-with-retry.sh") ||
						restDispatch
					if !dispatches {
						continue
					}
					// The ref can reach the dispatch through the run script directly OR staged in the
					// step's own `env:` (`env: REF: ${{ github.ref_name }}` + `--ref "$REF"`).
					refText := code
					for _, v := range s.Env {
						refText += "\n" + v.Value
					}
					if strings.Contains(refText, "github.ref_name") {
						t.Errorf("%s job %q step %q is reachable from a pull_request/review event, where "+
							"`github.ref_name` is the PR merge ref `<pr>/merge`, yet it dispatches a "+
							"workflow with `github.ref_name` (in its run or env) — that 422s and "+
							"dead-ends the loop (#1751). Dispatch on the resolved delivery branch "+
							"instead. Step:\n%s", phase, job, s.Name, refText)
					}
				}
			}
		})
	}

	if !sawReachable {
		t.Fatal("no workflow was detected as reachable from a pull_request/review event; the `on:` " +
			"block is not being parsed, so this guard would silently pass on the #1751 defect")
	}
}

// triggerNames returns the event names in a workflow's `on:` block, which YAML allows as a scalar
// (`on: push`), a sequence (`on: [push, pull_request]`) or a mapping (`on:\n  push:\n  ...`).
func triggerNames(n yaml.Node) []string {
	switch n.Kind {
	case yaml.ScalarNode:
		return []string{n.Value}
	case yaml.SequenceNode:
		var out []string
		for _, c := range n.Content {
			out = append(out, c.Value)
		}
		return out
	case yaml.MappingNode:
		var out []string
		for i := 0; i+1 < len(n.Content); i += 2 {
			out = append(out, n.Content[i].Value)
		}
		return out
	}
	return nil
}

// Every delivery phase hand-off must use the shared retry script (#1757).
//
// A single un-retried `gh workflow run` terminally stalls the delivery loop on a transient API
// failure — observed live on PR #1736. This test asserts every hand-off calls
// scripts/dispatch-with-retry.sh with the correct target workflow, preventing a future edit from
// regressing one back to a bare call.
func TestDeliveryHandoffDispatchesUseRetryScript(t *testing.T) {
	handoffs := []struct {
		file     string
		stepName string
		target   string
	}{
		{"deliver-verify.yml", "Hand off to correct", "deliver-correct.yml"},
		{"deliver-correct.yml", "Hand back to verify", "deliver-verify.yml"},
		{"deliver-implement.yml", "Hand off to verify", "deliver-verify.yml"},
	}

	for _, h := range handoffs {
		t.Run(h.file+"/"+h.stepName, func(t *testing.T) {
			path := filepath.Join("..", ".github", "workflows", h.file)
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("reading %s: %v", path, err)
			}
			body := string(raw)

			stepIdx := strings.Index(body, "name: "+h.stepName)
			if stepIdx < 0 {
				t.Fatalf("%s has no step named %q", h.file, h.stepName)
			}
			section := body[stepIdx:]
			nextStep := strings.Index(section[1:], "\n      - name:")
			if nextStep > 0 {
				section = section[:nextStep+1]
			}

			if !strings.Contains(section, "scripts/dispatch-with-retry.sh "+h.target) {
				t.Errorf("%s step %q does not call scripts/dispatch-with-retry.sh %s. "+
					"A bare gh workflow run terminally stalls the delivery loop (#1757)",
					h.file, h.stepName, h.target)
			}
		})
	}
}

// deliver-verify.yml must derive the branch's mergeability and feed it to the gate BEFORE the
// gate decides (#1758). Without it a PR that conflicts with main — GitHub cannot compute its
// merge ref — is invisible to the loop, which can then mark it `ready-for-merge`, a stale label
// a human cannot act on. The gate itself refuses to mark a conflicting branch ready
// (scripts/deliver_gate_test.go), but only if the signal reaches it, so this test guards the
// wiring: a step derives mergeable_state, and the `Decide` step passes MERGE_STATE.
//
// Asserted as ORDER (derive before decide) and as an explicit gate ENV wiring, not mere
// presence: a derivation step placed after the gate, or a MERGE_STATE the gate never receives,
// would satisfy a `strings.Contains` check while leaving the conflict signal disconnected.
func TestDeliverVerifyFeedsMergeStateToTheGate(t *testing.T) {
	path := filepath.Join("..", ".github", "workflows", "deliver-verify.yml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	var wf struct {
		Jobs map[string]struct {
			Steps []struct {
				ID   string            `yaml:"id"`
				Name string            `yaml:"name"`
				If   string            `yaml:"if"`
				Run  string            `yaml:"run"`
				Env  map[string]string `yaml:"env"`
			} `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		t.Fatalf("parsing %s: %v", path, err)
	}
	job, ok := wf.Jobs["verify"]
	if !ok {
		t.Fatal("deliver-verify.yml has no `verify` job")
	}

	idxByID := func(id string) int {
		for i, s := range job.Steps {
			if s.ID == id {
				return i
			}
		}
		return -1
	}

	gate := idxByID("gate")
	if gate < 0 {
		t.Fatal("deliver-verify.yml has no step with `id: gate` calling the decision brain")
	}
	merge := idxByID("mergestate")
	if merge < 0 {
		t.Fatal("deliver-verify.yml has no step with `id: mergestate`. The verify phase must " +
			"derive the branch's mergeability and feed it to the gate, or a PR conflicting with " +
			"main is invisible to the loop and can be marked ready-for-merge (#1758)")
	}
	if merge > gate {
		t.Errorf("the `mergestate` step is at index %d, AFTER the `gate` step at index %d — "+
			"the merge signal must be derived before the gate decides", merge, gate)
	}

	// The derivation must actually read GitHub's mergeable_state, and map it via the tested
	// script — not an inline `case` whose typo (dirty)→dirtry)) would silently mark a
	// conflicting branch mergeable with every test still green.
	if !strings.Contains(job.Steps[merge].Run, "mergeable_state") {
		t.Error("the `mergestate` step never reads `.mergeable_state`, so it cannot tell a " +
			"conflicting branch from a mergeable one")
	}
	if !strings.Contains(job.Steps[merge].Run, "map-merge-state.sh") {
		t.Error("the `mergestate` step does not map through scripts/map-merge-state.sh — the " +
			"mergeable_state→MERGE_STATE mapping must go through the tested script so a mapping " +
			"typo cannot silently mark a conflicting branch mergeable (#1758)")
	}

	// The derivation and the gate must share the same run condition. If the gate ran while the
	// derivation was skipped, MERGE_STATE would reach the gate empty (exit 2). They co-fire
	// today on `paused == false`; this pins that they stay in lockstep.
	if mg, gg := job.Steps[merge].If, job.Steps[gate].If; mg != gg {
		t.Errorf("the `mergestate` step's if (%q) differs from the `gate` step's if (%q); if they "+
			"diverge the gate could run without a derived MERGE_STATE", mg, gg)
	}

	// The gate must RECEIVE the derived signal, wired from the mergestate step's output.
	got := job.Steps[gate].Env["MERGE_STATE"]
	if got == "" {
		t.Fatal("the `gate` step's env does not set MERGE_STATE — deliver-gate.sh requires it, so " +
			"the gate would exit 2 (wiring error) on every round")
	}
	if !strings.Contains(got, "steps.mergestate.outputs") {
		t.Errorf("the `gate` step's MERGE_STATE is %q, not wired from steps.mergestate.outputs; the "+
			"gate would decide on a value unrelated to the branch's real mergeability", got)
	}
}

// deliver-correct.yml's agent must bring the branch up to date with main and resolve conflicts
// as part of its work (#1758). The gate routes a conflicting branch to this phase; if the prompt
// never tells the agent to merge main and resolve, the round accomplishes nothing and the branch
// stays conflicting until the round cap stops it at needs-human. This guards the prompt contract
// the same way TestDeliverImplementPromptContract guards the implement prompt.
func TestDeliverCorrectPromptResolvesMergeConflicts(t *testing.T) {
	path := filepath.Join("..", ".github", "workflows", "deliver-correct.yml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	var wf struct {
		Jobs map[string]struct {
			Steps []struct {
				Uses string `yaml:"uses"`
				With struct {
					Prompt string `yaml:"prompt"`
				} `yaml:"with"`
			} `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		t.Fatalf("parsing %s: %v", path, err)
	}
	job, ok := wf.Jobs["correct"]
	if !ok {
		t.Fatal("deliver-correct.yml has no `correct` job")
	}
	var prompt string
	for _, s := range job.Steps {
		if strings.HasPrefix(s.Uses, "anthropics/claude-code-action") {
			prompt = s.With.Prompt
			break
		}
	}
	if prompt == "" {
		t.Fatal("deliver-correct.yml has no anthropics/claude-code-action step with a prompt")
	}

	required := []struct{ needle, why string }{
		{"git merge", "the agent must merge to bring the branch up to date with main (#1758)"},
		{"origin/main", "the merge target must be main, so a conflicting branch is updated against it"},
		{"conflict", "the prompt must tell the agent to resolve merge conflicts, the whole point of #1758"},
		{"needs-human", "an unresolvable conflict must stop for a human, never a forced or guessed resolution"},
	}
	for _, r := range required {
		if !strings.Contains(prompt, r.needle) {
			t.Errorf("the correct prompt is missing %q: %s", r.needle, r.why)
		}
	}
}

// The gate returns `recheck` (non-terminal) when every signal is green but the branch's
// mergeability could not be read (#1758 G1). deliver-verify.yml's `Apply the decision` step
// MUST handle it explicitly: if it falls through to the `*)` arm it exits 1, the failure
// reporter fires, and the PR is stamped the terminal `needs-human` — reintroducing exactly the
// false-positive escalation `recheck` exists to prevent. This guards that the arm stays.
func TestDeliverVerifyHandlesRecheckNonTerminally(t *testing.T) {
	path := filepath.Join("..", ".github", "workflows", "deliver-verify.yml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	body := string(raw)

	// A `recheck)` case arm must exist in the decision handling — otherwise recheck is an
	// unexpected decision and the step errors into a terminal needs-human.
	if !strings.Contains(body, "recheck)") {
		t.Error("deliver-verify.yml's `Apply the decision` has no `recheck)` arm; an undeterminable " +
			"mergeability would hit the `*)` error path and be stamped terminal needs-human (#1758 G1)")
	}
	// recheck must NOT map to a terminal label. The only terminal labels are ready-for-merge and
	// needs-human; recheck applies neither (label=''). Guard that the recheck arm is not wired to
	// one of them by checking the arm assigns an empty label like `correct` does.
	recheckArm := regexp.MustCompile(`recheck\)\s+label=(''|"")`)
	if !recheckArm.MatchString(body) {
		t.Error("the `recheck)` arm does not set an empty label; recheck must apply no terminal " +
			"label so the PR stays re-checkable on the next event rather than being stopped")
	}
}
