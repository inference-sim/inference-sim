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
						if strings.Contains(code, "gh pr create") || strings.Contains(code, "pulls.create") {
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
