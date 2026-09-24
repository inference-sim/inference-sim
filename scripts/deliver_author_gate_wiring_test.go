package scripts_test

// The author gate is WIRED into every AI-flow workflow (#1813).
//
// deliver_author_gate_test.go exercises the DECISION (does this author pass?). These tests assert
// the four workflows actually CONSULT it before their agent, and refuse when it says no — the half
// that a green decision test cannot vouch for, since a workflow that never calls the gate would
// still let it pass its own unit tests. For a workflow file the declared content IS the behaviour
// (GitHub reads nothing else, and the file runs only inside Actions), so these read the YAML as
// text, the same approach claude_workflow_test.go and deliver_guards_test.go take.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func readWorkflow(t *testing.T, name string) string {
	t.Helper()
	path := filepath.Join("..", ".github", "workflows", name)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	return string(raw)
}

// wiringStep models the fields these tests assert on.
type wiringStep struct {
	Uses string `yaml:"uses"`
	Run  string `yaml:"run"`
	With struct {
		Ref  string `yaml:"ref"`
		Path string `yaml:"path"`
	} `yaml:"with"`
}

// gateJobSteps returns, for a workflow, the steps of the job that contains the author-gate step, in
// file order. Panics via t.Fatalf if no gate step is found.
func gateJobSteps(t *testing.T, wf string) []wiringStep {
	t.Helper()
	path := filepath.Join("..", ".github", "workflows", wf)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	var doc struct {
		Jobs map[string]struct {
			Steps []wiringStep `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("parsing %s: %v", path, err)
	}
	for _, job := range doc.Jobs {
		for _, s := range job.Steps {
			if strings.Contains(s.Run, "deliver-author-gate.sh") {
				return job.Steps
			}
		}
	}
	t.Fatalf("%s has no step invoking deliver-author-gate.sh — the author gate (#1813) must run "+
		"before the agent reads an outside-authored body/diff", wf)
	return nil
}

// The gate must run from a TRUSTED tree: in file order, the checkout IMMEDIATELY PRECEDING the gate
// step must be pinned to the repository default branch. This is the P0 fix. A bare checkout would
// leave the workspace at the event/delivery ref — for a `pull_request_review_comment` the PR MERGE
// tree, for `deliver-correct` the delivery branch — and the author could replace the gate script
// with an allow result. Asserting ORDER (not just "a pinned checkout exists somewhere") is what
// catches `deliver-correct`, where the delivery-branch checkout must come AFTER the gate, never
// before it.
func TestGateRunsFromTrustedDefaultBranchCheckout(t *testing.T) {
	for _, wf := range []string{
		"deliver-implement.yml",
		"deliver-verify.yml",
		"deliver-correct.yml",
		"claude.yml",
	} {
		t.Run(wf, func(t *testing.T) {
			steps := gateJobSteps(t, wf)
			var lastCheckoutRef string
			var sawCheckout, checked bool
			for _, s := range steps {
				if strings.Contains(s.Uses, "actions/checkout") {
					lastCheckoutRef = s.With.Ref
					sawCheckout = true
					continue
				}
				if strings.Contains(s.Run, "deliver-author-gate.sh") {
					checked = true
					if !sawCheckout {
						t.Errorf("%s: no checkout precedes the author-gate step — it would run the "+
							"workspace's implicit (event/delivery) ref", wf)
					} else if !strings.Contains(lastCheckoutRef, "default_branch") {
						t.Errorf("%s: the checkout immediately before the author-gate step pins ref %q, "+
							"not the repository default branch — the gate could run attacker-replaceable "+
							"code (P0 bypass)", wf, lastCheckoutRef)
					}
					break
				}
			}
			if !checked {
				t.Errorf("%s: never reached the author-gate step while scanning in order", wf)
			}
		})
	}
}

// deliver-correct must edit the delivery branch, so it checks that branch out — but ONLY after the
// gate. This pins the reviewer-requested ordering: a default-branch checkout, then the gate, then a
// checkout that is NOT pinned to the default branch (the delivery branch). Without it, moving the
// delivery checkout back before the gate would silently reopen the bypass.
func TestCorrectChecksOutDeliveryBranchOnlyAfterGate(t *testing.T) {
	steps := gateJobSteps(t, "deliver-correct.yml")
	gateIdx := -1
	for i, s := range steps {
		if strings.Contains(s.Run, "deliver-author-gate.sh") {
			gateIdx = i
			break
		}
	}
	if gateIdx < 0 {
		t.Fatal("deliver-correct.yml: no gate step found")
	}
	// Before the gate: every checkout must be default-branch-pinned.
	for _, s := range steps[:gateIdx] {
		if strings.Contains(s.Uses, "actions/checkout") && !strings.Contains(s.With.Ref, "default_branch") {
			t.Errorf("deliver-correct.yml: a checkout before the gate pins ref %q, not the default "+
				"branch — the delivery branch must not be present when the gate runs", s.With.Ref)
		}
	}
	// After the gate: the delivery-branch checkout (its ref is NOT the default branch).
	sawDeliveryCheckout := false
	for _, s := range steps[gateIdx+1:] {
		if strings.Contains(s.Uses, "actions/checkout") && !strings.Contains(s.With.Ref, "default_branch") {
			sawDeliveryCheckout = true
		}
	}
	if !sawDeliveryCheckout {
		t.Error("deliver-correct.yml: no delivery-branch checkout after the gate — the agent edits the " +
			"delivery branch, so it must be checked out (only once the gate has passed)")
	}
}

// Defense in depth (#1813, P0 from review): verify/correct read the PR diff AND the sub-issue, so
// BOTH authors are gated — same-repo + branch shape does not prove the PR is bot-authored. The PR
// author is resolved via REST in `Validate the target PR` (a GET, kept out of the gate step so its
// label POST does not trip the repo-wide PR-creation guard) and consumed by the gate as `PR_AUTHOR`;
// the sub-issue author is resolved in the gate. Assert the JOB resolves both and the gate checks both.
func TestVerifyAndCorrectGateBothAuthors(t *testing.T) {
	for _, wf := range []string{"deliver-verify.yml", "deliver-correct.yml"} {
		t.Run(wf, func(t *testing.T) {
			steps := gateJobSteps(t, wf)
			var gateRun, jobRun string
			for _, s := range steps {
				jobRun += s.Run + "\n"
				if strings.Contains(s.Run, "deliver-author-gate.sh") {
					gateRun = s.Run
				}
			}
			// PR author resolved in the job (REST GET) and exposed as an output.
			if !strings.Contains(jobRun, "/pulls/") || !strings.Contains(jobRun, "pr_author=") {
				t.Errorf("%s: the job does not resolve the delivery PR author via REST into a "+
					"pr_author output — the PR author would be unguarded (#1813)", wf)
			}
			// The gate consumes the PR author and resolves + checks the issue author too.
			if !strings.Contains(gateRun, "PR_AUTHOR") {
				t.Errorf("%s: the gate step does not consume the PR author (PR_AUTHOR)", wf)
			}
			if !strings.Contains(gateRun, "/issues/") {
				t.Errorf("%s: the gate does not resolve the sub-issue author (issues/…)", wf)
			}
			// Assert the two calls target DISTINCT authors, not the same one twice — a copy-paste
			// checking the PR author twice would satisfy a bare count. The gate labels them
			// "delivery PR" and "sub-issue".
			if !strings.Contains(gateRun, `check_author "delivery PR`) {
				t.Errorf("%s: the gate has no check_author call for the delivery PR author", wf)
			}
			if !strings.Contains(gateRun, `check_author "sub-issue`) {
				t.Errorf("%s: the gate has no check_author call for the sub-issue author", wf)
			}
		})
	}
}

// P1: on claude.yml's refusal path both agent jobs and report-status skip, so the gate's own comment
// is the ONLY signal. If that comment cannot be posted the job must fail LOUDLY (red run), not exit 0
// green-with-no-explanation — the silent-refusal class R1 exists for.
func TestClaudeGateFailsLoudWhenRefusalCommentCannotPost(t *testing.T) {
	body := readWorkflow(t, "claude.yml")
	if !strings.Contains(body, "if gh issue comment") {
		t.Error("claude.yml author gate does not condition on whether the refusal comment posted; a " +
			"`gh issue comment ... || true` followed by `exit 0` would refuse silently on an API outage")
	}
	if !strings.Contains(body, "::error::author gate refused") {
		t.Error("claude.yml author gate has no loud `::error::` failure when the refusal comment cannot " +
			"be posted — the run would be green with no explanation (R1)")
	}
}

// The two self-hosted delivery phases post their own specific refusal and move the label, so their
// generic failure reporter must EXCLUDE the handled author-block — otherwise a refusal draws two
// comments and a misleading "the automation failed, re-issue once fixed" verdict.
func TestVerifyAndCorrectReporterExcludeAuthorBlock(t *testing.T) {
	for _, wf := range []string{"deliver-verify.yml", "deliver-correct.yml"} {
		t.Run(wf, func(t *testing.T) {
			body := readWorkflow(t, wf)
			if !strings.Contains(body, "steps.author_gate.outputs.blocked != 'true'") {
				t.Errorf("%s failure reporter does not exclude the author-block "+
					"(`steps.author_gate.outputs.blocked != 'true'`); a refusal would then be "+
					"double-reported and mis-described as an automation failure", wf)
			}
		})
	}
}

// claude.yml expresses its gate as a job output ANDed into BOTH agent jobs (they exist as two jobs
// only because `permissions:` cannot be varied per trigger — see the file's own comment). Both must
// gate on author_allowed, or the untrusted-container hole stays open on one path.
func TestClaudeGatesBothAgentJobsOnAuthor(t *testing.T) {
	path := filepath.Join("..", ".github", "workflows", "claude.yml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}

	var wf struct {
		Jobs map[string]struct {
			If      string            `yaml:"if"`
			Outputs map[string]string `yaml:"outputs"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		t.Fatalf("parsing %s: %v", path, err)
	}

	cp, ok := wf.Jobs["check-permissions"]
	if !ok {
		t.Fatal("check-permissions job missing from claude.yml — re-derive these assertions")
	}
	if _, ok := cp.Outputs["author_allowed"]; !ok {
		t.Error("check-permissions does not export an `author_allowed` output — the author gate (#1813) is not wired")
	}

	for _, job := range []string{"claude", "claude-review"} {
		j, ok := wf.Jobs[job]
		if !ok {
			t.Fatalf("agent job %q missing from claude.yml", job)
		}
		if !strings.Contains(j.If, "author_allowed == 'true'") {
			t.Errorf("agent job %q does not gate on `author_allowed == 'true'` — @claude would still "+
				"run on an outside-authored PR/issue via this path (#1813). if:\n%s", job, j.If)
		}
	}
}
