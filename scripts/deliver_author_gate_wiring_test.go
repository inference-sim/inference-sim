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

// All four flows must reach the ONE gate script — never a re-implementation of the author-trust law
// (R23). deliver-verify/correct/implement invoke it directly; claude.yml invokes it too (asserted in
// TestClaudeGatesBothAgentJobsOnAuthor below via the gate step it wires).
func TestDeliveryFlowsInvokeTheAuthorGate(t *testing.T) {
	type step struct {
		Run              string `yaml:"run"`
		WorkingDirectory string `yaml:"working-directory"`
	}
	for _, wf := range []string{
		"deliver-implement.yml",
		"deliver-verify.yml",
		"deliver-correct.yml",
		"claude.yml",
	} {
		t.Run(wf, func(t *testing.T) {
			path := filepath.Join("..", ".github", "workflows", wf)
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("reading %s: %v", path, err)
			}
			var doc struct {
				Jobs map[string]struct {
					Steps []step `yaml:"steps"`
				} `yaml:"jobs"`
			}
			if err := yaml.Unmarshal(raw, &doc); err != nil {
				t.Fatalf("parsing %s: %v", path, err)
			}
			found := false
			for _, job := range doc.Jobs {
				for _, s := range job.Steps {
					if !strings.Contains(s.Run, "deliver-author-gate.sh") {
						continue
					}
					found = true
					// The gate must run FROM the trusted `.gate-trusted` checkout. A bare
					// `scripts/deliver-author-gate.sh` executed from the workspace root would run
					// whatever the event/delivery ref checked out (for a review event, the PR merge
					// tree) — the P0 bypass. `working-directory: .gate-trusted` is what pins it, so
					// even a relative path the script might read resolves inside the trusted tree.
					if s.WorkingDirectory != ".gate-trusted" {
						t.Errorf("%s: the author-gate step invokes deliver-author-gate.sh but its "+
							"working-directory is %q, not \".gate-trusted\" — the gate could then run "+
							"attacker-controlled content from the event/delivery checkout", wf, s.WorkingDirectory)
					}
				}
			}
			if !found {
				t.Errorf("%s does not invoke deliver-author-gate.sh — the author gate (#1813) must run "+
					"before the agent reads an outside-authored body/diff", wf)
			}
		})
	}
}

// The P0 fix: the gate script is executed from a checkout pinned to the default branch into its own
// `.gate-trusted` path — NOT the implicit event/delivery checkout. For a `pull_request_review_comment`
// GitHub checks out the PR MERGE tree, and `deliver-correct` is dispatched on the delivery branch, so
// a bare checkout would run attacker-replaceable code before the author decision. This asserts every
// gate-bearing workflow pins that trusted ref, so a future edit that drops the pin fails here.
func TestGateRunsFromTrustedDefaultBranchCheckout(t *testing.T) {
	type checkoutStep struct {
		Uses string `yaml:"uses"`
		With struct {
			Ref  string `yaml:"ref"`
			Path string `yaml:"path"`
		} `yaml:"with"`
	}
	for _, wf := range []string{
		"deliver-implement.yml",
		"deliver-verify.yml",
		"deliver-correct.yml",
		"claude.yml",
	} {
		t.Run(wf, func(t *testing.T) {
			path := filepath.Join("..", ".github", "workflows", wf)
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("reading %s: %v", path, err)
			}
			// Model just the jobs→steps shape and scan every step, so this does not depend on which
			// job the gate lives in.
			var doc struct {
				Jobs map[string]struct {
					Steps []checkoutStep `yaml:"steps"`
				} `yaml:"jobs"`
			}
			if err := yaml.Unmarshal(raw, &doc); err != nil {
				t.Fatalf("parsing %s: %v", path, err)
			}
			found := false
			for _, job := range doc.Jobs {
				for _, s := range job.Steps {
					if !strings.Contains(s.Uses, "actions/checkout") || s.With.Path != ".gate-trusted" {
						continue
					}
					found = true
					if !strings.Contains(s.With.Ref, "default_branch") {
						t.Errorf("%s: the `.gate-trusted` checkout pins ref %q, which is not the "+
							"repository default branch — for a review event the implicit ref is the PR "+
							"merge tree, so the gate could run attacker-replaceable code", wf, s.With.Ref)
					}
				}
			}
			if !found {
				t.Errorf("%s has no `path: .gate-trusted` checkout — the author gate must run from a "+
					"dedicated default-branch checkout, not the job's event/delivery checkout", wf)
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
