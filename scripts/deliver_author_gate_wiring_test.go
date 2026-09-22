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
	for _, wf := range []string{
		"deliver-implement.yml",
		"deliver-verify.yml",
		"deliver-correct.yml",
		"claude.yml",
	} {
		t.Run(wf, func(t *testing.T) {
			body := readWorkflow(t, wf)
			if !strings.Contains(body, "scripts/deliver-author-gate.sh") {
				t.Errorf("%s does not invoke scripts/deliver-author-gate.sh — the author gate (#1813) "+
					"is the container-level complement to the triggerer check and must run before the "+
					"agent reads an outside-authored body/diff", wf)
			}
		})
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

// The two ubuntu permission jobs run the gate script, so they must check out the repo first — and it
// must be the TRUSTED default-branch tree (no `ref:` pointing at PR head), so only BLIS's own script
// runs, never issue/PR content.
func TestUbuntuGateJobsCheckoutTrustedTree(t *testing.T) {
	for _, wf := range []string{"claude.yml", "deliver-implement.yml"} {
		t.Run(wf, func(t *testing.T) {
			body := readWorkflow(t, wf)
			if !strings.Contains(body, "actions/checkout") {
				t.Errorf("%s does not check out the repo, but the author gate runs a script from it", wf)
			}
		})
	}
}
