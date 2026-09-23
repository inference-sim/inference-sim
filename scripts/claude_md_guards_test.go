package scripts_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// CLAUDE.md-hygiene guards (#1818). Kept in their own small file — alongside the size-guard test
// in claude_md_size_test.go — rather than appended to the large deliver_guards_test.go, so the
// two new guards are easy to find and review together.

// deliver-verify.yml must withhold `ready-for-merge` from a delivery that edits CLAUDE.md unless
// the issue carries the `docs:claude-md` opt-in label (#1818, folding in #1824). CLAUDE.md is
// loaded into every Claude session; treating it as a per-PR changelog is what grew it past the
// context limit, so an unlabelled edit re-accretes the bloat the trim removed and must stop for a
// human. This mirrors the existing `ci_edit` guard exactly: a detection step sets an output and
// `Apply the decision` downgrades ready->needs-human when it is set — a post-gate override, so
// deliver-gate.sh is unchanged. Asserted as WIRING (step exists, reads the label, output reaches
// the decision env, override present) AND ORDER (producer before consumer), not mere presence, so
// a detached step, a step placed after the decision, or an unconsumed signal cannot pass.
func TestDeliverVerifyGuardsUnauthorizedClaudeMdEdits(t *testing.T) {
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

	// The detection step, and its POSITION: a step's output is visible only to LATER steps, so
	// claude_md_edit must run BEFORE `Apply the decision` or CLAUDE_MD_UNAUTHORIZED reaches it empty
	// and an unauthorized edit slips through. Asserted as order, not mere presence.
	guardRun, guardIdx := "", -1
	for i, s := range job.Steps {
		if s.ID == "claude_md_edit" {
			guardRun, guardIdx = s.Run, i
			break
		}
	}
	if guardIdx < 0 {
		t.Fatal("deliver-verify.yml has no step with `id: claude_md_edit`. A delivery that edits " +
			"CLAUDE.md without the `docs:claude-md` opt-in label must be detected so the gate can " +
			"withhold ready-for-merge (#1818/#1824)")
	}
	if !strings.Contains(guardRun, "docs:claude-md") {
		t.Error("the `claude_md_edit` step never checks the `docs:claude-md` opt-in label, so it " +
			"cannot tell an authorised CLAUDE.md edit from an unauthorised one")
	}
	if !strings.Contains(guardRun, "CLAUDE.md") {
		t.Error("the `claude_md_edit` step never references CLAUDE.md, so it detects nothing")
	}

	// The decision step must RECEIVE the signal and act on it, and must come AFTER the detection.
	applyRun, applyIdx := "", -1
	var applyEnv map[string]string
	for i, s := range job.Steps {
		if s.Name == "Apply the decision" {
			applyRun, applyEnv, applyIdx = s.Run, s.Env, i
			break
		}
	}
	if applyIdx < 0 {
		t.Fatal("deliver-verify.yml has no `Apply the decision` step to consume the CLAUDE.md guard")
	}
	if guardIdx > applyIdx {
		t.Errorf("the `claude_md_edit` step is at index %d, AFTER `Apply the decision` at index %d — "+
			"the detection output must be produced before the decision consumes it, or "+
			"CLAUDE_MD_UNAUTHORIZED is empty at the gate and an unauthorized edit reaches ready-for-merge",
			guardIdx, applyIdx)
	}
	wired := applyEnv["CLAUDE_MD_UNAUTHORIZED"]
	if wired == "" {
		t.Fatal("the `Apply the decision` step's env does not set CLAUDE_MD_UNAUTHORIZED, so the " +
			"CLAUDE.md guard's verdict never reaches the decision")
	}
	if !strings.Contains(wired, "steps.claude_md_edit.outputs") {
		t.Errorf("CLAUDE_MD_UNAUTHORIZED is %q, not wired from steps.claude_md_edit.outputs; the "+
			"decision would act on a value unrelated to the guard", wired)
	}
	if !strings.Contains(applyRun, "CLAUDE_MD_UNAUTHORIZED") {
		t.Error("`Apply the decision` receives CLAUDE_MD_UNAUTHORIZED but never reads it, so an " +
			"unauthorised CLAUDE.md edit could still be marked ready-for-merge")
	}
	if !strings.Contains(applyRun, "needs-human") {
		t.Error("the CLAUDE.md override does not route to needs-human")
	}
}
