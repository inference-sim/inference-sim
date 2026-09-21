package scripts_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// The workflow-wiring half of #1781. The two scripts it guards are behaviourally tested in
// scripts/deliver_branch_update_test.go; these tests assert the phases actually CALL them, in the
// right order, and act on their result — because a tested script nothing invokes is exactly the
// shape of the bug being fixed (the branch update was "specified" in a prompt and never happened).

// conflictStep is the subset of a delivery-workflow step these tests assert on.
type conflictStep struct {
	ID   string            `yaml:"id"`
	Name string            `yaml:"name"`
	If   string            `yaml:"if"`
	Run  string            `yaml:"run"`
	Uses string            `yaml:"uses"`
	Env  map[string]string `yaml:"env"`
}

func loadConflictSteps(t *testing.T, workflow, job string) []conflictStep {
	t.Helper()
	path := filepath.Join("..", ".github", "workflows", workflow)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	var wf struct {
		Jobs map[string]struct {
			Steps []conflictStep `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		t.Fatalf("parsing %s: %v", path, err)
	}
	j, ok := wf.Jobs[job]
	if !ok {
		t.Fatalf("%s has no `%s` job", workflow, job)
	}
	if len(j.Steps) == 0 {
		t.Fatalf("%s's `%s` job has no steps", workflow, job)
	}
	return j.Steps
}

func indexOfConflictStep(steps []conflictStep, match func(conflictStep) bool) int {
	for i, s := range steps {
		if match(s) {
			return i
		}
	}
	return -1
}

// The branch update must be a WORKFLOW STEP that runs BEFORE the agent, and it must go through the
// tested script (#1781).
//
// This is the #1778 regression in one assertion. There the update existed only as prose in the
// agent prompt; the round completed `success` with no commit, no push and no comment. Asserted as
// step ORDER, because an update placed after the agent would satisfy a `strings.Contains` check
// while leaving the agent to start from a stale branch — which is the failure mode.
func TestDeliverCorrectUpdatesTheBranchBeforeTheAgentRuns(t *testing.T) {
	steps := loadConflictSteps(t, "deliver-correct.yml", "correct")

	update := indexOfConflictStep(steps, func(s conflictStep) bool { return s.ID == "update" })
	if update < 0 {
		t.Fatal("deliver-correct.yml has no step with `id: update`. The branch update must be a " +
			"workflow step, not only a prompt instruction: on PR #1778 the agent's `git merge " +
			"origin/main` never happened and the round completed success with no progress (#1781)")
	}
	agent := indexOfConflictStep(steps, func(s conflictStep) bool {
		return strings.HasPrefix(s.Uses, "anthropics/claude-code-action")
	})
	if agent < 0 {
		t.Fatal("deliver-correct.yml has no anthropics/claude-code-action step")
	}
	if update > agent {
		t.Errorf("the `update` step is at index %d, AFTER the agent at index %d — the branch must be "+
			"brought up to date before the agent starts, or it corrects against a stale branch",
			update, agent)
	}
	if !strings.Contains(steps[update].Run, "deliver-update-branch.sh") {
		t.Error("the `update` step does not call scripts/deliver-update-branch.sh. The merge/push/abort " +
			"logic must go through the tested script (scripts/deliver_branch_update_test.go) rather " +
			"than being reimplemented inline, where it cannot be exercised")
	}
	// Its `state=`/`files=` output must reach GITHUB_OUTPUT, or the prompt and the hand-back cannot
	// read what happened.
	if !strings.Contains(steps[update].Run, "GITHUB_OUTPUT") {
		t.Error("the `update` step does not tee its state into GITHUB_OUTPUT, so nothing downstream " +
			"can tell whether the branch was merged, is current, or conflicts")
	}
}

// After the agent, the phase must re-check whether the conflict was actually resolved, report it,
// and withhold the hand-back — #1758(b), which PR #1778 did not satisfy.
func TestDeliverCorrectVerifiesTheConflictWasResolved(t *testing.T) {
	steps := loadConflictSteps(t, "deliver-correct.yml", "correct")

	check := indexOfConflictStep(steps, func(s conflictStep) bool { return s.ID == "conflictcheck" })
	if check < 0 {
		t.Fatal("deliver-correct.yml has no step with `id: conflictcheck`. Without it a correction " +
			"round that failed to resolve the conflict hands back and the next verify stops for a " +
			"human citing a missing review marker, never naming the conflict (#1781)")
	}
	agent := indexOfConflictStep(steps, func(s conflictStep) bool {
		return strings.HasPrefix(s.Uses, "anthropics/claude-code-action")
	})
	if check < agent {
		t.Errorf("the `conflictcheck` step is at index %d, BEFORE the agent at index %d — it must "+
			"establish the state the agent left behind", check, agent)
	}

	// `always()`: the agent step is where a timeout or crash lands, and an unresolved conflict must
	// be reported precisely then. An `if:` with no status function carries an implicit `success()`,
	// which would skip exactly the case this exists for.
	if !strings.Contains(steps[check].If, "always()") {
		t.Errorf("the `conflictcheck` step's if is %q; it must be guarded on always(), or a crashed "+
			"or timed-out agent skips the report and the round ends silently", steps[check].If)
	}
	if !strings.Contains(steps[check].Run, "deliver-conflict-check.sh") {
		t.Error("the `conflictcheck` step does not call scripts/deliver-conflict-check.sh")
	}
	// It must NAME the files and stop for a human — the two halves of #1758(b).
	if !strings.Contains(steps[check].Run, "needs-human") {
		t.Error("the `conflictcheck` step does not apply needs-human; a branch that still conflicts " +
			"cannot be verified to a real verdict, so the delivery must stop rather than loop")
	}
	if !strings.Contains(steps[check].Run, "Conflicting files") {
		t.Error("the `conflictcheck` step's comment does not name the conflicting files. #1758's " +
			"acceptance criterion is a needs-human that NAMES the conflict; on #1778 a human had " +
			"to go and find it")
	}

	// And the hand-back must be withheld on `conflicting`, or the delivery re-verifies a branch with
	// no merge ref and arrives at the same answer with a vaguer reason.
	handback := indexOfConflictStep(steps, func(s conflictStep) bool { return s.Name == "Hand back to verify" })
	if handback < 0 {
		t.Fatal("deliver-correct.yml has no `Hand back to verify` step")
	}
	if !strings.Contains(steps[handback].If, "conflictcheck") {
		t.Errorf("the hand-back's if is %q; it must be gated on the conflictcheck result, or a round "+
			"that already stopped the delivery at needs-human still triggers a full re-verification",
			steps[handback].If)
	}
}

// Verify must decide whether to run the agent reviews BEFORE running them, and must not gate on
// markers those skipped reviews could never produce (#1781, C3/C4).
func TestDeliverVerifySkipsTheReviewsOnAConflictingBranch(t *testing.T) {
	steps := loadConflictSteps(t, "deliver-verify.yml", "verify")

	hint := indexOfConflictStep(steps, func(s conflictStep) bool { return s.ID == "mergehint" })
	if hint < 0 {
		t.Fatal("deliver-verify.yml has no step with `id: mergehint`. A branch with no merge ref " +
			"cannot produce a verdict the gate may act on, so both agent reviews must be skipped " +
			"on one — and that decision has to be made before they run (#1781)")
	}

	for _, want := range []struct{ id, why string }{
		{"qa", "the cross-vendor qa-review"},
	} {
		i := indexOfConflictStep(steps, func(s conflictStep) bool { return s.ID == want.id })
		if i < 0 {
			t.Fatalf("deliver-verify.yml has no step with `id: %s` (%s)", want.id, want.why)
		}
		if i < hint {
			t.Errorf("%s runs at index %d, BEFORE the mergehint step at %d — the skip decision must precede it", want.id, i, hint)
		}
		if !strings.Contains(steps[i].If, "mergehint") {
			t.Errorf("%s's if is %q; it must be skipped when the branch already conflicts", want.id, steps[i].If)
		}
	}

	// The Anthropic reviewer has no step id, so it is matched by name.
	review := indexOfConflictStep(steps, func(s conflictStep) bool { return s.Name == "Review the PR" })
	if review < 0 {
		t.Fatal("deliver-verify.yml has no `Review the PR` step")
	}
	if review < hint {
		t.Errorf("`Review the PR` runs at index %d, before the mergehint step at %d", review, hint)
	}
	if !strings.Contains(steps[review].If, "mergehint") {
		t.Errorf("`Review the PR`'s if is %q; it must be skipped when the branch already conflicts", steps[review].If)
	}

	// The marker READERS must stay unconditional: they turn an absent marker into MISSING, which is
	// what the gate's #1781 row consumes. Skipping them would leave the gate's AGENT_VERDICT empty
	// and it would exit 2 (a wiring error) on every conflicting branch.
	for _, id := range []string{"verdict", "qa_verdict"} {
		i := indexOfConflictStep(steps, func(s conflictStep) bool { return s.ID == id })
		if i < 0 {
			t.Fatalf("deliver-verify.yml has no step with `id: %s`", id)
		}
		if strings.Contains(steps[i].If, "mergehint") {
			t.Errorf("the `%s` marker reader is gated on mergehint (%q); it must always run, or the "+
				"gate receives an empty verdict and exits 2 rather than reading MISSING", id, steps[i].If)
		}
	}
}

// The conflicting paths must reach the gate, or its reason names the conflict but not the files —
// half of #1758(b).
func TestDeliverVerifyFeedsConflictFilesToTheGate(t *testing.T) {
	steps := loadConflictSteps(t, "deliver-verify.yml", "verify")

	merge := indexOfConflictStep(steps, func(s conflictStep) bool { return s.ID == "mergestate" })
	if merge < 0 {
		t.Fatal("deliver-verify.yml has no step with `id: mergestate`")
	}
	if !strings.Contains(steps[merge].Run, "conflicting-files.sh") {
		t.Error("the `mergestate` step does not call scripts/conflicting-files.sh, so the conflicting " +
			"paths are never computed and a needs-human cannot name them (#1781)")
	}

	gate := indexOfConflictStep(steps, func(s conflictStep) bool { return s.ID == "gate" })
	if gate < 0 {
		t.Fatal("deliver-verify.yml has no step with `id: gate`")
	}
	got := steps[gate].Env["CONFLICT_FILES"]
	if got == "" {
		t.Fatal("the `gate` step's env does not set CONFLICT_FILES, so the conflicting paths never " +
			"reach the reason a human reads")
	}
	if !strings.Contains(got, "steps.mergestate.outputs") {
		t.Errorf("the `gate` step's CONFLICT_FILES is %q, not wired from steps.mergestate.outputs", got)
	}

	// #1781 G1: the gate needs REVIEWS_SKIPPED to tell "markers missing because reviews were
	// skipped on a conflict hint" (→ recheck if the branch is not conflicting) from "markers
	// missing because a review crashed" (→ needs-human). If verify stops feeding it, the gate
	// fails closed (exit 2) rather than mis-deciding — but that stops every delivery, so pin the
	// wiring here where the drift is visible.
	skipped := steps[gate].Env["REVIEWS_SKIPPED"]
	if skipped == "" {
		t.Fatal("the `gate` step's env does not set REVIEWS_SKIPPED, so a round that skipped its " +
			"reviews on a stale conflict hint cannot be told from a crashed review, and a stale hint " +
			"could dead-end the delivery at the round cap (#1781 G1)")
	}
	if !strings.Contains(skipped, "steps.mergestate.outputs") {
		t.Errorf("the `gate` step's REVIEWS_SKIPPED is %q, not wired from steps.mergestate.outputs", skipped)
	}
}
