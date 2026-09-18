package scripts_test

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// #1790: the delivery PR must record WHO ran `/approve-issue-for-pr-delivery`.
//
// The implement phase already knows — `check-permissions` gates on `github.actor` — but never put
// it on the PR. Every delivery PR is authored by `app/claude`, so with several in flight nobody can
// tell who started which, and the approver is not subscribed to the thread: they see none of the
// verdicts, correction rounds or needs-human stops that follow.
//
// WHY THE WORKFLOW HALF MAY TRAVEL AS A PATCH. The delivering credential is a GitHub App
// installation token without the `workflows` permission, and both `git push` and the contents API
// refuse it ("refusing to allow a GitHub App to create or update workflow … without `workflows`
// permission"). #1715 hit the same wall and set the convention followed here: the workflow edit is
// committed as an appliable patch, and the guard test holds the contract over whichever of the two
// currently carries it — asserting the real workflow the moment a human applies the patch, with no
// edit to this file.
//
// UNLIKE #1715's guard, this one does not match the patch's added lines. It reconstructs the
// POST-IMAGE by applying the diff, then asserts against that. Added-line matching cannot see step
// ORDER or parse the result as YAML, and — the reason that matters most here — it cannot tell a
// stale patch from a good one. Applying the diff checks every context line against the workflow as
// it stands today, so a patch that has drifted out of applying fails here rather than at the moment
// someone tries to land it.
const approverPatch = "scripts/deliver-implement-approver.patch"

// The step this change adds, and the pre-existing step it must follow.
const (
	recordStep = "Record the approver on the PR"
	locateStep = "Locate the PR"
)

func implementWorkflowPath() string {
	return filepath.Join("..", ".github", "workflows", "deliver-implement.yml")
}

func approverPatchPath() string {
	return filepath.Join("..", "scripts", "deliver-implement-approver.patch")
}

// implementWorkflowWithApproverWiring returns the text of deliver-implement.yml as it will run once
// #1790's wiring is in place, and whether that wiring is already live.
//
// Live if the workflow itself carries the step; otherwise the patch is applied to the workflow in
// memory. A tree with NEITHER is a failure, not a skip: this test is the only thing asserting the
// contract, so letting it pass on an empty tree would make the whole guard optional.
func implementWorkflowWithApproverWiring(t *testing.T) (workflow string, live bool) {
	t.Helper()

	current := readFileOrFail(t, implementWorkflowPath())
	if strings.Contains(current, "name: "+recordStep) {
		return current, true
	}

	patch, err := os.ReadFile(approverPatchPath())
	if err != nil {
		t.Fatalf("deliver-implement.yml carries no %q step and %s is missing (%v). One of the two "+
			"must hold #1790's wiring; with neither, nothing records who approved a delivery and "+
			"this guard would be asserting against an unchanged file",
			recordStep, approverPatch, err)
	}
	patched, err := applyUnifiedDiff(current, string(patch))
	if err != nil {
		t.Fatalf("%s no longer applies to deliver-implement.yml: %v\n\n"+
			"The patch carries #1790's workflow half because the delivering App token lacks the "+
			"`workflows` permission. A patch that does not apply cannot be landed, so regenerate it "+
			"against the current workflow — do not delete this test to make the failure go away",
			approverPatch, err)
	}
	if !strings.Contains(patched, "name: "+recordStep) {
		t.Fatalf("%s applies but the result still has no %q step, so it is not #1790's wiring",
			approverPatch, recordStep)
	}
	return patched, false
}

// applyUnifiedDiff applies the single-file unified diff embedded in patch to orig and returns the
// post-image.
//
// Deliberately hand-rolled rather than shelling out to `git apply`: this runs in the `test
// (scripts)` CI group, and a test that depends on an external binary to make an assertion is a test
// that can pass because the binary was missing.
//
// Parsing starts at `diff --git`, never at the top of the file: `git format-patch` puts the commit
// MESSAGE first, and this change's own message contains bullet lines beginning with `-` that would
// otherwise be read as removals.
func applyUnifiedDiff(orig, patch string) (string, error) {
	lines := strings.Split(patch, "\n")

	start := -1
	for i, l := range lines {
		if strings.HasPrefix(l, "diff --git ") {
			start = i
			break
		}
	}
	if start < 0 {
		return "", fmt.Errorf("no `diff --git` header found, so this is not a git patch")
	}
	// A second file header means the patch touches more than deliver-implement.yml. Refused rather
	// than partially applied: the caller is asserting that this patch IS the workflow half, and a
	// patch that quietly also edits something else is not that.
	for _, l := range lines[start+1:] {
		if strings.HasPrefix(l, "diff --git ") {
			return "", fmt.Errorf("patch contains more than one file diff; expected only the workflow")
		}
	}
	if !strings.Contains(lines[start], ".github/workflows/deliver-implement.yml") {
		return "", fmt.Errorf("patch's file header is %q, not deliver-implement.yml", lines[start])
	}

	src := strings.Split(orig, "\n")
	var out []string
	next := 0 // 0-based index of the next unconsumed line of src

	i := start
	for i < len(lines) {
		line := lines[i]
		if !strings.HasPrefix(line, "@@") {
			i++
			continue
		}
		oldStart, oldCount, err := parseHunkHeader(line)
		if err != nil {
			return "", err
		}
		if oldStart-1 < next {
			return "", fmt.Errorf("hunk %q overlaps or reverses a previous one", line)
		}
		if oldStart-1 > len(src) {
			return "", fmt.Errorf("hunk %q starts past the end of the file (%d lines)", line, len(src))
		}
		out = append(out, src[next:oldStart-1]...)
		next = oldStart - 1

		i++
		consumed := 0
		for consumed < oldCount || (i < len(lines) && strings.HasPrefix(lines[i], "+")) {
			if i >= len(lines) {
				return "", fmt.Errorf("patch ends mid-hunk at %q", line)
			}
			body := lines[i]
			switch {
			case strings.HasPrefix(body, "+"):
				out = append(out, body[1:])
			case strings.HasPrefix(body, "-"), strings.HasPrefix(body, " "):
				want := body[1:]
				if next >= len(src) {
					return "", fmt.Errorf("hunk %q reaches past the end of the file", line)
				}
				// The staleness check. A context or removal line that does not match means the
				// workflow has moved under the patch.
				if src[next] != want {
					return "", fmt.Errorf("context mismatch at line %d: patch expects %q, file has %q",
						next+1, want, src[next])
				}
				if strings.HasPrefix(body, " ") {
					out = append(out, want)
				}
				next++
				consumed++
			case body == "\\ No newline at end of file":
				// Nothing to apply.
			case body == "":
				// A context line for a blank line loses its leading space in some transports.
				if next >= len(src) {
					return "", fmt.Errorf("hunk %q reaches past the end of the file", line)
				}
				if src[next] != "" {
					return "", fmt.Errorf("context mismatch at line %d: patch expects a blank line, file has %q",
						next+1, src[next])
				}
				out = append(out, "")
				next++
				consumed++
			default:
				return "", fmt.Errorf("unexpected line in hunk %q: %q", line, body)
			}
			i++
		}
	}
	out = append(out, src[next:]...)
	return strings.Join(out, "\n"), nil
}

func parseHunkHeader(line string) (oldStart, oldCount int, err error) {
	// `@@ -a,b +c,d @@ …` — the counts default to 1 when omitted.
	fields := strings.Fields(line)
	if len(fields) < 3 || !strings.HasPrefix(fields[1], "-") {
		return 0, 0, fmt.Errorf("malformed hunk header %q", line)
	}
	old := strings.TrimPrefix(fields[1], "-")
	oldCount = 1
	if idx := strings.Index(old, ","); idx >= 0 {
		oldCount, err = strconv.Atoi(old[idx+1:])
		if err != nil {
			return 0, 0, fmt.Errorf("malformed hunk header %q: %v", line, err)
		}
		old = old[:idx]
	}
	oldStart, err = strconv.Atoi(old)
	if err != nil {
		return 0, 0, fmt.Errorf("malformed hunk header %q: %v", line, err)
	}
	return oldStart, oldCount, nil
}

// approverWiring is the parsed shape of deliver-implement.yml the assertions below read.
type approverWiring struct {
	CheckOutputs map[string]string
	DeliverEnv   map[string]string
	Steps        []approverStep
}

type approverStep struct {
	Name   string
	ID     string
	If     string
	Script string
}

func parseApproverWiring(t *testing.T, workflow string) approverWiring {
	t.Helper()
	var wf struct {
		Jobs map[string]struct {
			Outputs map[string]string `yaml:"outputs"`
			Env     map[string]string `yaml:"env"`
			Steps   []struct {
				Name string `yaml:"name"`
				ID   string `yaml:"id"`
				If   string `yaml:"if"`
				With struct {
					Script string `yaml:"script"`
				} `yaml:"with"`
			} `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal([]byte(workflow), &wf); err != nil {
		t.Fatalf("parsing deliver-implement.yml: %v", err)
	}
	check, ok := wf.Jobs["check-permissions"]
	if !ok {
		t.Fatal("deliver-implement.yml has no `check-permissions` job")
	}
	deliver, ok := wf.Jobs["deliver"]
	if !ok {
		t.Fatal("deliver-implement.yml has no `deliver` job")
	}
	out := approverWiring{
		CheckOutputs: check.Outputs,
		DeliverEnv:   deliver.Env,
	}
	// Both jobs' steps, so the permission check's own script is assertable alongside the recorder.
	for _, s := range append(append([]struct {
		Name string `yaml:"name"`
		ID   string `yaml:"id"`
		If   string `yaml:"if"`
		With struct {
			Script string `yaml:"script"`
		} `yaml:"with"`
	}{}, check.Steps...), deliver.Steps...) {
		out.Steps = append(out.Steps, approverStep{
			Name: s.Name, ID: s.ID, If: s.If,
			// Whole-line comments stripped for the reason the sibling guards give: this change's
			// comments have to NAME `addAssignees` and the provenance sentence in order to explain
			// them, and a raw substring match would then be satisfied by the prose that documents
			// the behaviour rather than by the code that implements it. Mutation-checked: deleting
			// the real `addAssignees` call must fail these assertions.
			Script: stripCommentLines(s.With.Script),
		})
	}
	return out
}

func (w approverWiring) step(t *testing.T, name string) approverStep {
	t.Helper()
	for _, s := range w.Steps {
		if s.Name == name {
			return s
		}
	}
	var names []string
	for _, s := range w.Steps {
		names = append(names, s.Name)
	}
	t.Fatalf("deliver-implement.yml has no %q step; steps: %v", name, names)
	return approverStep{}
}

func (w approverWiring) indexOf(name string) int {
	for i, s := range w.Steps {
		if s.Name == name {
			return i
		}
	}
	return -1
}

// The approver login must reach the recorder FROM THE EVENT PAYLOAD, by way of the step that
// checked its permission.
//
// Two properties in one. First, no untrusted string: reading the login out of the comment BODY
// would let anyone with read access name someone else as the approver, and the issue-number
// resolution in this same job already refuses to take input from comment text for that reason.
// Second, the login recorded is the one that passed the collaborator gate — `github.actor` is
// readable directly in the `deliver` job, but a second independent read is a second thing that can
// disagree with what was actually authorised.
func TestDeliverImplementThreadsTheApproverFromTheEventPayload(t *testing.T) {
	workflow, live := implementWorkflowWithApproverWiring(t)
	if !live {
		t.Logf("asserting against the reconstructed post-image of %s — deliver-implement.yml does "+
			"not yet carry the wiring", approverPatch)
	}
	w := parseApproverWiring(t, workflow)

	expr, ok := w.CheckOutputs["approver"]
	if !ok {
		t.Fatalf("the `check-permissions` job exposes no `approver` output; it has %v. Without it "+
			"the `deliver` job has nothing to record", sortedKeys(w.CheckOutputs))
	}
	if !strings.Contains(expr, "steps.check.outputs.approver") {
		t.Errorf("the `approver` output is %q, which does not come from the `check` step that "+
			"validates the permission. Sourcing it elsewhere breaks the tie between the login "+
			"recorded on the PR and the login that was actually authorised", expr)
	}

	check := w.step(t, "Check invoker permissions")
	if !strings.Contains(check.Script, "setOutput('approver', context.actor)") {
		t.Errorf("the permission check does not emit `context.actor` as the `approver` output. Its "+
			"script:\n%s", check.Script)
	}

	env, ok := w.DeliverEnv["APPROVER"]
	if !ok {
		t.Fatalf("the `deliver` job has no APPROVER env var; it has %v", sortedKeys(w.DeliverEnv))
	}
	if !strings.Contains(env, "needs.check-permissions.outputs.approver") {
		t.Errorf("APPROVER is %q rather than the permission check's output. Reading `github.actor` "+
			"directly here would record a login nothing in this workflow verified", env)
	}

	// The recorder must not take the login from comment text. Checked on the recorder's own script
	// so the `Resolve the target sub-issue` step — which legitimately reads the comment body to
	// refuse a disagreeing `#N` — cannot vouch for it.
	record := w.step(t, recordStep)
	for _, forbidden := range []string{"comment.body", "payload.comment", "github.event.comment"} {
		if strings.Contains(record.Script, forbidden) {
			t.Errorf("the %q step reads %q. The approver must come from the event payload via the "+
				"permission check, never from comment text — otherwise a comment can name whoever "+
				"it likes as the approver", recordStep, forbidden)
		}
	}
}

// The PR must carry the approver as an ASSIGNEE and as a line in its BODY.
//
// Both, because they do different jobs. The assignee is what makes a delivery filterable
// (`assignee:@me`, the Assigned tab) and is what subscribes the approver to the thread. The body
// line is the durable record: it survives someone clearing assignees, and it @-mentions the
// approver so the subscription does not depend on the assignment having stuck.
func TestDeliverImplementRecordsTheApproverAsAssigneeAndInTheBody(t *testing.T) {
	workflow, _ := implementWorkflowWithApproverWiring(t)
	record := parseApproverWiring(t, workflow).step(t, recordStep)

	if !strings.Contains(record.Script, "issues.addAssignees") {
		t.Errorf("the %q step does not call `issues.addAssignees`, so the approver is neither "+
			"filterable nor subscribed. Its script:\n%s", recordStep, record.Script)
	}
	// `addAssignees`, not `setAssignees`/`update`: a delivery resumed by a second collaborator must
	// end up with both of them on it rather than the first being replaced.
	for _, replacing := range []string{"setAssignees", "removeAssignees"} {
		if strings.Contains(record.Script, replacing) {
			t.Errorf("the %q step calls `%s`, which drops an earlier approver. A resumed delivery "+
				"can legitimately have two approvers and both should be recorded", recordStep, replacing)
		}
	}

	if !strings.Contains(record.Script, "Delivery approved by @") {
		t.Errorf("the %q step writes no `Delivery approved by @<login>` provenance line, so nothing "+
			"records the approver once assignees are cleared", recordStep)
	}
	if !strings.Contains(record.Script, "pulls.update") {
		t.Errorf("the %q step never updates the PR body, so the provenance line reaches nothing",
			recordStep)
	}
	// Idempotent. A resumed delivery re-runs this step, and an unconditional append would stack a
	// duplicate line on every attempt.
	if !strings.Contains(record.Script, "body.includes(") {
		t.Errorf("the %q step does not check the existing body before appending. Re-issuing the "+
			"command re-runs it, so an unconditional append duplicates the line each time", recordStep)
	}
}

// Recording must be deterministic, must survive a failed agent, and must never fail the delivery.
//
// Deterministic because the agent opening the PR is the least reliable link in this phase — asking
// it to record its own approver puts the provenance behind the one action that is not
// workflow-guaranteed. `always()` because a delivery that errored or timed out is exactly when
// knowing who to tell matters most. Never fatal because this is metadata about a run, and #1732
// settled that a run is not failed over a record of itself.
func TestDeliverImplementApproverRecordingIsDeterministicAndNonFatal(t *testing.T) {
	workflow, _ := implementWorkflowWithApproverWiring(t)
	w := parseApproverWiring(t, workflow)
	record := w.step(t, recordStep)

	if !strings.Contains(record.If, "always()") {
		t.Errorf("the %q step is guarded on %q, which does not include `always()`. An agent step "+
			"that failed or was cancelled would then leave the delivery with no approver recorded — "+
			"the runs where it is most needed", recordStep, record.If)
	}
	if !strings.Contains(record.If, "steps.pr.outputs.number != ''") {
		t.Errorf("the %q step is guarded on %q, which does not require a PR to exist. There is "+
			"nothing to assign or edit without one", recordStep, record.If)
	}
	if !strings.Contains(record.If, "steps.paused.outputs.paused == 'false'") {
		t.Errorf("the %q step is guarded on %q, which does not honour `deliver:paused`. BC-9 "+
			"requires a paused delivery to touch nothing", recordStep, record.If)
	}

	// It needs the PR number, so it must come after the step that resolves it.
	locate, rec := w.indexOf(locateStep), w.indexOf(recordStep)
	if locate < 0 {
		t.Fatalf("deliver-implement.yml has no %q step", locateStep)
	}
	if rec < locate {
		t.Errorf("the %q step is at index %d, BEFORE %q at index %d, so it would run without a PR "+
			"number", recordStep, rec, locateStep, locate)
	}

	// Non-fatal. `core.setFailed` here would turn a delivery that built and verified fine into a
	// failed phase over an assignment.
	if strings.Contains(record.Script, "core.setFailed") {
		t.Errorf("the %q step calls `core.setFailed`. Recording the approver is metadata: a failure "+
			"here must not fail the delivery", recordStep)
	}
	// R1 — never silent. Every way this can come up short has to say so.
	if strings.Count(record.Script, "core.warning") < 3 {
		t.Errorf("the %q step has %d `core.warning` calls. Three failure modes each need one: an "+
			"empty approver login, a failed or silently-dropped assignment, and a failed body "+
			"update. Swallowing any of them makes the provenance quietly unreliable",
			recordStep, strings.Count(record.Script, "core.warning"))
	}
	// GitHub returns 201 for `addAssignees` even when it drops an assignee it will not accept, so
	// the result has to be read back rather than assumed.
	if !strings.Contains(record.Script, "assignees") || !strings.Contains(record.Script, ".login === approver") {
		t.Errorf("the %q step does not read the assignment back. `addAssignees` silently drops an "+
			"assignee it will not accept and still returns success, so without this the phase "+
			"reports an assignment that never happened", recordStep)
	}
}

// The patch and the workflow must never both carry the wiring.
//
// A leftover patch file after the edit has landed is a trap: it reads as pending work that is
// already done, and it would make the staleness check above unreachable. #1715's application commit
// deleted its patch in the same commit for the same reason.
func TestApproverPatchIsRemovedOnceApplied(t *testing.T) {
	current := readFileOrFail(t, implementWorkflowPath())
	if !strings.Contains(current, "name: "+recordStep) {
		return
	}
	if _, err := os.Stat(approverPatchPath()); err == nil {
		t.Errorf("deliver-implement.yml carries the %q step AND %s still exists. Delete the patch "+
			"in the commit that applies it — a leftover patch describes work that is already done",
			recordStep, approverPatch)
	}
}
