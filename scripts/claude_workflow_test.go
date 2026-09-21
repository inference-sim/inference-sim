package scripts_test

// Pins the permission split in .github/workflows/claude.yml (#1697): the agent that
// REVIEWS a pull request must not hold a token that can push to it.
//
// These assertions look structural, but for a workflow file the declared structure IS the
// behaviour — GitHub reads nothing else, and the file runs only inside Actions, where no
// other test can reach it. Same reasoning as the shell-script tests in this package.
//
// What they defend against is one specific regression: the split is expressed as two jobs
// with identical steps and different `permissions:` blocks, because `permissions:` accepts
// no expressions and so cannot be varied per trigger inside a single job. A comment asks
// the next editor to keep the steps in sync; that comment is load-bearing and, without
// this test, unenforced.

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

type claudeJob struct {
	If          string            `yaml:"if"`
	Needs       yaml.Node         `yaml:"needs"`
	Permissions map[string]string `yaml:"permissions"`
	Steps       []yaml.Node       `yaml:"steps"`

	// Job-level configuration that must also match between the two agent jobs. Steps alone
	// are not the whole job: moving the review onto a different runner, or giving it a
	// shorter timeout, changes how it behaves without touching a single step.
	RunsOn         yaml.Node         `yaml:"runs-on"`
	TimeoutMinutes int               `yaml:"timeout-minutes"`
	Outputs        map[string]string `yaml:"outputs"`
}

type claudeWorkflow struct {
	Permissions map[string]string    `yaml:"permissions"`
	Jobs        map[string]claudeJob `yaml:"jobs"`
}

const (
	writeJob  = "claude"
	reviewJob = "claude-review"
)

func loadClaudeWorkflow(t *testing.T) claudeWorkflow {
	t.Helper()

	path := filepath.Join("..", ".github", "workflows", "claude.yml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}

	// Deliberately NOT strict: the struct models only the fields these tests assert on, and
	// the workflow carries many more. The job-existence check below is what catches drift.
	var wf claudeWorkflow
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	for _, name := range []string{writeJob, reviewJob, "report-status"} {
		if _, ok := wf.Jobs[name]; !ok {
			t.Fatalf("job %q missing from %s — the permission split was restructured; "+
				"re-derive these assertions rather than deleting them", name, path)
		}
	}
	return wf
}

// The security property itself: the review path cannot push, the general path still can.
func TestClaudeWorkflow_ReviewJobTokenCannotPush(t *testing.T) {
	wf := loadClaudeWorkflow(t)

	if got := wf.Jobs[reviewJob].Permissions["contents"]; got != "read" {
		t.Errorf("%s has contents: %q, want \"read\" — a review must not be able to push "+
			"to the branch it is reviewing (#1697)", reviewJob, got)
	}

	// A review must not be able to publish its own verdict either; report-status owns that.
	if _, ok := wf.Jobs[reviewJob].Permissions["statuses"]; ok {
		t.Errorf("%s declares a statuses permission — the commit status is published by "+
			"report-status so that the reviewer does not set its own verdict", reviewJob)
	}

	// Guards the other half: this must stay a SPLIT. If someone "simplifies" by giving the
	// general path read access too, @claude can no longer be asked to make a change, and the
	// pressure to widen the review job returns.
	if got := wf.Jobs[writeJob].Permissions["contents"]; got != "write" {
		t.Errorf("%s has contents: %q, want \"write\" — non-review @claude triggers may "+
			"legitimately be asked to change code", writeJob, got)
	}
}

// The workflow-level block is only ever inherited by a job that declares none of its own,
// so the single thing it can still do is hand push access to a future job that forgets its
// block — silently re-arming what #1697 closed.
func TestClaudeWorkflow_DefaultContentsIsNotWrite(t *testing.T) {
	wf := loadClaudeWorkflow(t)

	if got := wf.Permissions["contents"]; got != "read" {
		t.Errorf("workflow-level contents is %q, want \"read\"", got)
	}
}

// The routing must fail CLOSED. Both gates compare is_review to an explicit literal, so an
// unset value runs neither job. Written as `!=` on either side, an unset value would instead
// route a review down the write path — the exact bug this test exists to prevent.
func TestClaudeWorkflow_RoutingGatesFailClosed(t *testing.T) {
	wf := loadClaudeWorkflow(t)

	for job, wantLiteral := range map[string]string{
		writeJob:  "is_review == 'false'",
		reviewJob: "is_review == 'true'",
	} {
		gate := wf.Jobs[job].If
		if !strings.Contains(gate, wantLiteral) {
			t.Errorf("%s gate does not compare is_review to an explicit literal "+
				"(want %q):\n%s", job, wantLiteral, gate)
		}
		if strings.Contains(gate, "is_review !=") {
			t.Errorf("%s gate tests is_review with != — an unset value would then route "+
				"to this job. Compare to an explicit literal so unset runs neither:\n%s",
				job, gate)
		}
		// A routing gate is a conjunction of requirements: the caller is allowed AND the
		// command matched. An `||` can only ADD a way in, which is the shape an accidental
		// widening takes — including a substring-satisfying `true || ...` that the two
		// checks above would otherwise accept.
		if strings.Contains(gate, "||") {
			t.Errorf("%s gate contains || — a routing gate should only ever be a "+
				"conjunction, since a disjunct adds an additional way to enter the "+
				"job:\n%s", job, gate)
		}
	}
}

// The tests above pin which TOKEN each job gets. They say nothing about which job a review
// actually lands in — that is decided by the classifier in check-permissions, the third
// moving part of the split. Without this, breaking the command string or inverting the
// ternary routes every review to the write-capable job and all the other tests stay green,
// silently restoring what #1697 closed.
func TestClaudeWorkflow_ClassifierRoutesReviewsToTheReviewJob(t *testing.T) {
	wf := loadClaudeWorkflow(t)

	script := classifierScript(t, wf)

	// The command the routing actually keys on. A typo here sends every review to the
	// write path.
	if !strings.Contains(script, "includes('/blis-pr-review')") {
		t.Errorf("check-permissions does not test the body for '/blis-pr-review' — "+
			"reviews would route to the write-capable job:\n%s", script)
	}

	// The polarity. Inverted, reviews get write access and everything else gets read-only.
	if !strings.Contains(script, "isReview ? 'true' : 'false'") {
		t.Errorf("check-permissions does not map a matched command to is_review=true — "+
			"check the ternary has not been inverted:\n%s", script)
	}
}

// classifierScript returns the inline script of the check-permissions step that computes
// is_review, failing the test if no step does.
func classifierScript(t *testing.T, wf claudeWorkflow) string {
	t.Helper()

	for i, node := range wf.Jobs["check-permissions"].Steps {
		var step struct {
			With struct {
				Script string `yaml:"script"`
			} `yaml:"with"`
		}
		if err := node.Decode(&step); err != nil {
			t.Fatalf("decode check-permissions step %d: %v", i, err)
		}
		if strings.Contains(step.With.Script, "is_review") {
			return step.With.Script
		}
	}
	t.Fatal("no check-permissions step sets is_review — the routing classifier is gone, " +
		"so both agent gates compare against a value nothing produces")
	return ""
}

// The duplication the split forces. Everything except the permissions block and the routing
// gate must match, or a fix lands on one trigger and not the other.
func TestClaudeWorkflow_AgentJobsStayInSync(t *testing.T) {
	wf := loadClaudeWorkflow(t)

	// Job-level configuration, not just steps: a different runner or a shorter timeout on
	// the review job is drift the step comparison below cannot see.
	write, review := wf.Jobs[writeJob], wf.Jobs[reviewJob]
	var writeRunsOn, reviewRunsOn any
	if err := write.RunsOn.Decode(&writeRunsOn); err != nil {
		t.Fatalf("decode %s runs-on: %v", writeJob, err)
	}
	if err := review.RunsOn.Decode(&reviewRunsOn); err != nil {
		t.Fatalf("decode %s runs-on: %v", reviewJob, err)
	}
	if !reflect.DeepEqual(writeRunsOn, reviewRunsOn) {
		t.Errorf("runs-on differs: %s has %#v, %s has %#v",
			writeJob, writeRunsOn, reviewJob, reviewRunsOn)
	}
	if write.TimeoutMinutes != review.TimeoutMinutes {
		t.Errorf("timeout-minutes differs: %s has %d, %s has %d",
			writeJob, write.TimeoutMinutes, reviewJob, review.TimeoutMinutes)
	}
	if !reflect.DeepEqual(write.Outputs, review.Outputs) {
		t.Errorf("outputs differ: %s has %v, %s has %v",
			writeJob, write.Outputs, reviewJob, review.Outputs)
	}

	writeSteps, reviewSteps := write.Steps, review.Steps
	if len(writeSteps) != len(reviewSteps) {
		t.Fatalf("%s has %d steps, %s has %d — keep the two agent jobs in sync",
			writeJob, len(writeSteps), reviewJob, len(reviewSteps))
	}
	for i := range writeSteps {
		var w, r any
		if err := writeSteps[i].Decode(&w); err != nil {
			t.Fatalf("decode %s step %d: %v", writeJob, i, err)
		}
		if err := reviewSteps[i].Decode(&r); err != nil {
			t.Fatalf("decode %s step %d: %v", reviewJob, i, err)
		}
		// Comments are not part of the decoded value, so the two blocks may explain
		// themselves differently; only the effective configuration has to match.
		if !reflect.DeepEqual(w, r) {
			t.Errorf("step %d differs between %s and %s.\n%s: %#v\n%s: %#v",
				i, writeJob, reviewJob, writeJob, w, reviewJob, r)
		}
	}
}

// report-status must depend on BOTH agent jobs. Dropping either one loses the commit status
// for that path — and losing it for the review path is how a review silently stops
// reporting a verdict.
func TestClaudeWorkflow_ReportStatusNeedsBothAgentJobs(t *testing.T) {
	wf := loadClaudeWorkflow(t)

	reportStatus := wf.Jobs["report-status"]
	// `needs:` is a scalar or a sequence in Actions, and a regression to a single dependency
	// would take the scalar form — decode both so this test reports the missing dependency
	// rather than a decode error.
	var needs []string
	if reportStatus.Needs.Kind == yaml.ScalarNode {
		var one string
		if err := reportStatus.Needs.Decode(&one); err != nil {
			t.Fatalf("decode report-status needs: %v", err)
		}
		needs = []string{one}
	} else if err := reportStatus.Needs.Decode(&needs); err != nil {
		t.Fatalf("decode report-status needs: %v", err)
	}
	for _, want := range []string{writeJob, reviewJob} {
		found := false
		for _, got := range needs {
			if got == want {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("report-status does not need %q (needs: %v) — that path would "+
				"publish no commit status", want, needs)
		}
	}
}
