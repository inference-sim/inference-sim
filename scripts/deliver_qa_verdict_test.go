package scripts_test

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// #1715 has two halves that MUST land together: scripts/deliver-gate.sh gains QA_VERDICT as a
// required, fail-closed signal, and .github/workflows/deliver-verify.yml starts producing
// it. Merging the gate half alone bricks every delivery in the repository — the Decide step would
// exit 2 on a wiring error, the reporter would apply `needs-human`, and no PR could ever reach a
// verdict again.
//
// The delivering credential could not push the workflow half: a GitHub App installation token
// without the `workflows` permission is refused by both `git push` and the contents API. So the
// workflow half originally travelled as an appliable patch; it has since been applied by a
// workflows-scoped push and the patch deleted, so the wiring is now LIVE in deliver-verify.yml and
// the tests below assert it there directly.

// The three steps #1715 adds, plus the one it amends. Named as constants because every assertion
// below is scoped to a specific step: a needle satisfied anywhere in the job would let the
// PRE-EXISTING DELIVER-VERDICT reader vouch for the new QA reader, which is exactly the property
// under test (both readers must independently carry the Bot-author and last-line rules).
const (
	qaRunStep     = "Run qa-review"
	qaCleanupStep = "Remove the qa-review worktree"
	qaReadStep    = "Read the QA verdict marker"
	decideStep    = "Decide"
)

func verifyWorkflowPath() string {
	return filepath.Join("..", ".github", "workflows", "deliver-verify.yml")
}

func readFileOrFail(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	return string(raw)
}

// qaWiring returns the EXECUTABLE text of each step in the (now-live) qa-review wiring, keyed by
// step name.
//
// "Executable" is load-bearing in two ways. Whole-line comments are stripped, for the reason
// stripCommentLines exists in the sibling guard file: this change's own comments have to NAME
// `--no-exec` in order to explain the security invariant, and a raw substring match over the file
// would then be satisfied by the prose that documents the property rather than by the flag that
// implements it. (Found by mutation-testing this file: deleting the real `--no-exec` left every
// assertion passing.) And only `if`, `env` values and `run` are collected — YAML keys and step
// names carry no behaviour.
func qaWiring(t *testing.T) map[string]string {
	t.Helper()

	workflow := readFileOrFail(t, verifyWorkflowPath())
	if !strings.Contains(workflow, "name: "+qaRunStep) {
		t.Fatalf("deliver-verify.yml carries no %q step: #1715's gate half is merged and REQUIRES "+
			"QA_VERDICT, so a workflow without the qa-review wiring is one whose delivery loop cannot "+
			"reach a verdict", qaRunStep)
	}
	return liveStepCode(t, workflow)
}

// liveStepCode collects each step's `if`, `env` values and `run` from the parsed workflow.
func liveStepCode(t *testing.T, workflow string) map[string]string {
	t.Helper()
	var wf struct {
		Jobs map[string]struct {
			Env   map[string]yaml.Node `yaml:"env"`
			Steps []struct {
				Name string               `yaml:"name"`
				If   string               `yaml:"if"`
				Run  string               `yaml:"run"`
				Env  map[string]yaml.Node `yaml:"env"`
			} `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal([]byte(workflow), &wf); err != nil {
		t.Fatalf("parsing deliver-verify.yml: %v", err)
	}
	job, ok := wf.Jobs["verify"]
	if !ok {
		t.Fatal("deliver-verify.yml has no `verify` job")
	}
	out := map[string]string{}
	for _, s := range job.Steps {
		var b strings.Builder
		b.WriteString(s.If)
		b.WriteString("\n")
		// Env keys sorted, not iterated in map order: the assembled text reaches a failure
		// message, and a diagnostic that reorders itself between runs is the small version of
		// exactly what R2/INV-6 are about.
		env := map[string]string{}
		for k, v := range s.Env {
			env[k] = v.Value
		}
		for _, k := range sortedKeys(env) {
			b.WriteString(k + ": " + env[k] + "\n")
		}
		b.WriteString(stripCommentLines(s.Run))
		out[s.Name] = b.String()
	}
	return out
}

// requireStep returns a step's executable text, failing if the wiring does not define it.
func requireStep(t *testing.T, steps map[string]string, name string) string {
	t.Helper()
	code, ok := steps[name]
	if !ok {
		t.Fatalf("the qa-review wiring defines no %q step; known steps: %v", name, sortedKeys(steps))
	}
	return code
}

func sortedKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	// Sorted so a failure message is byte-identical across runs (R2/INV-6 in spirit).
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j] < out[j-1]; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}

// TestDeliverVerifySuppliesEveryGateRequiredVar is the guard that stops half of #1715 from being
// merged on its own.
//
// The required list is READ OUT OF deliver-gate.sh rather than restated here, so it cannot drift:
// adding an eighth signal to the gate without wiring it into the Decide step fails the moment it
// is written, instead of at the next delivery — where the symptom is the whole loop stopping with
// a wiring error on a PR that has nothing wrong with it.
//
// WHEN THIS FAILS, DO NOT "FIX" IT BY WEAKENING THE GATE. Fail-closed on an unwired signal is the
// contract (AC-1); the fix is to wire the signal into the Decide step.
func TestDeliverVerifySuppliesEveryGateRequiredVar(t *testing.T) {
	gate := readFileOrFail(t, scriptPath(t, "deliver-gate.sh"))

	m := regexp.MustCompile(`(?m)^for var in ([A-Z_ ]+); do`).FindStringSubmatch(gate)
	if m == nil {
		t.Fatal("deliver-gate.sh no longer declares its required variables as `for var in …; do`, so " +
			"this test cannot read them. Restore that shape or teach this test the new one — do not " +
			"delete the test, it is what keeps the gate and the workflow in step")
	}
	required := strings.Fields(m[1])
	// The gate half of #1715 is present iff QA_VERDICT is one of the required, fail-closed signals.
	// Check for it by name rather than by count: #1763 later added MERGE_STATE (eight signals in
	// total), and a count threshold would go stale every time the signal set grows.
	hasQA := false
	for _, v := range required {
		if v == "QA_VERDICT" {
			hasQA = true
		}
	}
	if !hasQA {
		t.Fatalf("deliver-gate.sh's required variables (%v) do not include QA_VERDICT; the gate half "+
			"of #1715 has been reverted", required)
	}

	// The Decide step's env is the whole surface the gate reads. MAX_ROUNDS is deliberately
	// job-level, so both scopes count as supplied.
	raw := readFileOrFail(t, verifyWorkflowPath())
	var wf struct {
		Jobs map[string]struct {
			Env   map[string]yaml.Node `yaml:"env"`
			Steps []struct {
				Name string               `yaml:"name"`
				Env  map[string]yaml.Node `yaml:"env"`
			} `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal([]byte(raw), &wf); err != nil {
		t.Fatalf("parsing deliver-verify.yml: %v", err)
	}
	job, ok := wf.Jobs["verify"]
	if !ok {
		t.Fatal("deliver-verify.yml has no `verify` job")
	}
	supplied := map[string]bool{}
	for k := range job.Env {
		supplied[k] = true
	}
	found := false
	for _, s := range job.Steps {
		if s.Name != decideStep {
			continue
		}
		found = true
		for k := range s.Env {
			supplied[k] = true
		}
	}
	if !found {
		t.Fatalf("deliver-verify.yml has no %q step, so nothing runs the gate", decideStep)
	}

	// Every required signal must be supplied to the gate. Now that the qa-review wiring is live,
	// a missing one is a defect IN it — there is no longer a pending-patch state to tolerate.
	for _, name := range required {
		if supplied[name] {
			continue
		}
		t.Errorf("deliver-gate.sh requires %s but the %q step does not supply it. The gate exits 2 "+
			"on an unset required variable, so EVERY delivery would stop with a wiring error rather "+
			"than a verdict. Wire the signal into the Decide step — do NOT remove it from the gate",
			name, decideStep)
	}
}

// TestQAWiringPreservesTheNoExecutionInvariant covers #1715's AC-5, the load-bearing security
// property of this whole change.
//
// deliver-verify.yml runs on a SELF-HOSTED runner with the LiteLLM secrets in its environment, so
// PR-authored code must never execute there. qa-review needs to READ the PR head, which narrows
// that invariant; these assertions are what keep the narrowing from becoming a weakening.
func TestQAWiringPreservesTheNoExecutionInvariant(t *testing.T) {
	steps := qaWiring(t)

	run := requireStep(t, steps, qaRunStep)
	for _, r := range []struct{ needle, why string }{
		{
			needle: "--no-exec",
			why: "the answerer must drop its `go` build/test tool. Without it the tool loop can compile " +
				"and run PR-authored code on a runner holding LITELLM_API_KEY — the exact reason #1714 " +
				"added the flag rather than having this change hand-edit the vendored script",
		},
		{
			needle: "git worktree add --detach",
			why: "the PR head must be materialised as a DETACHED worktree, not checked out over the " +
				"trusted workspace tree that supplies scripts/ and .archon-version",
		},
	} {
		if !strings.Contains(run, r.needle) {
			t.Errorf("the %q step does not contain %q: %s", qaRunStep, r.needle, r.why)
		}
	}

	// The worktree must live under RUNNER_TEMP, which the runner clears between jobs — not in the
	// workspace, where actions/checkout's `git clean -ffdx` and the trusted tree both live.
	if !strings.Contains(run, "runner.temp") && !strings.Contains(run, "RUNNER_TEMP") {
		t.Errorf("the %q step does not place its worktree under $RUNNER_TEMP. Anywhere else it "+
			"either collides with the trusted workspace checkout or survives the job", qaRunStep)
	}

	// Removal must be unconditional. A cleanup that only runs on success leaves the tree behind on
	// exactly the runs that broke part-way through reading PR code — and the self-hosted runner's
	// workspace persists between deliveries, so one PR's files stay readable to the next.
	cleanup := requireStep(t, steps, qaCleanupStep)
	if !strings.Contains(cleanup, "git worktree remove --force") {
		t.Errorf("the %q step does not remove the worktree", qaCleanupStep)
	}
	if !strings.Contains(cleanup, "always()") {
		t.Errorf("the %q step is not guarded on `always()`, so the worktree survives a failed or "+
			"cancelled run: %s", qaCleanupStep, cleanup)
	}
}

// TestQAAnswererCanVerifyIssueCompletion pins #1792: the round-0 answerer must be able to verify
// issue-completeness for fixed question F1 ("does this PR fully implement the issue it closes?").
// It gains the adjudicator's read-only gh_issue/pr_diff tools (the tool-inventory half is pinned
// model-free in qa_review_test.go's TestNoExecSeam), and this test pins the WIRING half: the
// `Run qa-review` step must hand the answerer the PR and closing-issue NUMBERS so those tools have
// something to query, and QA_REPO must be in scope so they target the right repository. Without
// this, F1 could only ever be a blocking CANNOT_ANSWER, so the loop's own qa-review could never
// reach QA_VERDICT=PASS from round 0.
//
// The number flags sit in the executable `run` and QA_REPO in the step `env`, both of which
// liveStepCode collects, so they survive stripCommentLines.
func TestQAAnswererCanVerifyIssueCompletion(t *testing.T) {
	run := requireStep(t, qaWiring(t), qaRunStep)
	for _, r := range []struct{ needle, why string }{
		{
			needle: "--issue",
			why: "the answerer must be handed the closing issue number so its read-only gh_issue tool " +
				"can fetch the acceptance criteria and answer F1 on the merits instead of a blocking " +
				"CANNOT_ANSWER (#1792)",
		},
		{
			needle: "--pr",
			why: "the answerer must be handed the PR number so its read-only pr_diff tool can read what " +
				"actually changed when judging the fixed policy questions (#1792)",
		},
		{
			needle: "QA_REPO",
			why: "the step must put QA_REPO in scope so the answerer's gh_issue/pr_diff calls target the " +
				"right repository (they default to inference-sim/inference-sim otherwise) (#1792)",
		},
	} {
		if !strings.Contains(run, r.needle) {
			t.Errorf("the %q step does not contain %q: %s", qaRunStep, r.needle, r.why)
		}
	}
}

// TestQAVerdictMarkerContract covers #1715's AC-4 and AC-6: the marker must be as hard to spoof
// and as hard to fake-pass as the DELIVER-VERDICT marker it sits beside.
//
// Every assertion is scoped to the NEW reader's own step. Asserting over the whole job would let
// the pre-existing DELIVER-VERDICT reader satisfy these needles while the QA reader carried none
// of them — which is the mutation this test had to be rewritten to catch.
func TestQAVerdictMarkerContract(t *testing.T) {
	steps := qaWiring(t)

	read := requireStep(t, steps, qaReadStep)
	for _, r := range []struct{ needle, why string }{
		{
			needle: `grep -xE 'QA-VERDICT: (PASS|BLOCK)'`,
			why: "`grep -x` anchors the WHOLE line, so a comment that merely mentions the marker — " +
				"including a report quoting this contract — cannot set the signal. An unanchored " +
				"match would let any prose containing the string decide the gate",
		},
		{
			needle: `select(.user.type == \"Bot\")`,
			why: "the marker must be read only from a Bot-authored comment. This is a PUBLIC " +
				"repository: without the author filter any GitHub user could pass their own PR by " +
				"posting `QA-VERDICT: PASS` as their comment's last line",
		},
		{
			needle: "marker=MISSING",
			why: "an absent marker must default to MISSING, which the gate turns into needs-human. " +
				"AC-6: a qa-review run that produced no verdict must never be read as a pass",
		},
		{
			needle: "created_at >",
			why: "the marker must be read only from comments newer than this round's start, or a " +
				"PREVIOUS round's PASS could be returned as this round's verdict",
		},
	} {
		if !strings.Contains(read, r.needle) {
			t.Errorf("the %q step does not contain %q: %s", qaReadStep, r.needle, r.why)
		}
	}

	run := requireStep(t, steps, qaRunStep)
	if !strings.Contains(run, `.status == "FLAW_FOUND" or .status == "CANNOT_ANSWER"`) {
		t.Errorf("the %q step does not derive the verdict from the answerer's statuses using "+
			"render_report.py's own rule. render_report.py is vendored verbatim and documents its "+
			"verdict as a header EMOJI rather than a machine marker, so deriving the gate signal by "+
			"grepping the rendered report would let a rendering change move the gate", qaRunStep)
	}
	// Appended, so it is the report's last non-empty line — the shape the reader requires. A marker
	// written into the body would be read by nothing.
	if !strings.Contains(run, `"QA-VERDICT: $verdict"`) {
		t.Errorf("the %q step never appends `QA-VERDICT: $verdict` to the report, so no marker "+
			"reaches the PR comment the reader scans", qaRunStep)
	}
}

// TestQAWiringUsesTheCommittedToolingAndConfiguredModels covers the rest of AC-4: the wiring must
// drive the scripts #1714 vendored, over the OpenAI-compatible surface, with the documented
// `vars.* || default` model selection.
func TestQAWiringUsesTheCommittedToolingAndConfiguredModels(t *testing.T) {
	steps := qaWiring(t)
	run := requireStep(t, steps, qaRunStep)

	for _, script := range []string{"questioner.py", "answerer.py", "render_report.py"} {
		if !strings.Contains(run, "scripts/qa-review/"+script) {
			t.Errorf("the %q step never runs scripts/qa-review/%s. The committed tooling is the point "+
				"of #1714; a re-implementation here would diverge from the prototype it vendors",
				qaRunStep, script)
		}
		// The script it names must exist, or the step fails at run time on the one self-hosted
		// runner the loop has.
		if _, err := os.Stat(filepath.Join("..", "scripts", "qa-review", script)); err != nil {
			t.Errorf("scripts/qa-review/%s does not exist: %v", script, err)
		}
	}

	for _, r := range []struct{ needle, why string }{
		{"OPENAI_BASE_URL", "the qa-review scripts read the OpenAI-compatible surface, not the Anthropic one"},
		{"OPENAI_API_KEY", "same: the answerer exits 2 without a key, which would make every verdict MISSING"},
		{"secrets.LITELLM_BASE_URL", "the proxy URL must come from the repository secret the workflow already uses"},
		{"secrets.LITELLM_API_KEY", "the proxy key must come from the repository secret, never be inlined"},
		{"vars.QA_QUESTIONER_MODEL", "model selection must follow the DELIVER_*_MODEL `vars.* || default` pattern"},
		{"vars.QA_ANSWERER_MODEL", "same, for the answerer"},
		{"gcp/gemini-3.6-flash", "the questioner's documented default — and a DIFFERENT vendor from the implementer and reviewer, which is the whole thesis of RFC #1603"},
		{"azure/gpt-5.6-sol", "the answerer's documented default, likewise cross-vendor"},
	} {
		if !strings.Contains(run, r.needle) {
			t.Errorf("the %q step does not reference %q: %s", qaRunStep, r.needle, r.why)
		}
	}
}
