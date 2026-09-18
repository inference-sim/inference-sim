package scripts_test

// #1716 makes the delivery verify phase's qa-review dimension ADJUDICATE-ONLY on a re-verify:
// round 0 runs the full questioner+answerer probe (#1715, unchanged), every later round runs only
// adjudicator.py over the findings that probe already raised.
//
// The change is entirely in .github/workflows/deliver-verify.yml, which the delivering credential
// (a GitHub App installation token without the `workflows` permission) cannot push. As with #1715,
// the workflow half was therefore applied by a workflows-scoped push rather than by the delivery
// agent; the tests below assert the contract over the live workflow.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

const (
	roundStep        = "Read the round counter"
	qaAdjudicateStep = "Run qa-review adjudication"
)

// adjudicateWorkflow returns the text of the live deliver-verify.yml, which carries the #1716
// adjudicate-only wiring (applied via a workflows-scoped push, since the delivering token lacks
// the `workflows` permission). A live workflow without the adjudication step means #1716 was
// reverted.
func adjudicateWorkflow(t *testing.T) string {
	t.Helper()
	workflow := readFileOrFail(t, verifyWorkflowPath())
	if !strings.Contains(workflow, "name: "+qaAdjudicateStep) {
		t.Fatalf("deliver-verify.yml carries no %q step, so #1716's adjudicate-only wiring has been "+
			"reverted", qaAdjudicateStep)
	}
	return workflow
}

// adjudicateWiring returns the per-step executable text of the #1716 wiring, plus the whole
// workflow text (comments included — one assertion is about a comment).
func adjudicateWiring(t *testing.T) (steps map[string]string, workflow string) {
	t.Helper()
	workflow = adjudicateWorkflow(t)
	return liveStepCode(t, workflow), workflow
}

// stepOrder lists the verify job's step names in file order, so a test can assert that one step
// runs before another — which is the whole point of relocating the round counter.
func stepOrder(t *testing.T, workflow string) []string {
	t.Helper()
	var wf struct {
		Jobs map[string]struct {
			Steps []struct {
				Name string `yaml:"name"`
			} `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal([]byte(workflow), &wf); err != nil {
		t.Fatalf("parsing the patched deliver-verify.yml: %v", err)
	}
	job, ok := wf.Jobs["verify"]
	if !ok {
		t.Fatal("the patched deliver-verify.yml has no `verify` job")
	}
	out := make([]string, 0, len(job.Steps))
	for _, s := range job.Steps {
		out = append(out, s.Name)
	}
	return out
}

func qaStepIndex(t *testing.T, order []string, name string) int {
	t.Helper()
	for i, n := range order {
		if n == name {
			return i
		}
	}
	t.Fatalf("the verify job has no %q step; steps: %v", name, order)
	return -1
}

// TestQARoundGateIsExclusiveAndExhaustive covers #1716 AC-1, AC-2 and AC-5.
//
// The two branches must be selected by ONE derived boolean rather than by two independently
// written expressions. With two, a value that satisfies neither (the empty string a failed
// `gh pr view` leaves behind) silently skips the qa dimension altogether — which reads as MISSING
// and stops the loop, but for a reason no log line explains.
func TestQARoundGateIsExclusiveAndExhaustive(t *testing.T) {
	steps, workflow := adjudicateWiring(t)

	// AC-5: the counter must be read BEFORE the step that branches on it. A step cannot read the
	// output of one that runs later — the expression would evaluate to the empty string.
	order := stepOrder(t, workflow)
	roundAt := qaStepIndex(t, order, roundStep)
	for _, consumer := range []string{qaRunStep, qaAdjudicateStep} {
		if at := qaStepIndex(t, order, consumer); at < roundAt {
			t.Errorf("%q runs at position %d, before %q at %d, so its round gate reads an empty "+
				"output and the branch is decided by accident", consumer, at, roundStep, roundAt)
		}
	}
	// Relocated, not duplicated: two counter steps would give the gate and the Decide step
	// different values on a PR whose labels change mid-job.
	if n := strings.Count(workflow, "name: "+roundStep); n != 1 {
		t.Errorf("the workflow defines %d %q steps, want exactly 1", n, roundStep)
	}

	round := requireStep(t, steps, roundStep)
	for _, r := range []struct{ needle, why string }{
		{
			needle: `echo "value=$round" >> "$GITHUB_OUTPUT"`,
			why: "the relocation must be behaviour-preserving: the round-cap and the Decide step read " +
				"this same `value` output, and AC-5 moves WHERE it is read, not WHAT it reports",
		},
		{
			needle: `reverify=$reverify" >> "$GITHUB_OUTPUT`,
			why:    "the qa-review round gate reads a derived `reverify` output; without it neither qa branch runs",
		},
		{
			needle: `^[0-9]+$`,
			why: "the derivation must require a NUMBER. An unparseable or empty round must resolve to " +
				"`false` (the full pass), never to an adjudication with no prior report to re-check",
		},
		{
			needle: "reverify=false",
			why:    "the non-re-verify case must be an explicit `false`, or the full-pass branch never runs",
		},
	} {
		if !strings.Contains(round, r.needle) {
			t.Errorf("the %q step does not contain %q: %s", roundStep, r.needle, r.why)
		}
	}

	// The two branches, on the same derived boolean and opposite values.
	full := requireStep(t, steps, qaRunStep)
	adj := requireStep(t, steps, qaAdjudicateStep)
	if !strings.Contains(full, "steps.round.outputs.reverify == 'false'") {
		t.Errorf("the %q step is not gated to round 0; it would run the full questioner+answerer pass "+
			"on every correction round, which is exactly the cost #1716 removes: %s", qaRunStep, full)
	}
	if !strings.Contains(adj, "steps.round.outputs.reverify == 'true'") {
		t.Errorf("the %q step is not gated to a re-verify; on round 0 it would adjudicate findings "+
			"that no probe has raised yet: %s", qaAdjudicateStep, adj)
	}

	// AC-2: a re-verify runs ONLY the adjudicator.
	for _, script := range []string{"questioner.py", "answerer.py", "render_report.py"} {
		if strings.Contains(adj, script) {
			t.Errorf("the %q step runs %s. A re-verify is adjudicate-only: re-probing the whole diff "+
				"every correction round is the cost this change exists to remove", qaAdjudicateStep, script)
		}
	}
	if !strings.Contains(adj, "scripts/qa-review/adjudicator.py") {
		t.Errorf("the %q step does not run scripts/qa-review/adjudicator.py, so nothing re-checks the "+
			"prior findings", qaAdjudicateStep)
	}
	if _, err := os.Stat(filepath.Join("qa-review", "adjudicator.py")); err != nil {
		t.Errorf("scripts/qa-review/adjudicator.py does not exist: %v", err)
	}
}

// TestQAAdjudicationPreservesTheNoExecutionInvariant covers AC-6's security half. deliver-verify.yml
// runs on a SELF-HOSTED runner with the LiteLLM secrets in its environment; the adjudicator reads
// the PR head, so it must read it the same way the answerer does — and nothing more.
func TestQAAdjudicationPreservesTheNoExecutionInvariant(t *testing.T) {
	steps, _ := adjudicateWiring(t)
	adj := requireStep(t, steps, qaAdjudicateStep)

	for _, r := range []struct{ needle, why string }{
		{
			needle: "--no-exec",
			why: "the adjudicator must drop its `go` build/test tool. Without it the tool loop can " +
				"compile and run PR-authored code on a runner holding LITELLM_API_KEY — the flag #1714 " +
				"added for exactly this, and which #1715 already passes on the answerer",
		},
		{
			needle: "git worktree add --detach",
			why: "the PR head must be materialised as a DETACHED worktree, not checked out over the " +
				"trusted workspace tree that supplies scripts/",
		},
	} {
		if !strings.Contains(adj, r.needle) {
			t.Errorf("the %q step does not contain %q: %s", qaAdjudicateStep, r.needle, r.why)
		}
	}

	// The same ephemeral worktree the round-0 branch uses, so the SAME always-run cleanup removes
	// it. A second path would survive the job on a runner whose workspace persists between
	// deliveries.
	const worktree = "WORKTREE: ${{ runner.temp }}/qa-head"
	for _, step := range []string{qaRunStep, qaAdjudicateStep, qaCleanupStep} {
		if !strings.Contains(requireStep(t, steps, step), worktree) {
			t.Errorf("the %q step does not use %q, so the re-verify's PR-head worktree is not the one "+
				"the always-run cleanup removes", step, worktree)
		}
	}
}

// TestQAAdjudicationMarkerReachesTheGateLikeRoundZero covers AC-4: the re-verify must produce the
// marker by the same render-to-file → append → post-once route as round 0, so the ONE existing
// reader serves both branches and the gate needs no knowledge of which ran.
func TestQAAdjudicationMarkerReachesTheGateLikeRoundZero(t *testing.T) {
	steps, workflow := adjudicateWiring(t)
	adj := requireStep(t, steps, qaAdjudicateStep)

	for _, r := range []struct{ needle, why string }{
		{
			needle: "--out",
			why: "the report must be RENDERED TO A FILE so the marker can be appended to it before it " +
				"is posted",
		},
		{
			needle: `"QA-VERDICT: $verdict"`,
			why: "the derived marker must be appended to the report, as the round-0 branch does. " +
				"adjudicator.py emits its aggregate verdict on stderr only and writes no marker of its " +
				"own, so without this the reader finds nothing and every re-verify reads MISSING",
		},
		{
			needle: "--body-file",
			why:    "the report and its marker must be posted as ONE comment whose last non-empty line is the marker",
		},
		{
			needle: `grep -xE '\[adjudication verdict: (PASS|BLOCK)\]'`,
			why: "the verdict must be read from the adjudicator's own stderr line, whole-line anchored: " +
				"the report body is model-written, so an unanchored match could be satisfied by a " +
				"rationale that quotes the line",
		},
		{
			needle: "MISSING",
			why: "the fail-closed path must be stated where it is taken: every early exit leaves no " +
				"marker, which the reader turns into MISSING and the gate into needs-human (AC-6)",
		},
	} {
		if !strings.Contains(adj, r.needle) {
			t.Errorf("the %q step does not contain %q: %s", qaAdjudicateStep, r.needle, r.why)
		}
	}

	if strings.Contains(adj, "--post-to-pr") {
		t.Errorf("the %q step passes --post-to-pr, which posts the report BEFORE the marker is "+
			"appended — leaving the comment the reader scans without a last-line marker (and forcing a "+
			"post-then-edit): %s", qaAdjudicateStep, adj)
	}

	// ONE reader for both branches (AC-4). A second one would be a second place for the marker
	// contract to drift.
	if n := strings.Count(workflow, `grep -xE 'QA-VERDICT: (PASS|BLOCK)'`); n != 1 {
		t.Errorf("the workflow reads the QA-VERDICT marker in %d places, want exactly 1: both round "+
			"types must feed the gate through the same %q step", n, qaReadStep)
	}
}

// TestQAAdjudicationEnvIsWired covers the rest of AC-6: the adjudicator needs the same
// OpenAI-compatible surface as the answerer, and its model must follow the repository's
// `vars.* || default` convention.
func TestQAAdjudicationEnvIsWired(t *testing.T) {
	steps, _ := adjudicateWiring(t)
	adj := requireStep(t, steps, qaAdjudicateStep)

	for _, r := range []struct{ needle, why string }{
		{"OPENAI_BASE_URL", "adjudicator.py reads the OpenAI-compatible surface and exits 2 without it"},
		{"OPENAI_API_KEY", "same: no key means no run, and therefore no verdict on every re-verify"},
		{"secrets.LITELLM_BASE_URL", "the proxy URL must come from the repository secret the workflow already uses"},
		{"secrets.LITELLM_API_KEY", "the proxy key must come from the repository secret, never be inlined"},
		{"vars.QA_ADJUDICATOR_MODEL", "model selection must follow the DELIVER_*_MODEL `vars.* || default` pattern"},
		{"azure/gpt-5.6-sol", "the adjudicator's documented default — cross-vendor from the implementer and reviewer"},
		{
			"QA_REPORT_AUTHOR",
			"the prior report must be looked for under the identity that POSTED it. This is a public " +
				"repository: without the restriction any commenter could post a report-shaped comment " +
				"with an empty Items-to-fix section and clear every outstanding finding",
		},
	} {
		if !strings.Contains(adj, r.needle) {
			t.Errorf("the %q step does not reference %q: %s", qaAdjudicateStep, r.needle, r.why)
		}
	}
}

// TestReVerifyTradeoffIsDocumentedAtTheRoundGate covers AC-7. The tradeoff — a re-verify does not
// re-probe the diff, so a regression a fix introduces is caught by the other three signals rather
// than by qa-review — is the pivotal decision of this change. It has to be readable at the gate
// that implements it, not only in an issue.
func TestReVerifyTradeoffIsDocumentedAtTheRoundGate(t *testing.T) {
	_, workflow := adjudicateWiring(t)

	start := strings.Index(workflow, "# ROUND GATE")
	if start < 0 {
		t.Fatal("the workflow has no `# ROUND GATE` comment block, so the round branch is undocumented")
	}
	end := strings.Index(workflow[start:], "- name: "+qaRunStep)
	if end < 0 {
		t.Fatalf("the `# ROUND GATE` comment block is not immediately above the %q step, so it does "+
			"not document the branch it explains", qaRunStep)
	}
	block := workflow[start : start+end]

	for _, r := range []struct{ needle, why string }{
		{"ADJUDICATE-ONLY", "the comment must say what a re-verify does instead of the full pass"},
		{"CI_STATUS", "it must name CI as one of the signals that DOES re-derive on the new head"},
		{"PLAN_GATE", "it must name the archon plan ratchet as another"},
		{"AGENT_VERDICT", "it must name blis-pr-review as the third"},
	} {
		if !strings.Contains(block, r.needle) {
			t.Errorf("the round-gate comment block does not mention %q: %s\n\nblock:\n%s",
				r.needle, r.why, block)
		}
	}
}

// TestRoundCounterRelocationCannotChangeWhatItReads is the other half of AC-5. Moving the read
// earlier is behaviour-preserving only because nothing between the two positions writes a round
// label; if a step ever does, the gate and the Decide step would disagree about the round.
func TestRoundCounterRelocationCannotChangeWhatItReads(t *testing.T) {
	_, workflow := adjudicateWiring(t)

	// The counter is the max of the `deliver:round-N` labels, and only the CORRECT phase applies
	// one. A write from this workflow — before or after the read — would make the position matter.
	for _, verb := range []string{`labels[]=deliver:round-`, `issues/$PR/labels/deliver:round-`} {
		if strings.Contains(stripCommentLines(workflow), verb) {
			t.Errorf("deliver-verify.yml writes a round label (%q). The round counter is read before "+
				"the qa-review step now, so a write anywhere in this job makes the read position "+
				"observable: the gate could see a different round than the qa branch did", verb)
		}
	}

	// And the consumers that were already there must still read the same output.
	if n := strings.Count(workflow, "steps.round.outputs.value"); n < 2 {
		t.Errorf("only %d step(s) read `steps.round.outputs.value`; the Decide step and the summary "+
			"both did before the relocation, so the move dropped a consumer", n)
	}
}
