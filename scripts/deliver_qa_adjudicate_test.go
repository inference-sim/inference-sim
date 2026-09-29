package scripts_test

// #1716 makes the delivery verify phase's qa-review dimension ADJUDICATE-ONLY on a re-verify:
// round 0 runs the full questioner+answerer probe (#1715, unchanged), every later round runs only
// adjudicator.py over the findings that probe already raised.
//
// The change is entirely in .github/workflows/deliver-verify.yml, which the delivering credential
// (a GitHub App installation token without the `workflows` permission) cannot push. As with #1715,
// the workflow half was therefore applied by a workflows-scoped push rather than by the delivery
// agent; the tests below assert the contract over the live workflow.
//
// #1834 adds guards for that same step at the bottom of this file, plus a behavioral test that
// EXECUTES the step under `bash -e`. Its workflow half — the canonical QA_REPORT_AUTHOR and the
// `-e`-safe exit-code capture — was applied by a workflows-scoped push in #1837 (the delivering
// token cannot push .github/workflows/*, the same permission reason as #1716/#1715). The guards
// still read the step from whichever of the two carries it, so they keep working if that half is
// ever regenerated as a patch. #1834's primary fix is live in scripts/qa-review/adjudicator.py, so
// the re-verify worked before this half was applied.

import (
	"bytes"
	"os"
	"os/exec"
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

// ---------------------------------------------------------------------------
// #1834 — the re-verify must be able to find its OWN report, and must say so when it cannot.
// ---------------------------------------------------------------------------

// The pending workflow half of #1834. Both of its hunks are in the adjudication step; see the
// patch's own header for why it is a patch (the delivering App token cannot push
// .github/workflows/*, measured again on this delivery) and for why the PRIMARY fix is not in it
// — that one is live in scripts/qa-review/adjudicator.py, so the loop is repaired whether or not
// a human ever applies this file.
const adjudicateRCPatch = "scripts/deliver-verify-adjudicate-rc.patch"

// errorExitSafeCapture is the `-e`-safe exit-code capture the step must use. `bash -e {0}` is the
// default shell and `set -uo pipefail` does not clear `-e`, so a bare `rc=$?` on the line AFTER
// the invocation is never reached when the invocation fails.
const errorExitSafeCapture = "|| rc=$?"

func adjudicateRCPatchPath() string {
	return filepath.Join("..", adjudicateRCPatch)
}

// adjudicationStepVariants returns the EXECUTABLE text of the adjudication step in every shape it
// can actually run in on this tree, labelled — the LIVE workflow, and (while the workflow half is
// still pending) the post-image of applying the patch to it.
//
// Both are returned rather than just the effective one because they differ in a way that matters:
// the live workflow is what runs on the next delivery, and the patched one is what runs after a
// human lands the patch. A guard that read only the post-image would go green on a tree whose LIVE
// wiring is still broken — which is the state this repository is in until the patch is applied.
//
// A tree with NEITHER the live capture nor the patch is a failure, not a skip: these tests are the
// only thing asserting the contract, so passing on a tree that has lost both halves would make the
// guard optional. Applying the diff — rather than matching its added lines — is what makes a STALE
// patch fail here, at every CI run, instead of at the moment a human tries to land it.
func adjudicationStepVariants(t *testing.T) (variants map[string]string, live bool) {
	t.Helper()

	current := readFileOrFail(t, verifyWorkflowPath())
	liveStep := requireStep(t, liveStepCode(t, current), qaAdjudicateStep)
	if strings.Contains(liveStep, errorExitSafeCapture) {
		// Applied. The patch must not also still be sitting in the tree: a leftover would read as
		// "still pending" to anyone scanning for one, and would be re-applied on top of itself.
		if _, err := os.Stat(adjudicateRCPatchPath()); err == nil {
			t.Errorf("deliver-verify.yml already carries #1834's %q capture, but %s is still in the "+
				"tree. Delete it in the commit that applies it, or a reader cannot tell which half is "+
				"in force", errorExitSafeCapture, adjudicateRCPatch)
		}
		return map[string]string{"the live deliver-verify.yml": liveStep}, true
	}

	patch, err := os.ReadFile(adjudicateRCPatchPath())
	if err != nil {
		t.Fatalf("the %q step carries no %q capture and %s is missing (%v). One of the two must hold "+
			"#1834's workflow half; with neither, a failing adjudication aborts the step under `bash -e` "+
			"before its own `skip` can say why, and the delivery loop reports only the generic \"the "+
			"verify phase failed or timed out\"",
			qaAdjudicateStep, errorExitSafeCapture, adjudicateRCPatch, err)
	}
	patched, err := applyUnifiedDiff(current, string(patch), "deliver-verify.yml")
	if err != nil {
		t.Fatalf("%s no longer applies to deliver-verify.yml: %v\n\n"+
			"It carries #1834's workflow half because the delivering App token lacks the `workflows` "+
			"permission. A patch that does not apply cannot be landed, so regenerate it against the "+
			"current workflow — do not delete this test to make the failure go away", adjudicateRCPatch, err)
	}
	patchedStep := requireStep(t, liveStepCode(t, patched), qaAdjudicateStep)
	if !strings.Contains(patchedStep, errorExitSafeCapture) {
		t.Fatalf("%s applies but the resulting %q step still has no %q capture, so it is not #1834's "+
			"wiring", adjudicateRCPatch, qaAdjudicateStep, errorExitSafeCapture)
	}
	return map[string]string{
		"the live deliver-verify.yml":               liveStep,
		"deliver-verify.yml + " + adjudicateRCPatch: patchedStep,
	}, false
}

// adjudicationStepWith1834Wiring returns the adjudication step as it will run once #1834's
// workflow half is in place — the live step when it is applied, the patch's post-image while it is
// pending — and whether it is already live.
func adjudicationStepWith1834Wiring(t *testing.T) (step string, live bool) {
	t.Helper()
	variants, live := adjudicationStepVariants(t)
	if live {
		return variants["the live deliver-verify.yml"], true
	}
	return variants["deliver-verify.yml + "+adjudicateRCPatch], false
}

// TestQAAdjudicationStepSurfacesTheAdjudicatorsExitCode is #1834's secondary half: the step's
// graceful degradation must actually run.
//
// The step is written to degrade — every early exit calls `skip`, which emits a ::warning:: naming
// the reason and leaves no QA-VERDICT marker, so the gate reads MISSING and stops for a human who
// can see WHY. That was defeated by `bash -e`: the adjudicator's invocation was followed by a bare
// `rc=$?` on the next line, so a nonzero exit aborted the step at the invocation — before the
// capture, before `cat`ing the captured stderr into the log, and before the `skip`. On PR #1832 the
// step died with "Process completed with exit code 3" and the intended message ("no prior report to
// adjudicate, or a failure") was never posted. That is why a deterministic, every-round regression
// read as an intermittent infrastructure failure.
func TestQAAdjudicationStepSurfacesTheAdjudicatorsExitCode(t *testing.T) {
	step, _ := adjudicationStepWith1834Wiring(t)

	// The invocation and the capture must be ONE command. A bare `rc=$?` statement is the bug.
	for _, line := range strings.Split(step, "\n") {
		if strings.TrimSpace(line) == "rc=$?" {
			t.Errorf("the %q step captures the adjudicator's exit code with a standalone `rc=$?`. "+
				"Under the default `bash -e {0}` — which `set -uo pipefail` does not clear — a nonzero "+
				"exit aborts the step before that line, so the `skip` below it never runs and the real "+
				"reason is swallowed. Use the inline %q form the sibling author-gate steps use",
				qaAdjudicateStep, errorExitSafeCapture)
		}
	}
	// And `rc` must be initialised, or `set -u` makes the `[[ "$rc" -eq 0 ]]` test below it an
	// unbound-variable error on the success path.
	if !strings.Contains(step, "rc=0") {
		t.Errorf("the %q step does not initialise `rc=0` before the adjudicator runs. With %q the "+
			"variable is only ever assigned on FAILURE, so under `set -u` the success path dies on the "+
			"unbound `$rc`", qaAdjudicateStep, errorExitSafeCapture)
	}
	// The degradation the capture exists to reach must still be there.
	for _, needle := range []string{`cat "$RUNNER_TEMP/qa-adjudication.err" >&2`, `[[ "$rc" -eq 0 ]]`} {
		if !strings.Contains(step, needle) {
			t.Errorf("the %q step no longer contains %q, so capturing the exit code buys nothing: the "+
				"adjudicator's own diagnostics never reach the job log", qaAdjudicateStep, needle)
		}
	}
}

// TestQAReportAuthorFindsTheReportTheDeliveryLoopPosts is #1834's primary half, asserted
// END TO END across the two halves that disagreed: the workflow's configured QA_REPORT_AUTHOR and
// the login the comment source actually reports.
//
// Neither half is wrong in isolation, which is exactly why #1806 could break this with every one
// of its own tests passing. So this guard does not assert a value — it takes whatever the workflow
// configures and checks that adjudicator.py, given a report posted under the login
// scripts/deliver-trusted-comments.sh emits, FINDS it. Before the fix that fails for the live
// workflow's `github-actions` against REST's `github-actions[bot]`; it fails again if a future
// edit changes either side alone.
func TestQAReportAuthorFindsTheReportTheDeliveryLoopPosts(t *testing.T) {
	requirePython3(t)

	// The fixture login must be the one the filter really emits, or this test would assert against
	// a spelling nothing produces. deliver-trusted-comments.sh reads REST precisely so its
	// automation allowlist can key on the canonical suffixed form.
	filter := readFileOrFail(t, filepath.Join("..", "scripts", "deliver-trusted-comments.sh"))
	if !strings.Contains(filter, `"`+filterReportLogin+`"`) {
		t.Fatalf("scripts/deliver-trusted-comments.sh no longer allowlists %q, so that is not the "+
			"login it reports for the report's poster and this guard is checking the wrong spelling",
			filterReportLogin)
	}

	// The THIRD side of the contract, since the `[bot]`-suffix equivalence became conditional on
	// the trust label (#1834 G2): the adjudicator grants it only to a comment labelled
	// `filterAutomationLabel`, so if the filter ever spells that label differently the equivalence
	// silently stops applying — and #1834's silent needs-human comes straight back. Asserted
	// against the emitting jq, which is where the string is produced.
	labels := readFileOrFail(t, filepath.Join("..", "scripts", "deliver-trusted-comments.jq"))
	if !strings.Contains(labels, `"`+filterAutomationLabel+`"`) {
		t.Fatalf("scripts/deliver-trusted-comments.jq no longer emits the trust label %q, which is "+
			"the evidence adjudicator.py requires before it treats `github-actions` and "+
			"`github-actions[bot]` as one actor. Without it the restriction is exact again and "+
			"every re-verify goes back to exiting 3", filterAutomationLabel)
	}

	// EVERY shape the step can run in, live included. The live workflow still carries the short
	// form while the workflow patch is pending, and that is the one the next delivery uses — so it
	// is the one that must work, not merely the post-patch shape.
	variants, _ := adjudicationStepVariants(t)
	for _, name := range sortedKeys(variants) {
		t.Run(name, func(t *testing.T) {
			var configured string
			for _, line := range strings.Split(variants[name], "\n") {
				if rest, ok := strings.CutPrefix(line, "QA_REPORT_AUTHOR: "); ok {
					configured = strings.TrimSpace(rest)
				}
			}
			// Retained, not dropped. It is defence in depth over the write-access filter now rather
			// than the only barrier, but removing it is a decision, not a side effect of fixing a
			// login form.
			if configured == "" {
				t.Fatalf("in %s the %q step sets no QA_REPORT_AUTHOR. Without it any comment the "+
					"trusted-comments filter admits can supply the prior findings, and one with an "+
					"empty Items-to-fix section clears every outstanding finding as a vacuous PASS",
					name, qaAdjudicateStep)
			}
			if !selectionFindsReport(t, filterReportLogin, filterAutomationLabel, configured) {
				t.Errorf("adjudicator.py does not find a qa-review report posted by %q when the "+
					"restriction is QA_REPORT_AUTHOR=%q, as %s configures it. That is #1834: the "+
					"report is skipped, fetch_comments returns None, main() exits 3 with no verdict "+
					"line, and the gate reads MISSING — so EVERY correction round ends at needs-human "+
					"no matter how good the corrections were", filterReportLogin, configured, name)
			}
		})
	}
}

// TestAdjudicateRCWiringStatusMatchesReality mirrors TestApproverWiringStatusMatchesReality for
// #1834's patch: the patch and the doc note describing it as pending must both be present while the
// workflow half has not landed, and both gone once it has.
//
// A leftover patch reads as pending work that is already done, and would make the staleness check
// in adjudicationStepVariants unreachable. A leftover doc note is worse: it tells a reader the
// adjudication step still swallows its failure reason when it no longer does. #1715 shipped exactly
// that stale status and needed a follow-up commit to remove it.
func TestAdjudicateRCWiringStatusMatchesReality(t *testing.T) {
	const docPath = "docs/contributing/automated-delivery.md"
	doc := readFileOrFail(t, filepath.Join("..", "docs", "contributing", "automated-delivery.md"))
	// Matched on the patch FILENAME rather than on prose: the paragraph can be reworded freely, but
	// naming a patch file that no longer exists is the specific staleness this catches.
	docSaysPending := strings.Contains(doc, "deliver-verify-adjudicate-rc.patch")
	_, patchErr := os.Stat(adjudicateRCPatchPath())
	patchExists := patchErr == nil
	liveStep := requireStep(t, liveStepCode(t, readFileOrFail(t, verifyWorkflowPath())), qaAdjudicateStep)
	live := strings.Contains(liveStep, errorExitSafeCapture)

	if live {
		if patchExists {
			t.Errorf("deliver-verify.yml carries #1834's %q capture AND %s still exists. Delete the "+
				"patch in the commit that applies it — a leftover patch describes work that is already "+
				"done", errorExitSafeCapture, adjudicateRCPatch)
		}
		if docSaysPending {
			t.Errorf("deliver-verify.yml carries #1834's %q capture but %s still describes it as "+
				"pending in a patch. Remove that text in the same commit: a doc saying a live fix is "+
				"unwired is worse than no doc", errorExitSafeCapture, docPath)
		}
		return
	}

	if !patchExists {
		t.Errorf("#1834's workflow half is neither live nor in %s (%v)", adjudicateRCPatch, patchErr)
	}
	if !docSaysPending {
		t.Errorf("#1834's workflow half is pending in %s but %s does not say so. A reader hitting the "+
			"generic \"the verify phase failed or timed out\" has no way to learn that the step's own "+
			"reason is being swallowed by `bash -e`", adjudicateRCPatch, docPath)
	}
}

// adjudicationRunBody returns the `run:` shell body of the adjudication step (parsed out of the
// YAML so it excludes the `if:`/`env:` lines and `${{ }}` expressions, which are not shell), and
// whether that body carries #1834's `-e`-safe capture. Only the body is executable.
func adjudicationRunBody(t *testing.T) (body string, wired bool) {
	t.Helper()
	var wf struct {
		Jobs map[string]struct {
			Steps []struct {
				Name string `yaml:"name"`
				Run  string `yaml:"run"`
			} `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal([]byte(readFileOrFail(t, verifyWorkflowPath())), &wf); err != nil {
		t.Fatalf("parsing deliver-verify.yml: %v", err)
	}
	for _, job := range wf.Jobs {
		for _, s := range job.Steps {
			if s.Name == qaAdjudicateStep {
				return s.Run, strings.Contains(s.Run, errorExitSafeCapture)
			}
		}
	}
	t.Fatalf("no %q step with a run: body found in deliver-verify.yml", qaAdjudicateStep)
	return "", false
}

// TestQAAdjudicationStepReachesSkipWhenTheAdjudicatorFails EXECUTES the adjudication step's run body
// under `bash -e` with a stub adjudicator that exits nonzero, and asserts the step reaches its
// graceful `skip` — a ::warning:: naming the exit code, the adjudicator's stderr replayed, and NO
// verdict marker posted — rather than aborting at the invocation.
//
// This is the behavioral counterpart to TestQAAdjudicationStepSurfacesTheAdjudicatorsExitCode,
// which only inspects the step's TEXT. #1834's defect was behavioral: under the default
// `bash -e {0}` (which `set -uo pipefail` does not clear) a bare `rc=$?` on the line AFTER the
// invocation is never reached, so the step died at the invocation and the `skip` never ran. A text
// guard can be satisfied by a step that still misbehaves at runtime; this one runs the step, so
// reverting to the bare-`rc=$?` shape makes it fail — the step exits nonzero and prints no warning.
// python3/git/gh are replaced by PATH stubs so nothing real is invoked.
func TestQAAdjudicationStepReachesSkipWhenTheAdjudicatorFails(t *testing.T) {
	body, wired := adjudicationRunBody(t)
	if !wired {
		// The workflow half is still pending in the patch; the text guards and the patch-apply guard
		// cover that state, and this executable test applies once the half is live.
		t.Skipf("the %q step is not yet #1834-wired (no %q); pending-state guards cover it",
			qaAdjudicateStep, errorExitSafeCapture)
	}

	dir := t.TempDir()
	binDir := filepath.Join(dir, "bin")
	runnerTemp := filepath.Join(dir, "runner")
	for _, d := range []string{binDir, runnerTemp} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	// gh must NOT run: the failure-path `skip` precedes the `gh pr comment`. The stub records a call
	// so the test can prove it never happened.
	ghRan := filepath.Join(runnerTemp, "gh-was-called")
	stubs := map[string]string{
		"git":     "#!/usr/bin/env bash\nexit 0\n",                                               // worktree add: no-op success
		"python3": "#!/usr/bin/env bash\necho 'stub-adjudicator: no prior report' >&2\nexit 3\n", // the adjudicator's exit 3
		"gh":      "#!/usr/bin/env bash\ntouch \"" + ghRan + "\"\nexit 0\n",
	}
	for name, script := range stubs {
		if err := os.WriteFile(filepath.Join(binDir, name), []byte(script), 0o755); err != nil {
			t.Fatal(err)
		}
	}

	// Match GitHub's `bash -e {0}` for a `run:` step; the body's own `set -uo pipefail` adds -u, so
	// every referenced variable must be set.
	cmd := exec.Command("bash", "-e", "-c", body)
	cmd.Env = append(os.Environ(),
		"PATH="+binDir+string(os.PathListSeparator)+os.Getenv("PATH"),
		"RUNNER_TEMP="+runnerTemp,
		"WORKTREE="+filepath.Join(runnerTemp, "qa-head"),
		"HEAD_SHA=0000000000000000000000000000000000000000",
		"PR=0",
		"REPO=example/repo",
		"ROUND=1",
	)
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	runErr := cmd.Run()
	got := out.String()

	// 1. Graceful degradation: `skip`'s `exit 0`, not the adjudicator's 3 aborting the step.
	if runErr != nil {
		t.Fatalf("the %q step exited nonzero (%v) when the adjudicator failed, instead of reaching its "+
			"`skip`. A bare `rc=$?` after the invocation reproduces this — the #1834 defect.\n\nOutput:\n%s",
			qaAdjudicateStep, runErr, got)
	}
	// 2. The skip fired and named the exit code, so the gate reads MISSING with a stated reason.
	if !strings.Contains(got, "the qa-review adjudicator exited 3") {
		t.Errorf("the %q step did not emit its exit-3 skip warning; the intended MISSING reason was "+
			"swallowed.\n\nOutput:\n%s", qaAdjudicateStep, got)
	}
	// 3. The adjudicator's captured stderr was replayed into the log (`cat ...err >&2`), or the
	//    failure would be undiagnosable.
	if !strings.Contains(got, "stub-adjudicator: no prior report") {
		t.Errorf("the %q step did not replay the adjudicator's stderr into the log.\n\nOutput:\n%s",
			qaAdjudicateStep, got)
	}
	// 4. No verdict marker posted: a skipped adjudication must not reach `gh pr comment`.
	if _, statErr := os.Stat(ghRan); statErr == nil {
		t.Errorf("the %q step ran `gh pr comment` despite the adjudicator failing; a skipped "+
			"adjudication must leave no QA-VERDICT marker for the gate to read", qaAdjudicateStep)
	}
}
