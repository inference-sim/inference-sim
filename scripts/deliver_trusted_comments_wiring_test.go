package scripts_test

// Pins that the trusted-comment filter (#1806) is actually WIRED into the AI flows, not just
// present as a standalone helper. deliver_trusted_comments_test.go proves the helper selects the
// right comments; this file proves the workflows CALL it and feed the agent its output instead of
// reading comments raw.
//
// Like the sibling workflow tests, these assert on file text: for a workflow the declared text IS
// the behaviour — GitHub reads nothing else, and the file only ever runs inside Actions, where no
// other test can reach it. The regression guarded is concrete and has already happened once (the
// helper shipped unwired): these pin that the digest step is PRESENT and its producer path matches
// the path the prompt reads, so deleting the step or drifting the path fails.
//
// Scope, stated honestly: this checks the digest wiring is present, NOT that a raw comment read is
// ABSENT. A negative "no `gh api .../comments`" assertion is deliberately not made — the prompts
// legitimately NAME that endpoint in a "do NOT run" instruction, which a text check cannot tell
// apart from an actual fetch. So a future edit that KEPT the digest but ALSO added a raw fetch would
// pass here; that is out of a text test's reach and left to review.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func readWorkflow(t *testing.T, name string) string {
	t.Helper()
	path := filepath.Join("..", ".github", "workflows", name)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(raw)
}

// The two delivery-loop phases read comments through a prompt this repository controls, so both
// MUST assemble the filtered digest in a step and point the agent at THE SAME FILE.
//
// The producer path (`$RUNNER_TEMP/trusted-comments.md`, written by the step's shell) and the
// consumer path (`${{ runner.temp }}/trusted-comments.md`, named in the prompt) are asserted as
// exact strings, not as two independent substrings: `$RUNNER_TEMP` and `${{ runner.temp }}` are the
// same directory, so pinning both spellings of the full path is what catches a producer/consumer
// mismatch (e.g. writing a.md and reading b.md) that a bare "does `trusted-comments.md` appear
// anywhere" check would pass.
func TestWiring_VerifyAndCorrect_AssembleAndReadTheTrustedDigest(t *testing.T) {
	const (
		producer = `> "$RUNNER_TEMP/trusted-comments.md"`      // the digest step's redirect
		consumer = "${{ runner.temp }}/trusted-comments.md"   // the path named in the agent prompt
	)
	for _, wf := range []string{"deliver-verify.yml", "deliver-correct.yml"} {
		text := readWorkflow(t, wf)

		if !strings.Contains(text, "Assemble the trusted comment digest") {
			t.Errorf("%s is missing the `Assemble the trusted comment digest` step (#1806)", wf)
		}
		// The step invokes the helper and writes the digest to the producer path.
		if !strings.Contains(text, "deliver-trusted-comments.sh --pr") {
			t.Errorf("%s does not invoke `deliver-trusted-comments.sh --pr` — the comment filter "+
				"(#1806) is not wired; the agent would read comments unfiltered", wf)
		}
		if !strings.Contains(text, producer) {
			t.Errorf("%s does not write the digest to %s — the producer path drifted from the "+
				"consumer path the prompt reads (#1806)", wf, producer)
		}
		// The prompt points the agent at THAT SAME file rather than fetching comments itself.
		if !strings.Contains(text, consumer) {
			t.Errorf("%s does not point the agent at %s — the prompt reads a different path than "+
				"the digest step writes, so the filter is bypassed (#1806)", wf, consumer)
		}
		// The empty-file guard is load-bearing: `|| true` swallows a helper that fails to RUN AT ALL
		// (not found / not executable), so this `[ ! -s ]` fallback is the only thing that turns the
		// resulting zero-byte digest into the fail-closed marker rather than a file the agent reads as
		// "nobody commented" (the silent-empty class this PR exists to close, R1).
		if !strings.Contains(text, `[ ! -s "$RUNNER_TEMP/trusted-comments.md" ]`) {
			t.Errorf("%s has no empty-digest guard (`[ ! -s ... ]`) — a helper that fails to run leaves "+
				"a zero-byte digest the agent reads as 'no comments' (#1806)", wf)
		}
		if !strings.Contains(text, "COMMENT-READ-FAILED") {
			t.Errorf("%s does not synthesize COMMENT-READ-FAILED for an empty digest, so an unproduced "+
				"digest is indistinguishable from an empty thread (#1806)", wf)
		}
		// Ordering: the digest MUST be assembled before the agent runs, or the agent reads a missing
		// or stale file. A presence check alone passes even if the step is moved below the agent, so
		// pin the byte order of the step name vs the agent action (the first — and only — one on the
		// digest-consuming path in each of these two workflows).
		stepIdx := strings.Index(text, "Assemble the trusted comment digest")
		agentIdx := strings.Index(text, "anthropics/claude-code-action")
		if stepIdx < 0 || agentIdx < 0 || stepIdx > agentIdx {
			t.Errorf("%s does not assemble the digest BEFORE the agent step (stepIdx=%d agentIdx=%d) — "+
				"the agent would read a missing or stale digest (#1806)", wf, stepIdx, agentIdx)
		}
	}
}

// The digest trusts only the automation logins on `AUTOMATION_LOGINS` (#1806 G2). That set MUST
// cover every identity the delivery loop posts its verdict comments under (`DELIVER-VERDICT` /
// `QA-VERDICT`); if one drops off, the correction phase reads the digest, finds its work list
// filtered out, and silently does nothing. This pins the coupling the helper's "keep in sync"
// comment only asks for — a future action-identity change that isn't mirrored here trips this.
func TestWiring_AutomationAllowlistCoversVerdictPosters(t *testing.T) {
	helper := readWorkflowFile(t, filepath.Join("..", "scripts", "deliver-trusted-comments.sh"))
	for _, login := range []string{"github-actions[bot]", "claude[bot]"} {
		if !strings.Contains(helper, `"`+login+`"`) {
			t.Errorf("scripts/deliver-trusted-comments.sh AUTOMATION_LOGINS does not include %q — the "+
				"delivery loop posts verdict comments under it, so dropping it empties the correction "+
				"phase's work list (#1806 G2)", login)
		}
	}
}

func readWorkflowFile(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(raw)
}

// claude.yml cannot be wired the same way — its jobs run claude-code-action in tag mode and the
// action assembles comment context itself. That is a deliberate, documented limitation, so the
// marker MUST be present: its removal would mean either the limitation was resolved (update this
// test) or the record was silently dropped (the thing this guards against).
func TestWiring_ClaudeYmlDocumentsTheTagModeLimitation(t *testing.T) {
	text := readWorkflow(t, "claude.yml")
	if !strings.Contains(text, "TRUSTED-COMMENTS-LIMITATION (#1806)") {
		t.Errorf("claude.yml is missing the TRUSTED-COMMENTS-LIMITATION (#1806) marker — the " +
			"tag-mode comment-filter gap must stay a marked, deliberate decision, not an unmarked hole")
	}
}
