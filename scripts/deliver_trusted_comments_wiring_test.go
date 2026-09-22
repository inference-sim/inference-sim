package scripts_test

// Pins that the trusted-comment filter (#1806) is actually WIRED into the AI flows, not just
// present as a standalone helper. deliver_trusted_comments_test.go proves the helper selects the
// right comments; this file proves the workflows CALL it and feed the agent its output instead of
// reading comments raw.
//
// Like the sibling workflow tests, these assert on file text: for a workflow the declared text IS
// the behaviour — GitHub reads nothing else, and the file only ever runs inside Actions, where no
// other test can reach it. The regression each guards is concrete and has already happened once
// (the helper shipped unwired): a future edit that deletes the digest step, or re-points a prompt
// at `gh pr view --comments`, would silently reopen the prompt-injection surface this closed.

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
	}
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
