package scripts_test

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// The delivery phases' failure reporters must be guarded on `always() && !success()`.
//
// The SEMANTICS of that expression are exercised against real GitHub infrastructure by
// .github/workflows/deliver-guards-selftest.yml, which cancels a step via a job timeout and asserts
// the reporter is reached. That workflow cannot run on every pull request: cancelling a job is the
// condition under test, so it necessarily reports a non-green job and would leave a permanently red
// check on every PR.
//
// This test is the half that CAN run everywhere. It asserts the workflows still use the guard the
// self-test exercises, which is what stops a green self-test result from vouching for a file that
// has since been reverted. Cheap, and it runs in the existing `test (scripts)` CI group.
func TestDeliveryReportersUseTheExercisedGuard(t *testing.T) {
	phases := []string{
		"deliver-implement.yml",
		"deliver-verify.yml",
		"deliver-correct.yml",
	}

	// Matched as a step condition rather than anywhere in the file, so the explanatory prose in the
	// surrounding comments — which necessarily quotes the old form to explain why it was replaced —
	// does not trip this.
	oldGuard := regexp.MustCompile(`(?m)^\s*if:.*failure\(\)\s*\|\|\s*cancelled\(\)`)

	for _, phase := range phases {
		t.Run(phase, func(t *testing.T) {
			path := filepath.Join("..", ".github", "workflows", phase)
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("reading %s: %v", path, err)
			}
			body := string(raw)

			if !strings.Contains(body, "if: always() && !success()") {
				t.Errorf("%s has no reporter guarded on `always() && !success()`. "+
					"deliver-guards-selftest.yml exercises that guard; a phase using a different "+
					"one is not covered by it, and a cancelled step could go unreported as on #1685",
					phase)
			}

			if loc := oldGuard.FindString(body); loc != "" {
				t.Errorf("%s reintroduces `failure() || cancelled()` as a step condition (%q). "+
					"It is a strict subset of the exercised guard, so replacing it silently narrows "+
					"which failures get reported",
					phase, strings.TrimSpace(loc))
			}
		})
	}
}
