package scripts_test

import (
	"errors"
	"os"
	"os/exec"
	"strings"
	"testing"
)

// map-merge-state.sh is the pure half of the #1758 mergeability read: it maps GitHub's raw
// REST `mergeable_state` into the gate's MERGE_STATE domain. It is a script, not inline
// workflow YAML, for the same reason deliver-gate.sh is: a bug here can mark a branch that
// conflicts with main as `mergeable`, which the gate then carries to `ready-for-merge` — the
// exact silent-stall #1758 fixes. Inline in the workflow it was proven only by its inverse
// (the gate rejecting raw values); here every GitHub state is pinned to its domain value.

func runMapMergeState(t *testing.T, raw string) (string, int) {
	t.Helper()
	cmd := exec.Command(scriptPath(t, "map-merge-state.sh"), raw)
	cmd.Env = []string{"PATH=" + os.Getenv("PATH")}
	out, err := cmd.Output()
	code := 0
	var exitErr *exec.ExitError
	if err != nil {
		if errors.As(err, &exitErr) {
			code = exitErr.ExitCode()
		} else {
			t.Fatalf("running map-merge-state.sh %q: %v", raw, err)
		}
	}
	return strings.TrimSpace(string(out)), code
}

// TestMapMergeState pins the full GitHub mergeable_state surface to the gate domain. Only a
// true conflict ("dirty") is `conflicting`; the enumerated non-conflict states are `mergeable`;
// an unresolved read ("" / "unknown") is `unknown`. A typo in the case arms (e.g. `dirty)` →
// `dirtry)`) would map a conflicting branch to `unknown` (no longer to `mergeable`, since the
// wildcard now fails closed) — still caught here because it would not be `conflicting`.
func TestMapMergeState(t *testing.T) {
	cases := []struct {
		raw, want string
	}{
		// The one blocking state.
		{"dirty", "conflicting"},
		// Unresolved reads — the poll in the workflow retries these, and if they never
		// resolve it passes "" here; a literal "unknown" must map the same way.
		{"", "unknown"},
		{"unknown", "unknown"},
		// Everything else GitHub can report is mergeable as far as this loop is concerned:
		// a conflict is the only thing that blocks the merge (#1758, conflicts-only scope).
		{"clean", "mergeable"},
		{"behind", "mergeable"},
		{"blocked", "mergeable"},
		{"unstable", "mergeable"},
		{"has_hooks", "mergeable"},
		{"draft", "mergeable"},
	}
	valid := map[string]bool{"mergeable": true, "conflicting": true, "unknown": true}
	for _, tc := range cases {
		t.Run("state-"+tc.raw, func(t *testing.T) {
			got, code := runMapMergeState(t, tc.raw)
			if code != 0 {
				t.Errorf("exit code = %d, want 0", code)
			}
			if got != tc.want {
				t.Errorf("map %q = %q, want %q", tc.raw, got, tc.want)
			}
			if !valid[got] {
				t.Errorf("map %q produced %q, outside the gate's MERGE_STATE domain", tc.raw, got)
			}
		})
	}
}

// A future GitHub state this loop has not seen must FAIL CLOSED to `unknown` (G4), not be
// assumed `mergeable`. This mirrors deliver-gate.sh's rule that an out-of-domain value is not
// trusted: a new GitHub state with different semantics must stop for a human (recoverable on
// the next run) until it is classified in the mapper deliberately, never silently treated as
// safe to merge. It must still be an in-domain value and exit 0, so the gate keeps deciding.
func TestMapMergeStateUnknownFutureValueFailsClosed(t *testing.T) {
	got, code := runMapMergeState(t, "some_future_state")
	if code != 0 {
		t.Errorf("exit code = %d, want 0", code)
	}
	if got != "unknown" {
		t.Errorf("map of an unseen state = %q, want unknown (fail closed)", got)
	}
}
