package scripts_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The CLAUDE.md size guard (ci.yml calls scripts/claude-md-size.sh) must FAIL CLOSED: reject a
// file over the ceiling AND — the case an earlier inline `set -uo pipefail` version got wrong
// (#1818) — a MISSING CLAUDE.md, which a deleting branch must not be able to slip past. YAML-
// parsing guard tests cannot see this; only executing the script can, so this runs it for real.
func TestClaudeMdSizeGuard(t *testing.T) {
	const script = "claude-md-size.sh" // the test's cwd is the scripts/ package directory
	if _, err := os.Stat(script); err != nil {
		t.Fatalf("size guard script missing (%s): %v", script, err)
	}

	run := func(t *testing.T, file, ceiling string) (int, string) {
		t.Helper()
		// Invoked via `bash` because the script uses bash-isms (`[[ =~ ]]`); ci.yml runs it
		// directly, which the committed executable bit covers.
		out, err := exec.Command("bash", script, file, ceiling).CombinedOutput()
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode(), string(out)
		}
		if err != nil {
			t.Fatalf("running %s: %v", script, err)
		}
		return 0, string(out)
	}

	fileOf := func(t *testing.T, n int) string {
		t.Helper()
		p := filepath.Join(t.TempDir(), "CLAUDE.md")
		if err := os.WriteFile(p, []byte(strings.Repeat("x", n)), 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}

	t.Run("under the ceiling passes", func(t *testing.T) {
		if code, out := run(t, fileOf(t, 100), "40000"); code != 0 {
			t.Errorf("expected exit 0 for a small file, got %d\n%s", code, out)
		}
	})

	t.Run("over the ceiling fails and points at the charter", func(t *testing.T) {
		code, out := run(t, fileOf(t, 41000), "40000")
		if code == 0 {
			t.Fatalf("expected non-zero exit for an over-ceiling file, got 0\n%s", out)
		}
		if !strings.Contains(out, "over the 40000-byte ceiling") {
			t.Errorf("over-ceiling failure does not name the breach:\n%s", out)
		}
		if !strings.Contains(out, "not a changelog") {
			t.Errorf("over-ceiling failure does not point at the charter:\n%s", out)
		}
	})

	t.Run("a missing file fails closed", func(t *testing.T) {
		// The #1818 hole: a deleted CLAUDE.md must FAIL the guard, not pass it exit 0.
		code, out := run(t, filepath.Join(t.TempDir(), "gone.md"), "40000")
		if code == 0 {
			t.Fatalf("a missing file must fail the guard (a deleted CLAUDE.md must not pass), got exit 0\n%s", out)
		}
		if !strings.Contains(out, "missing") {
			t.Errorf("missing-file failure does not say the file is missing:\n%s", out)
		}
	})
}
