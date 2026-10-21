package scripts_test

import (
	"errors"
	"os"
	"os/exec"
	"strings"
	"testing"
)

const regenGoldensScript = "regen-main-single-goldens.sh"

// gitWorktrees lists the repository's registered worktrees (porcelain), so a test can tell whether
// the script left one behind.
func gitWorktrees(t *testing.T) string {
	t.Helper()
	out, err := exec.Command("git", "worktree", "list", "--porcelain").Output()
	if err != nil {
		t.Fatalf("git worktree list: %v", err)
	}
	return string(out)
}

// regen-main-single-goldens.sh needs its commit: no published commit both routes natively and
// prices with this branch's kernel, so a default would regenerate nothing. With no argument it
// prints its usage (naming the commit the current goldens come from) and exits 2 before touching
// git; with a commit that does not exist it fails, and its cleanup leaves no worktree registered.
// Neither path writes a golden. The full regeneration builds a binary at another commit and is
// verified by hand (the script's header and #1984); this test pins its failure paths.
func TestRegenMainSingleGoldens_FailsCleanly(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Fatal("git is required")
	}
	before := gitWorktrees(t)
	goldenDir := "../cmd/testdata/golden"
	beforeGoldens, err := os.ReadDir(goldenDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name     string
		args     []string
		wantCode int
		wantErr  string
	}{
		{"no commit", nil, 2, "usage: scripts/regen-main-single-goldens.sh <commit>"},
		{"a commit that does not exist", []string{"0000000000000000000000000000000000000000"}, 128, ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cmd := exec.Command("bash", append([]string{regenGoldensScript}, tt.args...)...)
			out, err := cmd.CombinedOutput()
			var exitErr *exec.ExitError
			if !errors.As(err, &exitErr) {
				t.Fatalf("want a failing exit, got %v:\n%s", err, out)
			}
			if tt.wantCode != 0 && exitErr.ExitCode() != tt.wantCode {
				t.Errorf("exit %d, want %d:\n%s", exitErr.ExitCode(), tt.wantCode, out)
			}
			if tt.wantErr != "" && !strings.Contains(string(out), tt.wantErr) {
				t.Errorf("output lacks %q:\n%s", tt.wantErr, out)
			}
			if strings.Contains(string(out), "wrote ") {
				t.Errorf("a failing run wrote goldens:\n%s", out)
			}
		})
	}
	if after := gitWorktrees(t); after != before {
		t.Errorf("the script left a worktree registered:\n--- before\n%s\n--- after\n%s", before, after)
	}
	afterGoldens, err := os.ReadDir(goldenDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(afterGoldens) != len(beforeGoldens) {
		t.Errorf("the golden directory changed: %d files, was %d", len(afterGoldens), len(beforeGoldens))
	}
}
