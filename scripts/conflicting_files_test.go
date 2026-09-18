package scripts_test

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// conflicting-files.sh names the paths that conflict when BASE is merged into HEAD (#1781).
//
// It exists because #1758's acceptance criterion is that a conflicting delivery PR is either
// auto-resolved or "explicitly flagged needs-human NAMING THE CONFLICT" — and on PR #1778 it
// stopped for a human without the conflict ever being named. Neither GitHub's one-word
// `mergeable_state` nor the gate (a pure function of its environment) can supply the paths, so
// this is the one place that computes them, used by both the verify and the correct phase.
//
// Driven against real throwaway repositories rather than mocked: the whole value of the script is
// that it agrees with what git would actually do on a merge, and it deliberately does that on a
// git old enough to lack `merge-tree --write-tree` (2.34 on the runners).

// runConflictingFiles executes the script inside dir and returns its sorted stdout lines and
// exit code. Stderr is returned separately so the diagnostic can be asserted on.
func runConflictingFiles(t *testing.T, dir, base, head string) (lines []string, stderr string, code int) {
	t.Helper()
	cmd := exec.Command(scriptPath(t, "conflicting-files.sh"), base, head)
	cmd.Dir = dir
	cmd.Env = []string{"PATH=" + os.Getenv("PATH"), "HOME=" + dir}
	var out, errb strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &errb

	err := cmd.Run()
	var exitErr *exec.ExitError
	switch {
	case err == nil:
	case errors.As(err, &exitErr):
		code = exitErr.ExitCode()
	default:
		t.Fatalf("running conflicting-files.sh %s %s: %v", base, head, err)
	}
	for _, l := range strings.Split(strings.TrimSpace(out.String()), "\n") {
		if l != "" {
			lines = append(lines, l)
		}
	}
	return lines, errb.String(), code
}

// diverge builds two branches from one seed that both touch `shared` — the shape of the real
// #1778 conflict, where main and the delivery branch both edited CLAUDE.md. Returns the repo.
//
// `base` is left as a branch named `base`, standing in for origin/main; the returned HEAD is on
// branch `feature`.
func divergedRepo(t *testing.T, conflicting bool) string {
	t.Helper()
	dir := newRepo(t)
	commitFileAt(t, dir, "shared.md", "line one\nline two\nline three\n")
	gitCmd(t, dir, "branch", "base")
	gitCmd(t, dir, "checkout", "-q", "-b", "feature")

	// The feature branch edits line one.
	commitFileAt(t, dir, "shared.md", "line one — feature\nline two\nline three\n")

	gitCmd(t, dir, "checkout", "-q", "base")
	if conflicting {
		// base edits THE SAME line — a real content conflict.
		commitFileAt(t, dir, "shared.md", "line one — base\nline two\nline three\n")
	} else {
		// base edits a different file entirely — merges cleanly.
		commitFileAt(t, dir, "elsewhere.md", "unrelated\n")
	}
	gitCmd(t, dir, "checkout", "-q", "feature")
	return dir
}

// A real content conflict is DETERMINED (exit 0) and the conflicting path is printed. This is the
// contract the needs-human comment depends on: the conflict must be nameable.
func TestConflictingFilesNamesARealConflict(t *testing.T) {
	dir := divergedRepo(t, true)

	lines, stderr, code := runConflictingFiles(t, dir, "base", "feature")
	if code != 0 {
		t.Fatalf("exit %d, want 0 (determined); stderr: %s", code, stderr)
	}
	if len(lines) != 1 || lines[0] != "shared.md" {
		t.Errorf("got %v, want exactly [shared.md] — the comment a human reads is built from this list", lines)
	}
}

// A clean merge is DETERMINED with an EMPTY list. Exit 0 with no output must mean "clean", never
// "could not tell" — the caller distinguishes the two by exit code alone.
func TestConflictingFilesReportsACleanMergeAsDeterminedAndEmpty(t *testing.T) {
	dir := divergedRepo(t, false)

	lines, stderr, code := runConflictingFiles(t, dir, "base", "feature")
	if code != 0 {
		t.Fatalf("exit %d, want 0 for a clean merge; stderr: %s", code, stderr)
	}
	if len(lines) != 0 {
		t.Errorf("got %v, want no paths for a cleanly-merging pair", lines)
	}
}

// The caller's working tree, index and HEAD must be untouched. The verify phase asks this question
// in the middle of a job whose later steps run archon against that same checkout, so a script that
// merged in place would corrupt the run it was meant to diagnose.
func TestConflictingFilesLeavesTheCallersTreeUntouched(t *testing.T) {
	dir := divergedRepo(t, true)

	headBefore := gitCmd(t, dir, "rev-parse", "HEAD")
	statusBefore := gitCmd(t, dir, "status", "--porcelain")
	branchBefore := gitCmd(t, dir, "rev-parse", "--abbrev-ref", "HEAD")

	if _, stderr, code := runConflictingFiles(t, dir, "base", "feature"); code != 0 {
		t.Fatalf("exit %d, want 0; stderr: %s", code, stderr)
	}

	if got := gitCmd(t, dir, "rev-parse", "HEAD"); got != headBefore {
		t.Errorf("HEAD moved from %s to %s", headBefore, got)
	}
	if got := gitCmd(t, dir, "rev-parse", "--abbrev-ref", "HEAD"); got != branchBefore {
		t.Errorf("checked-out branch changed from %s to %s", branchBefore, got)
	}
	if got := gitCmd(t, dir, "status", "--porcelain"); got != statusBefore {
		t.Errorf("working tree changed: %q -> %q (a trial merge must not be left in place)", statusBefore, got)
	}
	// And no worktree administrative entry may survive: the runner's workspace persists between
	// deliveries, so a leaked worktree accumulates and eventually blocks `git worktree add`.
	if list := gitCmd(t, dir, "worktree", "list"); strings.Count(list, "\n") != 0 {
		t.Errorf("a temporary worktree was left registered:\n%s", list)
	}
	if entries, err := os.ReadDir(filepath.Join(dir, ".git", "worktrees")); err == nil && len(entries) != 0 {
		t.Errorf(".git/worktrees still holds %d entry/entries after the script ran", len(entries))
	}
}

// A conflicting DELETE/MODIFY pair must be named too. This is why the script reads the index
// (`git ls-files -u`) rather than a name-only diff: the paths a merge cannot resolve are not all
// content conflicts, and a delivery whose branch deleted a file main then edited is exactly the
// drift this loop encounters.
func TestConflictingFilesNamesADeleteModifyConflict(t *testing.T) {
	dir := newRepo(t)
	commitFileAt(t, dir, "doomed.md", "original\n")
	gitCmd(t, dir, "branch", "base")

	gitCmd(t, dir, "checkout", "-q", "-b", "feature")
	commitRemovalAt(t, dir, "doomed.md")

	gitCmd(t, dir, "checkout", "-q", "base")
	commitFileAt(t, dir, "doomed.md", "edited on base\n")
	gitCmd(t, dir, "checkout", "-q", "feature")

	lines, stderr, code := runConflictingFiles(t, dir, "base", "feature")
	if code != 0 {
		t.Fatalf("exit %d, want 0; stderr: %s", code, stderr)
	}
	if len(lines) != 1 || lines[0] != "doomed.md" {
		t.Errorf("got %v, want [doomed.md] — a delete/modify conflict must be named", lines)
	}
}

// A committish the caller never fetched is UNDETERMINED (exit 3) with a diagnostic, NOT a
// silent empty list. Exit 3 is what stops a caller reading "no output" as "the branch is clean"
// — the single most dangerous misreading of this script.
func TestConflictingFilesUnresolvableRefIsUndetermined(t *testing.T) {
	dir := divergedRepo(t, true)

	lines, stderr, code := runConflictingFiles(t, dir, "origin/does-not-exist", "feature")
	if code != 3 {
		t.Fatalf("exit %d, want 3 (undetermined) for an unresolvable base", code)
	}
	if len(lines) != 0 {
		t.Errorf("an undetermined run printed paths %v; it must print none", lines)
	}
	if !strings.Contains(stderr, "does-not-exist") {
		t.Errorf("the diagnostic does not name the ref it could not resolve: %q", stderr)
	}
}

// Missing arguments are a usage error, and they take the same undetermined exit rather than
// looking like a clean merge.
func TestConflictingFilesRequiresBothArguments(t *testing.T) {
	dir := divergedRepo(t, true)

	for _, args := range [][2]string{{"", ""}, {"base", ""}, {"", "feature"}} {
		lines, stderr, code := runConflictingFiles(t, dir, args[0], args[1])
		if code != 3 {
			t.Errorf("args %v: exit %d, want 3", args, code)
		}
		if len(lines) != 0 {
			t.Errorf("args %v: printed %v, want nothing", args, lines)
		}
		if !strings.Contains(stderr, "usage") {
			t.Errorf("args %v: no usage diagnostic: %q", args, stderr)
		}
	}
}

// Unrelated histories: git refuses the merge outright and records no conflicted entry. Reporting
// that as "determined, clean" would tell the caller the opposite of the truth, so it must be
// undetermined.
func TestConflictingFilesUnrelatedHistoriesIsUndetermined(t *testing.T) {
	dir := divergedRepo(t, true)
	// An orphan branch's first commit is a new ROOT, so it shares no commit with feature and git
	// refuses the merge outright rather than reporting conflicts.
	gitCmd(t, dir, "checkout", "-q", "--orphan", "orphan")
	commitFileAt(t, dir, "orphan.md", "unrelated history\n")
	gitCmd(t, dir, "checkout", "-q", "feature")

	lines, _, code := runConflictingFiles(t, dir, "orphan", "feature")
	if code != 3 {
		t.Errorf("exit %d, want 3 — a merge that fails without conflicted paths is undetermined, not clean", code)
	}
	if len(lines) != 0 {
		t.Errorf("printed %v, want nothing", lines)
	}
}
