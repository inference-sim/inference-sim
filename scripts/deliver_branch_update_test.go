package scripts_test

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// deliver-update-branch.sh and deliver-conflict-check.sh are the two halves of #1781's
// deterministic conflict handling: one brings a delivery branch up to date with main BEFORE the
// correction agent runs, the other establishes AFTER it whether the conflict actually got resolved.
//
// They are scripts rather than inline workflow YAML for the reason this repository already applies
// to deliver-gate.sh and map-merge-state.sh — and #1778 is the evidence. There the branch update
// was a prompt instruction to the agent, the round completed `success` with no commit and no
// comment, and the next verify stopped for a human citing a missing review marker. An instruction
// an agent may quietly not follow is not a mechanism, and a mechanism with no test is not much
// better, so both are exercised here against real repositories with a real remote.

// deliveryRepos builds a bare "remote" plus a working clone holding `main` and a delivery branch
// that has already diverged from it. Returns the working clone and the branch name.
//
// A real bare remote rather than a stub: pushing is part of what deliver-update-branch.sh
// guarantees ("merged" means merged AND pushed), and a test that could not observe the remote tip
// would not be testing that.
func deliveryRepos(t *testing.T) (work, branch string) {
	t.Helper()
	requireGit(t)

	root := t.TempDir()
	remote := filepath.Join(root, "remote.git")
	gitCmd(t, root, "init", "-q", "--bare", remote)

	work = filepath.Join(root, "work")
	gitCmd(t, root, "clone", "-q", remote, work)

	writeInRepo(t, work, "CLAUDE.md", []byte("line one\nline two\n"))
	commitAll(t, work, "seed")
	// The initial branch name depends on the git version's default; normalise it.
	gitCmd(t, work, "branch", "-M", "main")
	gitCmd(t, work, "push", "-q", "origin", "main")

	branch = "deliver/issue-1"
	gitCmd(t, work, "checkout", "-q", "-b", branch)
	writeInRepo(t, work, "CLAUDE.md", []byte("line one — the PR's change\nline two\n"))
	commitAll(t, work, "the PR's change")
	gitCmd(t, work, "push", "-q", "-u", "origin", branch)
	return work, branch
}

// advanceMain commits on main, pushes it, and returns to the delivery branch — the drift that puts
// a delivery PR behind (or in conflict with) main while it is open.
func advanceMain(t *testing.T, work, branch, path, content string) {
	t.Helper()
	gitCmd(t, work, "checkout", "-q", "main")
	writeInRepo(t, work, path, []byte(content))
	commitAll(t, work, "main advances: "+path)
	gitCmd(t, work, "push", "-q", "origin", "main")
	gitCmd(t, work, "checkout", "-q", branch)
}

// runStateScript runs a `state=`/`files=` script in dir and parses its two output lines. It is
// STRICT about stdout: a line that is not part of the state/files payload fails the test, because
// the caller tees this stdout straight into $GITHUB_OUTPUT (see runStateScriptRaw).
func runStateScript(t *testing.T, dir, script string, args ...string) (state, files string, code int) {
	t.Helper()
	stdout, state, files, code := runStateScriptRaw(t, dir, script, args...)
	if leaked := nonPayloadStdoutLines(stdout); len(leaked) > 0 {
		t.Errorf("%s %v leaked %d non-payload line(s) on stdout, which $GITHUB_OUTPUT rejects (#1799): %q",
			script, args, len(leaked), leaked)
	}
	return state, files, code
}

// runStateScriptRaw is runStateScript without the strict-stdout assertion, returning the raw stdout
// so a test can make its own claim about it.
func runStateScriptRaw(t *testing.T, dir, script string, args ...string) (stdout, state, files string, code int) {
	t.Helper()
	cmd := exec.Command(scriptPath(t, script), args...)
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
		t.Fatalf("running %s %v: %v", script, args, err)
	}

	// The scripts emit `state=` as a single line and `files` in GITHUB_OUTPUT's multiline heredoc
	// form (`files<<DELIM` … `DELIM`, #1781 G4), so a conflicting path may contain a space or
	// comma without a lossy join. Parse both: single-line `files=` is still accepted for
	// robustness, and the heredoc block is collected verbatim between its delimiters.
	lines := strings.Split(out.String(), "\n")
	for i := 0; i < len(lines); i++ {
		line := lines[i]
		switch {
		case strings.HasPrefix(line, "state="):
			state = strings.TrimPrefix(line, "state=")
		case strings.HasPrefix(line, "files="):
			files = strings.TrimPrefix(line, "files=")
		case strings.HasPrefix(line, "files<<"):
			delim := strings.TrimPrefix(line, "files<<")
			var collected []string
			for j := i + 1; j < len(lines); j++ {
				if lines[j] == delim {
					i = j
					break
				}
				collected = append(collected, lines[j])
			}
			files = strings.Join(collected, "\n")
		}
	}
	t.Logf("%s %v -> state=%q files=%q (exit %d)\n%s", script, args, state, files, code, errb.String())
	return out.String(), state, files, code
}

// nonPayloadStdoutLines returns every stdout line that is NOT part of the `state=`/`files<<DELIM …
// DELIM` payload — i.e. every line GitHub's $GITHUB_OUTPUT parser would reject with
// `Invalid format '<line>'`. Lines INSIDE a heredoc block are payload whatever they contain (a
// conflicting path is arbitrary text), so the walk tracks the block rather than pattern-matching
// each line independently. This is deliberately stricter than the parser above: the pre-#1799
// lenient `switch` skipped unrecognized lines, so the leaked `Already up to date.` was invisible to
// the tests and fatal in production — the harness being more forgiving than its consumer is exactly
// what let the bug ship.
func nonPayloadStdoutLines(stdout string) []string {
	lines := strings.Split(strings.TrimSuffix(stdout, "\n"), "\n")
	if len(lines) == 1 && lines[0] == "" {
		return nil // no output at all is not a grammar violation; the state assertions cover that
	}
	var leaked []string
	for i := 0; i < len(lines); i++ {
		switch {
		case strings.HasPrefix(lines[i], "state="), strings.HasPrefix(lines[i], "files="):
			// A single-line `key=value`: valid GITHUB_OUTPUT grammar.
		case strings.HasPrefix(lines[i], "files<<"):
			delim := strings.TrimPrefix(lines[i], "files<<")
			closed := false
			for j := i + 1; j < len(lines); j++ {
				if lines[j] == delim {
					i, closed = j, true
					break
				}
			}
			if !closed {
				// An unterminated heredoc swallows everything after it, so report the opener.
				leaked = append(leaked, lines[i])
				return leaked
			}
		default:
			leaked = append(leaked, lines[i])
		}
	}
	return leaked
}

// #1799 — on EVERY path the script's stdout must hold ONLY `state=<value>` and (when applicable) the
// `files<<DELIM`/`DELIM` heredoc, because deliver-correct.yml tees it straight into $GITHUB_OUTPUT,
// whose grammar admits nothing else. The bug this pins: `git merge` writes its own success text to
// stdout — `Already up to date.` when the branch is current, `Updating …`/`Fast-forward`/a diffstat
// on clean drift — and that reached $GITHUB_OUTPUT as `Invalid format 'Already up to date.'`, which
// failed the step and ended the correction round at needs-human WITHOUT reading a single finding.
//
// Asserted on the raw stdout rather than through the parsing helper, and over all four states,
// because "the state parsed correctly" is exactly the weaker claim that passed while the leak
// shipped. The `current` and `merged` cases are the two the merge actually printed on.
func TestUpdateBranchStdoutCarriesOnlyGitHubOutputGrammar(t *testing.T) {
	cases := []struct {
		name  string
		setup func(t *testing.T) (work, branch string)
		want  string
	}{
		{
			// The reproducing case: nothing to merge, so `git merge` prints "Already up to date."
			name: "current",
			setup: func(t *testing.T) (string, string) {
				return deliveryRepos(t)
			},
			want: "current",
		},
		{
			// The other leaking case: a clean fast-forward-style merge prints "Updating …",
			// "Fast-forward" and a diffstat — several stray lines rather than one.
			name: "merged",
			setup: func(t *testing.T) (string, string) {
				work, branch := deliveryRepos(t)
				advanceMain(t, work, branch, "unrelated.md", "main moved on\n")
				return work, branch
			},
			want: "merged",
		},
		{
			name: "conflicting",
			setup: func(t *testing.T) (string, string) {
				work, branch := deliveryRepos(t)
				advanceMain(t, work, branch, "CLAUDE.md", "line one — main's change\nline two\n")
				return work, branch
			},
			want: "conflicting",
		},
		{
			name: "unknown",
			setup: func(t *testing.T) (string, string) {
				work, branch := deliveryRepos(t)
				gitCmd(t, work, "remote", "set-url", "origin", filepath.Join(work, "..", "does-not-exist.git"))
				return work, branch
			},
			want: "unknown",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			work, branch := tc.setup(t)
			stdout, state, _, code := runStateScriptRaw(t, work, "deliver-update-branch.sh", branch)
			if code != 0 {
				t.Fatalf("exit %d, want 0", code)
			}
			// Non-vacuity: a test that asserted grammar on the wrong path would pass trivially.
			if state != tc.want {
				t.Fatalf("state = %q, want %q — this subtest is not exercising the intended path", state, tc.want)
			}
			if leaked := nonPayloadStdoutLines(stdout); len(leaked) > 0 {
				t.Errorf("stdout carries %d line(s) $GITHUB_OUTPUT would reject with `Invalid format`: %q\nfull stdout:\n%s",
					len(leaked), leaked, stdout)
			}
		})
	}
}

func remoteTip(t *testing.T, work, branch string) string {
	t.Helper()
	out := gitCmd(t, work, "ls-remote", "origin", "refs/heads/"+branch)
	if out == "" {
		t.Fatalf("origin has no %s", branch)
	}
	return strings.Fields(out)[0]
}

// A branch already up to date reports `current` and pushes nothing. This is the no-op case that
// must be distinguishable from a failure — a round that reports "unknown" here would send a
// perfectly healthy delivery to the agent to redo work that was not needed.
func TestUpdateBranchAlreadyCurrent(t *testing.T) {
	work, branch := deliveryRepos(t)
	tipBefore := remoteTip(t, work, branch)

	state, files, code := runStateScript(t, work, "deliver-update-branch.sh", branch)
	if code != 0 {
		t.Fatalf("exit %d, want 0", code)
	}
	if state != "current" {
		t.Errorf("state = %q, want current", state)
	}
	if files != "" {
		t.Errorf("files = %q, want empty", files)
	}
	if got := remoteTip(t, work, branch); got != tipBefore {
		t.Errorf("the remote tip moved from %s to %s; nothing should be pushed when already current", tipBefore, got)
	}
}

// ORDINARY DRIFT — main advanced without touching what the PR touched — is merged AND PUSHED by
// this step, with no agent involved. This is the case that silently no-op'd on #1778, and it is
// the majority of rounds, so it must not depend on anything an agent chooses to do.
func TestUpdateBranchMergesAndPushesCleanDrift(t *testing.T) {
	work, branch := deliveryRepos(t)
	advanceMain(t, work, branch, "unrelated.md", "main moved on\n")
	tipBefore := remoteTip(t, work, branch)

	state, files, code := runStateScript(t, work, "deliver-update-branch.sh", branch)
	if code != 0 {
		t.Fatalf("exit %d, want 0", code)
	}
	if state != "merged" {
		t.Fatalf("state = %q, want merged", state)
	}
	if files != "" {
		t.Errorf("files = %q, want empty for a clean merge", files)
	}

	// "merged" must mean merged AND pushed: an unpushed merge is invisible to the PR and to the
	// re-verification that follows, so it is not progress.
	tipAfter := remoteTip(t, work, branch)
	if tipAfter == tipBefore {
		t.Error("the remote tip did not move, so the merge was not pushed — the exact success-with-no-progress #1781 removes")
	}
	if local := gitCmd(t, work, "rev-parse", "HEAD"); local != tipAfter {
		t.Errorf("local HEAD %s != pushed tip %s", local, tipAfter)
	}
	// And main's content is now present on the branch.
	if _, err := os.Stat(filepath.Join(work, "unrelated.md")); err != nil {
		t.Errorf("main's file is not on the branch after the merge: %v", err)
	}
}

// A REAL content conflict is reported with the conflicting paths named, and the working tree is
// left CLEAN. Both halves matter: the paths are what the needs-human comment is built from, and a
// half-merged tree would break the agent's `gh pr checkout` and turn a resolvable conflict into a
// confusing checkout failure.
func TestUpdateBranchReportsAConflictAndLeavesTheTreeClean(t *testing.T) {
	work, branch := deliveryRepos(t)
	// main edits the SAME line the PR edited — the shape of the real #1778 CLAUDE.md conflict.
	advanceMain(t, work, branch, "CLAUDE.md", "line one — main's change\nline two\n")
	tipBefore := remoteTip(t, work, branch)

	state, files, code := runStateScript(t, work, "deliver-update-branch.sh", branch)
	if code != 0 {
		t.Fatalf("exit %d, want 0 — a conflict is a state, not a phase failure", code)
	}
	if state != "conflicting" {
		t.Fatalf("state = %q, want conflicting", state)
	}
	if files != "CLAUDE.md" {
		t.Errorf("files = %q, want CLAUDE.md — the comment a human reads is built from this", files)
	}
	if status := gitCmd(t, work, "status", "--porcelain"); status != "" {
		t.Errorf("the working tree is not clean after the abort:\n%s", status)
	}
	if head := gitCmd(t, work, "rev-parse", "HEAD"); head != tipBefore {
		t.Errorf("HEAD moved to %s despite the conflict; the aborted merge must leave no commit", head)
	}
	if got := remoteTip(t, work, branch); got != tipBefore {
		t.Errorf("the remote tip moved to %s; a conflicting merge must push nothing", got)
	}
}

// G6 — a DIRTY tracked worktree is refused, not merged or reset. The push-race path runs
// `git reset --hard`, which would DESTROY uncommitted work, and a merge against a dirty index can
// fail confusingly. The delivery checkout is always clean, but the script must be safe to invoke
// anywhere, so it reports `unknown` and leaves the working tree exactly as it found it — proven
// here by dirtying a tracked file over a drift that WOULD otherwise merge cleanly.
func TestUpdateBranchRefusesADirtyWorktreeAndPreservesTheEdit(t *testing.T) {
	work, branch := deliveryRepos(t)
	advanceMain(t, work, branch, "unrelated.md", "main moved on\n") // a merge here would be clean
	tipBefore := remoteTip(t, work, branch)

	dirty := "line one — the PR's change\nUNCOMMITTED WORK IN PROGRESS\n"
	writeInRepo(t, work, "CLAUDE.md", []byte(dirty))

	state, _, code := runStateScript(t, work, "deliver-update-branch.sh", branch)
	if code != 0 {
		t.Fatalf("exit %d, want 0 — a dirty tree is a state, not a phase failure", code)
	}
	if state != "unknown" {
		t.Fatalf("state = %q, want unknown — a dirty tree must not be merged or reset", state)
	}
	got, err := os.ReadFile(filepath.Join(work, "CLAUDE.md"))
	if err != nil {
		t.Fatalf("reading CLAUDE.md: %v", err)
	}
	if string(got) != dirty {
		t.Errorf("the uncommitted edit was destroyed (reset --hard on a dirty tree):\n got %q\nwant %q", string(got), dirty)
	}
	if now := remoteTip(t, work, branch); now != tipBefore {
		t.Errorf("the remote tip moved to %s; a refused update must push nothing", now)
	}
}

// #1781 G4 — a conflicting path that contains a SPACE and a COMMA is named verbatim, never split.
// git forbids only NUL in a path, so the newline-delimited transport must carry spaces and commas
// through untouched (an earlier comma-join + whitespace-split mangled them).
func TestUpdateBranchNamesAConflictingPathWithSpaceAndComma(t *testing.T) {
	requireGit(t)
	root := t.TempDir()
	remote := filepath.Join(root, "remote.git")
	gitCmd(t, root, "init", "-q", "--bare", remote)
	work := filepath.Join(root, "work")
	gitCmd(t, root, "clone", "-q", remote, work)

	const path = "docs/a file, notes.md" // a space AND a comma, both legal in a git path
	writeInRepo(t, work, path, []byte("base line\n"))
	commitAll(t, work, "seed")
	gitCmd(t, work, "branch", "-M", "main")
	gitCmd(t, work, "push", "-q", "origin", "main")

	branch := "deliver/issue-1"
	gitCmd(t, work, "checkout", "-q", "-b", branch)
	writeInRepo(t, work, path, []byte("the PR's change\n"))
	commitAll(t, work, "branch change")
	gitCmd(t, work, "push", "-q", "-u", "origin", branch)

	gitCmd(t, work, "checkout", "-q", "main")
	writeInRepo(t, work, path, []byte("main's change\n"))
	commitAll(t, work, "main change")
	gitCmd(t, work, "push", "-q", "origin", "main")
	gitCmd(t, work, "checkout", "-q", branch)

	state, files, code := runStateScript(t, work, "deliver-update-branch.sh", branch)
	if code != 0 || state != "conflicting" {
		t.Fatalf("exit %d state %q, want 0 conflicting", code, state)
	}
	if files != path {
		t.Errorf("files = %q, want %q reproduced verbatim — a path with a space or comma must survive the transport (#1781 G4)", files, path)
	}
}

// An unreachable origin is `unknown` — never `current` (which would claim the branch is up to
// date) and never `conflicting` (which would stop the delivery for a human over an outage).
func TestUpdateBranchUnreachableOriginIsUnknown(t *testing.T) {
	work, branch := deliveryRepos(t)
	gitCmd(t, work, "remote", "set-url", "origin", filepath.Join(work, "..", "does-not-exist.git"))

	state, files, code := runStateScript(t, work, "deliver-update-branch.sh", branch)
	if code != 0 {
		t.Fatalf("exit %d, want 0", code)
	}
	if state != "unknown" {
		t.Errorf("state = %q, want unknown for an unreachable origin", state)
	}
	if files != "" {
		t.Errorf("files = %q, want empty", files)
	}
}

func TestUpdateBranchRequiresABranchArgument(t *testing.T) {
	work, _ := deliveryRepos(t)
	if _, _, code := runStateScript(t, work, "deliver-update-branch.sh"); code != 2 {
		t.Errorf("exit %d with no branch argument, want 2 (a wiring error, not a state)", code)
	}
}

// The post-agent check must see the REMOTE tip, because the correction agent pushes from inside
// claude-code-action and the workflow's own checkout does not have that commit. A check that read
// the local HEAD would report a resolved conflict as still conflicting and stop a healthy delivery.
func TestConflictCheckReadsTheRemoteTipNotTheLocalCheckout(t *testing.T) {
	work, branch := deliveryRepos(t)
	advanceMain(t, work, branch, "CLAUDE.md", "line one — main's change\nline two\n")

	// Resolve and push, then rewind the LOCAL branch so the local checkout still conflicts. This is
	// the divergence a real correction round produces.
	gitCmd(t, work, "fetch", "-q", "--no-tags", "origin", "main")
	resolved := "line one — the PR's change, reconciled with main\nline two\n"
	if err := runMerge(work, "origin/main"); err == nil {
		t.Fatal("expected the merge to conflict, so this test is not exercising a resolution")
	}
	writeInRepo(t, work, "CLAUDE.md", []byte(resolved))
	gitCmd(t, work, "add", "CLAUDE.md")
	gitCmd(t, work, "commit", "-q", "-m", "resolve the conflict")
	gitCmd(t, work, "push", "-q", "origin", "HEAD:refs/heads/"+branch)
	pushed := gitCmd(t, work, "rev-parse", "HEAD")

	// Rewind local to before the resolution; the remote keeps it.
	gitCmd(t, work, "reset", "-q", "--hard", "HEAD~1")
	if local := gitCmd(t, work, "rev-parse", "HEAD"); local == pushed {
		t.Fatal("the local checkout was not rewound, so this test would pass trivially")
	}

	state, files, code := runStateScript(t, work, "deliver-conflict-check.sh", branch)
	if code != 0 {
		t.Fatalf("exit %d, want 0", code)
	}
	if state != "clean" {
		t.Errorf("state = %q, want clean — the pushed resolution is on the remote, which is what this must read (files=%q)", state, files)
	}
}

// A conflict that was NOT resolved is reported as conflicting, with the paths named. This is the
// signal that makes #1758(b) hold without depending on the agent: the caller posts the naming
// comment from `files` and withholds the hand-back.
func TestConflictCheckNamesAnUnresolvedConflict(t *testing.T) {
	work, branch := deliveryRepos(t)
	advanceMain(t, work, branch, "CLAUDE.md", "line one — main's change\nline two\n")

	state, files, code := runStateScript(t, work, "deliver-conflict-check.sh", branch)
	if code != 0 {
		t.Fatalf("exit %d, want 0", code)
	}
	if state != "conflicting" {
		t.Fatalf("state = %q, want conflicting", state)
	}
	if files != "CLAUDE.md" {
		t.Errorf("files = %q, want CLAUDE.md; #1758(b) requires the stop to NAME the conflict", files)
	}
}

// Two conflicting paths are emitted NEWLINE-delimited, one per line (#1781 G4). Newline is the one
// delimiter that survives a git path containing a space or comma, so the transport is one path per
// line rather than a comma- or space-joined single line that could not be split back losslessly.
func TestConflictCheckEmitsMultiplePathsOnePerLine(t *testing.T) {
	work, branch := deliveryRepos(t)

	// Both sides touch two files.
	writeInRepo(t, work, "second.md", []byte("pr side\n"))
	commitAll(t, work, "the PR touches a second file")
	gitCmd(t, work, "push", "-q", "origin", "HEAD:refs/heads/"+branch)

	gitCmd(t, work, "checkout", "-q", "main")
	writeInRepo(t, work, "CLAUDE.md", []byte("line one — main\nline two\n"))
	writeInRepo(t, work, "second.md", []byte("main side\n"))
	commitAll(t, work, "main touches both")
	gitCmd(t, work, "push", "-q", "origin", "main")
	gitCmd(t, work, "checkout", "-q", branch)

	state, files, code := runStateScript(t, work, "deliver-conflict-check.sh", branch)
	if code != 0 {
		t.Fatalf("exit %d, want 0", code)
	}
	if state != "conflicting" {
		t.Fatalf("state = %q, want conflicting", state)
	}
	// Sorted, one path per line — the exact newline-delimited value conflicting-files.sh produces.
	if want := "CLAUDE.md\nsecond.md"; files != want {
		t.Errorf("files = %q, want %q (newline-delimited, sorted)", files, want)
	}
}

// A branch that is not on the remote is `unknown`, NOT `conflicting`. The check may only escalate
// on positive evidence of a conflict; treating "could not tell" as a conflict would stop healthy
// deliveries at needs-human over a transient fetch failure.
func TestConflictCheckMissingBranchIsUnknownNotConflicting(t *testing.T) {
	work, _ := deliveryRepos(t)

	state, files, code := runStateScript(t, work, "deliver-conflict-check.sh", "deliver/issue-999")
	if code != 0 {
		t.Fatalf("exit %d, want 0", code)
	}
	if state != "unknown" {
		t.Errorf("state = %q, want unknown", state)
	}
	if state == "conflicting" {
		t.Error("an undeterminable check must never escalate to conflicting; that stops a healthy delivery")
	}
	if files != "" {
		t.Errorf("files = %q, want empty", files)
	}
}

func TestConflictCheckRequiresABranchArgument(t *testing.T) {
	work, _ := deliveryRepos(t)
	if _, _, code := runStateScript(t, work, "deliver-conflict-check.sh"); code != 2 {
		t.Errorf("exit %d with no branch argument, want 2", code)
	}
}

// runMerge attempts a merge and returns its error, so a test can assert a conflict occurred without
// gitCmd's t.Fatalf on non-zero exit.
func runMerge(dir, ref string) error {
	cmd := exec.Command("git",
		"-c", "user.name=blis-test",
		"-c", "user.email=blis-test@example.invalid",
		"-c", "commit.gpgsign=false",
		"merge", "--no-edit", ref)
	cmd.Dir = dir
	return cmd.Run()
}
