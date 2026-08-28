package scripts_test

import (
	"errors"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// reviewMarker is written into the stub's review.md so a test can tell which invocation
// produced the bundle that ended up in the comment.
const reviewMarker = "STUB REVIEW"

// stubArchon writes a fake archon-go. It records each invocation's argv on one line of
// $STUB_LOG, writes a review bundle naming whether --plan was in that argv, and exits with
// $STUB_EXIT. When $STUB_EXIT_ONCE is set only the first invocation fails, which is what
// the retry contract needs.
func stubArchon(t *testing.T, dir string) (binPath, logPath string) {
	t.Helper()
	binPath = filepath.Join(dir, "archon-go-stub")
	logPath = filepath.Join(dir, "stub.log")
	script := `#!/usr/bin/env bash
set -uo pipefail
echo "$*" >> "$STUB_LOG"
invocation=$(wc -l < "$STUB_LOG" | tr -d '[:space:]')

plan_flag=absent
for arg in "$@"; do [[ "$arg" == "--plan" ]] && plan_flag=present; done

exit_code=${STUB_EXIT:-0}
# STUB_EXIT_ONCE fails only the first invocation, which is what the retry contract needs.
if [[ -n "${STUB_EXIT_ONCE:-}" && "$invocation" == "1" ]]; then exit_code=1; fi

# STUB_WRITE_ONCE makes only the first invocation produce a bundle, so a caller that does
# not clear .archon between runs would post the first invocation's output.
if [[ -z "${STUB_WRITE_ONCE:-}" || "$invocation" == "1" ]]; then
  mkdir -p .archon
  printf '# ` + reviewMarker + ` (--plan %s)\n\nBody of the review.\n' "$plan_flag" > .archon/review.md
  if [[ -n "${STUB_PAD:-}" ]]; then
    head -c "$STUB_PAD" /dev/zero | tr '\0' 'x' >> .archon/review.md
  fi
fi
exit "$exit_code"
`
	if err := os.WriteFile(binPath, []byte(script), 0o755); err != nil {
		t.Fatalf("writing stub archon: %v", err)
	}
	return binPath, logPath
}

type reviewResult struct {
	body     string
	summary  string
	stubArgs []string
	exitCode int
	stderr   string
}

func (r reviewResult) invocations() int { return len(r.stubArgs) }

// runReview executes archon-review.sh with repoDir as the working directory.
//
// Both scripts are copied into a scratch directory and the copy is executed, because
// archon-review.sh locates the resolver next to itself. A test that needs a broken
// resolver can then break the copy instead of the repository's own file.
func runReview(t *testing.T, repoDir string, env map[string]string, mutate func(scriptDir string)) reviewResult {
	t.Helper()
	requireGit(t)

	scratch := t.TempDir()
	scriptDir := filepath.Join(scratch, "scripts")
	if err := os.MkdirAll(scriptDir, 0o755); err != nil {
		t.Fatalf("creating script dir: %v", err)
	}
	for _, name := range []string{"archon-review.sh", "archon-plan-resolve.sh"} {
		src, err := os.ReadFile(scriptPath(t, name))
		if err != nil {
			t.Fatalf("reading %s: %v", name, err)
		}
		if err := os.WriteFile(filepath.Join(scriptDir, name), src, 0o755); err != nil {
			t.Fatalf("copying %s: %v", name, err)
		}
	}
	if mutate != nil {
		mutate(scriptDir)
	}

	outputFile := filepath.Join(scratch, "archon-output.md")
	summaryFile := filepath.Join(scratch, "step-summary.md")
	stubBin, stubLog := stubArchon(t, scratch)

	full := map[string]string{
		"ARCHON_BIN":          stubBin,
		"DECL_FILE":           writeDecl(t, ""),
		"OUTPUT_FILE":         outputFile,
		"RUN_URL":             "https://example.invalid/run/1",
		"GITHUB_STEP_SUMMARY": summaryFile,
		"STUB_LOG":            stubLog,
		"RUNNER_TEMP":         scratch,
	}
	for k, v := range env {
		full[k] = v
	}

	cmd := exec.Command(filepath.Join(scriptDir, "archon-review.sh"))
	cmd.Dir = repoDir
	cmd.Env = append(os.Environ(), "PATH="+os.Getenv("PATH"))
	for k, v := range full {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	var stderr strings.Builder
	cmd.Stderr = &stderr
	cmd.Stdout = &strings.Builder{}

	var res reviewResult
	err := cmd.Run()
	res.stderr = stderr.String()
	var exitErr *exec.ExitError
	switch {
	case err == nil:
	case errors.As(err, &exitErr):
		res.exitCode = exitErr.ExitCode()
	default:
		t.Fatalf("running review script: %v", err)
	}

	body, readErr := os.ReadFile(outputFile)
	if readErr != nil && !errors.Is(readErr, fs.ErrNotExist) {
		t.Fatalf("reading the comment body: %v", readErr)
	}
	res.body = string(body)
	summary, readErr := os.ReadFile(summaryFile)
	if readErr != nil && !errors.Is(readErr, fs.ErrNotExist) {
		t.Fatalf("reading the job summary: %v", readErr)
	}
	res.summary = string(summary)
	if b, readErr := os.ReadFile(stubLog); readErr == nil {
		for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
			if line != "" {
				res.stubArgs = append(res.stubArgs, line)
			}
		}
	}
	return res
}

// planRepo builds a repository whose base and head share a merge-base, with the plan
// committed wherever the caller asks.
func planRepo(t *testing.T, planAt string) (dir, base, head string) {
	t.Helper()
	dir = newRepo(t)
	switch planAt {
	case "base":
		base = commitFileAt(t, dir, declaredPlan, `{"holes":["h1"]}`)
		head = commitFileAt(t, dir, "sim/thing.go", "package sim\n")
	case "head":
		base = gitCmd(t, dir, "rev-parse", "HEAD")
		head = commitFileAt(t, dir, declaredPlan, `{"holes":["h1"]}`)
	case "nowhere":
		base = gitCmd(t, dir, "rev-parse", "HEAD")
		head = commitFileAt(t, dir, "sim/thing.go", "package sim\n")
	default:
		t.Fatalf("unknown planAt %q", planAt)
	}
	return dir, base, head
}

func TestReview_NoDeclaration_DeltaOnly(t *testing.T) {
	repo, base, head := planRepo(t, "base")

	got := runReview(t, repo, map[string]string{
		"BASE_SHA":  base,
		"HEAD_SHA":  head,
		"DECL_FILE": writeDecl(t, "Fixes #1631\n\nNo plan declared.\n"),
	}, nil)

	if got.exitCode != 0 {
		t.Fatalf("exit code = %d, want 0 (stderr %q)", got.exitCode, got.stderr)
	}
	if got.invocations() != 1 {
		t.Fatalf("archon invoked %d times, want 1: %v", got.invocations(), got.stubArgs)
	}
	if strings.Contains(got.stubArgs[0], "--plan") {
		t.Errorf("--plan was passed for a PR that declared no plan: %q", got.stubArgs[0])
	}
	if !strings.Contains(got.body, reviewMarker) {
		t.Errorf("comment body does not contain the review: %q", got.body)
	}
	// A PR with no plan must read exactly as it did before this feature existed.
	if strings.Contains(got.body, "[!WARNING]") || strings.Contains(got.body, "Plan check") {
		t.Errorf("comment body carries a plan note or warning:\n%s", got.body)
	}
	if !strings.Contains(got.summary, reviewMarker) {
		t.Errorf("job summary does not contain the review: %q", got.summary)
	}
}

func TestReview_PlanOnBase_PassesPlanOnce(t *testing.T) {
	repo, base, head := planRepo(t, "base")

	got := runReview(t, repo, map[string]string{
		"BASE_SHA":  base,
		"HEAD_SHA":  head,
		"DECL_FILE": writeDecl(t, "archon-plan: "+declaredPlan+"\n"),
	}, nil)

	if got.exitCode != 0 {
		t.Fatalf("exit code = %d, want 0 (stderr %q)", got.exitCode, got.stderr)
	}
	if got.invocations() != 1 {
		t.Fatalf("archon invoked %d times, want 1: %v", got.invocations(), got.stubArgs)
	}
	if n := strings.Count(got.stubArgs[0], "--plan"); n != 1 {
		t.Errorf("--plan appears %d times in %q, want 1", n, got.stubArgs[0])
	}
	if !strings.Contains(got.body, declaredPlan) {
		t.Errorf("comment body does not name the plan:\n%s", got.body)
	}
	if !strings.Contains(got.body, base) {
		t.Errorf("comment body does not name the commit the plan came from:\n%s", got.body)
	}
	if strings.Contains(got.body, "[!WARNING]") {
		t.Errorf("comment body carries a warning for a successful plan check:\n%s", got.body)
	}
}

// A fork PR controls its own head, so a head-sourced plan is not an independent gate and
// the comment has to say so.
func TestReview_PlanFromHead_LabelsUnverified(t *testing.T) {
	repo, base, head := planRepo(t, "head")

	got := runReview(t, repo, map[string]string{
		"BASE_SHA":  base,
		"HEAD_SHA":  head,
		"DECL_FILE": writeDecl(t, "archon-plan: "+declaredPlan+"\n"),
	}, nil)

	if got.invocations() != 1 || !strings.Contains(got.stubArgs[0], "--plan") {
		t.Fatalf("want one plan-aware invocation, got %v", got.stubArgs)
	}
	if !strings.Contains(got.body, "not independently verified") {
		t.Errorf("comment body does not flag the head-sourced plan:\n%s", got.body)
	}
	if !strings.Contains(got.body, head) {
		t.Errorf("comment body does not name the head commit:\n%s", got.body)
	}
}

// A PR that asks to be checked against a plan and silently is not would be
// indistinguishable from a PR with no plan, which removes the gate without telling anyone.
func TestReview_UnusablePlan_WarnsAndReviewsWithoutPlan(t *testing.T) {
	repo, base, head := planRepo(t, "nowhere")

	got := runReview(t, repo, map[string]string{
		"BASE_SHA":  base,
		"HEAD_SHA":  head,
		"DECL_FILE": writeDecl(t, "archon-plan: "+declaredPlan+"\n"),
	}, nil)

	if got.exitCode != 0 {
		t.Fatalf("exit code = %d, want 0 (stderr %q)", got.exitCode, got.stderr)
	}
	if got.invocations() != 1 {
		t.Fatalf("archon invoked %d times, want 1: %v", got.invocations(), got.stubArgs)
	}
	if strings.Contains(got.stubArgs[0], "--plan") {
		t.Errorf("--plan was passed for an unresolvable plan: %q", got.stubArgs[0])
	}
	if !strings.Contains(got.body, "[!WARNING]") {
		t.Errorf("comment body carries no warning:\n%s", got.body)
	}
	if !strings.Contains(got.body, declaredPlan) {
		t.Errorf("warning does not name the declared path:\n%s", got.body)
	}
	if !strings.Contains(got.body, reviewMarker) {
		t.Errorf("comment body lost the delta review:\n%s", got.body)
	}
}

// The three views must survive a plan file archon rejects.
func TestReview_PlanAwareRunFails_RetriesWithoutPlan(t *testing.T) {
	repo, base, head := planRepo(t, "base")

	got := runReview(t, repo, map[string]string{
		"BASE_SHA":       base,
		"HEAD_SHA":       head,
		"DECL_FILE":      writeDecl(t, "archon-plan: "+declaredPlan+"\n"),
		"STUB_EXIT_ONCE": "1",
	}, nil)

	if got.exitCode != 0 {
		t.Fatalf("exit code = %d, want 0 (stderr %q)", got.exitCode, got.stderr)
	}
	if got.invocations() != 2 {
		t.Fatalf("archon invoked %d times, want 2: %v", got.invocations(), got.stubArgs)
	}
	if !strings.Contains(got.stubArgs[0], "--plan") {
		t.Errorf("first invocation should have carried --plan: %q", got.stubArgs[0])
	}
	if strings.Contains(got.stubArgs[1], "--plan") {
		t.Errorf("retry should not carry --plan: %q", got.stubArgs[1])
	}
	if !strings.Contains(got.body, "[!WARNING]") {
		t.Errorf("comment body carries no warning about the failed plan check:\n%s", got.body)
	}
	// The failed invocation's bundle must not be what gets posted.
	if !strings.Contains(got.body, "--plan absent") {
		t.Errorf("comment body is not the retry's output:\n%s", got.body)
	}
	if strings.Contains(got.body, "--plan present") {
		t.Errorf("comment body contains the failed invocation's output:\n%s", got.body)
	}
}

// The plan-aware invocation can leave a partial bundle behind. If the output directory is
// not cleared before the retry, that bundle gets posted as though it were the delta review —
// a plan-aware verdict presented under a "plan check failed" warning.
func TestReview_RetryProducesNoBundle_DoesNotPostTheFailedRun(t *testing.T) {
	repo, base, head := planRepo(t, "base")

	got := runReview(t, repo, map[string]string{
		"BASE_SHA":        base,
		"HEAD_SHA":        head,
		"DECL_FILE":       writeDecl(t, "archon-plan: "+declaredPlan+"\n"),
		"STUB_EXIT_ONCE":  "1",
		"STUB_WRITE_ONCE": "1",
	}, nil)

	if got.exitCode != 0 {
		t.Fatalf("exit code = %d, want 0 (stderr %q)", got.exitCode, got.stderr)
	}
	if got.invocations() != 2 {
		t.Fatalf("archon invoked %d times, want 2: %v", got.invocations(), got.stubArgs)
	}
	if strings.Contains(got.body, "--plan present") {
		t.Errorf("the failed plan-aware bundle was posted:\n%s", got.body)
	}
	if !strings.Contains(got.body, "did not produce") {
		t.Errorf("comment body does not report the missing bundle:\n%s", got.body)
	}
	if !strings.Contains(got.body, "[!WARNING]") {
		t.Errorf("comment body dropped the plan warning:\n%s", got.body)
	}
}

// The workflow shell is `bash -eo pipefail`, so an unguarded resolver failure would abort
// the step and post nothing at all.
func TestReview_ResolverNotExecutable_WarnsAndReviews(t *testing.T) {
	repo, base, head := planRepo(t, "base")

	got := runReview(t, repo, map[string]string{
		"BASE_SHA":  base,
		"HEAD_SHA":  head,
		"DECL_FILE": writeDecl(t, "archon-plan: "+declaredPlan+"\n"),
	}, func(scriptDir string) {
		if err := os.Chmod(filepath.Join(scriptDir, "archon-plan-resolve.sh"), 0o000); err != nil {
			t.Fatalf("breaking the resolver copy: %v", err)
		}
	})

	if got.exitCode != 0 {
		t.Fatalf("exit code = %d, want 0 (stderr %q)", got.exitCode, got.stderr)
	}
	if got.invocations() != 1 || strings.Contains(got.stubArgs[0], "--plan") {
		t.Fatalf("want one delta-only invocation, got %v", got.stubArgs)
	}
	if !strings.Contains(got.body, "[!WARNING]") {
		t.Errorf("comment body carries no warning about plan resolution:\n%s", got.body)
	}
	if !strings.Contains(got.body, reviewMarker) {
		t.Errorf("comment body lost the delta review:\n%s", got.body)
	}
}

func TestReview_UnreachableMergeBase_ReportsError(t *testing.T) {
	repo := newRepo(t)
	base := gitCmd(t, repo, "rev-parse", "HEAD")
	// An orphan root shares no history, so git merge-base genuinely fails.
	gitCmd(t, repo, "checkout", "-q", "--orphan", "unrelated")
	gitCmd(t, repo, "rm", "-q", "-rf", ".")
	writeInRepo(t, repo, "other.txt", []byte("other\n"))
	head := commitAll(t, repo, "unrelated root")

	got := runReview(t, repo, map[string]string{"BASE_SHA": base, "HEAD_SHA": head}, nil)

	if got.exitCode != 0 {
		t.Fatalf("exit code = %d, want 0 (stderr %q)", got.exitCode, got.stderr)
	}
	if got.invocations() != 0 {
		t.Errorf("archon was invoked despite an unreachable merge-base: %v", got.stubArgs)
	}
	if !strings.Contains(got.body, "Archon Error") || !strings.Contains(got.body, "merge-base") {
		t.Errorf("comment body does not report the merge-base failure:\n%s", got.body)
	}
	if !strings.Contains(got.summary, "merge-base") {
		t.Errorf("job summary does not report the merge-base failure:\n%s", got.summary)
	}
}

func TestReview_ArchonFails_ReportsError(t *testing.T) {
	repo, base, head := planRepo(t, "nowhere")

	got := runReview(t, repo, map[string]string{
		"BASE_SHA":  base,
		"HEAD_SHA":  head,
		"STUB_EXIT": "1",
	}, nil)

	if got.exitCode != 0 {
		t.Fatalf("exit code = %d, want 0 (stderr %q)", got.exitCode, got.stderr)
	}
	if !strings.Contains(got.body, "Archon Error") {
		t.Errorf("comment body does not report the failure:\n%s", got.body)
	}
}

// A plan warning must survive a failing review, or the reason the gate did not run is lost.
func TestReview_ArchonFailsWithPlanWarning_KeepsWarning(t *testing.T) {
	repo, base, head := planRepo(t, "nowhere")

	got := runReview(t, repo, map[string]string{
		"BASE_SHA":  base,
		"HEAD_SHA":  head,
		"DECL_FILE": writeDecl(t, "archon-plan: "+declaredPlan+"\n"),
		"STUB_EXIT": "1",
	}, nil)

	if got.exitCode != 0 {
		t.Fatalf("exit code = %d, want 0 (stderr %q)", got.exitCode, got.stderr)
	}
	if !strings.Contains(got.body, "Archon Error") {
		t.Errorf("comment body does not report the archon failure:\n%s", got.body)
	}
	if !strings.Contains(got.body, "[!WARNING]") {
		t.Errorf("comment body dropped the plan warning:\n%s", got.body)
	}
}

// The runner is self-hosted and the output path is fixed, so a body left by an earlier run
// must not survive into a run that aborts before composing its own.
func TestReview_AbortedRun_LeavesNoStaleBody(t *testing.T) {
	repo := newRepo(t)
	base := gitCmd(t, repo, "rev-parse", "HEAD")
	scratch := t.TempDir()
	outputFile := filepath.Join(scratch, "archon-output.md")
	if err := os.WriteFile(outputFile, []byte("## Stale review from an earlier run\n"), 0o644); err != nil {
		t.Fatalf("seeding a stale body: %v", err)
	}

	// An unset RUN_URL aborts the script after the truncation but before any body exists.
	got := runReview(t, repo, map[string]string{
		"BASE_SHA":    base,
		"HEAD_SHA":    base,
		"OUTPUT_FILE": outputFile,
		"RUN_URL":     "",
	}, nil)

	if got.exitCode == 0 {
		t.Fatalf("exit code = 0, want non-zero for a missing RUN_URL")
	}
	body, err := os.ReadFile(outputFile)
	if err != nil {
		t.Fatalf("reading the output file: %v", err)
	}
	if len(body) != 0 {
		t.Errorf("stale body survived an aborted run: %q", body)
	}
}

func TestReview_MissingRequiredEnv_ExitsTwo(t *testing.T) {
	repo, _, head := planRepo(t, "base")

	got := runReview(t, repo, map[string]string{"BASE_SHA": "", "HEAD_SHA": head}, nil)

	if got.exitCode != 2 {
		t.Fatalf("exit code = %d, want 2 (stderr %q)", got.exitCode, got.stderr)
	}
	if !strings.Contains(got.stderr, "BASE_SHA") {
		t.Errorf("stderr does not name the missing variable: %q", got.stderr)
	}
}

// GitHub caps a comment at 65536 characters, so an oversize review is truncated for the
// comment while the job summary keeps the whole thing.
func TestReview_OversizeReview_TruncatesWithRunLink(t *testing.T) {
	repo, base, head := planRepo(t, "nowhere")
	const runURL = "https://example.invalid/run/42"

	got := runReview(t, repo, map[string]string{
		"BASE_SHA": base,
		"HEAD_SHA": head,
		"STUB_PAD": "70000",
		"RUN_URL":  runURL,
	}, nil)

	if got.exitCode != 0 {
		t.Fatalf("exit code = %d, want 0 (stderr %q)", got.exitCode, got.stderr)
	}
	if len(got.body) > 65536 {
		t.Errorf("comment body is %d characters, over GitHub's 65536 limit", len(got.body))
	}
	if !strings.Contains(got.body, "Output truncated") || !strings.Contains(got.body, runURL) {
		t.Errorf("truncated body does not link the workflow run:\n%s", got.body[max(0, len(got.body)-300):])
	}
	if len(got.summary) < 70000 {
		t.Errorf("job summary is %d characters; it should keep the untruncated review", len(got.summary))
	}
}
