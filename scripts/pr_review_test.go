package scripts_test

// Guard tests for the external-PR reviewer (.github/workflows/pr-review.yml, #1879).
//
// GitHub Actions cannot be executed here, so these assert the workflow's
// security-load-bearing STRUCTURE (the properties a reviewer would check) and the
// BEHAVIOUR of the helper scripts it runs. End-to-end proof is the pre-merge
// adversarial dry-run on real infra, per the PR's validation section.

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func prReviewWorkflowPath() string {
	return filepath.Join("..", ".github", "workflows", "pr-review.yml")
}

func prReviewWorkflow(t *testing.T) string {
	t.Helper()
	return readFileOrFail(t, prReviewWorkflowPath())
}

// runPy runs a python3 script from the scripts/ dir (test cwd). The shared
// runPython helper forces cwd to scripts/qa-review, which these scripts are not in.
func runPy(t *testing.T, stdin string, args ...string) (string, int) {
	t.Helper()
	cmd := exec.Command("python3", args...)
	cmd.Env = os.Environ()
	if stdin != "" {
		cmd.Stdin = strings.NewReader(stdin)
	}
	var out strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &out
	err := cmd.Run()
	code := 0
	var exitErr *exec.ExitError
	switch {
	case err == nil:
	case errors.As(err, &exitErr):
		code = exitErr.ExitCode()
	default:
		t.Fatalf("running python3 %v: %v", args, err)
	}
	return out.String(), code
}

// runBash runs a repo script via bash from the scripts/ dir (test cwd).
func runBash(t *testing.T, stdin string, script string, args ...string) (string, int) {
	t.Helper()
	cmd := exec.Command("bash", append([]string{script}, args...)...)
	if stdin != "" {
		cmd.Stdin = strings.NewReader(stdin)
	}
	var out strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &out
	err := cmd.Run()
	code := 0
	var exitErr *exec.ExitError
	switch {
	case err == nil:
	case errors.As(err, &exitErr):
		code = exitErr.ExitCode()
	default:
		t.Fatalf("running bash %s %v: %v", script, args, err)
	}
	return out.String(), code
}

// --- Workflow structure: the trigger is precise and write-access gated (BC-1) ---

func TestPrReviewTriggerIsPreciseAndWriteGated(t *testing.T) {
	wf := prReviewWorkflow(t)
	// The precise token matcher must be present — a loose contains() alone would
	// also fire on /pr-review-foo.
	if !strings.Contains(wf, `/(^|\s)\/pr-review(\s|$)/`) {
		t.Error("pr-review.yml must match /pr-review as a precise token (regex (^|\\s)/pr-review(\\s|$)), not only a loose contains()")
	}
	// The commenter's write access must be checked (same boundary as #1813).
	if !strings.Contains(wf, "getCollaboratorPermissionLevel") {
		t.Error("pr-review.yml must gate on the commenter's collaborator permission")
	}
}

// TestPrReviewPreciseMatchSemantics documents and verifies the token rule the
// workflow's regex encodes: /pr-review (optionally with args) triggers; the
// sibling commands and a longer /pr-review-* do NOT.
func TestPrReviewPreciseMatchSemantics(t *testing.T) {
	re := regexp.MustCompile(`(^|\s)/pr-review(\s|$)`)
	cases := []struct {
		body string
		want bool
	}{
		{"/pr-review", true},
		{"please /pr-review this", true},
		{"/pr-review --post", true},
		{"line1\n/pr-review\nline2", true},
		{"/blis-pr-review", false},
		{"/archon-pr-review", false},
		{"/pr-review-experimental", false},
		{"nope", false},
	}
	for _, c := range cases {
		if got := re.MatchString(c.body); got != c.want {
			t.Errorf("precise match of %q = %v, want %v", c.body, got, c.want)
		}
	}
}

// --- Workflow structure: read-never-execute + containment (BC-2..BC-6) ---

type wfParsed struct {
	Jobs map[string]struct {
		RunsOn      yaml.Node `yaml:"runs-on"`
		Permissions yaml.Node `yaml:"permissions"`
	} `yaml:"jobs"`
}

func parsePrReview(t *testing.T) wfParsed {
	t.Helper()
	var p wfParsed
	if err := yaml.Unmarshal([]byte(prReviewWorkflow(t)), &p); err != nil {
		t.Fatalf("parsing pr-review.yml: %v", err)
	}
	return p
}

func TestPrReviewNoPullRequestWriteOnTheUntrustedJob(t *testing.T) {
	p := parsePrReview(t)
	review, ok := p.Jobs["review"]
	if !ok {
		t.Fatal("no `review` job in pr-review.yml")
	}
	perms := review.Permissions
	var permMap map[string]string
	_ = perms.Decode(&permMap)
	if _, hasWrite := permMap["pull-requests"]; hasWrite && permMap["pull-requests"] == "write" {
		t.Error("the `review` job (reads untrusted code) must NOT have pull-requests: write; only the `post` job may")
	}
	post, ok := p.Jobs["post"]
	if !ok {
		t.Fatal("no `post` job in pr-review.yml")
	}
	var postPerms map[string]string
	_ = post.Permissions.Decode(&postPerms)
	if postPerms["pull-requests"] != "write" {
		t.Error("the `post` job must have pull-requests: write to comment")
	}
}

func TestPrReviewReviewJobRunsOnDedicatedUntrustedPool(t *testing.T) {
	p := parsePrReview(t)
	review := p.Jobs["review"]
	raw := strings.Join(flattenScalars(review.RunsOn), ",")
	if !strings.Contains(raw, "pr-review-untrusted") {
		t.Errorf("the `review` job must run on the dedicated `pr-review-untrusted` pool, got %q", raw)
	}
	if raw == "self-hosted" {
		t.Error("the `review` job must NOT share the delivery loop's `self-hosted` pool")
	}
}

func flattenScalars(n yaml.Node) []string {
	var out []string
	if n.Kind == yaml.ScalarNode {
		out = append(out, n.Value)
	}
	for _, c := range n.Content {
		out = append(out, flattenScalars(*c)...)
	}
	return out
}

func TestPrReviewNeverExecutesPRCode(t *testing.T) {
	wf := prReviewWorkflow(t)
	checks := []struct {
		needle string
		why    string
	}{
		{"git fetch origin \"pull/${PR}/head\"", "archon must fetch the head into the object store"},
		{"core.hooksPath=/dev/null", "the PR-head worktree must be created with git hooks disabled"},
		{"scripts/pr_review/scrub_symlinks.sh", "escaping symlinks must be scrubbed before any reviewer reads the tree"},
		{"persist-credentials: false", "the trusted checkout must not write a token into .git/config"},
		{"--no-exec", "qa-review's answerer must run with --no-exec (drops its code-executing tool)"},
	}
	for _, c := range checks {
		if !strings.Contains(wf, c.needle) {
			t.Errorf("pr-review.yml missing %q — %s", c.needle, c.why)
		}
	}
	// There must be NO checkout of the PR head ref in the review job: a reviewer
	// reads the fetched worktree, never a `ref:`-checked-out PR tree that could
	// run setup actions.
	if strings.Contains(wf, "ref: ${{ needs.gate.outputs.head_sha }}") {
		t.Error("pr-review.yml must not actions/checkout the PR head ref; use the object-store fetch + detached worktree")
	}
}

func TestPrReviewBlisHasNoBash(t *testing.T) {
	wf := prReviewWorkflow(t)
	if !strings.Contains(wf, "--disallowedTools") || !regexp.MustCompile(`--disallowedTools\s+"[^"]*Bash`).MatchString(wf) {
		t.Error("blis must disallow Bash (and friends) in claude_args")
	}
	// A bare `Bash` must not appear in the allow list.
	allow := regexp.MustCompile(`--allowedTools\s+"([^"]*)"`).FindStringSubmatch(wf)
	if allow == nil {
		t.Fatal("no --allowedTools found for blis")
	}
	for _, tool := range strings.Split(allow[1], ",") {
		if strings.TrimSpace(tool) == "Bash" {
			t.Error("blis must not be granted a bare Bash tool")
		}
	}
	if !strings.Contains(wf, "scripts/pr_review/restrict-write.sh") {
		t.Error("blis must confine writes via the restrict-write PreToolUse hook")
	}
}

// TestPrReviewActionsArePinnedToSHA enforces #1879's requirement that every
// third-party action — especially the one that creates the tool-restricted blis
// session — is pinned to a full 40-char commit SHA, not a mutable tag.
func TestPrReviewActionsArePinnedToSHA(t *testing.T) {
	wf := prReviewWorkflow(t)
	uses := regexp.MustCompile(`uses:\s+(\S+)`)
	pinned := regexp.MustCompile(`^[^@]+@[0-9a-f]{40}(\s|$)`)
	found := false
	for _, m := range uses.FindAllStringSubmatch(wf, -1) {
		ref := m[1]
		if strings.HasPrefix(ref, "./") {
			continue // local action, no SHA
		}
		found = true
		if !pinned.MatchString(ref + " ") {
			t.Errorf("action %q is not pinned to a 40-char SHA (mutable ref)", ref)
		}
	}
	if !found {
		t.Error("no third-party actions found to check — did the parse break?")
	}
}

func TestPrReviewKeyNeverInSession(t *testing.T) {
	wf := prReviewWorkflow(t)
	// The real LiteLLM key is a GitHub Actions secret used by the delivery loop;
	// the external reviewer must NOT reference it — it uses the sidecar + a dummy.
	if strings.Contains(wf, "secrets.LITELLM_API_KEY") {
		t.Error("pr-review.yml must NOT reference secrets.LITELLM_API_KEY; the key lives in the sidecar")
	}
	if !strings.Contains(wf, "Assert no LiteLLM key in the session") {
		t.Error("pr-review.yml must assert at runtime that no real LiteLLM key is in the review job env")
	}
	// The LLM clients must point at the in-pod proxy.
	if !strings.Contains(wf, "LLM_PROXY_URL") {
		t.Error("pr-review.yml must route LLM calls through the sidecar proxy (LLM_PROXY_URL)")
	}
}

func TestPrReviewVerdictIsAdvisoryOnePost(t *testing.T) {
	wf := prReviewWorkflow(t)
	// Must NOT use --edit-last: this workflow posts as github-actions[bot], the
	// same identity archon.yml and deliver-verify.yml use, so --edit-last would
	// overwrite an existing archon or qa-review comment on the PR. Post a fresh
	// comment instead.
	if strings.Contains(wf, "--edit-last") {
		t.Error("the post job must NOT use --edit-last; it would clobber archon/deliver-verify comments posted under the same bot identity")
	}
	if !strings.Contains(wf, "scripts/pr_review/scrub_secrets.py") {
		t.Error("the combined comment must be secret-scrubbed before posting")
	}
}

// --- Helper-script behaviour ---

func TestScrubSecretsRedactsButKeepsProse(t *testing.T) {
	requirePython3(t)
	in := "normal prose line\nsk-abcdef1234567890 and Bearer AbCdEf123456xyz789\nx-api-key: supersecretvalue123\nkeep this\n"
	out, code := runPy(t, in, "pr_review/scrub_secrets.py")
	if code != 0 {
		t.Fatalf("scrub_secrets exited %d", code)
	}
	for _, leaked := range []string{"sk-abcdef1234567890", "AbCdEf123456xyz789", "supersecretvalue123"} {
		if strings.Contains(out, leaked) {
			t.Errorf("scrub_secrets leaked %q", leaked)
		}
	}
	for _, kept := range []string{"normal prose line", "keep this"} {
		if !strings.Contains(out, kept) {
			t.Errorf("scrub_secrets dropped benign text %q", kept)
		}
	}
}

func TestAssembleCommentReportsMissingReviewer(t *testing.T) {
	requirePython3(t)
	dir := t.TempDir()
	ar := filepath.Join(dir, "ar.md")
	bl := filepath.Join(dir, "bl.md")
	out := filepath.Join(dir, "out.md")
	_ = os.WriteFile(ar, []byte("archon ok"), 0o644)
	_ = os.WriteFile(bl, []byte("READY"), 0o644)
	_, code := runPy(t, "", "pr_review/assemble_comment.py",
		"--pr", "7", "--archon", ar, "--qa", filepath.Join(dir, "missing.md"), "--blis", bl, "--out", out)
	if code != 0 {
		t.Fatalf("assemble_comment exited %d", code)
	}
	body := readFileOrFail(t, out)
	if !strings.Contains(body, "archon ok") || !strings.Contains(body, "READY") {
		t.Error("assemble_comment dropped a present reviewer's output")
	}
	if !strings.Contains(body, "did not complete") {
		t.Error("assemble_comment must report a missing reviewer, not silently drop it")
	}
}

func TestScrubSymlinksRemovesEscapersKeepsInternal(t *testing.T) {
	dir := t.TempDir()
	tree := filepath.Join(dir, "tree")
	_ = os.MkdirAll(filepath.Join(tree, "sub"), 0o755)
	_ = os.WriteFile(filepath.Join(tree, "real.txt"), []byte("hi"), 0o644)
	_ = os.Symlink("real.txt", filepath.Join(tree, "good.lnk"))
	_ = os.Symlink("/etc/passwd", filepath.Join(tree, "bad.lnk"))
	_ = os.Symlink("../../outside", filepath.Join(tree, "sub", "escape.lnk"))
	out, code := runBash(t, "", "pr_review/scrub_symlinks.sh", tree)
	if code != 0 {
		t.Fatalf("scrub_symlinks exited %d: %s", code, out)
	}
	if _, err := os.Lstat(filepath.Join(tree, "good.lnk")); err != nil {
		t.Error("scrub_symlinks removed a legitimate internal symlink")
	}
	for _, gone := range []string{"bad.lnk", filepath.Join("sub", "escape.lnk")} {
		if _, err := os.Lstat(filepath.Join(tree, gone)); err == nil {
			t.Errorf("scrub_symlinks left an escaping symlink: %s", gone)
		}
	}
}

func TestRestrictWriteAllowsOnlyVerdictFile(t *testing.T) {
	verdict := filepath.Join(t.TempDir(), "verdict.md")
	t.Setenv("PR_REVIEW_VERDICT_FILE", verdict)
	// Allowed path -> exit 0.
	if _, code := runBashEnv(t, `{"tool_input":{"file_path":"`+verdict+`"}}`, "pr_review/restrict-write.sh"); code != 0 {
		t.Errorf("restrict-write blocked the allowed verdict path (exit %d)", code)
	}
	// Any other path -> exit 2 (deny).
	if _, code := runBashEnv(t, `{"tool_input":{"file_path":"/etc/evil"}}`, "pr_review/restrict-write.sh"); code != 2 {
		t.Errorf("restrict-write must deny a non-verdict path with exit 2, got %d", code)
	}
}

// runBashEnv is runBash but inheriting the test process env (for t.Setenv).
func runBashEnv(t *testing.T, stdin, script string, args ...string) (string, int) {
	t.Helper()
	cmd := exec.Command("bash", append([]string{script}, args...)...)
	cmd.Env = os.Environ()
	cmd.Stdin = strings.NewReader(stdin)
	var out strings.Builder
	cmd.Stdout = &out
	cmd.Stderr = &out
	err := cmd.Run()
	code := 0
	var exitErr *exec.ExitError
	switch {
	case err == nil:
	case errors.As(err, &exitErr):
		code = exitErr.ExitCode()
	default:
		t.Fatalf("running bash %s: %v", script, err)
	}
	return out.String(), code
}
