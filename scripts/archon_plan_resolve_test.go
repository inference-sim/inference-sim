// Package scripts_test drives the repository's CI shell scripts against throwaway git
// repositories. The scripts run inside a GitHub Actions workflow where nothing else can
// exercise them, so their behaviour is pinned here instead.
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

const declaredPlan = "specs/008-x/feature.plan.json"

// scriptPath returns the absolute path of a script in this directory. Go runs tests with
// the package directory as the working directory.
func scriptPath(t *testing.T, name string) string {
	t.Helper()
	abs, err := filepath.Abs(name)
	if err != nil {
		t.Fatalf("resolving %s: %v", name, err)
	}
	return abs
}

func requireGit(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is not on PATH")
	}
}

// gitCmd runs git with an identity and signing config of its own, so neither a missing nor
// a signing-enabled global config changes the result.
func gitCmd(t *testing.T, dir string, args ...string) string {
	t.Helper()
	full := append([]string{
		"-c", "user.name=blis-test",
		"-c", "user.email=blis-test@example.invalid",
		"-c", "commit.gpgsign=false",
	}, args...)
	cmd := exec.Command("git", full...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
	}
	return strings.TrimSpace(string(out))
}

// newRepo creates a repository with a single seed commit. Callers build every further
// commit explicitly, so no returned SHA can go stale.
func newRepo(t *testing.T) string {
	t.Helper()
	requireGit(t)
	dir := t.TempDir()
	gitCmd(t, dir, "init", "-b", "main")
	if err := os.WriteFile(filepath.Join(dir, "seed.txt"), []byte("seed\n"), 0o644); err != nil {
		t.Fatalf("writing seed: %v", err)
	}
	gitCmd(t, dir, "add", "seed.txt")
	gitCmd(t, dir, "commit", "-m", "seed")
	return dir
}

func commitAll(t *testing.T, dir, message string) string {
	t.Helper()
	gitCmd(t, dir, "add", "-A")
	gitCmd(t, dir, "commit", "-m", message)
	return gitCmd(t, dir, "rev-parse", "HEAD")
}

func writeInRepo(t *testing.T, dir, path string, content []byte) {
	t.Helper()
	full := filepath.Join(dir, path)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatalf("mkdir for %s: %v", path, err)
	}
	if err := os.WriteFile(full, content, 0o644); err != nil {
		t.Fatalf("writing %s: %v", path, err)
	}
}

// commitFileAt commits a regular file at path and returns the new commit SHA.
func commitFileAt(t *testing.T, dir, path, content string) string {
	t.Helper()
	writeInRepo(t, dir, path, []byte(content))
	return commitAll(t, dir, "add "+path)
}

func commitRemovalAt(t *testing.T, dir, path string) string {
	t.Helper()
	gitCmd(t, dir, "rm", "-q", path)
	return commitAll(t, dir, "remove "+path)
}

// commitSymlinkAt commits a symlink at path, which git records with mode 120000.
func commitSymlinkAt(t *testing.T, dir, path, target string) string {
	t.Helper()
	full := filepath.Join(dir, path)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatalf("mkdir for %s: %v", path, err)
	}
	if err := os.Symlink(target, full); err != nil {
		t.Fatalf("symlinking %s: %v", path, err)
	}
	return commitAll(t, dir, "symlink "+path)
}

// commitTreeAt makes the declared path itself a directory, by committing a file inside it.
// The .json suffix rule cannot short-circuit the object-type gate this way.
func commitTreeAt(t *testing.T, dir, path string) string {
	t.Helper()
	writeInRepo(t, dir, filepath.Join(path, "inner.txt"), []byte("inner\n"))
	return commitAll(t, dir, "tree at "+path)
}

func writeDecl(t *testing.T, text string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "plan-decl.txt")
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatalf("writing declaration: %v", err)
	}
	return path
}

type resolveResult struct {
	fields   map[string]string
	outPath  string
	exitCode int
	stderr   string
}

func (r resolveResult) status() string { return r.fields["status"] }

func (r resolveResult) outExists(t *testing.T) bool {
	t.Helper()
	_, err := os.Stat(r.outPath)
	switch {
	case err == nil:
		return true
	case errors.Is(err, fs.ErrNotExist):
		return false
	default:
		t.Fatalf("stat %s: %v", r.outPath, err)
		return false
	}
}

func (r resolveResult) outBytes(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile(r.outPath)
	if err != nil {
		t.Fatalf("reading extracted plan: %v", err)
	}
	return string(b)
}

// runResolve runs the resolver with repoDir as the working directory. Without that the
// script would anchor to the BLIS repository itself and the extraction tests would be
// meaningless.
func runResolve(t *testing.T, repoDir, baseRef, headRef, declPath string, extraArgs ...string) resolveResult {
	t.Helper()
	outPath := filepath.Join(t.TempDir(), "plan.json")
	args := []string{baseRef, headRef, declPath, outPath}
	if extraArgs != nil {
		args = extraArgs
	}
	cmd := exec.Command(scriptPath(t, "archon-plan-resolve.sh"), args...)
	cmd.Dir = repoDir
	var stderr strings.Builder
	cmd.Stderr = &stderr
	stdout, err := cmd.Output()

	res := resolveResult{fields: map[string]string{}, outPath: outPath, stderr: stderr.String()}
	var exitErr *exec.ExitError
	switch {
	case err == nil:
	case errors.As(err, &exitErr):
		res.exitCode = exitErr.ExitCode()
	default:
		t.Fatalf("running resolver: %v", err)
	}
	for _, line := range strings.Split(strings.TrimSpace(string(stdout)), "\n") {
		if key, value, ok := strings.Cut(line, "="); ok {
			res.fields[key] = value
		}
	}
	return res
}

func TestResolve_NoDeclaration_ReportsNone(t *testing.T) {
	repo := newRepo(t)
	base := commitFileAt(t, repo, declaredPlan, `{"holes":2}`)

	got := runResolve(t, repo, base, base, writeDecl(t, "Fixes #1631\n\nNo plan here.\n"))

	if got.status() != "none" {
		t.Fatalf("status = %q, want none (fields %v)", got.status(), got.fields)
	}
	if got.outExists(t) {
		t.Error("a plan file was produced for a PR that declared none")
	}
}

// A declaration file that was never written must not look like "no declaration": that
// would drop the plan gate exactly when the collecting step misbehaved.
func TestResolve_MissingDeclarationFile_ReportsError(t *testing.T) {
	repo := newRepo(t)
	base := gitCmd(t, repo, "rev-parse", "HEAD")

	got := runResolve(t, repo, base, base, filepath.Join(t.TempDir(), "absent.txt"))

	if got.status() != "error" {
		t.Fatalf("status = %q, want error", got.status())
	}
	if got.fields["message"] == "" {
		t.Error("no message explaining the failure")
	}
}

func TestResolve_WrongArgCount_ExitsTwo(t *testing.T) {
	repo := newRepo(t)

	got := runResolve(t, repo, "", "", "", "only-one-argument")

	if got.exitCode != 2 {
		t.Fatalf("exit code = %d, want 2", got.exitCode)
	}
	if !strings.Contains(got.stderr, "usage:") {
		t.Errorf("stderr = %q, want a usage message", got.stderr)
	}
}

func TestResolve_UnsafePath_Rejected(t *testing.T) {
	for _, tc := range []struct {
		name string
		path string
	}{
		{"absolute", "/etc/passwd"},
		{"traversal", "../../etc/passwd"},
		{"embedded traversal", "specs/../../x.json"},
		{"leading dash", "-rf.json"},
		{"space", "specs/a b.json"},
		{"shell metacharacter", "specs/a;b.json"},
		{"command substitution", "specs/a$(id).json"},
		{"unsubstituted placeholder", "specs/[NNN-feature]/[feature].plan.json"},
		{"wrong extension", "specs/plan.txt"},
		{"over length", "specs/" + strings.Repeat("a", 300) + ".json"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			repo := newRepo(t)
			base := gitCmd(t, repo, "rev-parse", "HEAD")

			got := runResolve(t, repo, base, base, writeDecl(t, "archon-plan: "+tc.path+"\n"))

			if got.status() != "error" {
				t.Fatalf("status = %q, want error (fields %v)", got.status(), got.fields)
			}
			if got.outExists(t) {
				t.Error("a plan file was produced for a rejected path")
			}
			if got.fields["message"] == "" {
				t.Error("no message explaining the rejection")
			}
			// Whatever is echoed back must be reduced to the allowlist, so a rejected
			// path cannot smuggle markdown or a workflow command into the comment.
			if p := got.fields["plan_path"]; strings.ContainsAny(p, "$;() []`\n") {
				t.Errorf("plan_path = %q still contains unsafe characters", p)
			}
		})
	}
}

func TestResolve_DeclarationWithNoPath_ReportsError(t *testing.T) {
	repo := newRepo(t)
	base := gitCmd(t, repo, "rev-parse", "HEAD")

	got := runResolve(t, repo, base, base, writeDecl(t, "archon-plan:\n"))

	if got.status() != "error" {
		t.Fatalf("status = %q, want error", got.status())
	}
	if got.outExists(t) {
		t.Error("a plan file was produced for a declaration with no path")
	}
}

func TestResolve_PlanOnBase_ResolvesFromBase(t *testing.T) {
	repo := newRepo(t)
	const content = `{"holes":["h1","h2"]}`
	base := commitFileAt(t, repo, declaredPlan, content)
	head := commitFileAt(t, repo, "sim/thing.go", "package sim\n")

	got := runResolve(t, repo, base, head, writeDecl(t, "archon-plan: "+declaredPlan+"\n"))

	if got.status() != "resolved" {
		t.Fatalf("status = %q, want resolved (fields %v)", got.status(), got.fields)
	}
	if got.fields["plan_source"] != "base" {
		t.Errorf("plan_source = %q, want base", got.fields["plan_source"])
	}
	if got.fields["plan_commit"] != base {
		t.Errorf("plan_commit = %q, want %q", got.fields["plan_commit"], base)
	}
	if got.outBytes(t) != content {
		t.Errorf("extracted plan = %q, want %q", got.outBytes(t), content)
	}
}

func TestResolve_PlanOnlyOnHead_ResolvesFromHead(t *testing.T) {
	repo := newRepo(t)
	base := gitCmd(t, repo, "rev-parse", "HEAD")
	const content = `{"holes":[]}`
	head := commitFileAt(t, repo, declaredPlan, content)

	got := runResolve(t, repo, base, head, writeDecl(t, "archon-plan: "+declaredPlan+"\n"))

	if got.status() != "resolved" {
		t.Fatalf("status = %q, want resolved (fields %v)", got.status(), got.fields)
	}
	if got.fields["plan_source"] != "head" {
		t.Errorf("plan_source = %q, want head", got.fields["plan_source"])
	}
	if got.fields["plan_commit"] != head {
		t.Errorf("plan_commit = %q, want %q", got.fields["plan_commit"], head)
	}
	if got.outBytes(t) != content {
		t.Errorf("extracted plan = %q, want %q", got.outBytes(t), content)
	}
}

// A hole PR must not be able to grade itself against a plan it rewrote.
func TestResolve_PlanOnBoth_PrefersBase(t *testing.T) {
	repo := newRepo(t)
	const onBase = `{"holes":["still-open"]}`
	const onHead = `{"holes":[]}`
	base := commitFileAt(t, repo, declaredPlan, onBase)
	head := commitFileAt(t, repo, declaredPlan, onHead)

	got := runResolve(t, repo, base, head, writeDecl(t, "archon-plan: "+declaredPlan+"\n"))

	if got.status() != "resolved" {
		t.Fatalf("status = %q, want resolved", got.status())
	}
	if got.fields["plan_source"] != "base" {
		t.Errorf("plan_source = %q, want base", got.fields["plan_source"])
	}
	if got.outBytes(t) != onBase {
		t.Errorf("extracted the head copy (%q); the base copy must win", got.outBytes(t))
	}
}

func TestResolve_DeclarationInSecondSection_Resolves(t *testing.T) {
	repo := newRepo(t)
	base := commitFileAt(t, repo, declaredPlan, `{"holes":1}`)

	// The PR body comes first and declares nothing; the closing issue's body follows.
	decl := writeDecl(t, "Implements hole 2.\n\nCloses #99\n"+"\n"+"### Hole 2\n\narchon-plan: "+declaredPlan+"\n")
	got := runResolve(t, repo, base, base, decl)

	if got.status() != "resolved" {
		t.Fatalf("status = %q, want resolved (fields %v)", got.status(), got.fields)
	}
	if got.fields["plan_path"] != declaredPlan {
		t.Errorf("plan_path = %q, want %q", got.fields["plan_path"], declaredPlan)
	}
}

// The PR body is collected first, so its declaration outranks one in a third-party issue.
func TestResolve_MultipleDeclarations_FirstWins(t *testing.T) {
	repo := newRepo(t)
	const other = "specs/999-other/other.plan.json"
	commitFileAt(t, repo, other, `{"holes":"other"}`)
	const mine = `{"holes":"mine"}`
	base := commitFileAt(t, repo, declaredPlan, mine)

	decl := writeDecl(t, "archon-plan: "+declaredPlan+"\n\n---\n\narchon-plan: "+other+"\n")
	got := runResolve(t, repo, base, base, decl)

	if got.fields["plan_path"] != declaredPlan {
		t.Fatalf("plan_path = %q, want the first declaration %q", got.fields["plan_path"], declaredPlan)
	}
	if got.outBytes(t) != mine {
		t.Errorf("extracted %q, want the first declaration's content", got.outBytes(t))
	}
}

// GitHub API bodies use CRLF; an unstripped \r fails validation on every real PR.
func TestResolve_CRLFDeclaration_Resolves(t *testing.T) {
	repo := newRepo(t)
	base := commitFileAt(t, repo, declaredPlan, `{"holes":1}`)

	got := runResolve(t, repo, base, base, writeDecl(t, "## Summary\r\n\r\narchon-plan: "+declaredPlan+"\r\n"))

	if got.status() != "resolved" {
		t.Fatalf("status = %q, want resolved (fields %v)", got.status(), got.fields)
	}
}

// The templates wrap the declaration in backticks and bold markers.
func TestResolve_MarkdownWrappedDeclaration_Resolves(t *testing.T) {
	for _, decl := range []string{
		"`archon-plan: " + declaredPlan + "`",
		"- archon-plan: " + declaredPlan,
		"> archon-plan: " + declaredPlan,
		"**archon-plan:** " + declaredPlan,
	} {
		t.Run(decl, func(t *testing.T) {
			repo := newRepo(t)
			base := commitFileAt(t, repo, declaredPlan, `{"holes":1}`)

			got := runResolve(t, repo, base, base, writeDecl(t, decl+"\n"))

			if got.status() != "resolved" {
				t.Fatalf("status = %q, want resolved (fields %v)", got.status(), got.fields)
			}
			if got.fields["plan_path"] != declaredPlan {
				t.Errorf("plan_path = %q, want %q", got.fields["plan_path"], declaredPlan)
			}
		})
	}
}

// Prose mentioning the convention must not be read as a declaration.
func TestResolve_ProseMention_ReportsNone(t *testing.T) {
	repo := newRepo(t)
	base := commitFileAt(t, repo, declaredPlan, `{"holes":1}`)

	got := runResolve(t, repo, base, base, writeDecl(t, "This PR has no archon-plan: line yet.\n"))

	if got.status() != "none" {
		t.Fatalf("status = %q, want none (fields %v)", got.status(), got.fields)
	}
}

func TestResolve_PlanInNeitherCommit_ReportsError(t *testing.T) {
	repo := newRepo(t)
	base := gitCmd(t, repo, "rev-parse", "HEAD")

	got := runResolve(t, repo, base, base, writeDecl(t, "archon-plan: "+declaredPlan+"\n"))

	if got.status() != "error" {
		t.Fatalf("status = %q, want error", got.status())
	}
	if !strings.Contains(got.fields["message"], declaredPlan) {
		t.Errorf("message %q does not name the declared path", got.fields["message"])
	}
	if got.outExists(t) {
		t.Error("a plan file was produced for a path committed nowhere")
	}
}

// The declared path exists in the working tree and is committed nowhere. Any
// implementation that reads the filesystem instead of git resolves it and fails here.
func TestResolve_PlanOnlyInWorkingTree_ReportsError(t *testing.T) {
	repo := newRepo(t)
	base := gitCmd(t, repo, "rev-parse", "HEAD")
	writeInRepo(t, repo, declaredPlan, []byte(`{"holes":"uncommitted"}`))

	got := runResolve(t, repo, base, base, writeDecl(t, "archon-plan: "+declaredPlan+"\n"))

	if got.status() != "error" {
		t.Fatalf("status = %q, want error; the plan is not committed", got.status())
	}
	if got.outExists(t) {
		t.Error("the working-tree copy was read; extraction must go through git")
	}
}

// A broken base copy must not hand grading back to the PR's own head copy.
func TestResolve_EmptyBlobOnBase_ReportsError(t *testing.T) {
	repo := newRepo(t)
	base := commitFileAt(t, repo, declaredPlan, "")
	head := commitFileAt(t, repo, declaredPlan, `{"holes":[]}`)

	got := runResolve(t, repo, base, head, writeDecl(t, "archon-plan: "+declaredPlan+"\n"))

	if got.status() != "error" {
		t.Fatalf("status = %q, want error", got.status())
	}
	if got.outExists(t) {
		t.Error("the head copy was used to replace an unusable base copy")
	}
}

func TestResolve_PathIsDirectory_ReportsError(t *testing.T) {
	repo := newRepo(t)
	base := commitTreeAt(t, repo, declaredPlan)

	got := runResolve(t, repo, base, base, writeDecl(t, "archon-plan: "+declaredPlan+"\n"))

	if got.status() != "error" {
		t.Fatalf("status = %q, want error", got.status())
	}
	if got.outExists(t) {
		t.Error("a plan file was produced for a directory")
	}
}

func TestResolve_PathIsSymlink_ReportsError(t *testing.T) {
	repo := newRepo(t)
	base := commitSymlinkAt(t, repo, declaredPlan, "/etc/passwd")

	got := runResolve(t, repo, base, base, writeDecl(t, "archon-plan: "+declaredPlan+"\n"))

	if got.status() != "error" {
		t.Fatalf("status = %q, want error", got.status())
	}
	if got.outExists(t) {
		t.Error("a symlink was dereferenced")
	}
}

func TestResolve_OversizePlan_ReportsError(t *testing.T) {
	repo := newRepo(t)
	base := commitFileAt(t, repo, declaredPlan, strings.Repeat("x", 1024*1024+1))

	got := runResolve(t, repo, base, base, writeDecl(t, "archon-plan: "+declaredPlan+"\n"))

	if got.status() != "error" {
		t.Fatalf("status = %q, want error", got.status())
	}
	if got.outExists(t) {
		t.Error("an oversized blob was extracted")
	}
}

// git ls-tree prints nothing both for an absent path and for an unknown commit. Without a
// reachability check an unfetched base tip would silently degrade to the head copy.
func TestResolve_UnknownBaseCommit_ReportsError(t *testing.T) {
	repo := newRepo(t)
	const unknown = "0123456789abcdef0123456789abcdef01234567"
	head := commitFileAt(t, repo, declaredPlan, `{"holes":[]}`)

	got := runResolve(t, repo, unknown, head, writeDecl(t, "archon-plan: "+declaredPlan+"\n"))

	if got.status() != "error" {
		t.Fatalf("status = %q, want error", got.status())
	}
	if got.outExists(t) {
		t.Error("an unreachable base commit fell through to the head copy")
	}
}

// A plan removed on the head is still resolved from the base.
func TestResolve_PlanRemovedOnHead_ResolvesFromBase(t *testing.T) {
	repo := newRepo(t)
	const content = `{"holes":["h1"]}`
	base := commitFileAt(t, repo, declaredPlan, content)
	head := commitRemovalAt(t, repo, declaredPlan)

	got := runResolve(t, repo, base, head, writeDecl(t, "archon-plan: "+declaredPlan+"\n"))

	if got.status() != "resolved" {
		t.Fatalf("status = %q, want resolved", got.status())
	}
	if got.outBytes(t) != content {
		t.Errorf("extracted plan = %q, want %q", got.outBytes(t), content)
	}
}
