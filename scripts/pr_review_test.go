package scripts_test

// Guard tests for the external-PR reviewer (.github/workflows/pr-review.yml, #1879).
//
// GitHub Actions cannot be executed here, so these assert the workflow's
// security-load-bearing STRUCTURE (the properties a reviewer would check) and the
// BEHAVIOUR of the helper scripts it runs. End-to-end proof is the pre-merge
// adversarial dry-run on real infra, per the PR's validation section.

import (
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// Minimal NetworkPolicy shape for parse-and-assert egress checks (refactor-proof:
// asserts the except list on the actual :443 rule, not substring presence).
type npParsed struct {
	Kind     string `yaml:"kind"`
	Metadata struct {
		Name string `yaml:"name"`
	} `yaml:"metadata"`
	Spec struct {
		PodSelector struct {
			MatchLabels map[string]string `yaml:"matchLabels"`
		} `yaml:"podSelector"`
		Egress []struct {
			To []struct {
				PodSelector *struct {
					MatchLabels map[string]string `yaml:"matchLabels"`
				} `yaml:"podSelector"`
				IPBlock *struct {
					CIDR   string   `yaml:"cidr"`
					Except []string `yaml:"except"`
				} `yaml:"ipBlock"`
			} `yaml:"to"`
			Ports []struct {
				Protocol string `yaml:"protocol"`
				Port     int    `yaml:"port"`
			} `yaml:"ports"`
		} `yaml:"egress"`
	} `yaml:"spec"`
}

// findNetworkPolicy decodes the multi-doc runner manifest and returns the
// NetworkPolicy with the given name.
func findNetworkPolicy(t *testing.T, name string) npParsed {
	t.Helper()
	data := readFileOrFail(t, filepath.Join("..", "k8s", "pr-review-runner.yaml"))
	dec := yaml.NewDecoder(strings.NewReader(data))
	for {
		var np npParsed
		err := dec.Decode(&np)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("decoding pr-review-runner.yaml: %v", err)
		}
		if np.Kind == "NetworkPolicy" && np.Metadata.Name == name {
			return np
		}
	}
	t.Fatalf("NetworkPolicy %q not found in pr-review-runner.yaml", name)
	return npParsed{}
}

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

// TestPrReviewBlisSkipsOIDC pins the #1883 fix: the blis action must be given the
// job's read-only github_token so claude-code-action takes its OVERRIDE_GITHUB_TOKEN
// path and SKIPS the OIDC exchange. Without it the action calls getIDToken(), which
// needs `id-token: write` — a privilege we withhold from the untrusted review job —
// and aborts before the model runs. The review job must therefore ALSO not grant
// id-token: write (the whole point is to stay least-privilege, not to add it).
func TestPrReviewBlisSkipsOIDC(t *testing.T) {
	// Parse the blis step precisely and assert ITS `with.github_token` is the
	// read-only job token. A file-global Contains would also be satisfied by a
	// github_token: on any OTHER step even if the blis step regressed (jgchn).
	type step struct {
		Uses string            `yaml:"uses"`
		With map[string]string `yaml:"with"`
	}
	var wf struct {
		Jobs map[string]struct {
			Permissions yaml.Node `yaml:"permissions"`
			Steps       []step    `yaml:"steps"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal([]byte(prReviewWorkflow(t)), &wf); err != nil {
		t.Fatalf("parsing pr-review.yml: %v", err)
	}
	review, ok := wf.Jobs["review"]
	if !ok {
		t.Fatal("no `review` job in pr-review.yml")
	}
	// Assert on EVERY claude-code-action step, not just the first: a second such
	// step added later without github_token must not slip through on the first
	// match (code-reviewer note).
	var found bool
	for i := range review.Steps {
		if !strings.Contains(review.Steps[i].Uses, "claude-code-action") {
			continue
		}
		found = true
		if got := review.Steps[i].With["github_token"]; !strings.Contains(got, "secrets.GITHUB_TOKEN") {
			t.Errorf("the claude-code-action (blis) step must pass github_token: ${{ secrets.GITHUB_TOKEN }} so it skips the OIDC exchange (#1883); got %q", got)
		}
	}
	if !found {
		t.Fatal("no claude-code-action (blis) step in the review job")
	}
	// The review job must NOT grant id-token: write — neither as an explicit key
	// nor via a `write-all` scalar, which implicitly includes it and decodes to an
	// empty map (jgchn). Check the node shape before decoding.
	assertNoIDTokenGrant(t, review.Permissions)
}

// assertNoIDTokenGrant fails if the job permissions grant the id-token: write
// scope that enables OIDC minting — either an explicit `id-token: write` key or
// the `write-all` scalar shorthand. A `read-all` scalar or `id-token: read`
// grants no minting scope (GitHub's getIDToken needs write), so it is allowed.
func assertNoIDTokenGrant(t *testing.T, perms yaml.Node) {
	t.Helper()
	if perms.Kind == yaml.ScalarNode {
		if perms.Value == "write-all" {
			t.Error("the review job uses `permissions: write-all`, which implicitly grants id-token: write; it must be an explicit least-privilege block")
		}
		return
	}
	var m map[string]string
	if err := perms.Decode(&m); err != nil {
		t.Fatalf("decoding review permissions: %v", err)
	}
	if m["id-token"] == "write" {
		t.Errorf("the review job must NOT grant id-token: write (the minting scope); the github_token override removes the need for OIDC")
	}
}

// TestPrReviewWorktreeIsIdempotent pins the #1888 fix: the worktree step must
// clear a leaked registration (prune) and any leftover directory (rm) BEFORE
// `worktree add`, so a restarted ephemeral runner that kept the _work emptyDir
// cannot exit-128 on a "missing but already registered" worktree.
func TestPrReviewWorktreeIsIdempotent(t *testing.T) {
	wf := prReviewWorkflow(t)
	rm := strings.Index(wf, `rm -rf "$WORKTREE"`)
	prune := strings.Index(wf, "git worktree prune")
	add := strings.Index(wf, "worktree add --detach")
	if rm < 0 || prune < 0 || add < 0 {
		t.Fatal(`worktree step must run rm -rf "$WORKTREE" and git worktree prune before worktree add --detach (#1888)`)
	}
	if rm > add || prune > add {
		t.Error("the worktree cleanup (rm + prune) must come BEFORE `worktree add` to be idempotent (#1888)")
	}
}

// TestPrReviewBlisWriteIsBareWithHook pins the #1888 fix: blis must grant a BARE
// Write tool — a path-qualified Write(...) allow-rule refuses silently in the
// action (see scripts/pr_review/restrict-write.sh), so the verdict never gets
// written — with the restrict-write PreToolUse hook as the actual path boundary.
func TestPrReviewBlisWriteIsBareWithHook(t *testing.T) {
	wf := prReviewWorkflow(t)
	// blis is the only --allowedTools in the workflow; assert that so this test
	// can't silently validate some other step's allow-list (code-reviewer note).
	if n := strings.Count(wf, "--allowedTools"); n != 1 {
		t.Fatalf("expected exactly one --allowedTools (blis), found %d", n)
	}
	allow := regexp.MustCompile(`--allowedTools\s+"([^"]*)"`).FindStringSubmatch(wf)
	if allow == nil {
		t.Fatal("no --allowedTools found for blis")
	}
	var hasBareWrite bool
	for _, tool := range strings.Split(allow[1], ",") {
		tool = strings.TrimSpace(tool)
		if tool == "Write" {
			hasBareWrite = true
		} else if strings.HasPrefix(tool, "Write(") {
			t.Errorf("blis must grant a BARE Write, not a path-qualified %q (it refuses silently in the action, #1888)", tool)
		}
	}
	if !hasBareWrite {
		t.Error("blis must grant a bare Write tool so the verdict file can be written (#1888)")
	}
	if !strings.Contains(wf, "scripts/pr_review/restrict-write.sh") {
		t.Error("blis must confine the bare Write via the restrict-write PreToolUse hook")
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
	// the external reviewer must NOT reference it — it uses the litellm-proxy + a dummy.
	if strings.Contains(wf, "secrets.LITELLM_API_KEY") {
		t.Error("pr-review.yml must NOT reference secrets.LITELLM_API_KEY; the key lives in the litellm-proxy pod")
	}
	if !strings.Contains(wf, "Assert no LiteLLM key in the session") {
		t.Error("pr-review.yml must assert at runtime that no real LiteLLM key is in the review job env")
	}
	// The LLM clients must point at the litellm-proxy Service.
	if !strings.Contains(wf, "LLM_PROXY_URL") {
		t.Error("pr-review.yml must route LLM calls through the litellm-proxy (LLM_PROXY_URL)")
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

// Pins the fixes from the second namasl review round.
func TestPrReviewSecondRoundHardening(t *testing.T) {
	wf := prReviewWorkflow(t)
	checks := []struct{ needle, why string }{
		{"git merge-base", "the qa diff must be against the merge base, not the base tip (diverged-branch correctness)"},
		{`-z "$cur"`, "the freshness check must fail CLOSED (empty live-head lookup => stale)"},
		{`gh pr comment "$PR_NUMBER" --repo "$REPO"`, "the post job has no checkout, so gh pr comment must pass --repo"},
		{"mkdir -p out", "the post fallback must create out/ (download continues-on-error)"},
	}
	for _, c := range checks {
		if !strings.Contains(wf, c.needle) {
			t.Errorf("pr-review.yml missing %q — %s", c.needle, c.why)
		}
	}
	// The untrusted runner must be egress-locked (k8s manifest). The proxy now
	// lives in its OWN pod (so the per-pod NetworkPolicy can confine the runner
	// tighter than the proxy), replacing the old loopback-bind invariant. Parse
	// the policy and assert the actual rules — NOT substring presence — so a
	// refactor that moved an except CIDR onto the proxy's rule or into a comment
	// would fail here.
	np := findNetworkPolicy(t, "pr-review-runner-egress")
	if got := np.Spec.PodSelector.MatchLabels["app"]; got != "github-runner-pr-review" {
		t.Errorf("runner egress policy must select app=github-runner-pr-review, got %q", got)
	}
	// Find the :443 egress rule and assert its ipBlock excludes metadata + the VPC.
	var except []string
	var found443, proxyByLabel bool
	for _, rule := range np.Spec.Egress {
		has443, has4000 := false, false
		for _, p := range rule.Ports {
			if p.Port == 443 {
				has443 = true
			}
			if p.Port == 4000 {
				has4000 = true
			}
		}
		if has443 {
			found443 = true
			for _, peer := range rule.To {
				if peer.IPBlock != nil {
					except = peer.IPBlock.Except
				}
			}
		}
		if has4000 {
			for _, peer := range rule.To {
				if peer.PodSelector != nil && peer.PodSelector.MatchLabels["app"] == "litellm-proxy" {
					proxyByLabel = true
				}
			}
		}
	}
	if !found443 {
		t.Fatal("runner egress policy has no :443 rule")
	}
	for _, cidr := range []string{"169.254.0.0/16", "9.0.0.0/8"} {
		if !slicesContains(except, cidr) {
			t.Errorf("runner :443 egress must except %s (metadata + IBM VPC must be unreachable); except=%v", cidr, except)
		}
	}
	if !proxyByLabel {
		t.Error("runner egress must allow :4000 to the litellm-proxy pod by label (app=litellm-proxy)")
	}
}

func slicesContains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

func TestAssembleCommentCapsOversizeBody(t *testing.T) {
	requirePython3(t)
	dir := t.TempDir()
	big := filepath.Join(dir, "big.md")
	out := filepath.Join(dir, "out.md")
	_ = os.WriteFile(big, []byte(strings.Repeat("x", 80000)), 0o644)
	_, code := runPy(t, "", "pr_review/assemble_comment.py",
		"--pr", "1", "--archon", big, "--out", out)
	if code != 0 {
		t.Fatalf("assemble_comment exited %d", code)
	}
	body := readFileOrFail(t, out)
	if len(body) > 65536 {
		t.Errorf("combined comment is %d chars, over GitHub's 65536 limit", len(body))
	}
	if !strings.Contains(body, "truncated") {
		t.Error("an over-limit comment must carry a truncation notice")
	}
}

// When the head moved (stale), the comment must publish ONLY a re-run notice —
// never the reviewer sections under a SHA they may not reflect (review finding).
func TestAssembleCommentStaleWithholdsSections(t *testing.T) {
	requirePython3(t)
	dir := t.TempDir()
	ar := filepath.Join(dir, "ar.md")
	out := filepath.Join(dir, "out.md")
	_ = os.WriteFile(ar, []byte("ARCHON-SECTION-CONTENT"), 0o644)
	_, code := runPy(t, "", "pr_review/assemble_comment.py",
		"--pr", "9", "--head-sha", "deadbeef", "--stale", "--archon", ar, "--out", out)
	if code != 0 {
		t.Fatalf("assemble_comment exited %d", code)
	}
	body := readFileOrFail(t, out)
	if strings.Contains(body, "ARCHON-SECTION-CONTENT") {
		t.Error("stale comment must NOT publish reviewer section content")
	}
	for _, h := range []string{"Architecture (archon)", "Correctness (blis", "Cross-vendor (qa"} {
		if strings.Contains(body, h) {
			t.Errorf("stale comment must omit section header %q", h)
		}
	}
	if !strings.Contains(body, "Re-run") && !strings.Contains(body, "re-run") {
		t.Error("stale comment must tell the reader to re-run")
	}
}

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
