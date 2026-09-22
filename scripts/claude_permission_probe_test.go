package scripts_test

// Pins the error handling of the collaborator probe in .github/workflows/claude.yml's
// check-permissions job (#1707).
//
// The defect these tests exist to prevent: the probe's `catch` mapped EVERY error to
// `allowed=false` and logged "is not a collaborator — skipping". A 404 means that; a 403, a
// 5xx and a network error do not. Because `allowed=false` skips both agent jobs, and
// report-status skips with them (it requires an agent job to have produced an
// execution_file), an unreported probe error dropped a collaborator's request with no
// comment, no commit status, and a stated reason that was false. Fail-closed was never the
// problem and is preserved below; being unobservable and misattributed was.
//
// These assertions read the workflow's inline script as text, for the reason the sibling
// file states: for a workflow file the declared content IS the behaviour, GitHub reads
// nothing else, and the file runs only inside Actions where no other test can reach it. They
// are written against the LAW in each case rather than one spelling of it — a status check
// may be `=== 404` or `[404, 403].includes(status)`, a throttling signal may be read off any
// of four headers or the message — so a rewrite that preserves the behaviour keeps them
// green.

import (
	"regexp"
	"strings"
	"testing"
)

// codeOnly strips whole-line `//` comments from an inline script.
//
// Every assertion below runs on the stripped form, and that is load-bearing rather than
// tidiness. The comments in check-permissions explain the classification, and in doing so they
// name the very statuses and headers these tests look for — so asserting on the raw text lets
// a comment satisfy a contract the code no longer meets. Found by mutation testing, not by
// inspection: deleting the 403 clause outright left both `DistinguishesDenialFromFailure`
// assertions green, because "403" and "rate limit" still appeared in the prose above them.
//
// Only whole-line comments are removed. A trailing `//` cannot be stripped without a JS
// parser (it may sit inside a string or a regex literal, and this script has both), and
// whole-line is what the workflow actually uses.
func codeOnly(script string) string {
	var kept []string
	for _, line := range strings.Split(script, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "//") {
			continue
		}
		kept = append(kept, line)
	}
	return strings.Join(kept, "\n")
}

// callArgs returns the text of the first `callee(...)` call in code, from the callee name to
// the `);` that closes it.
//
// Assertions about what a call REPORTS have to be scoped to the call. Asserting on the whole
// path instead passes on any interpolation anywhere in it — including the statement that
// computes the reason — which is how the denial log's `${detail}` could be deleted with the
// test still green (mutation testing again).
func callArgs(t *testing.T, code, callee string) string {
	t.Helper()

	start := strings.Index(code, callee+"(")
	if start < 0 {
		t.Fatalf("no %s(...) call found in:\n%s", callee, code)
	}
	rest := code[start:]
	end := strings.Index(rest, ");")
	if end < 0 {
		t.Fatalf("%s(...) call is never closed in:\n%s", callee, code)
	}
	return rest[:end]
}

// pendingHunkNote is appended to every failure that means "the script does not classify its
// errors at all". The automated delivery loop cannot push .github/workflows/* — both App
// tokens are rejected with "refusing to allow a GitHub App to create or update workflow ...
// without `workflows` permission" — so the #1707 hunk is applied by a human, and until it is
// these tests fail. That is deliberate: this issue is about a failure being unobservable, so
// pinning it with something that can pass while unapplied would repeat the defect.
const pendingHunkNote = "\n\nIf this is a fresh checkout of the #1707 delivery branch: the " +
	"hunk to .github/workflows/claude.yml has not been applied yet. The delivery loop cannot " +
	"push workflow files (docs/contributing/automated-delivery.md), so it is applied by hand " +
	"from the PR body. These tests go green once it lands."

// permissionProbeScript returns the inline script of the check-permissions step that probes
// the caller's permission, failing the test if no step does.
func permissionProbeScript(t *testing.T, wf claudeWorkflow) string {
	t.Helper()

	for i, node := range wf.Jobs["check-permissions"].Steps {
		var step struct {
			With struct {
				Script string `yaml:"script"`
			} `yaml:"with"`
		}
		if err := node.Decode(&step); err != nil {
			t.Fatalf("decode check-permissions step %d: %v", i, err)
		}
		if strings.Contains(step.With.Script, "getCollaboratorPermissionLevel") {
			return step.With.Script
		}
	}
	t.Fatal("no check-permissions step calls getCollaboratorPermissionLevel — the permission " +
		"probe is gone, so both agent gates compare against a value nothing produces")
	return ""
}

// probeCatchBlock returns the probe's error handler as CODE: everything from the `catch`
// onwards, with whole-line comments stripped (see codeOnly for why that matters). The catch is
// the last thing in the script, so this is the whole handler.
func probeCatchBlock(t *testing.T, script string) string {
	t.Helper()

	script = codeOnly(script)
	i := strings.Index(script, "catch (")
	if i < 0 {
		t.Fatalf("the permission probe has no catch block — an API error would now fail the "+
			"job with a raw stack trace instead of failing closed.%s", pendingHunkNote)
	}
	return script[i:]
}

// probeCatchPaths splits the error handler at its `else` into the path a genuine DENIAL takes
// and the path everything else takes.
//
// The first return value is everything BEFORE the else, so it holds the classification
// preamble as well as the denial arm. That is deliberate and harmless for the assertions
// here: whatever the preamble does, it must not escalate or comment either, so "does not
// appear before the else" is the property actually wanted in both cases.
func probeCatchPaths(t *testing.T, catchBlock string) (denialPath, escalationPath string) {
	t.Helper()

	i := strings.LastIndex(catchBlock, "else")
	if i < 0 {
		t.Fatalf("the probe's catch block has a single path — it cannot be distinguishing a "+
			"genuine denial from a probe that failed to answer, which is the whole of "+
			"#1707.%s", pendingHunkNote)
	}
	return catchBlock[:i], catchBlock[i:]
}

// The invariant that must survive every change here: an error NEVER admits the caller. The
// fix is about how a failure is reported, and a reporting change that also widened access
// would be a security regression wearing an observability hat.
func TestClaudePermissionProbe_FailsClosedOnEveryError(t *testing.T) {
	wf := loadClaudeWorkflow(t)
	catchBlock := probeCatchBlock(t, permissionProbeScript(t, wf))

	// Scoped to the catch: the try legitimately sets 'true' for a real collaborator.
	for _, admit := range []string{`'allowed', 'true'`, `"allowed", "true"`} {
		if strings.Contains(catchBlock, admit) {
			t.Errorf("the probe's error handler contains %s — a failed or denied probe must "+
				"never admit the caller:\n%s", admit, catchBlock)
		}
	}

	if !strings.Contains(catchBlock, `'allowed', 'false'`) {
		t.Errorf("the probe's error handler does not set allowed='false' — downstream jobs "+
			"would compare against an unset output, which claude.yml's own gates reason "+
			"about explicitly:\n%s", catchBlock)
	}

	// Unconditionally, before the branch, so neither path can forget it.
	denialPath, _ := probeCatchPaths(t, catchBlock)
	if !strings.Contains(denialPath, `'allowed', 'false'`) {
		t.Errorf("allowed='false' is set inside a branch rather than before it — one path "+
			"could then leave the output unset:\n%s", catchBlock)
	}
}

// The classification itself. Without it every error is one verdict, which is the bug.
func TestClaudePermissionProbe_DistinguishesDenialFromFailure(t *testing.T) {
	wf := loadClaudeWorkflow(t)
	catchBlock := probeCatchBlock(t, permissionProbeScript(t, wf))

	// 404 is the unambiguous denial; a 403 needs separate treatment because it is the one
	// status that can mean either thing. Both must be TESTED for — stricter than merely
	// appearing, so an unrelated mention cannot satisfy the contract. The alternation accepts
	// the spellings a rewrite would plausibly use (`status === 404`, `status == 404`,
	// `[404, 403].includes(status)`, `includes(404)`), and the trailing \b stops a longer
	// number from matching a prefix of it.
	for _, status := range []string{"404", "403"} {
		tested := regexp.MustCompile(`(?:==\s*|\[\s*|,\s*|\(\s*)` + status + `\b`)
		if !tested.MatchString(catchBlock) {
			t.Errorf("the probe's error handler never compares the status against %s — it "+
				"cannot be separating a genuine denial from a transient failure:\n%s",
				status, catchBlock)
		}
	}

	// A throttled probe did not answer, so it must not read as a denial however GitHub
	// spelled it. Any one of these discriminators satisfies the law.
	throttleSignals := []string{"429", "x-ratelimit", "retry-after", "rate limit"}
	found := false
	for _, s := range throttleSignals {
		if strings.Contains(strings.ToLower(catchBlock), s) {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("the probe's error handler names none of %v — a rate-limited probe would be "+
			"reported as a permission decision, which is the exact misattribution #1707 is "+
			"about:\n%s", throttleSignals, catchBlock)
	}
}

// The escalation: a probe that could not answer must fail the job, because nothing else on
// this path reports anything. And it must escalate ONLY there — turning every drive-by
// mention by a non-collaborator into a red run would make the signal worthless.
func TestClaudePermissionProbe_OnlyANonDenialFailsTheJob(t *testing.T) {
	wf := loadClaudeWorkflow(t)
	denialPath, escalationPath := probeCatchPaths(t, probeCatchBlock(t, permissionProbeScript(t, wf)))

	if !strings.Contains(escalationPath, "core.setFailed") {
		t.Errorf("the probe's non-denial path does not call core.setFailed — the run stays "+
			"green and the request is dropped in silence, since every downstream job skips "+
			"on allowed=false and report-status skips with them:\n%s",
			escalationPath)
	}

	if strings.Contains(denialPath, "core.setFailed") {
		t.Errorf("the probe escalates on the denial path — an ordinary non-collaborator "+
			"comment would turn the run red:\n%s", denialPath)
	}
}

// The log line must state the reason that occurred. A fixed reason is half the defect: the
// run log asserted "is not a collaborator" for errors that were nothing of the kind.
func TestClaudePermissionProbe_ReportsTheStatusThatOccurred(t *testing.T) {
	wf := loadClaudeWorkflow(t)
	denialPath, escalationPath := probeCatchPaths(t, probeCatchBlock(t, permissionProbeScript(t, wf)))

	// Any of these references the real outcome rather than a hardcoded one. Scoped to the
	// reporting CALL, not the path: the path also contains the statement that computes the
	// reason, which would satisfy the check on its own.
	reasonRefs := []string{"detail", "status", "e.message"}
	for _, tc := range []struct {
		name   string
		path   string
		callee string
	}{
		{"denial", denialPath, "core.info"},
		{"non-denial", escalationPath, "core.setFailed"},
	} {
		reported := callArgs(t, tc.path, tc.callee)
		found := false
		for _, ref := range reasonRefs {
			if strings.Contains(reported, "${"+ref) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("the %s path's %s(...) interpolates none of %v — the reason it states is "+
				"then fixed rather than the one that occurred, which is half of #1707:\n%s",
				tc.name, tc.callee, reasonRefs, reported)
		}
	}
}

// The requester reads the issue or the PR, not the Actions tab, and this path publishes no
// commit status — so a red run alone still looks like silence from where they are standing.
func TestClaudePermissionProbe_CommentsOnTheTriggerOnlyWhenTheProbeFails(t *testing.T) {
	wf := loadClaudeWorkflow(t)
	denialPath, escalationPath := probeCatchPaths(t, probeCatchBlock(t, permissionProbeScript(t, wf)))

	if !strings.Contains(escalationPath, "createComment") {
		t.Errorf("the probe's non-denial path posts no comment — the requester sees nothing "+
			"at all, because report-status publishes no status on a path where neither "+
			"agent job ran:\n%s", escalationPath)
	}

	if strings.Contains(denialPath, "createComment") {
		t.Errorf("the probe comments on the denial path — every drive-by mention by a "+
			"non-collaborator would get a bot reply:\n%s", denialPath)
	}
}

// A comment that cannot be posted must not become the diagnostic. If the escalation's own
// error handling swallowed the probe failure, the fix would reintroduce the bug it fixes.
func TestClaudePermissionProbe_CommentFailureDoesNotReplaceTheDiagnostic(t *testing.T) {
	wf := loadClaudeWorkflow(t)
	_, escalationPath := probeCatchPaths(t, probeCatchBlock(t, permissionProbeScript(t, wf)))

	comment := strings.Index(escalationPath, "createComment")
	if comment < 0 {
		t.Fatalf("the probe's non-denial path posts no comment:\n%s", escalationPath)
	}
	if !strings.Contains(escalationPath[:comment], "try") {
		t.Errorf("the comment is not attempted inside its own try — a failed comment would "+
			"throw past core.setFailed and the probe's real error would never be "+
			"reported:\n%s", escalationPath)
	}

	warn := strings.Index(escalationPath, "core.warning")
	if warn < comment {
		t.Errorf("the comment's failure handler does not warn after the attempt — a comment "+
			"that cannot be posted would fail silently:\n%s", escalationPath)
	}
	if failed := strings.Index(escalationPath, "core.setFailed"); failed < warn {
		t.Errorf("core.setFailed is reported before the comment is attempted — the ordering "+
			"must be comment, then warn if that failed, then fail the job, so the real "+
			"diagnostic is always the last word:\n%s", escalationPath)
	}
}

// The comment is posted by a job whose token comes from the workflow-level block, since
// check-permissions declares none of its own. A future editor adding a narrow block to this
// job would turn the comment into a 403 that only warns — silence again, one level down.
func TestClaudePermissionProbe_TokenCanCommentOnTheTrigger(t *testing.T) {
	wf := loadClaudeWorkflow(t)

	// A job-level permissions block REPLACES the workflow-level one wholesale, so the
	// effective token is the job's own block when it has one.
	effective := wf.Jobs["check-permissions"].Permissions
	if effective == nil {
		effective = wf.Permissions
	}

	// issues: write covers all three triggers — issues.createComment posts to a PR as well,
	// since every PR is an issue.
	if got := effective["issues"]; got != "write" {
		t.Errorf("check-permissions' effective issues permission is %q, want \"write\" — "+
			"the probe-failure comment would 403 and only warn, so the request would be "+
			"dropped silently again (#1707)", got)
	}
}

// The comment must not be able to re-enter this workflow. GITHUB_TOKEN-authored comments do
// not start workflow runs, so today this is belt and braces — but claude.yml's own trigger is
// a bare substring test, and a comment naming a trigger command would also read to a human as
// a fresh request rather than a report about a failed one.
func TestClaudePermissionProbe_CommentCannotRetriggerTheWorkflow(t *testing.T) {
	wf := loadClaudeWorkflow(t)
	_, escalationPath := probeCatchPaths(t, probeCatchBlock(t, permissionProbeScript(t, wf)))

	for _, trigger := range []string{"@claude", "/blis-pr-review"} {
		if strings.Contains(escalationPath, trigger) {
			t.Errorf("the probe's non-denial path contains the trigger string %q — a comment "+
				"posted from here could re-enter this workflow's own substring trigger, and "+
				"would read as a new request rather than a report of a failed one. Write it "+
				"so it cannot match:\n%s", trigger, escalationPath)
		}
	}
}
