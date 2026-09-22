package scripts_test

// Pins the error handling of the collaborator probe in .github/workflows/claude.yml's
// check-permissions job (#1707).
//
// The defect these tests exist to prevent: the probe's `catch` mapped EVERY error to
// `allowed=false` and logged "is not a collaborator — skipping". A 404 means that; a 5xx and a
// network error do not; and a 403 means it only sometimes — on a PRIVATE repo, where a 403 is
// also how "cannot see the collaborator list" surfaces, and then only when it carries no
// throttling signal. On this PUBLIC repo a non-collaborator gets 404, so a bare 403 here is a
// probe that could not answer rather than one that said no. Because `allowed=false` skips both
// agent jobs, and report-status skips with them (it requires an agent job to have produced an
// execution_file), an unreported probe error dropped a collaborator's request with no comment,
// no commit status, and a stated reason that was false. Fail-closed was never the problem and
// is preserved below; being unobservable and misattributed was.
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
	"unicode"
)

// codeOnly strips every JavaScript comment from an inline script — whole-line, trailing and
// block — leaving string, template and regex literals intact.
//
// Every assertion below runs on the stripped form, and that is load-bearing rather than
// tidiness. The comments in check-permissions explain the classification, and in doing so they
// name the very statuses and headers these tests look for — so asserting on the raw text lets
// a comment satisfy a contract the code no longer meets. Found by mutation testing, not by
// inspection: deleting the 403 clause outright left both `DistinguishesDenialFromFailure`
// assertions green, because "403" and "rate limit" still appeared in the prose above them.
//
// An earlier version removed only whole-line comments, reasoning that a trailing `//` cannot be
// stripped without a parser and that whole-line is what the workflow happens to use. That left
// the same hole one spelling over — `/* status === 403; x-ratelimit */` satisfies two of the
// assertions below with no code behind it — and it made the assertions describe today's
// formatting rather than the law they claim to pin. Hence a scanner.
//
// It is NOT a JavaScript parser, and the two places it guesses are worth knowing:
//
//   - whether `/` opens a regex or divides is decided by which character precedes it, the
//     standard heuristic, not by a grammar. A regex after a keyword (`return /x/`) is read as
//     division;
//   - a template literal is copied verbatim, `${}` substitutions included, so a comment written
//     inside a substitution would survive.
//
// Both guesses are conservative in the one direction that matters: every branch here COPIES
// what it cannot classify, so a misjudgement can only leave a comment un-stripped — it can
// never delete code and quietly turn an assertion vacuous the other way.
// TestCodeOnly_LeavesTheRealScriptsCodeIntact is the guard, stripping the actual workflow
// script and asserting its code survives.
func codeOnly(script string) string {
	var out strings.Builder
	src := []rune(script)

	// The most recent non-whitespace rune emitted, which is what decides whether a `/` can
	// open a regex literal. Zero at the start of the script, where one can.
	var prev rune

	for i := 0; i < len(src); i++ {
		r := src[i]
		switch {
		case r == '/' && i+1 < len(src) && src[i+1] == '/':
			for i < len(src) && src[i] != '\n' {
				i++
			}
			// Keep the newline, so stripping never joins two statements onto one line and
			// failure messages still read like the script.
			if i < len(src) {
				out.WriteRune('\n')
			}
		case r == '/' && i+1 < len(src) && src[i+1] == '*':
			i = skipBlockComment(src, i)
		case r == '\'' || r == '"':
			i = copyQuoted(&out, src, i, r)
			prev = r
		case r == '`':
			i = copyTemplate(&out, src, i)
			prev = r
		case r == '/' && regexCanFollow(prev):
			i = copyRegex(&out, src, i)
			prev = '/'
		default:
			out.WriteRune(r)
			if !unicode.IsSpace(r) {
				prev = r
			}
		}
	}
	return out.String()
}

// regexCanFollow reports whether a `/` appearing after prev opens a regex literal rather than
// dividing. A regex can only follow an operator, an opening bracket or a statement boundary; an
// identifier, a literal, `)` or `]` means division. prev is zero at the start of the script.
func regexCanFollow(prev rune) bool {
	return prev == 0 || strings.ContainsRune("(,=:[!&|?{};+-*%^~<>", prev)
}

// skipBlockComment returns the index of the `/` closing the `/*` comment that starts at i, or
// the last index of src if it is never closed.
func skipBlockComment(src []rune, i int) int {
	for i += 2; i+1 < len(src); i++ {
		if src[i] == '*' && src[i+1] == '/' {
			return i + 1
		}
	}
	return len(src) - 1
}

// copyQuoted copies the '...' or "..." literal starting at i verbatim, honouring backslash
// escapes, and returns the index of its closing quote.
func copyQuoted(out *strings.Builder, src []rune, i int, quote rune) int {
	out.WriteRune(src[i])
	for i++; i < len(src); i++ {
		out.WriteRune(src[i])
		if src[i] == '\\' && i+1 < len(src) {
			i++
			out.WriteRune(src[i])
			continue
		}
		if src[i] == quote {
			return i
		}
	}
	return len(src) - 1
}

// copyTemplate copies the `...` literal starting at i verbatim and returns the index of its
// closing backtick. `${}` depth is tracked so a backtick inside a substitution cannot close the
// outer literal early.
func copyTemplate(out *strings.Builder, src []rune, i int) int {
	out.WriteRune(src[i])
	depth := 0
	for i++; i < len(src); i++ {
		out.WriteRune(src[i])
		switch {
		case src[i] == '\\' && i+1 < len(src):
			i++
			out.WriteRune(src[i])
		case src[i] == '$' && i+1 < len(src) && src[i+1] == '{':
			i++
			out.WriteRune(src[i])
			depth++
		case src[i] == '}' && depth > 0:
			depth--
		case src[i] == '`' && depth == 0:
			return i
		}
	}
	return len(src) - 1
}

// copyRegex copies the /.../flags literal starting at i verbatim and returns the index of its
// last rune. A newline ends it: a regex literal cannot span lines, so reaching one means the `/`
// was a division after all — and since everything has been copied, nothing is lost either way.
func copyRegex(out *strings.Builder, src []rune, i int) int {
	out.WriteRune(src[i])
	inClass := false
	for i++; i < len(src); i++ {
		out.WriteRune(src[i])
		switch {
		case src[i] == '\\' && i+1 < len(src):
			i++
			out.WriteRune(src[i])
		case src[i] == '[':
			inClass = true
		case src[i] == ']':
			inClass = false
		case src[i] == '\n':
			return i
		case src[i] == '/' && !inClass:
			// Consume the flags too, so a following `/` is not read as opening another literal.
			for i+1 < len(src) && unicode.IsLetter(src[i+1]) {
				i++
				out.WriteRune(src[i])
			}
			return i
		}
	}
	return len(src) - 1
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

// The escalation comment is published to a PUBLIC thread, where it persists and notifies long
// after the run log has scrolled past. So it names a BOUNDED reason — a status, or a category
// when there is no status — and never the raw exception: a transport failure's text is
// unbounded and carries incidental request detail that has no business in a permanent public
// notification. The precise text still reaches core.setFailed, which is the diagnostic channel.
func TestClaudePermissionProbe_CommentPublishesOnlyABoundedReason(t *testing.T) {
	wf := loadClaudeWorkflow(t)
	catchBlock := probeCatchBlock(t, permissionProbeScript(t, wf))
	_, escalationPath := probeCatchPaths(t, catchBlock)

	comment := callArgs(t, escalationPath, "createComment")

	// The spellings that put the caught exception itself into the text.
	rawExceptions := []string{"String(e)", "${e}", "${e.", "JSON.stringify(e"}
	for _, expr := range rawExceptions {
		if strings.Contains(comment, expr) {
			t.Errorf("the probe's escalation comment interpolates %s — the raw exception text "+
				"is unbounded and this comment is public and permanent. Publish the status or "+
				"a category, and leave the exception to core.setFailed:\n%s", expr, comment)
		}
	}

	// And the same thing one hop through a variable, which is how the check above would be
	// satisfied without changing a word of what actually gets published.
	for _, name := range interpolatedIdents(comment) {
		decl := declarationOf(catchBlock, name)
		if decl == "" {
			continue
		}
		for _, expr := range rawExceptions {
			if strings.Contains(decl, expr) {
				t.Errorf("the escalation comment interpolates ${%s}, and %s is assigned %s — "+
					"the raw exception reaches the public comment through a variable:\n%s",
					name, name, expr, decl)
			}
		}
	}
}

// interpolation matches the leading identifier of a `${...}` substitution: `detail` in
// `${detail}`, `commentErr` in `${commentErr.message}`.
var interpolation = regexp.MustCompile(`\$\{\s*([A-Za-z_$][\w$]*)`)

// interpolatedIdents returns those identifiers, in order, without repeats.
func interpolatedIdents(code string) []string {
	var names []string
	seen := map[string]bool{}
	for _, m := range interpolation.FindAllStringSubmatch(code, -1) {
		if !seen[m[1]] {
			seen[m[1]] = true
			names = append(names, m[1])
		}
	}
	return names
}

// declarationOf returns the text of code's `const|let|var <name> = ...` statement up to its
// terminating `;`, or "" when code declares no such variable (`context`, for instance, is
// interpolated everywhere and declared nowhere).
func declarationOf(code, name string) string {
	decl := regexp.MustCompile(`(?:const|let|var)\s+` + regexp.QuoteMeta(name) + `\s*=`)
	loc := decl.FindStringIndex(code)
	if loc == nil {
		return ""
	}
	rest := code[loc[0]:]
	if end := strings.Index(rest, ";"); end >= 0 {
		return rest[:end+1]
	}
	return rest
}

// codeOnly's own contract. It is asserted on directly because every assertion above runs on its
// output, so a hole in it weakens all of them at once — which is exactly what happened to the
// line-based version it replaced.
func TestCodeOnly_RemovesCommentsAndKeepsLiterals(t *testing.T) {
	for _, tc := range []struct {
		name        string
		script      string
		wantAbsent  []string
		wantPresent []string
	}{
		{
			name:       "whole-line comment",
			script:     "// status === 403\nconst x = 1;",
			wantAbsent: []string{"403"},
		},
		{
			name:       "trailing comment",
			script:     "const x = 1; // status === 403 and x-ratelimit-remaining",
			wantAbsent: []string{"403", "x-ratelimit"},
			// This is the case the line-based version could not reach.
			wantPresent: []string{"const x = 1;"},
		},
		{
			name:       "block comment on one line",
			script:     "const x = /* status === 403; x-ratelimit */ 1;",
			wantAbsent: []string{"403", "x-ratelimit"},
			// The concrete evasion the reviewer named.
			wantPresent: []string{"const x =", "1;"},
		},
		{
			name:        "block comment spanning lines",
			script:      "const a = 1;\n/*\n429 retry-after\n*/\nconst b = 2;",
			wantAbsent:  []string{"429", "retry-after"},
			wantPresent: []string{"const a = 1;", "const b = 2;"},
		},
		{
			name:        "// inside a single-quoted string is not a comment",
			script:      "const u = 'https://api.github.com/403';",
			wantPresent: []string{"'https://api.github.com/403'"},
		},
		{
			name:        "// inside a template literal is not a comment",
			script:      "core.info(`see https://x/403 ${status}`);",
			wantPresent: []string{"https://x/403", "${status}"},
		},
		{
			name:        "/* inside a string does not open a comment",
			script:      "const s = '/*'; const keep = 403;",
			wantPresent: []string{"'/*'", "403"},
		},
		{
			name:        "regex literal survives and its slashes do not open a comment",
			script:      "const t = /rate limit|abuse detection/i.test(e.message ?? '');",
			wantPresent: []string{"/rate limit|abuse detection/i", "e.message"},
		},
		{
			name:        "division is not mistaken for a regex",
			script:      "const r = a / b; // 403\nconst keep = 429;",
			wantAbsent:  []string{"403"},
			wantPresent: []string{"a / b;", "429"},
		},
		{
			name:        "escaped quote does not end a string early",
			script:      `const s = 'it\'s // not a comment'; const keep = 403;`,
			wantPresent: []string{"not a comment", "403"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := codeOnly(tc.script)
			for _, absent := range tc.wantAbsent {
				if strings.Contains(got, absent) {
					t.Errorf("codeOnly kept %q, so a comment can still satisfy an assertion "+
						"about code:\ninput:  %s\noutput: %s", absent, tc.script, got)
				}
			}
			for _, present := range tc.wantPresent {
				if !strings.Contains(got, present) {
					t.Errorf("codeOnly dropped %q — it deleted code, which would make an "+
						"assertion fail for a reason that has nothing to do with the "+
						"contract:\ninput:  %s\noutput: %s", present, tc.script, got)
				}
			}
		})
	}
}

// The guard on codeOnly's two guesses (regex-vs-division, and verbatim template
// substitutions), run against the script it actually has to handle. A heuristic that misjudged
// this script would fail here, loudly, rather than silently weakening every assertion above.
//
// The tokens asserted on are ones the probe has in EVERY state — before the #1707 hunk and
// after — so this test does not depend on the pending hunk.
func TestCodeOnly_LeavesTheRealScriptsCodeIntact(t *testing.T) {
	script := permissionProbeScript(t, loadClaudeWorkflow(t))
	stripped := codeOnly(script)

	// Code, including a string literal holding a slash and an optional-chaining expression.
	for _, code := range []string{
		"getCollaboratorPermissionLevel",
		"core.setOutput('allowed'",
		"'/blis-pr-review'",
		"context.payload.comment?.body",
		"catch (",
	} {
		if !strings.Contains(stripped, code) {
			t.Errorf("codeOnly removed %q from the real permission probe — it is deleting "+
				"code, so every assertion running on its output is unsound:\n%s", code, stripped)
		}
	}

	// And the comments really are gone, so the stripping is not a no-op that only looks safe.
	if !strings.Contains(script, "//") {
		t.Fatal("the permission probe contains no comments at all — this test can no longer " +
			"tell stripping from a no-op; assert on whatever prose it does carry instead")
	}
	if strings.Contains(stripped, "//") {
		t.Errorf("codeOnly left a `//` in the real permission probe, so prose can still "+
			"satisfy the assertions above:\n%s", stripped)
	}
}
