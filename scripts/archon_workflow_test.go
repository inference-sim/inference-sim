package scripts_test

// Pins the trigger gate in .github/workflows/archon.yml (#1675): a comment that merely
// MENTIONS `/archon-pr-review` must not start a self-hosted archon run; a comment that uses
// it as a directive must.
//
// Why these tests evaluate the expression instead of asserting its text. A job-level `if:`
// runs only inside GitHub Actions, so no other test in this repository can reach it — the
// same position the shell-script tests in this package are in. But unlike a permission block
// (see claude_workflow_test.go, which pins literals because the declared structure IS the
// behaviour), this gate's whole point is a DECISION over an input, and the regression it
// guards against — widening back to a whole-body `contains()` — has many textual spellings
// and one behaviour. So the tests read the expression out of the workflow file and evaluate
// it against real comment bodies taken from this repository's history.
//
// The evaluator below covers exactly the operators and functions this one gate uses. It is a
// model of GitHub's documented expression semantics, not GitHub's implementation: it proves
// the RULE is the intended one and that the file still declares that rule. It cannot prove
// GitHub's parser accepts the syntax — that is what the first real invocation after merge
// shows, and why the gate keeps `startsWith` (plain, documented) as its first alternative.
// Where the documented semantics are surprising they are modelled rather than simplified:
// `contains`/`startsWith`/`endsWith` compare case-INSENSITIVELY (see foldExprCase), so the
// cases below cover the mixed-case bodies that behaviour makes reachable.
//
// WHERE THE GATE IS READ FROM. #1675's fix could not be committed into archon.yml: the L1
// delivery runner's App token has no `workflows` permission, so it is carried as a pending
// patch (.github/archon-trigger-anchor.patch) for a human to apply, following the convention
// .github/deliver-conflict-wiring.patch set on #1781. So these tests resolve the gate that is
// (or, once that patch is applied, will be) in force — see effectiveArchonGate. Both terminal
// states pass; the two ways of losing half the change fail.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// --- the gate under test -----------------------------------------------------------------

type archonWorkflow struct {
	Jobs map[string]struct {
		If string `yaml:"if"`
	} `yaml:"jobs"`
}

const (
	archonWorkflowRelPath = ".github/workflows/archon.yml"
	// Repo-relative for diagnostics a human acts on; the tests run one directory down.
	archonPatchRelPath = ".github/archon-trigger-anchor.patch"
	archonPendingPatch = "../" + archonPatchRelPath

	// The discriminator between the pre- and post-#1675 gate, used to decide which source is
	// authoritative. Stated behaviourally rather than as a text match: "anchored" has many
	// spellings and exactly one observable meaning.
	proseProbe = "Once the rebase lands, please run /archon-pr-review again."
)

// readArchonTriggerGate returns the `if:` expression of the job that decides whether a comment
// is an invocation at all, from the archon.yml at the given path.
func readArchonTriggerGate(t *testing.T, path string) string {
	t.Helper()

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	// Deliberately not strict: the struct models only the field these tests assert on.
	var wf archonWorkflow
	if err := yaml.Unmarshal(raw, &wf); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
	job, ok := wf.Jobs["check-permissions"]
	if !ok {
		t.Fatalf("job \"check-permissions\" missing from %s — the trigger gate was "+
			"restructured; re-derive these assertions rather than deleting them", path)
	}
	if strings.TrimSpace(job.If) == "" {
		t.Fatalf("job \"check-permissions\" in %s declares no `if:` — every issue_comment "+
			"on every PR would now start this job (#1675)", path)
	}
	return job.If
}

// effectiveArchonGate resolves the trigger gate these tests assert on: the one archon.yml
// declares, or — while #1675's fix is still carried as a pending patch — the one that patch
// produces. It fails on either way of ending up with half the change:
//
//	patch present, workflow anchored   -> applied but not deleted (stale patch)
//	patch absent,  workflow unanchored -> deleted unapplied, or the gate was widened back
func effectiveArchonGate(t *testing.T) string {
	t.Helper()

	livePath := filepath.Join("..", filepath.FromSlash(archonWorkflowRelPath))
	live := readArchonTriggerGate(t, livePath)
	liveAnchored := !gateFires(t, live, proseProbe, true)

	_, err := os.Stat(archonPendingPatch)
	switch {
	case err == nil && liveAnchored:
		t.Fatalf("%s is anchored, but %s is still present — the patch was applied and not "+
			"deleted. Run `git rm %s`.", archonWorkflowRelPath, archonPatchRelPath,
			archonPatchRelPath)
		return ""
	case err == nil:
		// The pending state. Applying the patch here also proves on every CI run that it
		// still applies cleanly, which a human would otherwise have to check by hand.
		t.Logf("#1675 is still pending: reading the gate from %s applied to %s. "+
			"Apply and delete the patch to put it in force.",
			archonPatchRelPath, archonWorkflowRelPath)
		patched := applyPendingPatch(t, livePath)
		if gateFires(t, patched, proseProbe, true) {
			t.Fatalf("%s does not anchor the trigger: the gate it produces still fires on a "+
				"body that only mentions the command (#1675)", archonPatchRelPath)
		}
		return patched
	case errors.Is(err, fs.ErrNotExist) && liveAnchored:
		return live // the terminal state: applied and the patch removed
	case errors.Is(err, fs.ErrNotExist):
		t.Fatalf("%s fires on a body that only MENTIONS /archon-pr-review, and no pending "+
			"%s carries the fix. Either the anchor was widened back to whole-body "+
			"containment, or the patch was deleted without being applied — each costs a "+
			"self-hosted archon run per comment that discusses the trigger (#1675).",
			archonWorkflowRelPath, archonPatchRelPath)
		return ""
	default:
		t.Fatalf("stat %s: %v", archonPendingPatch, err)
		return ""
	}
}

// applyPendingPatch applies the pending workflow patch to a throwaway copy of archon.yml and
// returns the trigger gate of the result.
func applyPendingPatch(t *testing.T, livePath string) string {
	t.Helper()

	dir := t.TempDir()
	dst := filepath.Join(dir, filepath.FromSlash(archonWorkflowRelPath))
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		t.Fatalf("prepare temp tree: %v", err)
	}
	raw, err := os.ReadFile(livePath)
	if err != nil {
		t.Fatalf("read %s: %v", livePath, err)
	}
	if err := os.WriteFile(dst, raw, 0o644); err != nil {
		t.Fatalf("write %s: %v", dst, err)
	}

	patch, err := filepath.Abs(archonPendingPatch)
	if err != nil {
		t.Fatalf("resolve %s: %v", archonPendingPatch, err)
	}
	// `git apply` outside a repository is a plain patch applier, and it ignores the `#`
	// preamble the patch carries above its first `diff --git` line.
	cmd := exec.Command("git", "apply", patch)
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("`git apply %s` failed (%v) — the patch no longer applies to %s, so #1675's "+
			"fix can no longer be landed as written:\n%s",
			archonPatchRelPath, err, archonWorkflowRelPath, out)
	}
	return readArchonTriggerGate(t, dst)
}

// gateFires evaluates the workflow's own gate for one comment.
func gateFires(t *testing.T, gate, body string, onPullRequest bool) bool {
	t.Helper()

	ctx := map[string]any{"github.event.comment.body": body}
	// GitHub supplies `issue.pull_request` only for a comment on a pull request; on a plain
	// issue the path is absent, which an expression treats as falsy. An unknown path
	// resolving to nil models that, and makes the gate fail closed.
	if onPullRequest {
		ctx["github.event.issue.pull_request"] = true
	}

	got, err := evalGitHubExpr(gate, ctx)
	if err != nil {
		t.Fatalf("evaluate the gate from archon.yml: %v\n\ngate was:\n%s", err, gate)
	}
	return got
}

// --- the contract ------------------------------------------------------------------------

// The two acceptance criteria of #1675, over bodies drawn from this repository's history.
func TestArchonWorkflow_TriggersOnDirectivesNotMentions(t *testing.T) {
	gate := effectiveArchonGate(t)

	cases := []struct {
		name string
		body string
		want bool
		why  string
	}{
		{
			name: "bare invocation",
			body: "/archon-pr-review",
			want: true,
			why:  "the overwhelmingly common invocation — 134 of them in this repo's history",
		},
		{
			name: "invocation with an archon ref argument",
			body: "/archon-pr-review --archon-ref feature/1585-kv-offload",
			want: true,
			why:  "the `Parse archon ref` step exists to read exactly this",
		},
		{
			name: "invocation as the last line of a long report",
			body: "Rebased onto `main` after #1660 (`017fcd98`) merged. New head `ce1f1505`.\n\n" +
				"**Verification on `4be14710`:** `go build ./...` clean, 14 packages ok.\n\n" +
				"/archon-pr-review",
			want: true,
			why: "the established habit on #1667/#1664 — a first-line-only anchor would " +
				"reject this and break the workflow for its heaviest user",
		},
		{
			name: "prose: the command inline, mid-sentence",
			body: "This also explains the ratio noted on #1667 — 14 archon comments for 9 " +
				"`/archon-pr-review` triggers.",
			want: false,
			why:  "the #1675 report itself; under the old gate, writing it fired a run",
		},
		{
			name: "prose: an agent review quoted back",
			body: "> 🤖 **qa-review** — experimental two-agent AI-review demo\n" +
				"> The `/archon-pr-review` trigger fired twice for one head.\n",
			want: false,
			why:  "fired run 33785123327 on #1674 four seconds after it was posted",
		},
		{
			name: "prose: a review heading and findings list",
			body: "## Review — the two fixes are correct, but one claim is not\n\n" +
				"- `--edit-last` bounds the thread, but each `/archon-pr-review` still " +
				"burns a self-hosted runner.\n",
			want: false,
			why:  "fired run 33785598745 on #1674, byte-identical to the previous report",
		},
		{
			name: "prose: an imperative asking someone else to run it",
			body: "Once the rebase lands, please run /archon-pr-review again.",
			want: false,
			why: "the defining false positive: whole-body containment cannot tell a request " +
				"to invoke later from an invocation now",
		},
		{
			name: "the -claude variant is routed elsewhere",
			body: "/archon-pr-review-claude",
			want: false,
			why:  "claude.yml owns it; archon.yml must not double-handle it",
		},
		{
			name: "an invocation that also names the -claude variant is excluded",
			body: "/archon-pr-review\n\nSee also /archon-pr-review-claude for the LLM read.",
			want: false,
			why: "the whole-body -claude exclusion is unchanged by #1675 and fails closed; " +
				"pinned so a future edit to the anchor cannot quietly drop it",
		},
		{
			name: "a mixed-case directive fires, because GitHub folds case",
			body: "/Archon-PR-Review",
			want: true,
			why: "contains/startsWith are documented \"not case sensitive\", so this IS an " +
				"invocation at the runtime; pinned so the evaluator keeps modelling that " +
				"rather than quietly comparing case-sensitively",
		},
		{
			name: "a mixed-case -claude variant is still routed elsewhere",
			body: "/Archon-PR-Review-Claude",
			want: false,
			why: "the direction case-folding actually matters in: a case-sensitive exclusion " +
				"would let this through the `!contains` conjunct and double-handle a comment " +
				"claude.yml owns",
		},
		{
			name: "CRLF line endings do not break the later-line anchor",
			body: "Rebased onto `main`; new head `ce1f1505`.\r\n\r\n/archon-pr-review",
			want: true,
			why: "comment payloads from some clients carry \\r\\n. The anchor looks for a " +
				"newline immediately before the command, and in CRLF that newline is the LF " +
				"that follows the CR — so it still matches. A tighter rule requiring the " +
				"command be FOLLOWED by \\n would instead reject this real invocation, which " +
				"is why the residual below is left open rather than closed that way",
		},
		{
			name: "a directive on an issue rather than a pull request",
			body: "/archon-pr-review",
			want: false,
			why:  "there is no PR to review; the gate must fail closed on a plain issue",
		},
		{
			name: "empty body",
			body: "",
			want: false,
			why:  "degenerate input must not fire",
		},
		{
			name: "known residual: a sentence that begins with the command still fires",
			body: "/archon-pr-review issued `NO_CHANGE` (fast-track eligible), noting 2 " +
				"guarded promises changed.\n",
			want: true,
			why: "expressions have no regex, so \"alone on its line\" is not expressible. " +
				"One comment in 268 hit this. Recorded as behaviour rather than left " +
				"undocumented — if a scripted gate ever closes it, flip this case",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			onPR := tc.name != "a directive on an issue rather than a pull request"
			if got := gateFires(t, gate, tc.body, onPR); got != tc.want {
				verb := map[bool]string{true: "fires", false: "does not fire"}
				t.Errorf("the gate in archon.yml %s for this body, want %s — %s\n\nbody:\n%s",
					verb[got], verb[tc.want], tc.why, tc.body)
			}
		})
	}
}

// The gate must not be satisfiable by containment alone. Stated separately from the table
// because it is the regression #1675 names: if a future edit restores a whole-body
// `contains()` — as its own conjunct, as an extra `||`, or by dropping the anchor — every
// one of these bodies starts a run again.
func TestArchonWorkflow_TriggerIsNotWholeBodyContainment(t *testing.T) {
	gate := effectiveArchonGate(t)

	mentions := []string{
		"See `/archon-pr-review` above.",
		"The /archon-pr-review command reports dist.",
		"Cancelled the `/archon-pr-review` I just triggered — my mistake.",
		"| `/archon-pr-review` | structural |",
	}
	for _, body := range mentions {
		if gateFires(t, gate, body, true) {
			t.Errorf("the gate fires on a body that only mentions the command:\n%s\n\n"+
				"The match must stay anchored to the start of a line. A whole-body "+
				"contains() spends a self-hosted archon run on every comment that "+
				"discusses the trigger (#1675).", body)
		}
	}
}

// Non-vacuity: a gate that never fires would pass every assertion above except this one.
func TestArchonWorkflow_TriggerStillAcceptsARealInvocation(t *testing.T) {
	gate := effectiveArchonGate(t)

	if !gateFires(t, gate, "/archon-pr-review", true) {
		t.Fatal("the gate rejects a bare `/archon-pr-review` on a pull request — archon " +
			"can no longer be invoked at all")
	}
}

// --- a minimal evaluator for the subset of GitHub expression syntax this gate uses -------
//
// Grammar: or := and ('||' and)* ; and := unary ('&&' unary)* ; unary := '!' unary | primary
// primary := '(' or ')' | call | path | string
// Supported calls: contains, startsWith, endsWith, format, fromJSON.
// Anything outside that is an error rather than a guess, so a gate this cannot model fails
// the test loudly instead of being silently mis-evaluated.

func evalGitHubExpr(src string, ctx map[string]any) (bool, error) {
	toks, err := lexGitHubExpr(src)
	if err != nil {
		return false, err
	}
	p := &exprParser{toks: toks, ctx: ctx}
	v, err := p.parseOr()
	if err != nil {
		return false, err
	}
	if p.pos != len(p.toks) {
		return false, fmt.Errorf("trailing tokens after the expression (at token %d)", p.pos)
	}
	return truthy(v), nil
}

type exprToken struct {
	kind string // "(" ")" "," "op" "ident" "str"
	val  string
}

func lexGitHubExpr(src string) ([]exprToken, error) {
	var toks []exprToken
	for i := 0; i < len(src); {
		c := src[i]
		switch {
		case c == ' ' || c == '\t' || c == '\n' || c == '\r':
			i++
		case c == '(' || c == ')' || c == ',':
			toks = append(toks, exprToken{kind: string(c)})
			i++
		case c == '!':
			toks = append(toks, exprToken{kind: "op", val: "!"})
			i++
		case strings.HasPrefix(src[i:], "&&"), strings.HasPrefix(src[i:], "||"):
			toks = append(toks, exprToken{kind: "op", val: src[i : i+2]})
			i += 2
		case c == '\'':
			// Single-quoted literal; '' is an escaped quote and there are no others. That
			// absence is why the gate needs fromJSON to obtain a newline.
			var sb strings.Builder
			j := i + 1
			for j < len(src) {
				if src[j] == '\'' {
					if j+1 < len(src) && src[j+1] == '\'' {
						sb.WriteByte('\'')
						j += 2
						continue
					}
					break
				}
				sb.WriteByte(src[j])
				j++
			}
			if j >= len(src) {
				return nil, fmt.Errorf("unterminated string literal at offset %d", i)
			}
			toks = append(toks, exprToken{kind: "str", val: sb.String()})
			i = j + 1
		default:
			j := i
			for j < len(src) && (isIdentByte(src[j])) {
				j++
			}
			if j == i {
				return nil, fmt.Errorf("unsupported character %q at offset %d", c, i)
			}
			toks = append(toks, exprToken{kind: "ident", val: src[i:j]})
			i = j
		}
	}
	return toks, nil
}

func isIdentByte(c byte) bool {
	return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' ||
		c == '.' || c == '_' || c == '-' || c == '*'
}

type exprParser struct {
	toks []exprToken
	pos  int
	ctx  map[string]any
}

func (p *exprParser) peek() (exprToken, bool) {
	if p.pos >= len(p.toks) {
		return exprToken{}, false
	}
	return p.toks[p.pos], true
}

func (p *exprParser) parseOr() (any, error) {
	left, err := p.parseAnd()
	if err != nil {
		return nil, err
	}
	for {
		t, ok := p.peek()
		if !ok || t.kind != "op" || t.val != "||" {
			return left, nil
		}
		p.pos++
		right, err := p.parseAnd()
		if err != nil {
			return nil, err
		}
		left = truthy(left) || truthy(right)
	}
}

func (p *exprParser) parseAnd() (any, error) {
	left, err := p.parseUnary()
	if err != nil {
		return nil, err
	}
	for {
		t, ok := p.peek()
		if !ok || t.kind != "op" || t.val != "&&" {
			return left, nil
		}
		p.pos++
		right, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		left = truthy(left) && truthy(right)
	}
}

func (p *exprParser) parseUnary() (any, error) {
	if t, ok := p.peek(); ok && t.kind == "op" && t.val == "!" {
		p.pos++
		v, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		return !truthy(v), nil
	}
	return p.parsePrimary()
}

func (p *exprParser) parsePrimary() (any, error) {
	t, ok := p.peek()
	if !ok {
		return nil, fmt.Errorf("expression ended where a value was expected")
	}
	switch t.kind {
	case "(":
		p.pos++
		v, err := p.parseOr()
		if err != nil {
			return nil, err
		}
		next, ok := p.peek()
		if !ok || next.kind != ")" {
			return nil, fmt.Errorf("unclosed parenthesis")
		}
		p.pos++
		return v, nil
	case "str":
		p.pos++
		return t.val, nil
	case "ident":
		p.pos++
		if next, ok := p.peek(); ok && next.kind == "(" {
			args, err := p.parseArgs()
			if err != nil {
				return nil, err
			}
			return callExprFunc(t.val, args)
		}
		// A context path. An unknown one is nil (falsy), exactly as an absent payload field
		// is in Actions — so the gate fails closed rather than erroring here.
		return p.ctx[t.val], nil
	default:
		return nil, fmt.Errorf("unexpected %q where a value was expected", t.kind)
	}
}

func (p *exprParser) parseArgs() ([]any, error) {
	p.pos++ // consume "("
	var args []any
	if t, ok := p.peek(); ok && t.kind == ")" {
		p.pos++
		return args, nil
	}
	for {
		v, err := p.parseOr()
		if err != nil {
			return nil, err
		}
		args = append(args, v)
		t, ok := p.peek()
		if !ok {
			return nil, fmt.Errorf("unterminated argument list")
		}
		p.pos++
		switch t.kind {
		case ",":
			continue
		case ")":
			return args, nil
		default:
			return nil, fmt.Errorf("unexpected %q in an argument list", t.kind)
		}
	}
}

// foldExprCase normalizes both operands of a string comparison. GitHub documents contains,
// startsWith and endsWith as "not case sensitive", so a model that compared case-sensitively
// would disagree with the runtime in the permissive direction: it would report that a
// mixed-case `/Archon-PR-Review` does not fire when GitHub fires on it, and — the direction
// that actually matters — that a mixed-case `/archon-pr-review-CLAUDE` escapes the exclusion
// conjunct when GitHub excludes it. Both operands are folded, matching the runtime.
func foldExprCase(a, b any) (string, string) {
	return strings.ToLower(asString(a)), strings.ToLower(asString(b))
}

func callExprFunc(name string, args []any) (any, error) {
	switch strings.ToLower(name) {
	case "contains":
		if len(args) != 2 {
			return nil, fmt.Errorf("contains takes 2 arguments, got %d", len(args))
		}
		s, sub := foldExprCase(args[0], args[1])
		return strings.Contains(s, sub), nil
	case "startswith":
		if len(args) != 2 {
			return nil, fmt.Errorf("startsWith takes 2 arguments, got %d", len(args))
		}
		s, prefix := foldExprCase(args[0], args[1])
		return strings.HasPrefix(s, prefix), nil
	case "endswith":
		if len(args) != 2 {
			return nil, fmt.Errorf("endsWith takes 2 arguments, got %d", len(args))
		}
		s, suffix := foldExprCase(args[0], args[1])
		return strings.HasSuffix(s, suffix), nil
	case "format":
		if len(args) == 0 {
			return nil, fmt.Errorf("format takes at least 1 argument")
		}
		out := asString(args[0])
		for i, a := range args[1:] {
			out = strings.ReplaceAll(out, fmt.Sprintf("{%d}", i), asString(a))
		}
		return out, nil
	case "fromjson":
		if len(args) != 1 {
			return nil, fmt.Errorf("fromJSON takes 1 argument, got %d", len(args))
		}
		var v any
		if err := json.Unmarshal([]byte(asString(args[0])), &v); err != nil {
			return nil, fmt.Errorf("fromJSON(%q): %v", asString(args[0]), err)
		}
		return v, nil
	default:
		return nil, fmt.Errorf("this evaluator does not model the function %q — it was "+
			"written for the functions archon.yml's trigger gate uses; extend it "+
			"deliberately rather than loosening the test", name)
	}
}

func asString(v any) string {
	switch s := v.(type) {
	case nil:
		return ""
	case string:
		return s
	case bool:
		if s {
			return "true"
		}
		return "false"
	default:
		return fmt.Sprint(v)
	}
}

func truthy(v any) bool {
	switch b := v.(type) {
	case nil:
		return false
	case bool:
		return b
	case string:
		return b != ""
	default:
		return true
	}
}
