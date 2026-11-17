package scripts_test

// Executes the small inline github-script bodies of the workflows against a stubbed github,
// context and core, and asserts what each one outputs, posts or refuses (#1811).
// github_script_parse_test.go only parses them; claude_permission_probe_exec_test.go runs
// claude.yml's permission probe. Like those, it needs a pinned Node (REQUIRE_NODE=1 in CI).

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// execHarness runs argv[2] (a body) as actions/github-script does, with a github whose every
// rest.<group>.<method> call is recorded and answered from cfg.responses["group.method"] (or
// thrown, when the entry is {"throw": {...}}), a context from cfg.context and process.env from cfg.env.
const execHarness = `const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const body = require('fs').readFileSync(process.argv[2], 'utf8');
const cfg = JSON.parse(process.argv[3]);
Object.assign(process.env, cfg.env || {});
const out = {}, infos = [], warnings = [], failed = [], calls = [];
const core = { setOutput: (k, v) => { out[k] = v; }, info: (m) => infos.push(m),
  warning: (m) => warnings.push(m), setFailed: (m) => failed.push(m) };
const group = (g) => new Proxy({}, { get: (_, m) => async (args) => {
  const key = g + '.' + m; calls.push({ call: key, args });
  const r = (cfg.responses || {})[key];
  if (r && r.throw) { const e = new Error(r.throw.message || 'boom'); if (r.throw.status) e.status = r.throw.status; throw e; }
  return r === undefined ? { data: {} } : r;
} });
const github = { rest: new Proxy({}, { get: (_, g) => group(g) }) };
const context = Object.assign({ actor: 'someone', repo: { owner: 'o', repo: 'r' }, issue: { number: 7 },
  serverUrl: 'https://x', runId: 1, payload: {} }, cfg.context || {});
(async () => {
  await new AsyncFunction('github', 'context', 'core', body)(github, context, core);
  console.log(JSON.stringify({ out, infos, warnings, failed, calls }));
})().catch((e) => { console.log(JSON.stringify({ crashed: String(e) })); });
`

type execResult struct {
	Out      map[string]string `json:"out"`
	Infos    []string          `json:"infos"`
	Warnings []string          `json:"warnings"`
	Failed   []string          `json:"failed"`
	Calls    []struct {
		Call string         `json:"call"`
		Args map[string]any `json:"args"`
	} `json:"calls"`
	Crashed string `json:"crashed"`
}

// scriptRunner returns a function that runs the workflow script whose key contains want.
func scriptRunner(t *testing.T, want string) func(cfg map[string]any, subst ...string) execResult {
	t.Helper()
	node, err := exec.LookPath("node")
	if err != nil {
		if os.Getenv("REQUIRE_NODE") != "" {
			t.Fatalf("REQUIRE_NODE is set and node is not installed: %v", err)
		}
		t.Skip("node is not installed; CI runs this with a pinned Node (REQUIRE_NODE=1)")
	}
	var body string
	n := 0
	for name, b := range githubScriptBodies(t) {
		if strings.Contains(name, want) {
			body, n = b, n+1
		}
	}
	if n != 1 {
		t.Fatalf("%d github-script steps match %q, want exactly one", n, want)
	}
	dir := t.TempDir()
	harness := filepath.Join(dir, "h.js")
	if err := os.WriteFile(harness, []byte(execHarness), 0o644); err != nil {
		t.Fatal(err)
	}
	return func(cfg map[string]any, subst ...string) execResult {
		t.Helper()
		b := body
		// Actions substitutes ${{ expr }} before the script runs: subst is expr, value pairs.
		for i := 0; i+1 < len(subst); i += 2 {
			b = strings.ReplaceAll(b, "${{ "+subst[i]+" }}", subst[i+1])
		}
		bf := filepath.Join(dir, "body.js")
		if err := os.WriteFile(bf, []byte(b), 0o644); err != nil {
			t.Fatal(err)
		}
		in, _ := json.Marshal(cfg)
		raw, err := exec.Command(node, harness, bf, string(in)).CombinedOutput()
		if err != nil {
			t.Fatalf("node: %v\n%s", err, raw)
		}
		var r execResult
		if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &r); err != nil {
			t.Fatalf("harness output %q: %v", raw, err)
		}
		if r.Crashed != "" {
			t.Fatalf("the script threw: %s", r.Crashed)
		}
		return r
	}
}

func perm(p string) map[string]any {
	return map[string]any{"repos.getCollaboratorPermissionLevel": map[string]any{"data": map[string]any{"permission": p}}}
}

var denied = map[string]any{"repos.getCollaboratorPermissionLevel": map[string]any{"throw": map[string]any{"status": 404, "message": "Not Found"}}}

// Every workflow that lets a comment start work admits only a commenter with admin, write or
// maintain permission, and denies a reader and a non-collaborator.
func TestCommentGates_AdmitOnlyWriters(t *testing.T) {
	for _, step := range []string{"archon.yml:check-permissions/Check invoker", "deliver-implement.yml:check-permissions/Check invoker", "pr-review.yml:gate/Precise"} {
		t.Run(step, func(t *testing.T) {
			run := scriptRunner(t, step)
			ctx := map[string]any{"payload": map[string]any{"comment": map[string]any{"body": "/pr-review please"}}}
			for _, p := range []string{"admin", "write", "maintain"} {
				if r := run(map[string]any{"responses": perm(p), "context": ctx}); r.Out["allowed"] != "true" {
					t.Errorf("%s permission: allowed = %q, want true", p, r.Out["allowed"])
				}
			}
			for _, p := range []string{"read", "triage", "none"} {
				if r := run(map[string]any{"responses": perm(p), "context": ctx}); r.Out["allowed"] != "false" {
					t.Errorf("%s permission: allowed = %q, want false", p, r.Out["allowed"])
				}
			}
			if r := run(map[string]any{"responses": denied, "context": ctx}); r.Out["allowed"] != "false" {
				t.Errorf("a non-collaborator (404): allowed = %q, want false", r.Out["allowed"])
			}
		})
	}
}

// The delivery loop records who approved: the actor, only when allowed.
func TestDeliverImplementPermission_NamesTheApproverOnlyWhenAllowed(t *testing.T) {
	run := scriptRunner(t, "deliver-implement.yml:check-permissions/Check invoker")
	ctx := map[string]any{"actor": "maintainer-1"}
	if r := run(map[string]any{"responses": perm("write"), "context": ctx}); r.Out["approver"] != "maintainer-1" {
		t.Errorf("approver = %q, want maintainer-1", r.Out["approver"])
	}
	if r := run(map[string]any{"responses": perm("read"), "context": ctx}); r.Out["approver"] != "" {
		t.Errorf("a reader was named approver: %q", r.Out["approver"])
	}
}

// /pr-review is matched as a whole token: lookalike commands and a missing command do not start a
// review, whatever the commenter's permission.
func TestPRReviewGate_MatchesTheCommandAsAWholeToken(t *testing.T) {
	run := scriptRunner(t, "pr-review.yml:gate/Precise")
	for body, want := range map[string]string{
		"/pr-review":                    "true",
		"please /pr-review now":         "true",
		"line one\n/pr-review\nline 3":  "true",
		"/blis-pr-review":               "false",
		"/archon-pr-review":             "false",
		"/pr-review-extra":              "false",
		"no command here":               "false",
		"":                              "false",
		"prefix/pr-review":              "false",
		"text containing /pr-reviewing": "false",
	} {
		r := run(map[string]any{"responses": perm("admin"),
			"context": map[string]any{"payload": map[string]any{"comment": map[string]any{"body": body}}}})
		if r.Out["allowed"] != want {
			t.Errorf("comment %q: allowed = %q, want %q", body, r.Out["allowed"], want)
		}
		if want == "false" && len(r.Calls) != 0 {
			t.Errorf("comment %q: the permission probe ran for a comment that is not the command", body)
		}
	}
}

func TestArchonRef_AcceptsOnlyAPlainRef(t *testing.T) {
	run := scriptRunner(t, "archon.yml:archon-review/Parse archon ref")
	cases := []struct {
		body, ref string
		failed    bool
	}{
		{"/archon-review --archon-ref feature/x-1.2_y", "feature/x-1.2_y", false},
		{"/archon-review", "", false},
		{"/archon-review --archon-ref $(rm -rf)", "", true},
		{"/archon-review --archon-ref a;b", "", true},
	}
	for _, c := range cases {
		r := run(map[string]any{"context": map[string]any{"payload": map[string]any{"comment": map[string]any{"body": c.body}}}})
		if (len(r.Failed) > 0) != c.failed {
			t.Errorf("%q: failed = %v, want %v", c.body, r.Failed, c.failed)
		}
		if !c.failed && r.Out["ref"] != c.ref {
			t.Errorf("%q: ref = %q, want %q", c.body, r.Out["ref"], c.ref)
		}
		if c.failed {
			if _, set := r.Out["ref"]; set {
				t.Errorf("%q: an invalid ref was still output as %q", c.body, r.Out["ref"])
			}
		}
	}
}

func TestResolvePRRefs_OutputsTheBaseAndHeadOfTheCommentedPR(t *testing.T) {
	pr := map[string]any{"pulls.get": map[string]any{"data": map[string]any{
		"base": map[string]any{"sha": "b" + strings.Repeat("0", 39)}, "head": map[string]any{"sha": "h" + strings.Repeat("1", 39)}, "body": nil}}}
	for _, step := range []string{"archon.yml:archon-review/Resolve PR refs", "pr-review.yml:gate/Resolve PR refs"} {
		r := scriptRunner(t, step)(map[string]any{"responses": pr, "context": map[string]any{"issue": map[string]any{"number": 42}}})
		if r.Out["base_sha"] != "b"+strings.Repeat("0", 39) || r.Out["head_sha"] != "h"+strings.Repeat("1", 39) {
			t.Errorf("%s: outputs %v", step, r.Out)
		}
		if len(r.Calls) != 1 || r.Calls[0].Args["pull_number"] != float64(42) {
			t.Errorf("%s: asked for %+v, want pull 42", step, r.Calls)
		}
	}
	// archon also passes the PR body on, empty when the PR has none.
	r := scriptRunner(t, "archon.yml:archon-review/Resolve PR refs")(map[string]any{"responses": pr})
	if body, set := r.Out["body"]; !set || body != "" {
		t.Errorf("a PR with no body: output body = %q (set %v), want the empty string", body, set)
	}
}

func TestLocateThePR_TakesTheNewestOpenPR(t *testing.T) {
	run := scriptRunner(t, "deliver-implement.yml:deliver/Locate the PR")
	env := map[string]any{"BRANCH": "m3/deliver-9"}
	list := func(nums ...int) map[string]any {
		l := []map[string]any{}
		for _, n := range nums {
			l = append(l, map[string]any{"number": n})
		}
		return map[string]any{"pulls.list": map[string]any{"data": l}}
	}
	if r := run(map[string]any{"env": env, "responses": list(12, 40, 7)}); r.Out["number"] != "40" {
		t.Errorf("number = %q, want the newest, 40", r.Out["number"])
	}
	r := run(map[string]any{"env": env, "responses": list()})
	if n, set := r.Out["number"]; !set || n != "" {
		t.Errorf("no open PR: number = %q (set %v), want the empty string", n, set)
	}
	if len(r.Calls) != 1 || r.Calls[0].Args["head"] != "o:m3/deliver-9" || r.Calls[0].Args["state"] != "open" {
		t.Errorf("the lookup asked for %+v, want open PRs from o:m3/deliver-9", r.Calls)
	}
}

// A failed implement phase tells the issue what is observable about the branch, and never claims
// a state it did not observe.
func TestReportFailedImplement_StatesOnlyWhatWasObserved(t *testing.T) {
	run := scriptRunner(t, "deliver-implement.yml:deliver/Report a failed implement")
	env := map[string]any{"BRANCH": "m3/deliver-9", "ISSUE_NUMBER": "9"}
	say := func(pr, changed string) string {
		r := run(map[string]any{"env": env}, "steps.pr.outputs.number", pr, "steps.work.outputs.changed", changed)
		if len(r.Calls) != 1 || r.Calls[0].Call != "issues.createComment" || r.Calls[0].Args["issue_number"] != float64(9) {
			t.Fatalf("calls %+v, want one comment on issue 9", r.Calls)
		}
		return r.Calls[0].Args["body"].(string)
	}
	for _, c := range []struct{ pr, changed, has, hasNot string }{
		{"31", "true", "PR #31 is open", "carries no file changes"},
		{"", "false", "No PR is open", "HAS committed work"},
		{"", "", "could not be determined", "carries no file changes"},
	} {
		body := say(c.pr, c.changed)
		if !strings.Contains(body, c.has) || strings.Contains(body, c.hasNot) {
			t.Errorf("pr %q changed %q: body\n%s\nwant it to say %q and not %q", c.pr, c.changed, body, c.has, c.hasNot)
		}
	}
}

// The commit status published for a Claude run maps the job's result, and an unknown result is
// an error status with a warning, not a stale pending or a success.
func TestReportCommitStatus_MapsTheJobResult(t *testing.T) {
	run := scriptRunner(t, "claude.yml:report-status")
	for result, want := range map[string]string{"success": "success", "failure": "failure", "cancelled": "error", "skipped": "error", "": "error"} {
		r := run(map[string]any{"env": map[string]any{"CLAUDE_RESULT": result, "REVIEWED_SHA": strings.Repeat("a", 40), "REVIEWED_TREE": strings.Repeat("c", 40)}})
		if len(r.Calls) != 1 || r.Calls[0].Args["state"] != want || r.Calls[0].Args["sha"] != strings.Repeat("a", 40) {
			t.Errorf("result %q: calls %+v, want state %q on the reviewed sha", result, r.Calls, want)
		}
		if (want == "error" && result != "cancelled") != (len(r.Warnings) > 0) {
			t.Errorf("result %q: warnings %v", result, r.Warnings)
		}
	}
}
