package scripts_test

// Executes claude.yml's check-permissions script against a stubbed github, context and core,
// one case per row of its decision table (#1811). claude_permission_probe_test.go reads the
// script as text; this runs it, so a script that no longer parses, or that classifies a status
// differently, fails here.

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

const probeHarness = `const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const body = require('fs').readFileSync(process.argv[2], 'utf8');
const c = JSON.parse(process.argv[3]);
const out = {}, infos = [], warnings = [], failed = [], comments = [];
const core = { setOutput: (k, v) => { out[k] = v; }, info: (m) => infos.push(m),
  warning: (m) => warnings.push(m), setFailed: (m) => failed.push(m) };
const mk = (e) => { if (e === null) return null; const err = new Error(e.message ?? 'boom');
  if (e.status !== undefined) err.status = e.status; if (e.headers) err.response = { headers: e.headers }; return err; };
const github = { rest: {
  repos: { getCollaboratorPermissionLevel: async () => { const e = mk(c.probeError ?? null); if (e) throw e;
    return { data: { permission: c.permission } }; } },
  issues: { createComment: async (a) => { if (c.commentFails) throw new Error('comment refused'); comments.push(a.body); return {}; } } } };
const context = { actor: 'someone', repo: { owner: 'o', repo: 'r' }, issue: { number: 7 }, serverUrl: 'https://x', runId: 1,
  payload: { comment: { body: c.body ?? '' }, repository: c.private === undefined ? undefined : { private: c.private } } };
(async () => {
  await new AsyncFunction('github', 'context', 'core', body)(github, context, core);
  console.log(JSON.stringify({ out, infos, warnings, failed, comments }));
})().catch((e) => { console.log(JSON.stringify({ crashed: String(e) })); });
`

type probeResult struct {
	Out      map[string]string `json:"out"`
	Infos    []string          `json:"infos"`
	Warnings []string          `json:"warnings"`
	Failed   []string          `json:"failed"`
	Comments []string          `json:"comments"`
	Crashed  string            `json:"crashed"`
}

func TestClaudeCheckPermissions_Executes(t *testing.T) {
	node, err := exec.LookPath("node")
	if err != nil {
		if os.Getenv("REQUIRE_NODE") != "" {
			t.Fatalf("REQUIRE_NODE is set and node is not installed: %v", err)
		}
		t.Skip("node is not installed; CI runs this with a pinned Node (REQUIRE_NODE=1)")
	}
	var script string
	for name, body := range githubScriptBodies(t) {
		if strings.HasPrefix(name, "claude.yml:") && strings.Contains(body, "getCollaboratorPermissionLevel") {
			script = body
		}
	}
	if script == "" {
		t.Fatal("claude.yml has no github-script step that probes collaborator permission")
	}
	dir := t.TempDir()
	harness, body := filepath.Join(dir, "h.js"), filepath.Join(dir, "body.js")
	if err := os.WriteFile(harness, []byte(probeHarness), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(body, []byte(script), 0o644); err != nil {
		t.Fatal(err)
	}
	run := func(c map[string]any) probeResult {
		t.Helper()
		in, _ := json.Marshal(c)
		raw, err := exec.Command(node, harness, body, string(in)).CombinedOutput()
		if err != nil {
			t.Fatalf("node: %v\n%s", err, raw)
		}
		var r probeResult
		if err := json.Unmarshal([]byte(strings.TrimSpace(string(raw))), &r); err != nil {
			t.Fatalf("harness output %q: %v", raw, err)
		}
		if r.Crashed != "" {
			t.Fatalf("the script threw: %s", r.Crashed)
		}
		return r
	}
	cases := []struct {
		name        string
		in          map[string]any
		allowed     string
		failed      bool // the job is failed and the requester told: a probe failure, not a decision
		commentPost bool
	}{
		{"write access allows", map[string]any{"permission": "write"}, "true", false, false},
		{"admin allows", map[string]any{"permission": "admin"}, "true", false, false},
		{"read access denies, quietly", map[string]any{"permission": "read"}, "false", false, false},
		{"404 is the denial a public repo gives", map[string]any{"probeError": map[string]any{"status": 404}}, "false", false, false},
		{"a bare 403 on a public repo is an escalation", map[string]any{"probeError": map[string]any{"status": 403}, "private": false}, "false", true, true},
		{"a 403 with no repository in the payload is an escalation", map[string]any{"probeError": map[string]any{"status": 403}}, "false", true, true},
		{"a bare 403 on a private repo is a denial", map[string]any{"probeError": map[string]any{"status": 403}, "private": true}, "false", false, false},
		{"a throttled 403 is never a denial (retry-after)", map[string]any{"probeError": map[string]any{"status": 403, "headers": map[string]any{"retry-after": "30"}}, "private": true}, "false", true, true},
		{"a throttled 403 is never a denial (budget header)", map[string]any{"probeError": map[string]any{"status": 403, "headers": map[string]any{"x-ratelimit-remaining": "0"}}, "private": true}, "false", true, true},
		{"a rate-limit message is throttling", map[string]any{"probeError": map[string]any{"status": 403, "message": "You have exceeded a secondary rate limit"}, "private": true}, "false", true, true},
		{"429 is a probe failure", map[string]any{"probeError": map[string]any{"status": 429}}, "false", true, true},
		{"500 is a probe failure", map[string]any{"probeError": map[string]any{"status": 500}}, "false", true, true},
		{"a transport error is a probe failure", map[string]any{"probeError": map[string]any{"message": "socket hang up"}}, "false", true, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := run(tc.in)
			if r.Out["allowed"] != tc.allowed {
				t.Errorf("allowed = %q, want %q", r.Out["allowed"], tc.allowed)
			}
			if got := len(r.Failed) > 0; got != tc.failed {
				t.Errorf("job failed = %v (%v), want %v", got, r.Failed, tc.failed)
			}
			if got := len(r.Comments) > 0; got != tc.commentPost {
				t.Errorf("requester told = %v, want %v", got, tc.commentPost)
			}
		})
	}
	t.Run("a comment that cannot be posted still fails the job with the real reason", func(t *testing.T) {
		r := run(map[string]any{"probeError": map[string]any{"status": 500}, "commentFails": true})
		if len(r.Failed) != 1 || !strings.Contains(r.Failed[0], "HTTP 500") || len(r.Warnings) == 0 {
			t.Errorf("failed=%v warnings=%v: want one failure naming HTTP 500 and a warning about the comment", r.Failed, r.Warnings)
		}
	})
	t.Run("a transport failure's text stays out of the public comment", func(t *testing.T) {
		r := run(map[string]any{"probeError": map[string]any{"message": "connect ECONNRESET 10.0.0.1:443"}})
		if len(r.Comments) != 1 || strings.Contains(r.Comments[0], "ECONNRESET") || !strings.Contains(r.Comments[0], "no HTTP status") {
			t.Errorf("comment %v: want the bounded category only", r.Comments)
		}
		if len(r.Failed) != 1 || !strings.Contains(r.Failed[0], "ECONNRESET") {
			t.Errorf("failure annotation %v: want the precise text", r.Failed)
		}
	})
	t.Run("a body quoting the review command routes to review, in any case", func(t *testing.T) {
		if r := run(map[string]any{"permission": "write", "body": "please /BLIS-PR-REVIEW this"}); r.Out["is_review"] != "true" {
			t.Errorf("is_review = %q, want true", r.Out["is_review"])
		}
		if r := run(map[string]any{"permission": "write", "body": "@claude fix it"}); r.Out["is_review"] != "false" {
			t.Errorf("is_review = %q, want false", r.Out["is_review"])
		}
	})
}
