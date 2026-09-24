package scripts_test

// The author-trust law for AI flows (#1813).
//
// Every AI flow (bot delivery implement/verify/correct; interactive @claude; /blis-pr-review) can
// be pointed at a PR or issue authored by someone WITHOUT write access. On such an item the
// untrusted content is the PR/issue BODY and, on a PR, the DIFF — and reading it is the whole task,
// so #1806's comment-text filter does not help. deliver-author-gate.sh is the container-level gate:
// it answers "may an agent run on content authored by this login?", so the four workflows can reach
// one decision instead of four re-implementations.
//
// The law is EXERCISED here, not reasoned about, because it sits on a public-repo trust boundary and
// both failure directions cost real work: over-trusting hands the agent — running on a persistent
// self-hosted runner with credentials — a stranger's body/diff; under-trusting refuses a
// maintainer. The offline half drives the decision through the script's `--permission` seam; the
// live half resolves a login against a `gh` stub on PATH, the same shape sibling
// deliver_issue_refinements_test.go uses.

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// gate runs deliver-author-gate.sh with the given args and returns (stdout+stderr, exit code). The
// PATH is passed through so a `gh` stub dir (when supplied) takes precedence.
func gate(t *testing.T, path string, args ...string) (string, int) {
	t.Helper()
	cmd := exec.Command("bash", append([]string{scriptPath(t, "deliver-author-gate.sh")}, args...)...)
	env := []string{"PATH=" + path, "GH_REPO=owner/repo"}
	cmd.Env = env
	var buf strings.Builder
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	err := cmd.Run()
	code := 0
	var exitErr *exec.ExitError
	if err != nil {
		if !errors.As(err, &exitErr) {
			t.Fatalf("running gate: %v\n%s", err, buf.String())
		}
		code = exitErr.ExitCode()
	}
	return buf.String(), code
}

// gateStubGh writes a fake `gh` whose `api …/permission` branch runs permissionScript, and returns a
// PATH with that dir first. A test that expects NO network passes a script that fails loudly if
// called.
func gateStubGh(t *testing.T, permissionScript string) string {
	t.Helper()
	dir := t.TempDir()
	script := `#!/usr/bin/env bash
set -uo pipefail
case "${1:-}" in
  "api")
` + permissionScript + `
    ;;
esac
echo "stub gh: unexpected invocation: $*" >&2
exit 1
`
	if err := os.WriteFile(filepath.Join(dir, "gh"), []byte(script), 0o700); err != nil {
		t.Fatalf("writing gh stub: %v", err)
	}
	return dir + string(os.PathListSeparator) + os.Getenv("PATH")
}

// ── The decision, offline via the --permission seam ─────────────────────────────────────────────

func TestAuthorGate_AllowsWriteAccessLevels(t *testing.T) {
	// A write-access author is who the flows exist to serve; all three levels must pass identically.
	for _, perm := range []string{"admin", "write", "maintain"} {
		t.Run(perm, func(t *testing.T) {
			out, code := gate(t, os.Getenv("PATH"), "--permission", perm, "someone")
			if code != 0 {
				t.Fatalf("permission=%q: exit %d, want 0\n%s", perm, code, out)
			}
			if !strings.Contains(out, "allowed") {
				t.Errorf("permission=%q: stdout does not say allowed\n%s", perm, out)
			}
		})
	}
}

func TestAuthorGate_BlocksReadAccessAndNone(t *testing.T) {
	// A genuine non-collaborator on a PUBLIC repo returns 200 with `read` (verified on this repo by
	// the sibling script) — the common untrusted case. `none` is the other definitive negative.
	// Both must be a DENIAL (exit 1), not a read failure (exit 3): the gate answered.
	for _, perm := range []string{"read", "none", "triage"} {
		t.Run(perm, func(t *testing.T) {
			out, code := gate(t, os.Getenv("PATH"), "--permission", perm, "outsider")
			if code != 1 {
				t.Fatalf("permission=%q: exit %d, want 1 (blocked)\n%s", perm, code, out)
			}
			if !strings.Contains(out, "blocked") {
				t.Errorf("permission=%q: stdout does not say blocked\n%s", perm, out)
			}
		})
	}
}

func TestAuthorGate_TrustsKnownBotsWithoutNetwork(t *testing.T) {
	// The permission API 404s for a bot login, so probing would DENY the delivery loop's own PRs.
	// The bots must be trusted BEFORE any probe — asserted by a stub that fails if queried.
	stub := gateStubGh(t, `    echo "stub gh: a bot author must not be probed" >&2; exit 1`)
	for _, bot := range []string{"github-actions[bot]", "claude[bot]"} {
		t.Run(bot, func(t *testing.T) {
			out, code := gate(t, stub, bot)
			if code != 0 {
				t.Fatalf("bot=%q: exit %d, want 0\n%s", bot, code, out)
			}
			if !strings.Contains(out, "allowed") {
				t.Errorf("bot=%q: stdout does not say allowed\n%s", bot, out)
			}
		})
	}
}

func TestAuthorGate_UntrustedBotIsBlockedNotTrusted(t *testing.T) {
	// A bot-shaped login that is NOT on the allowlist (a third-party App on a fork PR) must not get
	// a free pass off its `[bot]` suffix. It is a definitive denial (exit 1) WITHOUT a probe — an
	// App is not a collaborator, and this keeps the `[ ]` of a bot login out of a probe URL. The
	// stub fails loudly if queried, proving no probe happened.
	stub := gateStubGh(t, `    echo "stub gh: a non-allowlisted bot must not be probed" >&2; exit 1`)
	out, code := gate(t, stub, "dependabot[bot]")
	if code != 1 {
		t.Fatalf("exit %d, want 1 (blocked)\n%s", code, out)
	}
	if !strings.Contains(out, "blocked") {
		t.Errorf("stdout does not say blocked\n%s", out)
	}
}

// ── The live path, against a gh stub ────────────────────────────────────────────────────────────

func TestAuthorGate_LiveWriteAccessAllowed(t *testing.T) {
	stub := gateStubGh(t, `    echo "write"; exit 0`)
	out, code := gate(t, stub, "maintainer")
	if code != 0 {
		t.Fatalf("exit %d, want 0\n%s", code, out)
	}
	if !strings.Contains(out, "allowed") {
		t.Errorf("stdout does not say allowed\n%s", out)
	}
}

func TestAuthorGate_LiveNonCollaborator404IsDenial(t *testing.T) {
	// A clean 404 is a definitive "not a collaborator", not a probe failure: DENY (exit 1), do not
	// escalate. Distinguishing the two is the mistake the sibling script's comment records.
	stub := gateStubGh(t, `    echo "HTTP 404: Not Found" >&2; exit 1`)
	out, code := gate(t, stub, "outsider")
	if code != 1 {
		t.Fatalf("exit %d, want 1 (blocked, definitive)\n%s", code, out)
	}
	if !strings.Contains(out, "blocked") {
		t.Errorf("stdout does not say blocked\n%s", out)
	}
}

func TestAuthorGate_LiveProbeFailureFailsClosedLoud(t *testing.T) {
	// A 5xx / 403 / network error is NOT an answer. The agent must not run (fail closed), but the
	// outcome must be reported as a probe failure (exit 3, read-failed marker), never a silent drop
	// (R1) and never mistaken for a denial.
	for name, script := range map[string]string{
		"500":     `    echo "HTTP 500: Internal Server Error" >&2; exit 1`,
		"403":     `    echo "HTTP 403: Forbidden" >&2; exit 1`,
		"network": `    echo "could not resolve host" >&2; exit 1`,
	} {
		t.Run(name, func(t *testing.T) {
			stub := gateStubGh(t, script)
			out, code := gate(t, stub, "someone")
			if code != 3 {
				t.Fatalf("exit %d, want 3 (read-failed)\n%s", code, out)
			}
			if !strings.Contains(out, "AUTHOR-GATE-READ-FAILED") {
				t.Errorf("stdout lacks the read-failed marker\n%s", out)
			}
		})
	}
}

func TestAuthorGate_LiveDeadlineFailsClosed(t *testing.T) {
	// A stalled probe must be bounded and treated as a probe failure, not left to hang the job until
	// its 60/120-minute timeout. Force a short deadline against a stub that sleeps.
	stub := gateStubGh(t, `    exec sleep 30`)
	cmd := exec.Command("bash", scriptPath(t, "deliver-author-gate.sh"), "someone")
	cmd.Env = []string{"PATH=" + stub, "GH_REPO=owner/repo", "GH_DEADLINE_SECONDS=1", "GH_KILL_GRACE_SECONDS=1"}
	var buf strings.Builder
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	err := cmd.Run()
	code := 0
	var exitErr *exec.ExitError
	if err != nil {
		if !errors.As(err, &exitErr) {
			t.Fatalf("running gate: %v\n%s", err, buf.String())
		}
		code = exitErr.ExitCode()
	}
	if code != 3 {
		t.Fatalf("exit %d, want 3 (read-failed on deadline)\n%s", code, buf.String())
	}
}

// ── Input validation ────────────────────────────────────────────────────────────────────────────

func TestAuthorGate_EmptyOrMissingLoginFailsClosed(t *testing.T) {
	// No author to weigh is not "allowed" — it is a read failure the caller must surface. An empty
	// login must never fall through to a probe of `repos/…/collaborators//permission`.
	for _, args := range [][]string{{}, {""}, {"--permission", "write", ""}} {
		out, code := gate(t, os.Getenv("PATH"), args...)
		if code != 3 && code != 2 { // 3 read-failed, or 2 usage for the no-arg form
			t.Errorf("args=%v: exit %d, want 3 (read-failed) or 2 (usage)\n%s", args, code, out)
		}
	}
}

func TestAuthorGate_MalformedLoginRejected(t *testing.T) {
	// A login carrying shell metacharacters or path separators must be refused before it reaches a
	// URL or the shell, so a crafted value can neither be interpolated nor traverse.
	for _, bad := range []string{"a/b", "a;rm -rf", "a b", "../x", "a$(id)"} {
		stub := gateStubGh(t, `    echo "stub gh: a malformed login must not be probed" >&2; exit 1`)
		out, code := gate(t, stub, bad)
		if code != 3 {
			t.Errorf("login=%q: exit %d, want 3 (read-failed)\n%s", bad, code, out)
		}
	}
}
