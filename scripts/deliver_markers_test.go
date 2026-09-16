package scripts_test

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// deliver-methodology-markers.sh is the predicate deliver-verify.yml uses to decide whether a
// GREEN review actually ran the blis-pr-review methodology (its Q/A phase and findings table)
// or only improvised a verdict. A false "present" would let an improvised GREEN reach the gate
// and merge; a false "absent" would send a thorough review back for a needless correction
// round. Both directions are pinned here.

type markersOutcome struct {
	markers  string
	reason   string
	exitCode int
	stdout   string
	stderr   string
}

// runMarkers pipes body to the script on stdin, exactly as the workflow feeds it the winning
// verdict comment. The ambient environment is not inherited: the empty-input contract must be
// satisfied by stdin, not by some stray value.
func runMarkers(t *testing.T, body string) markersOutcome {
	t.Helper()

	cmd := exec.Command(scriptPath(t, "deliver-methodology-markers.sh"))
	cmd.Env = []string{"PATH=" + os.Getenv("PATH")}
	cmd.Stdin = strings.NewReader(body)

	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	var out markersOutcome
	err := cmd.Run()
	out.stdout = stdout.String()
	out.stderr = stderr.String()

	var exitErr *exec.ExitError
	switch {
	case err == nil:
	case errors.As(err, &exitErr):
		out.exitCode = exitErr.ExitCode()
	default:
		t.Fatalf("running deliver-methodology-markers.sh: %v", err)
	}

	for _, line := range strings.Split(out.stdout, "\n") {
		switch {
		case strings.HasPrefix(line, "markers="):
			out.markers = strings.TrimPrefix(line, "markers=")
		case strings.HasPrefix(line, "reason="):
			out.reason = strings.TrimPrefix(line, "reason=")
		}
	}
	return out
}

func readFixture(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("reading fixture %s: %v", name, err)
	}
	return string(b)
}

// The two fixtures are the real comments from the issue that motivated this check: #1723 round
// 4 ran the full methodology, #1725 skipped the Q/A phase and the findings table. If the
// predicate ever stops distinguishing these two actual reviews, the check is worthless.
func TestMarkers_RealReviews(t *testing.T) {
	if got := runMarkers(t, readFixture(t, "review_full_1723.md")); got.markers != "present" {
		t.Errorf("#1723 (full methodology): markers = %q, want present (stdout: %s)", got.markers, got.stdout)
	}
	if got := runMarkers(t, readFixture(t, "review_no_markers_1725.md")); got.markers != "absent" {
		t.Errorf("#1725 (no Q/A, no table): markers = %q, want absent (stdout: %s)", got.markers, got.stdout)
	}
}

// A minimal body that carries BOTH markers must pass, and dropping EITHER must fail — so the
// check requires the whole methodology, not just one visible half. Tags and heading are the
// only load-bearing tokens, so the fixtures are deliberately tiny.
func TestMarkers_RequiresBoth(t *testing.T) {
	const qa = "Q: is the boundary right?\nA: FLAW_FOUND — offload_chain.go:222 off by one\n"
	const table = "## Findings Summary\n\n| Finding | Location |\n| none | — |\n"

	cases := []struct {
		name string
		body string
		want string
	}{
		{"both present", qa + table, "present"},
		{"missing findings table", qa, "absent"},
		{"missing q/a phase", table, "absent"},
		{"neither", "just some prose with a DELIVER-VERDICT: GREEN line", "absent"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := runMarkers(t, tc.body)
			if got.exitCode != 0 {
				t.Fatalf("exit code = %d, want 0 (stderr: %s)", got.exitCode, got.stderr)
			}
			if got.markers != tc.want {
				t.Errorf("markers = %q, want %q (stdout: %s)", got.markers, tc.want, got.stdout)
			}
			if strings.TrimSpace(got.reason) == "" {
				t.Errorf("reason is empty; a verdict downgrade with no explanation is useless in a log")
			}
		})
	}
}

// Any of the three Q/A tags satisfies the Q/A half — a review with no FLAW_FOUND still ran the
// phase if it emitted CONFIDENT or CANNOT_ANSWER answers.
func TestMarkers_AnyQATag(t *testing.T) {
	const table = "### Findings Summary\n| Finding | Location |\n"
	for _, tag := range []string{"CONFIDENT", "FLAW_FOUND", "CANNOT_ANSWER"} {
		t.Run(tag, func(t *testing.T) {
			if got := runMarkers(t, "A: "+tag+" — see file.go:1\n"+table); got.markers != "present" {
				t.Errorf("tag %s: markers = %q, want present", tag, got.markers)
			}
		})
	}
}

// The findings table is accepted as a markdown heading OR a bold line — a real review uses
// either, and a false "absent" would send a thorough review through a needless correction
// round. Prose that merely mentions the phrase must NOT satisfy it.
func TestMarkers_FindingsHeadingShapes(t *testing.T) {
	const qa = "A: CONFIDENT — file.go:1\n"
	cases := []struct {
		name string
		line string
		want string
	}{
		{"markdown heading", "## Findings Summary", "present"},
		{"deep heading", "###### Findings Summary", "present"},
		{"bold line", "**Findings Summary**", "present"},
		{"prose mention only", "See the Findings Summary below.", "absent"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := runMarkers(t, qa+tc.line+"\n"); got.markers != tc.want {
				t.Errorf("%q: markers = %q, want %q", tc.line, got.markers, tc.want)
			}
		})
	}
}

// CRLF line endings (which the GitHub API can return) must not defeat detection: the trailing
// \r sits past the marker tokens, so a review with both markers still reads present.
func TestMarkers_CRLF(t *testing.T) {
	body := "A: FLAW_FOUND — file.go:1\r\n## Findings Summary\r\n| none | — |\r\n"
	if got := runMarkers(t, body); got.markers != "present" {
		t.Errorf("CRLF body: markers = %q, want present (stdout: %s)", got.markers, got.stdout)
	}
}

// Empty or whitespace-only stdin means the caller failed to capture the comment body. That is a
// wiring bug, not an absent-markers verdict: exit 2 so the workflow stops rather than silently
// downgrading a review it never actually read.
func TestMarkers_EmptyInputIsWiringError(t *testing.T) {
	for _, body := range []string{"", "   \n\t\n"} {
		got := runMarkers(t, body)
		if got.exitCode != 2 {
			t.Errorf("empty input %q: exit code = %d, want 2 (stdout: %s)", body, got.exitCode, got.stdout)
		}
		if got.markers != "" {
			t.Errorf("empty input %q: markers = %q, want no verdict emitted", body, got.markers)
		}
	}
}
