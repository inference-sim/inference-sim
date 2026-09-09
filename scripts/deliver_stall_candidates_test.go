package scripts_test

import (
	"errors"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"
)

// The stall sweep's selection filter decides which open PRs may be labelled `needs-human` and
// have their delivery halted. Selecting one PR too many stops a healthy delivery, so the filter
// is exercised here rather than reasoned about.
//
// Only the SELECTION is covered. The sweep's other decisions — the quiet-window comparison and
// the stand-down on a recently created phase run — are GitHub Actions and `gh` behaviour, and
// there is no harness in this repository that can evaluate a workflow expression or a live API
// response. Those remain covered by `actionlint` plus a live delivery.

func requireJq(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("jq"); err != nil {
		t.Skip("jq is not on PATH")
	}
}

// candidates runs the committed filter over a `gh pr list` payload and returns the selected PR
// numbers in the order emitted.
func candidates(t *testing.T, prListJSON string) []string {
	t.Helper()
	requireJq(t)

	cmd := exec.Command("jq", "-r", "-f", scriptPath(t, "deliver-stall-candidates.jq"))
	cmd.Stdin = strings.NewReader(prListJSON)
	cmd.Env = []string{"PATH=" + os.Getenv("PATH")}

	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			t.Fatalf("filter exited %d: %s", exitErr.ExitCode(), stderr.String())
		}
		t.Fatalf("running jq: %v", err)
	}

	var got []string
	for _, line := range strings.Split(strings.TrimSpace(stdout.String()), "\n") {
		if line == "" {
			continue
		}
		number, created, ok := strings.Cut(line, "\t")
		if !ok {
			t.Errorf("line %q is not <number>\\t<createdAt>; the sweep reads it as two "+
				"tab-separated fields", line)
			continue
		}
		if created == "" {
			t.Errorf("line %q carries no creation time; the sweep falls back to it for a PR "+
				"with no comments, which is exactly the never-got-started case", line)
		}
		got = append(got, number)
	}
	return got
}

// pr builds one `gh pr list` element.
func pr(number int, headRef string, labels ...string) string {
	var b strings.Builder
	b.WriteString(`{"number":`)
	b.WriteString(strconv.Itoa(number))
	b.WriteString(`,"headRefName":"`)
	b.WriteString(headRef)
	b.WriteString(`","createdAt":"2026-01-01T00:00:00Z","labels":[`)
	for i, l := range labels {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(`{"name":"`)
		b.WriteString(l)
		b.WriteString(`"}`)
	}
	b.WriteString(`]}`)
	return b.String()
}

func prList(elems ...string) string {
	return "[" + strings.Join(elems, ",") + "]"
}

func TestStallCandidatesSelectsOnlyDeliveriesInFlight(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []string
	}{
		{
			// The case the sweep exists for: a delivery that opened a PR and then went silent
			// without any phase reporting.
			name: "a delivery branch with no delivery labels at all is swept",
			in:   prList(pr(10, "deliver/issue-99")),
			want: []string{"10"},
		},
		{
			name: "a delivery mid-round is swept — a round label is not a verdict",
			in:   prList(pr(11, "deliver/issue-99", "deliver:round-2")),
			want: []string{"11"},
		},
		{
			// Paused is the operator's explicit hold. It goes quiet by design, so it crosses
			// every threshold; sweeping it would overrule the human who paused it.
			name: "a paused delivery is never swept",
			in:   prList(pr(12, "deliver/issue-99", "deliver:paused")),
			want: nil,
		},
		{
			name: "paused still wins when a round is in progress",
			in:   prList(pr(13, "deliver/issue-99", "deliver:round-1", "deliver:paused")),
			want: nil,
		},
		{
			name: "a delivery that reached ready-for-merge is not swept",
			in:   prList(pr(14, "deliver/issue-99", "ready-for-merge")),
			want: nil,
		},
		{
			name: "a delivery already stopped at needs-human is not swept",
			in:   prList(pr(15, "deliver/issue-99", "needs-human")),
			want: nil,
		},
		{
			name: "a non-delivery PR is not swept whatever it is labelled",
			in:   prList(pr(16, "feature/some-work", "deliver:round-1")),
			want: nil,
		},
		{
			// The ref test is anchored, so a branch that merely starts with the prefix is not a
			// delivery. This loop always names its branches exactly `deliver/issue-<N>`.
			name: "a near-miss branch name is not swept",
			in:   prList(pr(17, "deliver/issue-99-wip")),
			want: nil,
		},
		{
			name: "no open PRs at all selects nothing",
			in:   prList(),
			want: nil,
		},
		{
			name: "a mixed list selects exactly the deliveries in flight",
			in: prList(
				pr(20, "deliver/issue-1"),
				pr(21, "deliver/issue-2", "ready-for-merge"),
				pr(22, "deliver/issue-3", "deliver:paused"),
				pr(23, "deliver/issue-4", "deliver:round-3"),
				pr(24, "main-ish/branch"),
			),
			want: []string{"20", "23"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := candidates(t, tc.in)
			if len(got) != len(tc.want) {
				t.Fatalf("selected %v, want %v", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("selected %v, want %v", got, tc.want)
				}
			}
		})
	}
}

// A PR carrying no `labels` key at all must not kill the filter. `gh pr list --json labels`
// always emits the key, but the sweep exits non-zero if jq errors, which would silently disable
// stall detection for every delivery rather than for one PR.
func TestStallCandidatesToleratesAnEmptyLabelSet(t *testing.T) {
	got := candidates(t, `[{"number":30,"headRefName":"deliver/issue-7","createdAt":"2026-01-01T00:00:00Z","labels":[]}]`)
	if len(got) != 1 || got[0] != "30" {
		t.Fatalf("selected %v, want [30]", got)
	}
}
