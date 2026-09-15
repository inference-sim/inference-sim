package scripts_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// seedRefs runs scripts/deliver-seed-refs.sh over a body and returns the two values it prints.
func seedRefs(t *testing.T, body string) (targetBranch, archonPlan string) {
	t.Helper()
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("bash is not on PATH")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "body.md")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("writing body: %v", err)
	}
	out, err := exec.Command("bash", scriptPath(t, "deliver-seed-refs.sh"), path).CombinedOutput()
	if err != nil {
		t.Fatalf("deliver-seed-refs.sh failed: %v\n%s", err, out)
	}
	for _, line := range strings.Split(strings.TrimRight(string(out), "\n"), "\n") {
		key, value, found := strings.Cut(line, "=")
		if !found {
			t.Fatalf("output line %q is not key=value; the caller parses this line-by-line", line)
		}
		switch key {
		case "target_branch":
			targetBranch = value
		case "archon_plan":
			archonPlan = value
		default:
			t.Fatalf("unexpected key %q in output:\n%s", key, out)
		}
	}
	return targetBranch, archonPlan
}

// The two shapes documented in docs/contributing/templates/archon-issue-examples.md, plus the
// standalone issue that has neither section. deliver-implement.yml bases the delivery branch on
// target_branch and seeds archon_plan into the draft PR body, so a wrong answer here either
// fills the PR diff with unrelated commits or silently disables the dist ratchet.
func TestDeliverSeedRefs(t *testing.T) {
	cases := []struct {
		name       string
		body       string
		wantBranch string
		wantPlan   string
	}{
		{
			name: "hole sub-issue: section holds one ref",
			body: "## Dependencies\n\nNone\n\n" +
				"## Target branch\n\n" +
				"`feature/kv-offload` (PR against the feature branch, NOT main)\n\n" +
				"## Review\n\nSelf-reviewed\n\n---\n\n" +
				"archon-plan: specs/008-kv/kv.plan.json\n",
			wantBranch: "feature/kv-offload",
			wantPlan:   "archon-plan: specs/008-kv/kv.plan.json",
		},
		{
			// The arrow form names the eventual feature->main base second. Taking the LAST ref
			// would base a hole delivery on main and put the whole feature branch in its diff.
			name: "integration issue: arrow form takes the FIRST ref",
			body: "## Target branch\n\n`feature/kv-offload` → `main`\n\n" +
				"## Review\n\nMaintainer reviews\n\n---\n\n" +
				"archon-plan: specs/008-kv/kv.plan.json\n",
			wantBranch: "feature/kv-offload",
			wantPlan:   "archon-plan: specs/008-kv/kv.plan.json",
		},
		{
			// The common case in this repository: a standalone hardening/bug issue. Neither
			// value is an error, and the workflow falls back to the default branch.
			name:       "standalone issue: neither section",
			body:       "**What problem does this solve?**\n\nThe thing is broken.\n",
			wantBranch: "",
			wantPlan:   "",
		},
		{
			// A backticked ref under a LATER heading must not be mistaken for the target.
			name: "ref in a later section is not the target",
			body: "## Target branch\n\nNot stated.\n\n" +
				"## Notes\n\nSee `feature/unrelated` for context.\n",
			wantBranch: "",
			wantPlan:   "",
		},
		{
			name:       "plan declared without any target branch section",
			body:       "Some prose.\n\narchon-plan: specs/009-x/x.plan.json\n",
			wantBranch: "",
			wantPlan:   "archon-plan: specs/009-x/x.plan.json",
		},
		{
			// Matches archon-plan-resolve.sh, which requires a non-space after the colon. A bare
			// declaration seeded into a PR body would resolve to an empty path there.
			name:       "bare archon-plan: with no path is not a declaration",
			body:       "## Notes\n\narchon-plan:\n",
			wantBranch: "",
			wantPlan:   "",
		},
		{
			// Prose mentioning the key mid-line is not a declaration; the pattern is anchored.
			name:       "archon-plan mentioned in prose is not a declaration",
			body:       "This PR has no archon-plan: it is standalone.\n",
			wantBranch: "",
			wantPlan:   "",
		},
		{
			// A web-UI edit can leave CRLF. An unstripped \r rides inside the ref and makes
			// every `git ls-remote` lookup miss, silently falling back to the default branch.
			name:       "CRLF body yields a clean ref",
			body:       "## Target branch\r\n\r\n`feature/crlf`\r\n\r\n## Review\r\n\r\nx\r\n",
			wantBranch: "feature/crlf",
			wantPlan:   "",
		},
		{
			// The list form the bold/list variants of these templates produce.
			name:       "list-prefixed plan declaration is still a declaration",
			body:       "- archon-plan: specs/010-y/y.plan.json\n",
			wantBranch: "",
			wantPlan:   "- archon-plan: specs/010-y/y.plan.json",
		},
		{
			// Not a branch name. Passing it on would reach `git ls-remote` as two arguments.
			name:       "whitespace inside the backticks is refused",
			body:       "## Target branch\n\n`not a branch`\n",
			wantBranch: "",
			wantPlan:   "",
		},
	}

	// An issue body is attacker-writable and the caller feeds this ref to `git ls-remote` and
	// `gh pr create --base`, so a ref that is not shaped like a branch name must come back empty
	// and let the caller fall back to the default branch.
	for _, bad := range []struct{ name, ref string }{
		{"leading dash reads as an option", "-upload-pack=x"},
		{"command substitution characters", "feature/$(id)"},
		{"semicolon", "feature/x;y"},
		{"backslash", `feature\x`},
		{"double dot has revision-range meaning", "feature/a..b"},
		{"colon is a refspec separator", "feature/a:b"},
		{"tilde is a revision suffix", "main~1"},
		{"caret is a revision suffix", "main^"},
		{"absolute-looking ref", "/etc/passwd"},
		{"trailing slash", "feature/x/"},
		{"lock suffix", "feature/x.lock"},
	} {
		cases = append(cases, struct {
			name       string
			body       string
			wantBranch string
			wantPlan   string
		}{
			name:       "rejected ref: " + bad.name,
			body:       "## Target branch\n\n`" + bad.ref + "`\n",
			wantBranch: "",
			wantPlan:   "",
		})
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			branch, plan := seedRefs(t, tc.body)
			if branch != tc.wantBranch {
				t.Errorf("target_branch = %q, want %q", branch, tc.wantBranch)
			}
			if plan != tc.wantPlan {
				t.Errorf("archon_plan = %q, want %q", plan, tc.wantPlan)
			}
		})
	}
}

// The seeded plan line must be resolvable by the script that actually consumes it. If the two
// patterns drift apart, a plan seeded into the PR body stops being the plan the dist ratchet
// reads — a silent skip, which deliver-verify.yml then treats as a declared-but-unverified plan.
func TestDeliverSeedRefsPlanPatternMatchesResolver(t *testing.T) {
	raw, err := os.ReadFile("archon-plan-resolve.sh")
	if err != nil {
		t.Fatalf("reading archon-plan-resolve.sh: %v", err)
	}
	const pattern = `'^[^A-Za-z0-9]*archon-plan:`
	if !strings.Contains(string(raw), pattern) {
		t.Fatalf("archon-plan-resolve.sh no longer greps %s; deliver-seed-refs.sh seeds the PR "+
			"body using that same pattern, so the two must be changed together", pattern)
	}

	seed, err := os.ReadFile("deliver-seed-refs.sh")
	if err != nil {
		t.Fatalf("reading deliver-seed-refs.sh: %v", err)
	}
	if !strings.Contains(string(seed), pattern) {
		t.Fatalf("deliver-seed-refs.sh no longer greps %s, so it can seed a declaration that "+
			"archon-plan-resolve.sh will not resolve", pattern)
	}
}
