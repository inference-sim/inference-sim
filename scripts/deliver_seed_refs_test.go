package scripts_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// seedRefs runs scripts/deliver-seed-refs.sh over a body and returns the three values it prints.
func seedRefs(t *testing.T, body string) (targetBranch, archonPlan, headingSeen, planSeen, unclosedFence string) {
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
		case "heading_seen":
			headingSeen = value
		case "plan_seen":
			planSeen = value
		case "unclosed_fence":
			unclosedFence = value
		default:
			t.Fatalf("unexpected key %q in output:\n%s", key, out)
		}
	}
	for name, v := range map[string]string{"heading_seen": headingSeen, "plan_seen": planSeen, "unclosed_fence": unclosedFence} {
		if v != "true" && v != "false" {
			t.Fatalf("%s = %q, want \"true\" or \"false\"; the caller branches on these to decide "+
				"whether to warn, and an empty value would silently disable that warning", name, v)
		}
	}
	return targetBranch, archonPlan, headingSeen, planSeen, unclosedFence
}

type seedRefsCase struct {
	name         string
	body         string
	wantBranch   string
	wantPlan     string
	wantHeading  string
	wantPlan2    string // plan_seen
	wantUnclosed string // unclosed_fence; "" is treated as "false"
}

// The two shapes documented in docs/contributing/templates/archon-issue-examples.md, plus the
// standalone issue that has neither section. deliver-implement.yml bases the delivery branch on
// target_branch and seeds archon_plan into the draft PR body, so a wrong answer here either fills
// the PR diff with unrelated commits or silently disables the dist ratchet.
func TestDeliverSeedRefs(t *testing.T) {
	cases := []seedRefsCase{
		{
			name: "hole sub-issue: section holds one ref",
			body: "## Dependencies\n\nNone\n\n" +
				"## Target branch\n\n" +
				"`feature/kv-offload` (PR against the feature branch, NOT main)\n\n" +
				"## Review\n\nSelf-reviewed\n\n---\n\n" +
				"archon-plan: specs/008-kv/kv.plan.json\n",
			wantBranch:  "feature/kv-offload",
			wantPlan:    "archon-plan: specs/008-kv/kv.plan.json",
			wantHeading: "true",
			wantPlan2:   "true",
		},
		{
			// The arrow form names the eventual feature->main base second. Taking the LAST ref
			// would base a hole delivery on main and put the whole feature branch in its diff.
			name: "integration issue: arrow form takes the FIRST ref",
			body: "## Target branch\n\n`feature/kv-offload` → `main`\n\n" +
				"## Review\n\nMaintainer reviews\n\n---\n\n" +
				"archon-plan: specs/008-kv/kv.plan.json\n",
			wantBranch:  "feature/kv-offload",
			wantPlan:    "archon-plan: specs/008-kv/kv.plan.json",
			wantHeading: "true",
			wantPlan2:   "true",
		},
		{
			// The common case in this repository: a standalone hardening/bug issue. Neither value
			// is an error, and the workflow falls back to the default branch without warning.
			name:        "standalone issue: neither section",
			body:        "**What problem does this solve?**\n\nThe thing is broken.\n",
			wantBranch:  "",
			wantPlan:    "",
			wantHeading: "false",
			wantPlan2:   "false",
		},
		{
			// A backticked ref under a LATER heading must not be mistaken for the target.
			name: "ref in a later section is not the target",
			body: "## Target branch\n\nNot stated.\n\n" +
				"## Notes\n\nSee `feature/unrelated` for context.\n",
			wantBranch:  "",
			wantPlan:    "",
			wantHeading: "true",
			wantPlan2:   "false",
		},
		{
			name:        "plan declared without any target branch section",
			body:        "Some prose.\n\narchon-plan: specs/009-x/x.plan.json\n",
			wantBranch:  "",
			wantPlan:    "archon-plan: specs/009-x/x.plan.json",
			wantHeading: "false",
			wantPlan2:   "true",
		},
		{
			// Matches archon-plan-resolve.sh, which requires a non-space after the colon. A bare
			// declaration seeded into a PR body would resolve to an empty path there.
			name:        "bare archon-plan: with no path is not a declaration",
			body:        "## Notes\n\narchon-plan:\n",
			wantBranch:  "",
			wantPlan:    "",
			wantHeading: "false",
			wantPlan2:   "false",
		},
		{
			// Prose mentioning the key mid-line is not a declaration; the pattern is anchored.
			name:        "archon-plan mentioned in prose is not a declaration",
			body:        "This PR has no archon-plan: it is standalone.\n",
			wantBranch:  "",
			wantPlan:    "",
			wantHeading: "false",
			wantPlan2:   "false",
		},
		{
			// A web-UI edit can leave CRLF. An unstripped \r rides inside the ref and makes every
			// `git ls-remote` lookup miss, silently falling back to the default branch.
			name:        "CRLF body yields a clean ref",
			body:        "## Target branch\r\n\r\n`feature/crlf`\r\n\r\n## Review\r\n\r\nx\r\n",
			wantBranch:  "feature/crlf",
			wantPlan:    "",
			wantHeading: "true",
			wantPlan2:   "false",
		},
		{
			// The list form the bold/list variants of these templates produce.
			name:        "list-prefixed plan declaration is still a declaration",
			body:        "- archon-plan: specs/010-y/y.plan.json\n",
			wantBranch:  "",
			wantPlan:    "- archon-plan: specs/010-y/y.plan.json",
			wantHeading: "false",
			wantPlan2:   "true",
		},
		{
			// Not a branch name. Passing it on would reach `git ls-remote` as two arguments.
			name:        "whitespace inside the backticks is refused",
			body:        "## Target branch\n\n`not a branch`\n",
			wantBranch:  "",
			wantPlan:    "",
			wantHeading: "true",
			wantPlan2:   "false",
		},

		// --- Fenced code blocks (review finding on #1723) ---
		//
		// docs/contributing/templates/archon-issue-examples.md presents the whole sub-issue
		// template inside a fence, `## Target branch` and a `feature/...` ref included. A
		// contributor pasting that example in alongside their real content must not have the
		// EXAMPLE win. A fictional placeholder would fail the remote check and fall back, but a
		// fenced REAL branch name would otherwise silently become the delivery's base.
		{
			name: "a fenced Target branch section is ignored",
			body: "## Target branch\n\n`feature/real`\n\n" +
				"## Notes\n\nThe template looks like this:\n\n" +
				"```\n## Target branch\n\n`feature/example`\n```\n",
			wantBranch:  "feature/real",
			wantPlan:    "",
			wantHeading: "true",
			wantPlan2:   "false",
		},
		{
			// Only a fenced section, so nothing is declared AND nothing is seen: both are computed
			// on the stripped body, so quoting the template is silent rather than warning. An
			// earlier revision computed them on the raw body, which made an illustrative fence
			// indistinguishable from a real declaration (#1723 review, F2).
			name: "a body whose ONLY Target branch section is fenced declares nothing and is not SEEN",
			body: "Here is the template:\n\n" +
				"```markdown\n## Target branch\n\n`feature/example`\n```\n",
			wantBranch:  "",
			wantPlan:    "",
			wantHeading: "false",
			wantPlan2:   "false",
		},
		{
			// Tilde fences are equally valid CommonMark, and are used for blocks that themselves
			// contain backticks.
			name:        "tilde fences are honoured too",
			body:        "~~~\n## Target branch\n\n`feature/example`\n~~~\n",
			wantBranch:  "",
			wantPlan:    "",
			wantHeading: "false",
			wantPlan2:   "false",
		},
		{
			// A fenced declaration must not be seeded into the PR body either: the dist ratchet
			// would then resolve a plan path the author only quoted as an example.
			// plan_seen is computed on the STRIPPED body, so a quoted example is NOT reported as a
			// declaration. It used to be, on the raw body, and the caller turned that into a hard
			// error that refused a perfectly legitimate delivery (#1723 review, F2).
			name:        "a fenced archon-plan declaration is ignored and NOT reported as seen",
			body:        "Example:\n\n```\narchon-plan: specs/000-example/x.plan.json\n```\n",
			wantBranch:  "",
			wantPlan:    "",
			wantHeading: "false",
			wantPlan2:   "false",
		},

		{
			// THE #1723 REVIEW REPRODUCTION. The old line-parity toggle flipped on this single
			// INDENTED ``` — which GitHub renders as literal text inside an indented code block, so
			// the author sees nothing wrong — and discarded everything after it: target_branch and
			// archon_plan both came back empty AND heading_seen was false, so no warning fired. Two
			// silent harms at once: the delivery based on the default branch, and the dist ratchet
			// skipped. A marker indented 4+ spaces is not a fence, so nothing is stripped now.
			name: "an INDENTED ``` is not a fence and must not swallow the body",
			body: "## Notes\n\nTo open a fence you write:\n\n    ```\n\n" +
				"## Target branch\n\n`feature/real`\n\n---\n\narchon-plan: specs/x.plan.json\n",
			wantBranch:  "feature/real",
			wantPlan:    "archon-plan: specs/x.plan.json",
			wantHeading: "true",
			wantPlan2:   "true",
		},

		{
			// F4 FROM THE #1723 REVIEW. Fenced lines are removed, so a fence spanning a `## `
			// boundary deletes that boundary; the section range then ran on into what the author
			// sees as a LATER section and a ref from there became the delivery's base, SILENTLY.
			// Reproduced: this body yielded `feature/WRONG`. Reading only the first non-blank line
			// of the section fixes it — "Not stated here." has no ref, so nothing is declared and
			// heading_seen makes the caller warn.
			name: "a fence spanning a ## boundary cannot promote a later ref to the base",
			body: "## Target branch\n\nNot stated here.\n\n```\n## Notes\n```\n\n" +
				"See `feature/WRONG` for context.\n",
			wantBranch:  "",
			wantPlan:    "",
			wantHeading: "true",
			wantPlan2:   "false",
		},
		{
			// The templates all put the ref on the first line of the section, so this is the shape
			// that must keep working after the narrowing above.
			name:        "ref on the first line of the section is still read",
			body:        "## Target branch\n\n`feature/first-line` (PR against the feature branch)\n\n## Review\n\nx\n",
			wantBranch:  "feature/first-line",
			wantPlan:    "",
			wantHeading: "true",
			wantPlan2:   "false",
		},

		{
			// #1723 REVIEW, list-item fence. CommonMark measures fence indent RELATIVE to the
			// containing block, so a fence quoted under a bullet is indented 4+ from the margin and
			// is still a real fence. The `ind <= 3` rule (added to fix the top-level indented-code
			// case) therefore stopped stripping it, and the EXAMPLE's ref and plan line leaked out:
			// `feature/EXAMPLE-IN-LIST` became the delivery's base with no warning.
			//
			// The two cases are irreconcilable in a line-oriented fence rule, so the indent
			// constraint lives on the heading and declaration patterns instead: a heading indented
			// 4+ is not a heading, and a line indented 4+ is not a declaration.
			name: "a fence quoted under a list item cannot promote its example to the base",
			body: "## Target branch\n\n`feature/real`\n\n## Notes\n\n- Template looks like:\n\n" +
				"    ```\n    ## Target branch\n\n    `feature/EXAMPLE-IN-LIST`\n\n" +
				"    archon-plan: specs/000-EXAMPLE/x.plan.json\n    ```\n",
			wantBranch:  "feature/real",
			wantPlan:    "",
			wantHeading: "true",
			wantPlan2:   "false",
		},
		{
			// The same rule, isolated: an indented declaration is not a declaration. Without this a
			// quoted plan path could be seeded into the PR body and resolved by the dist ratchet.
			name:        "an indented archon-plan line is not a declaration",
			body:        "Example:\n\n    archon-plan: specs/000-EXAMPLE/x.plan.json\n",
			wantBranch:  "",
			wantPlan:    "",
			wantHeading: "false",
			wantPlan2:   "false",
		},
		{
			// And an indented heading is not a heading, so it must not open a section either.
			name:        "an indented Target branch heading is not a heading",
			body:        "Example:\n\n    ## Target branch\n\n    `feature/EXAMPLE`\n",
			wantBranch:  "",
			wantPlan:    "",
			wantHeading: "false",
			wantPlan2:   "false",
		},

		// --- heading present but unreadable (review finding on #1723) ---
		{
			// The strict section pattern requires the heading stand alone on its line, so this
			// reads as "no section". heading_seen is what lets the caller say so out loud instead
			// of silently delivering onto the default branch.
			name:        "heading with trailing text is seen but yields no ref",
			body:        "## Target branch (base)\n\n`feature/x`\n",
			wantBranch:  "",
			wantPlan:    "",
			wantHeading: "true",
			wantPlan2:   "false",
		},
		{
			name:        "heading present with an unbackticked ref is seen but yields no ref",
			body:        "## Target branch\n\nfeature/x\n",
			wantBranch:  "",
			wantPlan:    "",
			wantHeading: "true",
			wantPlan2:   "false",
		},

		// --- case and fence edge cases (round-2 review findings on #1723) ---
		{
			// heading_seen is computed with `grep -i`, while extraction used a case-SENSITIVE sed —
			// so a lowercase heading warned and fell back to the default branch even though the
			// author had written a perfectly usable section. The two now agree.
			name:        "lowercase heading is extracted, not merely detected",
			body:        "## target branch\n\n`feature/lower`\n",
			wantBranch:  "feature/lower",
			wantPlan:    "",
			wantHeading: "true",
			wantPlan2:   "false",
		},
		{
			name:        "mixed-case heading is extracted too",
			body:        "## Target Branch\n\n`feature/mixed`\n",
			wantBranch:  "feature/mixed",
			wantPlan:    "",
			wantHeading: "true",
			wantPlan2:   "false",
		},
		{
			// A ``` fence is NOT closed by ~~~ — per CommonMark the closer must be the same
			// character — so this fence is unclosed and runs to end of document, swallowing the
			// section that follows. That is correct parsing, and it is also the residual harm no
			// parser can remove: the extraction yields nothing. What saves it is heading_seen, which
			// is computed on the unstripped body, stays true, and makes the caller WARN rather than
			// silently base on the default branch (#1723 review).
			name:         "an unclosed fence swallows the rest of the body and is REPORTED",
			body:         "```\nexample\n~~~\n\n## Target branch\n\n`feature/after-mismatch`\n",
			wantBranch:   "",
			wantPlan:     "",
			wantHeading:  "false",
			wantUnclosed: "true",
			wantPlan2:    "false",
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
		cases = append(cases, seedRefsCase{
			name:        "rejected ref: " + bad.name,
			body:        "## Target branch\n\n`" + bad.ref + "`\n",
			wantBranch:  "",
			wantPlan:    "",
			wantHeading: "true",
			wantPlan2:   "false",
		})
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			branch, plan, heading, planSeen, unclosed := seedRefs(t, tc.body)
			if branch != tc.wantBranch {
				t.Errorf("target_branch = %q, want %q", branch, tc.wantBranch)
			}
			if plan != tc.wantPlan {
				t.Errorf("archon_plan = %q, want %q", plan, tc.wantPlan)
			}
			if heading != tc.wantHeading {
				t.Errorf("heading_seen = %q, want %q", heading, tc.wantHeading)
			}
			if planSeen != tc.wantPlan2 {
				t.Errorf("plan_seen = %q, want %q", planSeen, tc.wantPlan2)
			}
			wantUnclosed := tc.wantUnclosed
			if wantUnclosed == "" {
				wantUnclosed = "false"
			}
			if unclosed != wantUnclosed {
				t.Errorf("unclosed_fence = %q, want %q", unclosed, wantUnclosed)
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

// The caller must actually consume heading_seen. The script computing it is useless if the workflow
// does not branch on it, and the whole point of the key is turning a silent fallback into a warning.
func TestDeliverImplementWarnsOnUnreadableTargetBranchHeading(t *testing.T) {
	path := filepath.Join("..", ".github", "workflows", "deliver-implement.yml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	body := string(raw)

	if !strings.Contains(body, "heading_seen=") {
		t.Error("deliver-implement.yml does not read `heading_seen` out of deliver-seed-refs.sh, " +
			"so a Target branch section it could not parse falls back to the default branch " +
			"silently — the delivery lands on the wrong base with nothing said (R1)")
	}
	if !strings.Contains(body, `"$heading_seen" == "true"`) {
		t.Error("deliver-implement.yml never branches on `heading_seen`, so the value is computed " +
			"and discarded and no warning is emitted for an unreadable Target branch heading")
	}
}

// The workflow must actually REFUSE to seed when the script reports an unclosed fence.
//
// The review noted that nothing connected the script's output to the workflow's behaviour — they
// live in different files, so a correct signal could be computed and then ignored. An unclosed fence
// means an unknown amount of the issue body was discarded, possibly including the target branch or
// the plan declaration, so seeding anyway risks the wrong base AND a silently absent dist ratchet.
func TestDeliverImplementRefusesToSeedOnAnUnclosedFence(t *testing.T) {
	path := filepath.Join("..", ".github", "workflows", "deliver-implement.yml")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	body := string(raw)

	if !strings.Contains(body, "unclosed_fence=") {
		t.Fatal("deliver-implement.yml does not read `unclosed_fence` from deliver-seed-refs.sh, so " +
			"a body whose tail was discarded by an unclosed fence would be seeded as if complete")
	}
	if !strings.Contains(body, `"$unclosed_fence" == "true"`) {
		t.Error("deliver-implement.yml never branches on `unclosed_fence`, so the signal is computed " +
			"and discarded")
	}
	// The refusal must be an error-and-exit, not a warning: we cannot know what was lost.
	idx := strings.Index(body, `"$unclosed_fence" == "true"`)
	if idx < 0 {
		return
	}
	window := body[idx:min(idx+700, len(body))]
	if !strings.Contains(window, "::error::") || !strings.Contains(window, "exit 1") {
		t.Error("the unclosed-fence branch does not `::error::` and `exit 1`. A warning is not enough: " +
			"an unclosed fence hides an unknown amount of the body, so the target branch and the " +
			"archon-plan line are both unreliable and the delivery must not be seeded")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// A failure of the fence scan must be loud, never an empty answer.
//
// This closes the second instance of one class. The first was a `$(mktemp)` whose failure made the
// script report a valid issue body as entirely empty at exit 0 — silencing every downstream guard and
// basing the delivery on the default branch with no plan line (#1723 review). Removing the temp file
// fixed that route; ANY failure of the awk stage reached the same place, because empty output reads as
// "closed fence, empty body". The trigger for both is the runner's known storage exhaustion.
//
// Exercised by putting a failing `awk` first on PATH, which is the cheapest faithful stand-in for
// "the fence scan did not run".
// Every text tool the script depends on must make it FAIL, never answer emptily.
//
// Two instances of this class were reported on #1723. A failing `awk` made a valid body report as
// entirely empty at exit 0. A failing `sed` made a body that DID declare `archon-plan:` report
// `plan_seen=false` — silently disabling the dist ratchet, which `deliver-verify.yml` then reads as
// `absent` (a PASS) rather than `unverified` (a BLOCK). `set -e` is deliberately off and a legitimate
// no-match is also a non-zero exit, so the script proves the tools usable up front instead; this
// pins that for each of them.
func TestDeliverSeedRefsFailsLoudlyWhenAToolIsBroken(t *testing.T) {
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("bash is not on PATH")
	}
	for _, tool := range []string{"sed", "grep", "awk", "tr", "head", "tail"} {
		t.Run(tool, func(t *testing.T) {
			dir := t.TempDir()
			bin := filepath.Join(dir, "bin")
			if err := os.MkdirAll(bin, 0o700); err != nil {
				t.Fatalf("creating stub bin: %v", err)
			}
			if err := os.WriteFile(filepath.Join(bin, tool), []byte("#!/bin/sh\nexit 127\n"), 0o700); err != nil {
				t.Fatalf("writing %s stub: %v", tool, err)
			}
			// A body that declares BOTH values, so a silent empty answer would lose both the base
			// and the plan gate.
			body := filepath.Join(dir, "body.md")
			content := "## Target branch\n\n`feature/real`\n\narchon-plan: specs/x.plan.json\n"
			if err := os.WriteFile(body, []byte(content), 0o600); err != nil {
				t.Fatalf("writing body: %v", err)
			}

			cmd := exec.Command("bash", scriptPath(t, "deliver-seed-refs.sh"), body)
			cmd.Env = append(os.Environ(), "PATH="+bin+string(os.PathListSeparator)+os.Getenv("PATH"))
			out, err := cmd.CombinedOutput()

			if err == nil {
				t.Fatalf("script exited 0 with a broken %s; output was:\n%s\n"+
					"An empty answer at exit 0 is indistinguishable from a body that declares nothing, "+
					"so the caller seeds the default branch and the dist ratchet reads `absent` (a pass) "+
					"instead of `unverified` (a block)", tool, out)
			}
			if strings.Contains(string(out), "target_branch=") {
				t.Errorf("script printed its key=value protocol despite %s being broken:\n%s", tool, out)
			}
		})
	}
}
