package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
)

func has(t *testing.T, got []string, want string) {
	t.Helper()
	if !strings.Contains(strings.Join(got, "\n"), want) {
		t.Errorf("no finding contains %q; got:\n%s", want, strings.Join(got, "\n"))
	}
}

func none(t *testing.T, got []string) {
	t.Helper()
	if len(got) > 0 {
		t.Errorf("expected no findings, got:\n%s", strings.Join(got, "\n"))
	}
}

// X20. The good commit is the form the brief requires of agent-assisted work, once the
// X20: the commit-policy check covers the form an agent can get wrong, and nothing else.
// Signing and sign-off are encouraged, not required, so a commit with neither passes.
func TestCommitPolicy(t *testing.T) {
	good := Commit{Hash: strings.Repeat("a", 40), Author: "Ada Person <ada@example.com>",
		Message: "Add a check\n\nBody.\n\nAssisted-by: Claude Code (Claude Opus 5.5)\nSigned-off-by: Ada Person <ada@example.com>\n"}
	none(t, CommitPolicy(good))
	bot := Commit{Hash: "b", Author: "release bot <" + releaseBotEmail + ">",
		Committer: "release bot <" + releaseBotEmail + ">", Message: "Release v0.11.1\n"}
	none(t, CommitPolicy(bot))

	// What the design no longer requires: a sign-off, and a signature. A commit with an
	// Assisted-by line and no Signed-off-by passes, and so does one with neither trailer.
	none(t, CommitPolicy(Commit{Hash: "c", Author: "Ada Person <ada@example.com>", Message: "Add\n\nAssisted-by: Claude Code\n"}))
	none(t, CommitPolicy(Commit{Hash: "d", Author: "Ada Person <ada@example.com>", Message: "Add a thing\n"}))
	// A person's own sign-off, with or without an Assisted-by line, passes.
	none(t, CommitPolicy(Commit{Hash: "e", Author: "Ada Person <ada@example.com>", Message: "Add\n\nSigned-off-by: Ada Person <ada@example.com>\n"}))

	cases := []struct {
		name string
		edit func(*Commit)
		want string
	}{
		{"agent signs off", func(c *Commit) {
			c.Message = "Add\n\nAssisted-by: Claude Code\nSigned-off-by: Claude <noreply@anthropic.com>\n"
		}, "is not a person"},
		{"a sign-off that is no address", func(c *Commit) {
			c.Message = "Add\n\nSigned-off-by: Ada\n"
		}, "is not a person"},
		{"co-authored by a tool", func(c *Commit) {
			c.Message = "Add\n\nCo-authored-by: Claude Opus <noreply@anthropic.com>\nSigned-off-by: Ada Person <ada@example.com>\n"
		}, "names a tool"},
		{"two assisted-by", func(c *Commit) {
			c.Message = "Add\n\nAssisted-by: Claude Code\nAssisted-by: Copilot\nSigned-off-by: Ada Person <ada@example.com>\n"
		}, "2 Assisted-by lines"},
		{"an empty assisted-by", func(c *Commit) {
			c.Message = "Add\n\nAssisted-by:\n"
		}, "empty Assisted-by"},
		{"bot signs off", func(c *Commit) {
			c.Author = "release bot <" + releaseBotEmail + ">"
			c.Committer = "release bot <" + releaseBotEmail + ">"
			c.Message = "Release\n\nSigned-off-by: Ada Person <ada@example.com>\n"
		}, "the bot never signs off"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cm := good
			c.edit(&cm)
			has(t, CommitPolicy(cm), c.want)
		})
	}
}

const goodWorkflow = `name: ci
on:
  pull_request:
permissions:
  contents: read
jobs:
  go:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      - uses: ./.github/actions/go-toolchain
      - run: |
          set -euo pipefail
          # a comment that says || true is not code
          go test ./...
          pip install --require-hashes -r docs/requirements.txt
`

// X21, X22, X16 on workflow files.
func TestWorkflowLint(t *testing.T) {
	got, err := LintWorkflow("ci.yml", []byte(goodWorkflow))
	if err != nil {
		t.Fatal(err)
	}
	none(t, got)
	cases := []struct{ name, old, new, want string }{
		{"action by tag", "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", "actions/checkout@v7", "not pinned by a full commit hash"},
		{"action by short hash", "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", "actions/checkout@3d3c42e", "not pinned by a full commit hash"},
		{"docker by tag", "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", "docker://alpine:3", "not pinned by sha256 digest"},
		{"or true", "go test ./...", "go test ./... || true", "ignores an exit code"},
		{"or colon", "go test ./...", "go test ./... || :", "ignores an exit code"},
		{"or colon, trailing text", "go test ./...", "go test ./... ||: # best effort", "ignores an exit code"},
		{"or exit 0", "go test ./...", "go test ./... || exit 0", "ignores an exit code"},
		{"or subshell exit 0", "go test ./...", "go test ./... || (exit 0)", "ignores an exit code"},
		{"or return 0", "go test ./...", "go test ./... || return 0", "ignores an exit code"},
		{"or true with trailing text", "go test ./...", "go test ./... || true; echo done", "ignores an exit code"},
		{"semicolon true", "go test ./...", "go test ./...; true", "ignores an exit code"},
		{"semicolon true, trailing text", "go test ./...", "go test ./... ; true # keep going", "ignores an exit code"},
		{"set +e", "set -euo pipefail", "set +e", "ignores an exit code"},
		{"continue-on-error", "      - uses: ./.github/actions/go-toolchain\n", "      - uses: ./.github/actions/go-toolchain\n        continue-on-error: true\n", "continue-on-error"},
		{"pip without hashes", "pip install --require-hashes -r", "pip install -r", "without --require-hashes"},
		{"go test -short", "go test ./...", "go test -short ./...", "skips tests"},
		{"go test -skip", "go test ./...", "go test -skip TestSlow ./...", "skips tests"},
		{"skip env", "    runs-on: ubuntu-latest\n", "    runs-on: ubuntu-latest\n    env:\n      SKIP_SLOW: 1\n", "SKIP_SLOW turns a check off"},
		{"shell sh", "      - run: |", "      - shell: sh\n        run: |", "does not stop on the first failing command"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if !strings.Contains(goodWorkflow, c.old) {
				t.Fatalf("setup: %q", c.old)
			}
			got, err := LintWorkflow("ci.yml", []byte(strings.Replace(goodWorkflow, c.old, c.new, 1)))
			if err != nil {
				t.Fatal(err)
			}
			has(t, got, c.want)
		})
	}
	// Words that only begin like the ignored forms are not findings.
	for _, ok := range []string{"go test ./... || truefalse-tool", "go test ./... || exit 01", "echo a; trueblue"} {
		got, err := LintWorkflow("ci.yml", []byte(strings.Replace(goodWorkflow, "go test ./...", ok, 1)))
		if err != nil {
			t.Fatal(err)
		}
		none(t, got)
	}
	// An exemption names an invariant and a reason.
	ex := strings.Replace(goodWorkflow, "go test ./...", "go test ./... || true # lint-ok: X22 the step reports, a later step fails", 1)
	got, _ = LintWorkflow("ci.yml", []byte(ex))
	none(t, got)
}

// X16: a gate's pull_request paths equal its declared paths.
func TestGateDeclaresItsPaths(t *testing.T) {
	gate := "# gate-paths: registry/**, kernel/**\nname: gate\non:\n  pull_request:\n    paths: [registry/**, kernel/**]\njobs: {}\n"
	got, _ := LintWorkflow("gate-landing.yml", []byte(gate))
	none(t, got)
	got, _ = LintWorkflow("gate-landing.yml", []byte(strings.Replace(gate, "paths: [registry/**, kernel/**]", "paths: [registry/**]", 1)))
	has(t, got, "differ from the declared gate paths")
	got, _ = LintWorkflow("gate-landing.yml", []byte(strings.Replace(gate, "# gate-paths: registry/**, kernel/**\n", "", 1)))
	has(t, got, "declares its trigger paths")
}

// This repository passes its own checks, so a change that breaks one fails go test as well
// as CI.
func TestThisRepositoryPassesItsOwnChecks(t *testing.T) {
	cwd, _ := os.Getwd()
	defer os.Chdir(cwd)
	os.Chdir("../..")
	for _, c := range []string{"workflows", "versions", "docrefs", "quarantine"} {
		var out strings.Builder
		if err := run([]string{c}, &out); err != nil {
			t.Errorf("prcheck %s: %v\n%s", c, err, out.String())
		}
	}
}

// X14.
func TestDescriptionCheck(t *testing.T) {
	head, base := strings.Repeat("1", 40), strings.Repeat("2", 40)
	other := strings.Repeat("3", 40)
	heads := func(repo string, n int) (string, error) {
		if repo == "inference-sim/blis-substrate" && n == 1 {
			return other, nil
		}
		return "", fmt.Errorf("no such pull request")
	}
	in := DescriptionInput{
		Body: "## Depends on\n\nhttps://github.com/inference-sim/blis-substrate/pull/1 at " + other + "\n\nHead " + head + ".\n",
		Repo: "inference-sim/blis-substrate", Head: head, Base: base, Heads: heads, Merges: true,
	}
	none(t, CheckDescription(in))
	stale := in
	stale.Body = strings.Replace(in.Body, other, strings.Repeat("4", 40), 1)
	has(t, CheckDescription(stale), "but its head is "+other)
	short := in
	short.Body = "Depends on inference-sim/blis-substrate#1 at commit 6d79cc8.\n"
	has(t, CheckDescription(short), "is a short hash")
	stray := in
	stray.Body = "Built on " + strings.Repeat("5", 40) + ".\n"
	has(t, CheckDescription(stray), "is neither this pull request's head")
	conflict := in
	conflict.Merges = false
	has(t, CheckDescription(conflict), "does not merge cleanly")
	commented := in
	commented.Body = "<!-- at commit 6d79cc8 -->\n"
	none(t, CheckDescription(commented))
}

// X14: a pin written as a release tag resolves to the commit the tag points at, and a stale
// pin fails (blis-schemas#92). Known-bad inputs: a hash that is not the tag's commit, a tag
// that does not resolve, and a line with a tag but no hash for it.
func TestDescriptionChecksReleaseTagPins(t *testing.T) {
	head, base := strings.Repeat("1", 40), strings.Repeat("2", 40)
	tagged := strings.Repeat("a", 40)
	tags := func(repo, tag string) (string, error) {
		if repo == "inference-sim/blis-substrate" && tag == "v0.13.0" {
			return tagged, nil
		}
		if repo == "inference-sim/blis-vllm" && tag == "v0.32.0-blis.2" {
			return strings.Repeat("b", 40), nil
		}
		return "", fmt.Errorf("no such tag")
	}
	check := func(body string) []string {
		return CheckDescription(DescriptionInput{Body: body, Repo: "inference-sim/x", Head: head, Base: base,
			Tags: tags, Merges: true,
			Heads: func(string, int) (string, error) { return "", fmt.Errorf("no such pull request") }})
	}
	none(t, check("Pins inference-sim/blis-substrate v0.13.0 at "+tagged+".\n"))
	none(t, check("Pins inference-sim/blis-substrate@v0.13.0, commit "+tagged+".\n"))
	none(t, check("Pins https://github.com/inference-sim/blis-substrate/releases/tag/v0.13.0 at "+tagged+".\n"))
	none(t, check("Pins inference-sim/blis-vllm v0.32.0-blis.2 at "+strings.Repeat("b", 40)+".\n"))
	has(t, check("Pins inference-sim/blis-substrate v0.13.0 at "+strings.Repeat("c", 40)+".\n"), "but the tag points at "+tagged)
	has(t, check("Pins inference-sim/blis-substrate v9.9.9 at "+tagged+".\n"), "no such tag")
	has(t, check("Pins inference-sim/blis-substrate v0.13.0 and inference-sim/blis-vllm v0.32.0-blis.2 at "+tagged+".\n"), "2 release tags and 1 hashes")
	// A tag in prose with no hash needs no check, and a hash with no tag is the head rule.
	none(t, check("Pins inference-sim/blis-substrate v0.13.0.\n"))
	has(t, check("Built on "+strings.Repeat("5", 40)+".\n"), "is neither this pull request's head")
	// A pull request line keeps its own rule even when it mentions a tag.
	prLine := "inference-sim/blis-vllm#9 at " + strings.Repeat("7", 40) + ", built for v0.32.0\n"
	other := CheckDescription(DescriptionInput{Body: prLine, Repo: "inference-sim/x", Head: head, Base: base, Tags: tags, Merges: true,
		Heads: func(string, int) (string, error) { return strings.Repeat("7", 40), nil }})
	none(t, other)
}

// X28.
func TestCitedIssues(t *testing.T) {
	src := []byte(`// Refused: KV role kv_both is not modelled (inference-sim/blis-vllm#48).
// The gap is tracked in #7.
// See inference-sim/blis-vllm#3 for history.
x := 1 // unrelated #9
`)
	cites := Citations("inference-sim/blis-vllm", "core/a.go", src)
	if len(cites) != 2 || cites[0].Issue.N != 48 || cites[1].Issue.N != 7 {
		t.Fatalf("citations %+v; want #48 and #7, the lines that mark a refusal or gap", cites)
	}
	fields := []byte(`{"name":"kv_role","status":"not_modelled","evidence":{"issue":"inference-sim/blis-vllm#48"}}`)
	if c := Citations("inference-sim/blis-vllm", "fields.json", fields); len(c) != 1 || c[0].Issue.N != 48 {
		t.Fatalf("fields.json citations %+v", c)
	}
	if c := Citations("x/y", "a.go", []byte("// refused: llm-d/llm-d#5\n")); len(c) != 0 {
		t.Errorf("an issue outside BLIS is not covered: %+v", c)
	}
	state := func(i Issue) (bool, error) { return i.N != 7, nil }
	bad := CheckCitedOpen(cites, state)
	if len(bad) != 1 || !strings.Contains(bad[0], "cites inference-sim/blis-vllm#7, which is closed") {
		t.Fatalf("got %v", bad)
	}
	closes := ClosedBy("inference-sim/blis-vllm", "Closes #48\nCloses inference-sim/inference-sim#2\n<!-- Closes #7 -->\n")
	if len(closes) != 2 {
		t.Fatalf("closes %+v", closes)
	}
	has(t, CheckClosesUncited(closes, cites), "closes inference-sim/blis-vllm#48, which is still cited at core/a.go:1")
}

// X23.
func TestVersionLiterals(t *testing.T) {
	if f := VersionFindings("docs/install.md", []byte("Install Go 1.27 and blis-vllm v0.32.0-blis.2.\n")); len(f) != 2 {
		t.Fatalf("findings %v", f)
	}
	for _, p := range []string{"go.mod", "stack.json", "contracts/toolchain", "a/b_test.go", "x/testdata/go.mod"} {
		if f := VersionFindings(p, []byte("go 1.27.2\n")); len(f) != 0 {
			t.Errorf("%s is a source of versions or a test; got %v", p, f)
		}
	}
	if f := VersionFindings("docs/x.md", []byte("<!-- BEGIN GENERATED -->go1.27.2")); len(f) != 0 {
		t.Errorf("a generated file is exempt: %v", f)
	}
	// A record under specs/ is exempt (design 13.1); the same text outside it is a finding.
	if f := VersionFindings("specs/old/plan.md", []byte("Go 1.27")); len(f) != 0 {
		t.Errorf("a specs/ record is exempt: %v", f)
	}
	if f := VersionFindings("docs/plan.md", []byte("Go 1.27")); len(f) != 1 {
		t.Errorf("the same text outside specs/ is a finding: %v", f)
	}
	// A workflow reads the Go version from go.mod, so a literal one is a finding.
	wf := "      - uses: actions/setup-go@x\n        with:\n          go-version: '1.27'\n"
	has(t, VersionFindings(".github/workflows/ci.yml", []byte(wf)), "go-version: '1.27")
	none(t, VersionFindings(".github/workflows/ci.yml", []byte("          go-version-file: go.mod\n")))
	// No list of known findings hides one.
	has(t, CheckVersions([]string{"docs/a.md: Go 1.27"}), "is a version literal outside")
	none(t, CheckVersions(nil))
}

type fakeTree map[string]int

func (f fakeTree) Lines(p string) (int, bool) { n, ok := f[p]; return n, ok }

// X26.
func TestDocReferences(t *testing.T) {
	tree := fakeTree{"kernel": 0, "kernel/new.go": 100, "docs": 0, "docs/a.md": 5, "docs/b.md": 5}
	roots := map[string]bool{"kernel": true, "docs": true}
	good := "See `kernel/new.go:10-20` and [b](b.md).\n\n| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"
	none(t, DocFindings("docs/a.md", []byte(good), tree, roots))
	has(t, DocFindings("docs/a.md", []byte("See `kernel/old.go`.\n"), tree, roots), "kernel/old.go does not exist")
	has(t, DocFindings("docs/a.md", []byte("See `kernel/new.go:90-120`.\n"), tree, roots), "has 100 lines")
	has(t, DocFindings("docs/a.md", []byte("[c](c.md)\n"), tree, roots), "docs/c.md does not exist")
	has(t, DocFindings("docs/a.md", []byte("| x | y |\n|---|---|\n| 1 | 2 |\n| 1 | 2 |\n"), tree, roots), "is repeated")
	none(t, DocFindings("docs/a.md", []byte("```\n`kernel/old.go`\n```\nprose `config.json`\n"), tree, roots))
	none(t, DocFindings("docs/a.md", []byte("[m](catalog/reference/models/x.md) [t]([[catalog.tree]]/a.go)\n"), tree, roots))
}

// X14: an abbreviated hash is refused in any phrasing, and ordinary hex-looking text is not.
func TestShortHashPhrasingIndependent(t *testing.T) {
	check := func(body string) []string {
		return CheckDescription(DescriptionInput{Body: body, Repo: "o/r", Head: strings.Repeat("1", 40),
			Base: strings.Repeat("2", 40), Merges: true})
	}
	for name, body := range map[string]string{
		"after commit":     "Built at commit 6d79cc8.",
		"after rev":        "Rebased onto 6d79cc8 today.",
		"in parentheses":   "Fixed (6d79cc8).",
		"in backticks":     "See `6d79cc8` for the change.",
		"after a colon":    "Tip: 6d79cc8a1",
		"after sha":        "sha 6d79cc8",
		"line start":       "6d79cc8 is the tip",
		"line end":         "the tip is 6d79cc8",
		"long abbreviated": "tip 6d79cc8a1b2c3d4e5f60718293a4b5c6d7e8f90",
		"beside a PR ref":  "o/r#4 merged as 6d79cc8",
		"in a link text":   "[6d79cc8](https://example.com/x)",
		"after a url":      "https://example.com/p 6d79cc8",
		"all digits+a":     "tip 1234567a",
		"commit url":       "see https://github.com/o/r/commit/6d79cc8 for it",
		"blob url":         "https://github.com/o/r/blob/6d79cc8a1/x.go#L3",
		"tree url":         "https://github.com/o/r/tree/6d79cc8a1b",
		"commit url, end":  "(https://github.com/o/r/commit/6d79cc8)",
		"url all letters":  "https://github.com/o/r/commit/deadbee",
		"url all digits":   "https://github.com/o/r/commit/1234567",
		"url upper case":   "https://github.com/o/r/commit/DEADBEE1",
	} {
		has(t, check(body), "is a short hash")
		_ = name
	}
	for _, body := range []string{
		"defaced the decade facade; accede",
		"deadbeef is hex but no digit", // letters only: a word
		"On 20261105 we shipped",       // digits only: a date
		"Closes #1234567 and o/r#7654321",
		"go get example.com/m@v0.0.0-20261105123456-6d79cc8a1b2c",
		"example.com/m v1.2.3-pre.0.20261105123456-6d79cc8a1b2c",
		"digest sha256:6d79cc8a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f6071829",
		"see https://x.io/6d79cc8a and https://github.com/o/r/pull/12",
		"a full commit url https://github.com/o/r/commit/" + strings.Repeat("a", 40),
		"a branch https://github.com/o/r/tree/main and a tag https://github.com/o/r/blob/v1.2.3/x.go",
		"path internal/6d79cc8/file and file.6d79cc8.go",
		"full " + strings.Repeat("a", 40) + " is a different check",
		"too short 6d79cc",
		"in a longer run 6d79cc8a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2",
		"part of a word x6d79cc8y and g6d79cc8",
		"<!-- at commit 6d79cc8 -->",
	} {
		if bad := check(body); strings.Contains(strings.Join(bad, "\n"), "short hash") {
			t.Errorf("%q: wrongly flagged: %v", body, bad)
		}
	}
}

// X25.
func TestQuarantine(t *testing.T) {
	day := time.Date(2026, 11, 4, 0, 0, 0, 0, time.UTC)
	good := "- {test: TestA, package: ./kernel, issue: inference-sim/blis-substrate#9, added: 2026-11-01, expires: 2026-11-20, reason: flaky on arm64}\n"
	list, bad := CheckQuarantine([]byte(good), day, nil)
	none(t, bad)
	if p := GoSkipPattern(list, "./kernel"); p != "^TestA$" {
		t.Errorf("skip pattern %q", p)
	}
	for name, c := range map[string]struct{ doc, want string }{
		"expired":   {strings.Replace(good, "2026-11-20", "2026-11-04", 1), "expired on 2026-11-04"},
		"no issue":  {strings.Replace(good, "inference-sim/blis-substrate#9", "none", 1), "every entry has an owning issue"},
		"bad date":  {strings.Replace(good, "2026-11-20", "soon", 1), "is not a date"},
		"unknown":   {strings.Replace(good, "reason:", "owner: me, reason:", 1), "field owner not found"},
		"no reason": {strings.Replace(good, "flaky on arm64", "' '", 1), "needs test, package and reason"},
	} {
		if _, bad := CheckQuarantine([]byte(c.doc), day, nil); !strings.Contains(strings.Join(bad, "\n"), c.want) {
			t.Errorf("%s: want %q, got %v", name, c.want, bad)
		}
	}
	// An added day later than the entry's first appearance in git history is refused.
	_, bad = CheckQuarantine([]byte(good), day, map[string]time.Time{"TestA": day.AddDate(0, 0, -5)})
	has(t, bad, "when the entry first appeared")
	_, bad = CheckQuarantine([]byte(good), day, map[string]time.Time{"TestA": day.AddDate(0, 0, -3)})
	none(t, bad)
}

// X25: the cap holds through the same function the subcommands call, with no history
// supplied: a far expiry, a future added day, and a missing added day are each refused.
func TestQuarantineCapReachesTheCLI(t *testing.T) {
	dir := t.TempDir()
	entry := func(added, expires string) string {
		a := ""
		if added != "" {
			a = "  added: " + added + "\n"
		}
		return "- test: TestA\n  package: ./kernel\n  issue: inference-sim/blis-substrate#9\n" + a +
			"  expires: " + expires + "\n  reason: flaky\n"
	}
	for name, c := range map[string]struct{ doc, want string }{
		"expires 2099":    {entry("2026-11-01", "2099-01-01"), "more than 30 days after it was added"},
		"added in future": {entry("2026-11-30", "2026-12-10"), "is in the future"},
		"missing added":   {entry("", "2026-11-20"), "added is required"},
		"bad added":       {entry("yesterday", "2026-11-20"), "is not a date"},
	} {
		file := filepath.Join(dir, strings.ReplaceAll(name, " ", "-")+".yaml")
		if err := os.WriteFile(file, []byte(c.doc), 0o644); err != nil {
			t.Fatal(err)
		}
		var out strings.Builder
		err := run([]string{"quarantine", "-list", file, "-today", "2026-11-06"}, &out)
		if !errors.Is(err, errFindings) || !strings.Contains(out.String(), c.want) {
			t.Errorf("%s: want a finding naming %q, got %v\n%s", name, c.want, err, out.String())
		}
		// The skips subcommand reads the same list the same way.
		res := filepath.Join(dir, "results.json")
		if err := os.WriteFile(res, []byte(`{"Action":"pass","Package":"p","Test":"T"}`+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		out.Reset()
		err = run([]string{"skips", "-results", res, "-list", file, "-min", "0"}, &out)
		if name != "added in future" && (!errors.Is(err, errFindings) || !strings.Contains(out.String(), c.want)) {
			t.Errorf("skips, %s: want a finding naming %q, got %v\n%s", name, c.want, err, out.String())
		}
	}
	ok := filepath.Join(dir, "ok.yaml")
	if err := os.WriteFile(ok, []byte(entry("2026-11-01", "2026-11-20")), 0o644); err != nil {
		t.Fatal(err)
	}
	var out strings.Builder
	if err := run([]string{"quarantine", "-list", ok, "-today", "2026-11-06"}, &out); err != nil {
		t.Errorf("a well-formed entry was refused: %v\n%s", err, out.String())
	}
}

// X16, X25: a pytest -v run names each skipped test, so a quarantine entry covers that test
// and not its file's others; without -v the summary lines name only the file.
func TestPytestSkips(t *testing.T) {
	verbose := []byte("validator/test_c.py::test_ll SKIPPED (cited bodies absent)    [ 10%]\n" +
		"validator/test_c.py::test_gb200 SKIPPED (no data)    [ 20%]\n" +
		"validator/test_c.py::test_p[a b] SKIPPED (param)    [ 30%]\n" +
		"validator/test_c.py::test_ok PASSED    [ 40%]\n" +
		"SKIPPED [1] validator/test_c.py:12: cited bodies absent\n" +
		"SKIPPED [1] validator/test_c.py:20: no data\n" +
		"SKIPPED [1] validator/test_c.py:30: param\n")
	skips, _, err := ParseSkips(verbose, "registry")
	if err != nil {
		t.Fatal(err)
	}
	want := []SkipResult{{"registry/validator/test_c.py", "test_ll"}, {"registry/validator/test_c.py", "test_gb200"}, {"registry/validator/test_c.py", "test_p[a b]"}}
	if fmt.Sprint(skips) != fmt.Sprint(want) {
		t.Fatalf("skips %v, want %v", skips, want)
	}
	list := []QuarantineEntry{{Test: "test_ll", Package: "registry/validator/test_c.py"}, {Test: "test_p", Package: "registry/validator/test_c.py"}}
	bad := CheckSkips(skips, list)
	if len(bad) != 1 || !strings.Contains(bad[0], "test_gb200 was skipped") {
		t.Errorf("an entry for one test must not cover its file's others: %v", bad)
	}
	// A module skipped at import has a summary line and no test line; it is still a skip.
	mod, _, _ := ParseSkips(append(verbose, "SKIPPED [1] validator/test_m.py:2: whole module\n"...), "registry")
	if len(mod) != 4 || mod[3] != (SkipResult{"registry/validator/test_m.py", "(module)"}) {
		t.Errorf("module skip: %v", mod)
	}
	summary, _, err := ParseSkips([]byte("SKIPPED [1] validator/test_c.py:12: cited bodies absent\n"), "registry")
	if err != nil || len(summary) != 1 || summary[0] != (SkipResult{"registry", "validator/test_c.py"}) {
		t.Errorf("summary-only output: %v %v", summary, err)
	}
}

// The commits subcommand, against a local git repository.
func TestCommitsAgainstALocalRepository(t *testing.T) {
	dir := t.TempDir()
	g := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		cmd.Env = append(os.Environ(), "GIT_AUTHOR_NAME=Ada Person", "GIT_AUTHOR_EMAIL=ada@example.com",
			"GIT_COMMITTER_NAME=Ada Person", "GIT_COMMITTER_EMAIL=ada@example.com", "GIT_CONFIG_GLOBAL=/dev/null", "GIT_CONFIG_SYSTEM=/dev/null")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	g("init", "-q", "-b", "main")
	os.WriteFile(filepath.Join(dir, "a"), []byte("a\n"), 0o644)
	g("add", "a")
	g("commit", "-q", "-m", "base")
	base := g("rev-parse", "HEAD")
	os.WriteFile(filepath.Join(dir, "a"), []byte("b\n"), 0o644)
	g("commit", "-q", "-am", "Change a\n\nAssisted-by: Claude Code\nSigned-off-by: Ada Person <ada@example.com>")
	head := g("rev-parse", "HEAD")
	cwd, _ := os.Getwd()
	defer os.Chdir(cwd)
	os.Chdir(dir)
	// An unsigned commit with a person's sign-off and an Assisted-by line is clean: signing
	// is encouraged, not checked (X20).
	var out strings.Builder
	if err := run([]string{"commits", "-base", base, "-head", head}, &out); err != nil {
		t.Fatalf("%v\n%s", err, out.String())
	}
	if !strings.Contains(out.String(), "1 checked") {
		t.Errorf("output %q", out.String())
	}
	// Known-bad: the same change with a tool as the sign-off.
	if err := os.WriteFile(filepath.Join(dir, "a"), []byte("c\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	g("commit", "-q", "-am", "Change a again\n\nAssisted-by: Claude Code\nSigned-off-by: Claude <noreply@anthropic.com>")
	bad := g("rev-parse", "HEAD")
	out.Reset()
	if err := run([]string{"commits", "-base", head, "-head", bad}, &out); !errors.Is(err, errFindings) || !strings.Contains(out.String(), "is not a person") {
		t.Fatalf("a tool's sign-off passed: %v\n%s", err, out.String())
	}
	// Zero commits is not a pass (X22).
	if err := run([]string{"commits", "-base", head, "-head", head}, &out); err == nil || errors.Is(err, errFindings) {
		t.Fatalf("an empty range passed: %v", err)
	}
}

func TestTags(t *testing.T) {
	a, b := strings.Repeat("a", 40), strings.Repeat("b", 40)
	lock := "# recorded tags\nv0.11.0 " + a + "\n"
	none(t, CheckTags(lock, map[string]string{"v0.11.0": a}))
	has(t, CheckTags(lock, map[string]string{"v0.11.0": b}), "tags are never moved")
	has(t, CheckTags(lock, map[string]string{}), "no longer exists")
	has(t, CheckTags(lock, map[string]string{"v0.11.0": a, "v0.11.1": b}), "is not in tags.lock")
	has(t, CheckTags("v0.11.0 abc\n", nil), "is not \"<tag> <40-hex commit>\"")
}

// X16: the status page shows a gate not run for seven days, or never run, as failing; and a
// gate's declared paths must equal its workflow's triggers.
func TestStatusPage(t *testing.T) {
	reg, err := invariants.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	gs, err := ParseGates([]byte("- {name: landing, workflow: gate-landing.yml, paths: [registry/**], invariants: [VAL-6]}\n- {name: rederive, workflow: rederive.yml, paths: [registry/**], invariants: [SUB-4]}\n"), reg)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 11, 4, 0, 0, 0, 0, time.UTC)
	last := map[string]*Run{"landing": {When: now.AddDate(0, 0, -2), Conclusion: "success", URL: "u"}}
	page, failing := StatusPage("blis-substrate", now, gs, last, reg)
	if len(failing) != 1 || !strings.Contains(failing[0], "rederive: failing: never run") {
		t.Fatalf("failing %v", failing)
	}
	if !strings.Contains(page, "| VAL-6 |") || !strings.Contains(page, "| landing | `gate-landing.yml` | [2026-11-02](u) | success | passing |") {
		t.Errorf("page:\n%s", page)
	}
	last["landing"].When = now.AddDate(0, 0, -8)
	_, failing = StatusPage("blis-substrate", now, gs, last, reg)
	has(t, failing, "landing: failing: last run 8 days ago")
	has(t, CheckGatePaths(gs, map[string][]string{"gate-landing.yml": {"registry/**", "kernel/**"}}), "but gates.yaml declares")
	has(t, CheckGatePaths(gs, map[string][]string{}), "does not exist or has no pull_request paths")
	if _, err := ParseGates([]byte("- {name: x, workflow: y.yml, paths: [a/**], invariants: [VAL-99]}\n"), reg); err == nil {
		t.Error("a gate citing an unknown invariant was accepted")
	}
	if _, err := ParseGates([]byte("- {name: x, workflow: y.yml, paths: []}\n"), reg); err == nil {
		t.Error("a gate with no trigger paths was accepted")
	}
}

// X18: every known_bad entry for this repository names a test that exists and passes. The
// sweep itself runs nightly; here every named test is checked to exist, so a rename that
// orphans an entry fails go test.
func TestKnownBadRunsEachTest(t *testing.T) {
	reg, err := invariants.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	tests, _ := KnownBadTests(reg, "blis-substrate")
	if len(tests) == 0 {
		t.Fatal("the registry names no known-bad test for blis-substrate")
	}
	for pkg, names := range tests {
		out, err := exec.Command("go", "test", "-list", ".", "../../"+pkg).Output()
		if err != nil {
			t.Errorf("go test -list ./%s: %v", pkg, err)
			continue
		}
		for _, n := range names {
			if !strings.Contains("\n"+string(out), "\n"+n+"\n") {
				t.Errorf("%s: the registry names %s as a known-bad test, and it does not exist", pkg, n)
			}
		}
	}
	// Known-bad: an entry naming a test that does not exist is reported. runKnownBad runs
	// from the repository root.
	cwd, _ := os.Getwd()
	defer os.Chdir(cwd)
	os.Chdir("../..")
	bad := runKnownBad(map[string][]string{"tools/internal/bot": {"TestNoSuchTest"}})
	if len(bad) != 1 || !strings.Contains(bad[0], "TestNoSuchTest did not run") {
		t.Fatalf("an entry naming a missing test was not reported: %v", bad)
	}
}

// X21: LICENSE and an engine's NOTICE.
func TestLicenceFiles(t *testing.T) {
	dir := t.TempDir()
	has(t, CheckLicenceFiles(dir, "inference-sim"), "LICENSE is missing")
	os.WriteFile(filepath.Join(dir, "LICENSE"), []byte("Apache License\n"), 0o644)
	none(t, CheckLicenceFiles(dir, "inference-sim"))
	has(t, CheckLicenceFiles(dir, "blis-vllm"), "NOTICE is missing")
	os.WriteFile(filepath.Join(dir, "NOTICE"), []byte("Portions derived from SGLang.\n"), 0o644)
	has(t, CheckLicenceFiles(dir, "blis-vllm"), "does not name vLLM")
	none(t, CheckLicenceFiles("../..", "blis-substrate"))
}

// ClosedBy reads every form GitHub acts on and none it does not.
func TestClosedByForms(t *testing.T) {
	const repo = "o/r"
	for _, c := range []struct {
		name, body string
		want       []Issue
	}{
		{"plain", "Closes #1\n", []Issue{{repo, 1}}},
		{"colon", "Closes: #2\n", []Issue{{repo, 2}}},
		{"bullet", "- Closes #3\n* Fixes #4\n+ Resolves #5\n", []Issue{{repo, 3}, {repo, 4}, {repo, 5}}},
		{"quote", "> Closes #6\n", []Issue{{repo, 6}}},
		{"every keyword", "close #1\ncloses #2\nclosed #3\nfix #4\nfixes #5\nfixed #6\nresolve #7\nresolves #8\nresolved #9\n",
			[]Issue{{repo, 1}, {repo, 2}, {repo, 3}, {repo, 4}, {repo, 5}, {repo, 6}, {repo, 7}, {repo, 8}, {repo, 9}}},
		{"list", "Closes #1, #2, #3\n", []Issue{{repo, 1}, {repo, 2}, {repo, 3}}},
		{"list with and", "Closes #1 and #2\n", []Issue{{repo, 1}, {repo, 2}}},
		{"parenthetical", "Closes #7 (the rest is in #9)\n", []Issue{{repo, 7}}},
		{"cross repository", "Closes a/b#8\n", []Issue{{"a/b", 8}}},
		{"mixed list", "Fixes #1, a/b#2\n", []Issue{{repo, 1}, {"a/b", 2}}},
		{"comment", "<!-- Closes #7 -->\nCloses #1\n", []Issue{{repo, 1}}},
		{"unterminated comment", "Closes #1\n<!-- Closes #7\nCloses #8\n", []Issue{{repo, 1}}},
		{"fenced block", "```\nCloses #7\n```\nCloses #1\n", []Issue{{repo, 1}}},
		{"tilde fence", "~~~\nCloses #7\n~~~\n", nil},
		{"unterminated fence", "Closes #1\n```\nCloses #7\n", []Issue{{repo, 1}}},
		{"prose is not a closing line", "This closes #7 eventually.\nsee Closes #8 in the text\n", nil},
		{"no number", "Closes the loop\n", nil},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := ClosedBy(repo, c.body)
			if len(got) != len(c.want) {
				t.Fatalf("got %+v, want %+v", got, c.want)
			}
			for i := range got {
				if got[i] != c.want[i] {
					t.Fatalf("got %+v, want %+v", got, c.want)
				}
			}
		})
	}
}

// The aggregator reads the path-filtered workflows from the files. The real ones are found,
// each with the check-run name its job makes.
func TestPathGatesAreReadFromTheWorkflowFiles(t *testing.T) {
	files, err := filepath.Glob("../../.github/workflows/*.yml")
	if err != nil {
		t.Fatal(err)
	}
	gates, err := LoadPathGates(os.ReadFile, files)
	if err != nil {
		t.Fatal(err)
	}
	byFile := map[string]PathGate{}
	for _, g := range gates {
		byFile[filepath.Base(g.File)] = g
	}
	for _, f := range []string{"rederive.yml", "ledger-candidate.yml"} {
		if g, ok := byFile[f]; !ok || len(g.Paths) == 0 || len(g.Jobs) == 0 {
			t.Errorf("%s is not read as a path-filtered gate: %+v", f, g)
		}
	}
	if len(gates) != 2 {
		t.Errorf("%d path-filtered workflows; a new one needs a look at the aggregator: %v", len(gates), gates)
	}
	// The registry's gate list agrees with the file, and a change under it triggers it.
	if !byFile["rederive.yml"].Triggered([]string{"registry/scripts/x.py"}) || byFile["rederive.yml"].Triggered([]string{"docs/a.md", "kernel/a.go"}) {
		t.Error("rederive's trigger does not follow its paths")
	}
	if !byFile["ledger-candidate.yml"].Triggered([]string{"kernel/a.go"}) || byFile["ledger-candidate.yml"].Triggered([]string{"docs/a.md"}) {
		t.Error("ledger-candidate's trigger does not follow its paths")
	}
}

func TestCheckGates(t *testing.T) {
	gates := []PathGate{
		{File: "w/rederive.yml", Paths: []string{"registry/scripts/**", ".github/workflows/rederive.yml"}, Jobs: []string{"rederive"}},
		{File: "w/ledger-candidate.yml", Paths: []string{"kernel/**"}, Jobs: []string{"candidate"}},
	}
	ok := map[string]GateRun{"rederive": {true, "success"}, "candidate": {true, "success"}}
	t.Run("untouched gates need no run", func(t *testing.T) {
		none(t, CheckGates(gates, []string{"docs/a.md"}, nil))
	})
	t.Run("touched and passed", func(t *testing.T) {
		none(t, CheckGates(gates, []string{"registry/scripts/a.py", "kernel/k.go"}, ok))
	})
	t.Run("touched and never started", func(t *testing.T) {
		has(t, CheckGates(gates, []string{"kernel/k.go"}, nil), `no check run "candidate"`)
	})
	t.Run("touched and failed", func(t *testing.T) {
		runs := map[string]GateRun{"rederive": {true, "failure"}}
		has(t, CheckGates(gates, []string{"registry/scripts/a.py"}, runs), "concluded failure")
	})
	t.Run("touched and skipped or cancelled", func(t *testing.T) {
		runs := map[string]GateRun{"rederive": {true, "skipped"}}
		has(t, CheckGates(gates, []string{".github/workflows/rederive.yml"}, runs), "concluded skipped")
	})
	t.Run("touched and still running", func(t *testing.T) {
		runs := map[string]GateRun{"rederive": {false, ""}}
		has(t, CheckGates(gates, []string{"registry/scripts/a.py"}, runs), "has not completed")
		if p := Pending(gates, []string{"registry/scripts/a.py"}, runs); len(p) != 1 || p[0] != "rederive" {
			t.Errorf("pending %v", p)
		}
	})
	t.Run("an untouched gate that failed does not block", func(t *testing.T) {
		none(t, CheckGates(gates, []string{"docs/a.md"}, map[string]GateRun{"rederive": {true, "failure"}}))
	})
}

// The runs of a gate are read at their newest attempt (highest id), so a re-run on the same head
// is not poisoned by an older failure and an older pass does not hide a newer failure. Known-bad
// input: failure then success (refused before), success then failure, a running newest attempt,
// and a run that does not exist.
func TestGateRunsUseTheNewestAttempt(t *testing.T) {
	gates := []PathGate{{File: "w/rederive.yml", Paths: []string{"registry/**"}, Jobs: []string{"rederive", "candidate"}}}
	changed := []string{"registry/a.py"}
	rec := func(id int64, name, status, concl string) gateRunRecord {
		return gateRunRecord{ID: id, Name: name, Status: status, Conclusion: concl}
	}
	cases := []struct {
		name string
		recs []gateRunRecord
		want string // substring of the finding, or "" for none
	}{
		{"older failure then newer success passes", []gateRunRecord{rec(5, "rederive", "completed", "failure"), rec(9, "rederive", "completed", "success"), rec(6, "candidate", "completed", "success")}, ""},
		{"line order does not matter", []gateRunRecord{rec(9, "rederive", "completed", "success"), rec(5, "rederive", "completed", "failure"), rec(6, "candidate", "completed", "success")}, ""},
		{"older success then newer failure is refused", []gateRunRecord{rec(5, "rederive", "completed", "success"), rec(9, "rederive", "completed", "failure"), rec(6, "candidate", "completed", "success")}, "concluded failure"},
		{"a newest attempt still running is no result", []gateRunRecord{rec(5, "rederive", "completed", "success"), rec(9, "rederive", "in_progress", ""), rec(6, "candidate", "completed", "success")}, "has not completed"},
		{"a missing run is refused", []gateRunRecord{rec(5, "rederive", "completed", "success")}, `no check run "candidate"`},
		// A workflow's concurrency group cancels the run a newer one replaces. A cancelled run
		// proves nothing: when a newer run passed it is not the newest and does not matter; when
		// it is the newest it is a finished run that did not succeed, never a pass.
		{"an older cancelled run and a newer success passes", []gateRunRecord{rec(5, "rederive", "completed", "cancelled"), rec(9, "rederive", "completed", "success"), rec(6, "candidate", "completed", "success")}, ""},
		{"an older success and a newer cancelled run is no pass", []gateRunRecord{rec(5, "rederive", "completed", "success"), rec(9, "rederive", "completed", "cancelled"), rec(6, "candidate", "completed", "success")}, "concluded cancelled"},
		{"a cancelled run alone is no pass", []gateRunRecord{rec(5, "rederive", "completed", "cancelled"), rec(6, "candidate", "completed", "success")}, "concluded cancelled"},
		{"ids compare as numbers", []gateRunRecord{rec(9, "rederive", "completed", "failure"), rec(10, "rederive", "completed", "success"), rec(6, "candidate", "completed", "success")}, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := CheckGates(gates, changed, foldGateRuns(c.recs))
			if c.want == "" {
				none(t, got)
			} else {
				has(t, got, c.want)
			}
		})
	}
}

func TestPathPatterns(t *testing.T) {
	for _, c := range []struct {
		pat, path string
		want      bool
	}{
		{"registry/**", "registry/a/b.py", true},
		{"registry/**", "registryx/a.py", false},
		{"kernel/*.go", "kernel/a.go", true},
		{"kernel/*.go", "kernel/sub/a.go", false},
		{"a/**/b", "a/b", true},
		{"a/**/b", "a/x/y/b", true},
		{".github/quarantine.yaml", ".github/quarantine.yaml", true},
		{".github/quarantine.yaml", ".github/quarantineXyaml", false},
	} {
		if got := pathPattern(c.pat).MatchString(c.path); got != c.want {
			t.Errorf("%q against %q = %v, want %v", c.pat, c.path, got, c.want)
		}
	}
}

// X24: a review thread is resolved, and states an outcome.
func TestCheckThreads(t *testing.T) {
	good := []Thread{
		{URL: "t1", Resolved: true, Comments: []string{"nit", "Outcome: fixed in abc"}},
		{URL: "t2", Resolved: true, Comments: []string{"Outcome: filed inference-sim/blis-substrate#9"}},
		{URL: "t3", Resolved: true, Comments: []string{"Outcome: rejected: the datasheet states it"}},
	}
	none(t, CheckThreads(good))
	none(t, CheckThreads(nil))
	for _, c := range []struct {
		name string
		th   Thread
		want string
	}{
		{"open thread", Thread{URL: "t1", Comments: []string{"Outcome: fixed"}}, "is open"},
		{"resolved without outcome", Thread{URL: "t1", Resolved: true, Comments: []string{"done"}}, "resolved with no"},
		{"rejected without reason", Thread{URL: "t1", Resolved: true, Comments: []string{"Outcome: rejected:"}}, "resolved with no"},
		{"filed without issue", Thread{URL: "t1", Resolved: true, Comments: []string{"Outcome: filed"}}, "resolved with no"},
		{"outcome not at the start of a line", Thread{URL: "t1", Resolved: true, Comments: []string{"I think Outcome: fixed"}}, "resolved with no"},
	} {
		t.Run(c.name, func(t *testing.T) {
			has(t, CheckThreads(append([]Thread{good[0]}, c.th)), c.want)
		})
	}
}

func TestThreadsOfKeepsTheFirstCommentURL(t *testing.T) {
	var api apiThread
	api.IsResolved = true
	api.Comments.Nodes = append(api.Comments.Nodes,
		struct {
			Body string `json:"body"`
			URL  string `json:"url"`
		}{"first", "u1"},
		struct {
			Body string `json:"body"`
			URL  string `json:"url"`
		}{"Outcome: fixed", "u2"})
	got := threadsOf([]apiThread{api})
	if len(got) != 1 || got[0].URL != "u1" || !got[0].Resolved || len(got[0].Comments) != 2 {
		t.Fatalf("threadsOf = %+v", got)
	}
	none(t, CheckThreads(got))
}

// The pull-request check runs the three checks that moved out of the merge job, and nothing
// in the workflows still names a merge queue (X22, X24, X28).
func TestThePullRequestCheckHoldsTheChecksOfTheRemovedMergeJob(t *testing.T) {
	read := func(p string) string {
		t.Helper()
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	pr := read("../../.github/workflows/shared-pr-checks.yml")
	for _, want := range []string{
		`prcheck threads -repo "${{ github.repository }}" -pr "$PR"`,
		`prcheck issues -repo "${{ github.repository }}" -closes "$RUNNER_TEMP/body.md"`,
	} {
		if !strings.Contains(pr, want) {
			t.Errorf("shared-pr-checks.yml does not run %q", want)
		}
	}
	files, err := filepath.Glob("../../.github/workflows/*.yml")
	if err != nil || len(files) == 0 {
		t.Fatalf("no workflows: %v", err)
	}
	for _, f := range files {
		if strings.Contains(read(f), "merge_group") {
			t.Errorf("%s has a merge_group trigger or reads a merge-group event; BLIS repositories do not use a merge queue (X22)", filepath.Base(f))
		}
	}
	if _, err := os.Stat("../../.github/workflows/shared-merge-check.yml"); err == nil {
		t.Error("shared-merge-check.yml still exists")
	}
}
