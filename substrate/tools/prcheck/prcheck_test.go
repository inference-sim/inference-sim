package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v3"

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
// responsible person has signed off.
func TestCommitPolicy(t *testing.T) {
	good := Commit{Hash: strings.Repeat("a", 40), Signed: true, Author: "Ada Person <ada@example.com>",
		Message: "Add a check\n\nBody.\n\nAssisted-by: Claude Code (Claude Opus 5.5)\nSigned-off-by: Ada Person <ada@example.com>\n"}
	none(t, CommitPolicy(good))
	bot := Commit{Hash: "b", Signed: true, Author: "release bot <" + releaseBotEmail + ">",
		Committer: "release bot <" + releaseBotEmail + ">", Message: "Release v0.11.1\n"}
	none(t, CommitPolicy(bot))

	cases := []struct {
		name string
		edit func(*Commit)
		want string
	}{
		{"unsigned", func(c *Commit) { c.Signed = false }, "not signed"},
		{"no sign-off", func(c *Commit) { c.Message = "Add\n\nAssisted-by: Claude Code\n" }, "no Signed-off-by"},
		{"agent signs off", func(c *Commit) {
			c.Message = "Add\n\nAssisted-by: Claude Code\nSigned-off-by: Claude <noreply@anthropic.com>\n"
		}, "is not a person"},
		{"co-authored by a tool", func(c *Commit) {
			c.Message = "Add\n\nCo-authored-by: Claude Opus <noreply@anthropic.com>\nSigned-off-by: Ada Person <ada@example.com>\n"
		}, "names a tool"},
		{"two assisted-by", func(c *Commit) {
			c.Message = "Add\n\nAssisted-by: Claude Code\nAssisted-by: Copilot\nSigned-off-by: Ada Person <ada@example.com>\n"
		}, "2 Assisted-by lines"},
		{"bot author, person committer", func(c *Commit) {
			c.Author = "release bot <" + releaseBotEmail + ">"
			c.Committer = "Ada Person <ada@example.com>"
			c.Message = "Release\n"
		}, "no Signed-off-by"},
		{"bot author, no committer", func(c *Commit) {
			c.Author = "release bot <" + releaseBotEmail + ">"
			c.Committer = ""
			c.Message = "Release\n"
		}, "no Signed-off-by"},
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
	// A co-author who is a person is allowed.
	person := good
	person.Message = "Add\n\nCo-authored-by: Bea Person <bea@example.com>\nSigned-off-by: Ada Person <ada@example.com>\n"
	none(t, CommitPolicy(person))
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

// This repository passes its own checks, so a change that breaks one, or that fixes a
// baselined finding without shrinking the baseline, fails go test as well as CI.
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

// X22 and X24.
func TestMergeCheck(t *testing.T) {
	mt, ht := strings.Repeat("a", 40), strings.Repeat("b", 40)
	good := MergeInput{
		MergeTree: mt, HeadTree: ht, Required: []string{"required"},
		Verdicts: []Verdict{
			{Kind: "check", Name: "required", Body: "ok\nTree: " + mt + "\n"},
			{Kind: "review", Name: "ada", Body: "Looks right.\nTree: " + ht + "\n", Paths: []string{"kernel/a.go"}},
			{Kind: "answer", Name: "X2", Body: "yes: neutral\nTree: " + ht + "\n", Paths: []string{"kernel/**"}},
		},
		Threads:        []Thread{{URL: "t1", Resolved: true, Comments: []string{"nit", "Outcome: fixed in abc"}}},
		ChangedByMerge: []string{"registry/x.yaml"},
	}
	none(t, CheckMerge(good))
	cases := []struct {
		name string
		edit func(*MergeInput)
		want string
	}{
		{"check on the head tree", func(m *MergeInput) { m.Verdicts[0].Body = "Tree: " + ht }, "not the tree being merged"},
		{"no tree", func(m *MergeInput) { m.Verdicts[1].Body = "Looks right." }, "names no tree"},
		{"required check missing", func(m *MergeInput) { m.Verdicts = m.Verdicts[1:] }, `required check "required" has no successful run`},
		{"review on an old tree", func(m *MergeInput) { m.Verdicts[1].Body = "Tree: " + strings.Repeat("c", 40) }, "neither the head"},
		{"merge changes a reviewed path", func(m *MergeInput) { m.ChangedByMerge = []string{"kernel/a.go"} }, `answer "X2" judged the head tree, and merging changes paths it covers`},
		{"open thread", func(m *MergeInput) { m.Threads[0].Resolved = false }, "is open"},
		{"resolved without outcome", func(m *MergeInput) { m.Threads[0].Comments = []string{"done"} }, "resolved with no"},
		{"rejected without reason", func(m *MergeInput) { m.Threads[0].Comments = []string{"Outcome: rejected:"} }, "resolved with no"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			m := good
			m.Verdicts = append([]Verdict{}, good.Verdicts...)
			m.Threads = []Thread{{URL: "t1", Resolved: true, Comments: append([]string{}, good.Threads[0].Comments...)}}
			c.edit(&m)
			has(t, CheckMerge(m), c.want)
		})
	}
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
	found := []string{"docs/a.md: Go 1.27"}
	has(t, CheckVersions(found, map[string]bool{}), "is a version literal outside")
	none(t, CheckVersions(found, map[string]bool{"docs/a.md: Go 1.27": true}))
	has(t, CheckVersions(nil, map[string]bool{"docs/a.md: Go 1.27": true}), "no longer occurs")
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
	has(t, CheckBaseline([]string{"docs/a.md:3: x (X26)"}, map[string]bool{}, "X26"), "docs/a.md:3: x")
	none(t, CheckBaseline([]string{"docs/a.md:3: x (X26)"}, map[string]bool{"docs/a.md: x (X26)": true}, "X26"))
}

// X25.
func TestQuarantine(t *testing.T) {
	day := time.Date(2026, 11, 4, 0, 0, 0, 0, time.UTC)
	good := "- {test: TestA, package: ./kernel, issue: inference-sim/blis-substrate#9, expires: 2026-11-20, reason: flaky on arm64}\n"
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
	_, bad = CheckQuarantine([]byte(good), day, map[string]time.Time{"TestA": day.AddDate(0, -1, -1)})
	has(t, bad, "more than 30 days")
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
	sig := filepath.Join(dir, "sigs")
	// Signed according to the API: clean.
	os.WriteFile(sig, []byte(head+" G\n"), 0o644)
	var out strings.Builder
	if err := run([]string{"commits", "-base", base, "-head", head, "-signatures", sig}, &out); err != nil {
		t.Fatalf("%v\n%s", err, out.String())
	}
	// Known-bad: the same commit unsigned.
	os.WriteFile(sig, []byte(head+" N\n"), 0o644)
	out.Reset()
	if err := run([]string{"commits", "-base", base, "-head", head, "-signatures", sig}, &out); !errors.Is(err, errFindings) || !strings.Contains(out.String(), "not signed") {
		t.Fatalf("an unsigned commit passed: %v\n%s", err, out.String())
	}
	// Zero commits is not a pass (X22).
	if err := run([]string{"commits", "-base", head, "-head", head}, &out); err == nil || errors.Is(err, errFindings) {
		t.Fatalf("an empty range passed: %v", err)
	}
}

// The review-question automation keeps answers and reports open questions.
func TestMergeQuestions(t *testing.T) {
	body := "## Summary\n\nx\n\n## Review questions\n\nold text\n\n## Checklist\n\n- [ ] X13\n"
	qs := "- [ ] **X2** Does this hide policy? Reviewer: owners. (changed: `kernel/a.go`) Answer: \n"
	nb, open := MergeQuestions(body, qs)
	if len(open) != 1 || !strings.Contains(nb, "- [ ] **X2** Does this hide policy?") || strings.Contains(nb, "old text") || !strings.Contains(nb, "## Checklist\n\n- [ ] X13") {
		t.Fatalf("first pass: open=%v\n%s", open, nb)
	}
	answered := strings.Replace(nb, "- [ ] **X2**", "- [x] **X2**", 1)
	answered = strings.Replace(answered, "Answer: \n", "Answer: no, the threshold is the chip's datasheet figure\n", 1)
	// A later push changes the changed-path note; the answer stays.
	qs2 := strings.Replace(qs, "kernel/a.go", "kernel/b.go", 1)
	nb2, open2 := MergeQuestions(answered, qs2)
	if len(open2) != 0 || !strings.Contains(nb2, "- [x] **X2**") || !strings.Contains(nb2, "kernel/b.go") || !strings.Contains(nb2, "Answer: no, the threshold") {
		t.Fatalf("second pass: open=%v\n%s", open2, nb2)
	}
	// Known-bad: ticked without a yes or no.
	bad := strings.Replace(answered, "Answer: no, the threshold is the chip's datasheet figure", "Answer: looks fine", 1)
	if _, open3 := MergeQuestions(bad, qs); len(open3) != 1 {
		t.Fatalf("an answer without yes or no was accepted")
	}
	// A body with no section gets one at the end.
	nb4, _ := MergeQuestions("## Summary\n\nx", qs)
	if !strings.HasSuffix(nb4, "Answer: \n") || !strings.Contains(nb4, "x\n\n## Review questions") {
		t.Fatalf("appended section:\n%s", nb4)
	}
}

func TestBuildMergeInput(t *testing.T) {
	trees := map[string]string{"m": strings.Repeat("a", 40), "h": strings.Repeat("b", 40), "old": strings.Repeat("c", 40)}
	treeOf := func(s string) (string, error) {
		if t, ok := trees[s]; ok {
			return t, nil
		}
		return "", fmt.Errorf("unknown commit %s", s)
	}
	runs := []apiCheckRun{{Name: "blis checks", HeadSHA: "m", Conclusion: "success"}, {Name: "lint", HeadSHA: "m", Conclusion: "failure"}}
	reviews := []apiReview{{State: "APPROVED", CommitID: "old"}, {State: "COMMENTED", CommitID: "h"}}
	reviews[0].User.Login = "ada"
	in, err := BuildMergeInput("m", "h", []string{"blis checks", "lint"}, runs, reviews, nil, nil, []string{"a.go"}, answerSource{}, treeOf)
	if err != nil {
		t.Fatal(err)
	}
	got := CheckMerge(in)
	has(t, got, `required check "lint" has no successful run`)
	has(t, got, `review "ada" judged tree `+trees["old"])
	if len(got) != 2 {
		t.Errorf("want exactly the two findings, got:\n%s", strings.Join(got, "\n"))
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

// X17: a merge without a successful required check on its tree fails the audit.
func TestAuditMerges(t *testing.T) {
	req := map[string][]string{"blis-vllm": {"blis checks", "ci"}}
	ok := Merge{Repo: "blis-vllm", PR: 1, MergeSHA: strings.Repeat("a", 40), Successes: []string{"ci", "blis checks"}}
	none(t, AuditMerges([]Merge{ok}, req))
	bad := ok
	bad.Successes = []string{"ci"}
	has(t, AuditMerges([]Merge{bad}, req), "with no successful run of blis checks")
	other := ok
	other.Repo = "blis-sglang"
	has(t, AuditMerges([]Merge{other}, req), "names no required checks")
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

// X20 deferral: an unsigned commit and a missing sign-off are deferred while the file holds;
// every other finding fails; an expired or malformed file defers nothing.
func TestApplyDeferral(t *testing.T) {
	day := func(s string) time.Time { d, _ := time.Parse("2006-01-02", s); return d }
	const file = "issue: inference-sim/blis-substrate#12\nexpires: 2026-12-05\nreason: a person signs before merge\n"
	findings := []string{
		"abc123abc123: the commit is not signed (X20)",
		"abc123abc123: no Signed-off-by line; the person who takes responsibility for the commit adds the DCO sign-off (X20)",
		"abc123abc123: Signed-off-by: Claude <noreply@anthropic.com> is not a person; an agent never signs off (X20)",
	}
	fail, deferred := ApplyDeferral([]byte(file), day("2026-11-05"), findings)
	if len(deferred) != 2 || len(fail) != 1 || !strings.Contains(fail[0], "is not a person") {
		t.Fatalf("held: fail=%v deferred=%v", fail, deferred)
	}
	fail, deferred = ApplyDeferral(nil, day("2026-11-05"), findings)
	if len(fail) != 3 || len(deferred) != 0 {
		t.Fatalf("no file: fail=%v deferred=%v", fail, deferred)
	}
	// The last day before the expiry date still defers, at any time of day; the date itself
	// does not, from its first instant.
	for _, at := range []string{"2026-12-04T00:00:00Z", "2026-12-04T23:59:59Z"} {
		now, err := time.Parse(time.RFC3339, at)
		if err != nil {
			t.Fatal(err)
		}
		if fail, deferred = ApplyDeferral([]byte(file), now, findings); len(deferred) != 2 || len(fail) != 1 {
			t.Fatalf("%s: fail=%v deferred=%v, want 2 deferred", at, fail, deferred)
		}
	}
	if now, _ := time.Parse(time.RFC3339, "2026-12-05T00:00:01Z"); !now.IsZero() {
		if fail, deferred = ApplyDeferral([]byte(file), now, findings); len(deferred) != 0 || len(fail) != 4 {
			t.Fatalf("2026-12-05T00:00:01Z: fail=%v deferred=%v, want expired", fail, deferred)
		}
	}
	if fail, deferred = ApplyDeferral([]byte(file), day("2026-12-04"), findings); len(deferred) != 2 || len(fail) != 1 {
		t.Fatalf("2026-12-04: fail=%v deferred=%v, want 2 deferred", fail, deferred)
	}
	// A deferral file that is missing is no file: nothing is deferred and every finding fails.
	if fail, deferred = ApplyDeferral([]byte{}, day("2026-11-05"), findings); len(fail) != 3 || len(deferred) != 0 {
		t.Fatalf("empty file: fail=%v deferred=%v", fail, deferred)
	}
	fail, deferred = ApplyDeferral([]byte(file), day("2026-12-05"), findings)
	if len(deferred) != 0 || len(fail) != 4 || !strings.Contains(fail[3], "expired on 2026-12-05") {
		t.Fatalf("expired: fail=%v deferred=%v", fail, deferred)
	}
	fail, _ = ApplyDeferral([]byte("issue: nonsense\nexpires: 2026-12-05\nreason: x\n"), day("2026-11-05"), findings)
	if len(fail) != 4 || !strings.Contains(fail[3], "owning issue") {
		t.Fatalf("bad issue: %v", fail)
	}
	fail, _ = ApplyDeferral([]byte("issue: inference-sim/blis-substrate#12\nexpires: 2026-12-05\nreason: x\nextra: y\n"), day("2026-11-05"), findings)
	if len(fail) != 4 || !strings.Contains(fail[3], "x20-deferral.yaml") {
		t.Fatalf("unknown key: %v", fail)
	}
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

// X22: review-question answers are verdicts on the head tree. An answer removed from the
// body, or edited after the approving review, fails the merge check.
func TestAnswersAreVerdictsBoundToTheHead(t *testing.T) {
	trees := map[string]string{"m": strings.Repeat("a", 40), "h": strings.Repeat("b", 40)}
	treeOf := func(s string) (string, error) { return trees[s], nil }
	q := invariants.Question{ID: "X2", Review: invariants.Review{Question: "Is it so?", Reviewer: "r",
		Trigger: invariants.Trigger{Repos: []string{"blis-substrate"}, Paths: []string{"contracts/**"}}}}
	body := func(answer string) string {
		return "## Review questions\n\n- [x] **X2** Is it so? Reviewer: r. (changed: `contracts/a.go`) Answer: " + answer + "\n"
	}
	review := []apiReview{{State: "APPROVED", CommitID: "h", SubmittedAt: "2026-11-06T10:00:00Z"}}
	run := func(b, edited string) []string {
		ans := answerSource{Body: b, EditedAt: edited, Repo: "blis-substrate", Questions: []invariants.Question{q}}
		in, err := BuildMergeInput("m", "h", nil, nil, review, nil, nil, []string{"contracts/a.go"}, ans, treeOf)
		if err != nil {
			t.Fatal(err)
		}
		return CheckMerge(in)
	}
	none(t, run(body("yes, it is"), "2026-11-06T09:00:00Z"))
	has(t, run(body("yes, it is"), "2026-11-06T11:00:00Z"), "after the latest approving review")
	has(t, run("## Summary\n\nx\n", "2026-11-06T09:00:00Z"), "has no answer in the pull-request body")
	has(t, run(body(""), "2026-11-06T09:00:00Z"), "has no answer in the pull-request body")
	// A question no changed path triggers needs no answer.
	ans := answerSource{Body: "", Repo: "blis-substrate", Questions: []invariants.Question{q}}
	in, _ := BuildMergeInput("m", "h", nil, nil, nil, nil, nil, []string{"docs/a.md"}, ans, treeOf)
	none(t, CheckMerge(in))
}

// The names the merge check and the nightly audit compare are check-run names. A reusable
// workflow's job is named "<caller job> / <job name>" and never carries the workflow's name,
// so "blis / pr / blis checks" matches no run (X17, X22). The merge check's required list is
// the list the audit reads, plus nothing the merge queue cannot produce.
func TestRequiredCheckNamesAreCheckRunNames(t *testing.T) {
	raw, err := os.ReadFile("../../.github/required-checks.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var lists map[string][]string
	if err := yaml.Unmarshal(raw, &lists); err != nil {
		t.Fatal(err)
	}
	for repo, names := range lists {
		for _, n := range names {
			if strings.HasPrefix(n, "blis / ") {
				t.Errorf("%s: %q is prefixed with the workflow name; a check run is named %q", repo, n, strings.TrimPrefix(n, "blis / "))
			}
		}
	}
	wf, err := os.ReadFile("../../.github/workflows/blis.yml")
	if err != nil {
		t.Fatal(err)
	}
	m := regexp.MustCompile(`required: "([^"]+)"`).FindSubmatch(wf)
	if m == nil {
		t.Fatal("blis.yml names no required checks for the merge check")
	}
	have := map[string]bool{}
	for _, n := range lists["blis-substrate"] {
		have[n] = true
	}
	for _, n := range strings.Split(string(m[1]), ",") {
		if !have[n] {
			t.Errorf("the merge check requires %q, which required-checks.yaml does not list for blis-substrate", n)
		}
	}
	if !strings.Contains(string(wf), "github.event_name == 'merge_group'") || !strings.Contains(string(wf), "  pr:\n") {
		t.Error("the job that makes the check run \"pr / blis checks\" does not run on the merge-queue tree")
	}
}
