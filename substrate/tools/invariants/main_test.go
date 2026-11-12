package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/invariants"
)

func runIn(t *testing.T, args ...string) (string, string, error) {
	t.Helper()
	var out, errb bytes.Buffer
	err := run(args, &out, &errb)
	return out.String(), errb.String(), err
}

func writeFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

// The committed AGENTS.md block, pull-request template and rendered docs match the registry
// (design §13.1: a rendered copy that does not match its record fails CI).
func TestRenderedCopiesAreCurrent(t *testing.T) {
	_, stderr, err := runIn(t, "render", "-check", "-root", "../..")
	if err != nil {
		t.Fatalf("%v\n%s", err, stderr)
	}
}

// Known-bad input for the rendered-copy check (X18): a copy edited by hand fails.
func TestRenderCheckFailsOnAHandEditedCopy(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "AGENTS.md", "# Agent instructions\n")
	if _, stderr, err := runIn(t, "render", "-root", root); err != nil {
		t.Fatalf("%v\n%s", err, stderr)
	}
	if _, _, err := runIn(t, "render", "-check", "-root", root); err != nil {
		t.Fatalf("a fresh render does not pass its own check: %v", err)
	}
	tpl := filepath.Join(root, ".github/pull_request_template.md")
	raw, _ := os.ReadFile(tpl)
	os.WriteFile(tpl, bytes.Replace(raw, []byte("X13:"), []byte("X13 (optional):"), 1), 0o644)
	_, stderr, err := runIn(t, "render", "-check", "-root", root)
	if !errors.Is(err, errCheck) || !strings.Contains(stderr, "pull_request_template.md does not match") {
		t.Fatalf("a hand-edited template passed: err=%v stderr=%s", err, stderr)
	}
}

// render keeps the hand-written part of AGENTS.md and replaces only the generated block.
func TestRenderKeepsHandWrittenText(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "AGENTS.md", "# Rules\n\nKeep this.\n\n"+beginMarker+"\nold\n"+endMarker+"\n\nAnd this.\n")
	if _, stderr, err := runIn(t, "render", "-root", root); err != nil {
		t.Fatalf("%v\n%s", err, stderr)
	}
	raw, _ := os.ReadFile(filepath.Join(root, "AGENTS.md"))
	s := string(raw)
	if !strings.HasPrefix(s, "# Rules\n\nKeep this.\n\n"+beginMarker) || !strings.HasSuffix(s, endMarker+"\n\nAnd this.\n") || strings.Contains(s, "\nold\n") {
		t.Fatalf("render did not replace only the block:\n%s", s)
	}
	writeFile(t, root, "AGENTS.md", beginMarker+"\n"+beginMarker+"\n"+endMarker+"\n")
	if _, _, err := runIn(t, "render", "-root", root); err == nil {
		t.Fatal("render accepted repeated markers")
	}
}

// check-copy is how another repository confirms its copy matches the registry it pins.
func TestCheckCopyForAnotherRepository(t *testing.T) {
	dir := t.TempDir()
	agents, _, _ := runIn(t, "agents", "-repo", "blis-vllm")
	tpl, _, _ := runIn(t, "template", "-repo", "blis-vllm")
	a := writeFile(t, dir, "AGENTS.md", "# blis-vllm\n\n"+agents)
	p := writeFile(t, dir, "pr.md", tpl)
	if _, stderr, err := runIn(t, "check-copy", "-repo", "blis-vllm", "-agents", a, "-template", p); err != nil {
		t.Fatalf("%v\n%s", err, stderr)
	}
	if !strings.Contains(agents, "**VLM-1**") || strings.Contains(agents, "**SUB-6**") {
		t.Error("blis-vllm's section must list VLM-1 and must not list SUB-6, which it runs no check for")
	}
	// Known-bad input: the copy of another repository's section.
	other, _, _ := runIn(t, "agents", "-repo", "inference-sim")
	writeFile(t, dir, "AGENTS.md", other)
	_, stderr, err := runIn(t, "check-copy", "-repo", "blis-vllm", "-agents", a, "-template", p)
	if !errors.Is(err, errCheck) || !strings.Contains(stderr, "AGENTS.md") {
		t.Fatalf("a mismatched copy passed: %v %s", err, stderr)
	}
}

// X14: every invariant a pull request cites exists. Known-bad inputs: a body citing an
// identifier the registry does not hold, a body with no Invariants section, and one whose
// section cites nothing.
func TestCitedRefusesUnknownIdentifiers(t *testing.T) {
	dir := t.TempDir()
	good := writeFile(t, dir, "good.md", "## Summary\n\nX99 is not cited here.\n\n## Invariants\n\n<!-- e.g. SUB-99 -->\n- X14\n- SUB-6, LIF-2\n- T-A1\n\n## Tests\n")
	out, stderr, err := runIn(t, "cited", "-body", good)
	if err != nil {
		t.Fatalf("%v\n%s", err, stderr)
	}
	for _, want := range []string{"X14\tnew", "SUB-6\tnew", "LIF-2\tpartial", "T-A1\ttrigger outcome"} {
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q:\n%s", want, out)
		}
	}
	for name, body := range map[string]string{
		"unknown":    "## Invariants\n\n- SUB-21\n",
		"no section": "## Summary\n\n- SUB-6\n",
		"empty":      "## Invariants\n\n<!-- SUB-6 -->\n\n## Tests\n",
	} {
		p := writeFile(t, dir, name+".md", body)
		if _, _, err := runIn(t, "cited", "-body", p); !errors.Is(err, errCheck) {
			t.Errorf("%s: cited passed, err=%v", name, err)
		}
	}
}

// The review-question automation: a change to a path a question lists adds the question;
// an excluded path or another repository's path does not.
func TestQuestionsMatchChangedPaths(t *testing.T) {
	dir := t.TempDir()
	changed := writeFile(t, dir, "changed.txt", "kernel/new.go\nkernel/new_test.go\nREADME.md\n")
	out, _, err := runIn(t, "questions", "-repo", "blis-substrate", "-changed", changed)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "**X2**") || !strings.Contains(out, "`kernel/new.go`") || strings.Contains(out, "new_test.go") {
		t.Fatalf("the X2 question was not added for kernel/new.go alone:\n%s", out)
	}
	if strings.Count(out, "- [ ]") != 1 {
		t.Errorf("expected one question, got:\n%s", out)
	}
	none := writeFile(t, dir, "none.txt", "kernel/new_test.go\n")
	if out, _, _ := runIn(t, "questions", "-repo", "blis-substrate", "-changed", none); out != "" {
		t.Errorf("a test-only change added questions:\n%s", out)
	}
	vllm := writeFile(t, dir, "vllm.txt", "oracle/mocks.json\n")
	if out, _, _ := runIn(t, "questions", "-repo", "blis-vllm", "-changed", vllm); !strings.Contains(out, "**X5**") {
		t.Errorf("a manifest change in blis-vllm did not add X5's question:\n%s", out)
	}
}

// The parser the snapshot uses reads every standing invariant of the published design.
// The design is not in this repository, so the test runs on a small document of the same
// form; TestEveryDesignInvariantAppearsExactlyOnce compares the snapshot with the registry.
func TestParseDesign(t *testing.T) {
	doc := `<ul class="invlist">
<li><span class="inv">X1</span> <b>Title one.</b>
<span class="stmt">First &amp; only.</span>
<div class="tbl"><table><thead><tr><th>Check</th><th>Runs in</th></tr></thead><tbody><tr><td>c</td><td>r</td></tr></tbody></table></div>
<span class="enf"><b>Human review.</b> Q?</span></li>
<li><span class="inv">VAL-1</span> <span class="stmt">Methods:</span>
<div class="tbl"><table><thead><tr><th>Method</th><th>Required</th></tr></thead><tbody>
<tr><td><code>a</code></td><td>x</td></tr><tr><td>b</td><td>y</td></tr></tbody></table></div><span class="enf"><b>Checked by:</b> z.</span></li>
<li><span class="inv">T-A1</span> Outcome.</li>
</ul>`
	items, err := parseDesign(doc)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 2 {
		t.Fatalf("parsed %d items, want 2 (trigger outcomes are not records): %+v", len(items), items)
	}
	if items[0].Title != "Title one." || items[0].Statement != "First & only." {
		t.Errorf("X1: %+v", items[0])
	}
	if items[1].Statement != "Methods: a: x; b: y." {
		t.Errorf("VAL-1 statement %q", items[1].Statement)
	}
}

func TestFirstSentence(t *testing.T) {
	for in, want := range map[string]string{
		"One (a. b). Two.":                       "One (a. b).",
		"Each kind, and nothing else: a: b.":     "Each kind, and nothing else: a: b.",
		"Pins are exact release tags. The only…": "Pins are exact release tags.",
		"Ends with a colon:":                     "Ends with a colon.",
	} {
		if got := firstSentence(in); got != want {
			t.Errorf("firstSentence(%q) = %q, want %q", in, got, want)
		}
	}
}

// Every repository that runs a check has a section, and each section names only the
// invariants that have a check there.
func TestEveryRepositoryHasASection(t *testing.T) {
	reg, err := invariants.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, repo := range invariants.Repositories {
		s := renderAgents(reg, repo)
		n := strings.Count(s, "\n- **")
		want := 0
		for _, inv := range reg.Invariants {
			if applies(inv, repo) {
				want++
			}
		}
		if want == 0 {
			t.Errorf("%s runs no check", repo)
		}
		// questions also start with "- **"
		if n < want {
			t.Errorf("%s: %d entries, want at least %d", repo, n, want)
		}
	}
}

// Every repository's AGENTS.md lists each practice check that concerns it as a maintainer
// checklist item, and the guidance of every rule that has any (design §4.2, §13.1).
func TestEveryRepositoryGetsItsPracticesAndGuidance(t *testing.T) {
	reg, err := invariants.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, repo := range invariants.Repositories {
		s := renderAgents(reg, repo)
		for _, inv := range reg.Invariants {
			for _, c := range inv.Checks {
				if c.Repo == repo && c.Runs == invariants.RunsPractice {
					if want := "- [ ] **" + inv.ID + "** "; !strings.Contains(s, want) {
						t.Errorf("%s: no checklist item %q", repo, want)
					}
				}
			}
			if inv.Guidance != "" && applies(inv, repo) && !strings.Contains(s, "- **"+inv.ID+"** "+strings.TrimSpace(inv.Guidance)) {
				t.Errorf("%s: guidance of %s missing", repo, inv.ID)
			}
		}
	}
}

// A practice appears in the maintainer checklist, and a repository with no practice has no
// maintainer section.
func TestAPracticeIsListedInTheMaintainerChecklist(t *testing.T) {
	reg := &invariants.Registry{Invariants: []invariants.Invariant{{
		ID: "T-1", Scope: []string{"blis-substrate"}, Status: "new",
		Checks: []invariants.Check{{Repo: "blis-substrate", Check: "Tags are protected", Runs: invariants.RunsPractice, Status: "ci"}},
	}}}
	s := renderAgents(reg, "blis-substrate")
	if !strings.Contains(s, "### Maintainer practices") || !strings.Contains(s, "- [ ] **T-1** Tags are protected") {
		t.Errorf("practice not in the checklist:\n%s", s)
	}
	reg.Invariants[0].Checks = nil
	if s := renderAgents(reg, "blis-substrate"); strings.Contains(s, "Maintainer practices") {
		t.Errorf("empty checklist rendered:\n%s", s)
	}
}
