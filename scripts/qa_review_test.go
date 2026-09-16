package scripts_test

// qa-review's deterministic (model-free) surface is pinned here. Following the
// established shell-tool precedent (deliver_gate_test.go drives deliver-gate.sh
// via exec.Command), these tests shell out to python3 so they run under
// `go test ./scripts/...` with no Python test framework and no CI wiring. The
// model-calling paths (question generation, the answerer/adjudicator tool
// loops) need the live LiteLLM proxy and are intentionally NOT covered here.

import (
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// requirePython3 skips a test when python3 is unavailable, matching how
// requireGit guards the git-dependent script tests.
func requirePython3(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("python3"); err != nil {
		t.Skip("python3 is not on PATH")
	}
}

// qaScript returns the absolute path of a qa-review script.
func qaScript(t *testing.T, name string) string {
	t.Helper()
	abs, err := filepath.Abs(filepath.Join("qa-review", name))
	if err != nil {
		t.Fatalf("resolving %s: %v", name, err)
	}
	return abs
}

// runPython runs `python3 <args...>` from the qa-review directory (so `import`
// resolves the sibling modules) and returns stdout, stderr, and the exit code.
func runPython(t *testing.T, stdin string, args ...string) (string, string, int) {
	t.Helper()
	cmd := exec.Command("python3", args...)
	cmd.Dir = qaScript(t, ".")
	if stdin != "" {
		cmd.Stdin = strings.NewReader(stdin)
	}
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	code := 0
	var exitErr *exec.ExitError
	switch {
	case err == nil:
	case errors.As(err, &exitErr):
		code = exitErr.ExitCode()
	default:
		t.Fatalf("running python3 %v: %v", args, err)
	}
	return stdout.String(), stderr.String(), code
}

// writeFixture writes body to a temp file and returns its path.
func writeFixture(t *testing.T, name, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("writing fixture %s: %v", name, err)
	}
	return path
}

// ---------------------------------------------------------------------------
// render_report.py — verdict rule and output shape.
// ---------------------------------------------------------------------------

const rrQuestions = `{"model":"gcp/gemini-3.6-flash","questions":[` +
	`{"id":"F1","topic":"fixed","question":"Implements the issue?"},` +
	`{"id":"G1","topic":"tests","question":"Covered by tests?"}]}`

// renderFixture invokes render_report.py with the given answers JSON and
// returns its stdout.
func renderFixture(t *testing.T, answers string) string {
	t.Helper()
	q := writeFixture(t, "questions.json", rrQuestions)
	a := writeFixture(t, "answers.json", answers)
	stdout, stderr, code := runPython(t, "",
		qaScript(t, "render_report.py"),
		"--questions", q, "--answers", a,
		"--pr", "42", "--title", "My PR", "--amodel", "azure/gpt-5.6-sol",
	)
	if code != 0 {
		t.Fatalf("render_report.py exit=%d stderr=%s", code, stderr)
	}
	return stdout
}

func TestRenderVerdictBlocksOnBlockingStatus(t *testing.T) {
	requirePython3(t)

	cases := []struct {
		name     string
		answers  string
		wantHead string // the verdict header emoji+word this must contain
	}{
		{
			name:     "all-confident-passes",
			answers:  `[{"id":"F1","status":"CONFIDENT","answer":"Yes."},{"id":"G1","status":"CONFIDENT","answer":"Yes."}]`,
			wantHead: "## qa-review — PR #42: ✅ PASS",
		},
		{
			name:     "flaw-found-blocks",
			answers:  `[{"id":"F1","status":"CONFIDENT","answer":"Yes."},{"id":"G1","status":"FLAW_FOUND","answer":"No.","evidence":"b.go:2"}]`,
			wantHead: "## qa-review — PR #42: ⛔ BLOCK",
		},
		{
			name:     "cannot-answer-blocks",
			answers:  `[{"id":"F1","status":"CANNOT_ANSWER","answer":"Unclear.","evidence":"?"},{"id":"G1","status":"CONFIDENT","answer":"Yes."}]`,
			wantHead: "## qa-review — PR #42: ⛔ BLOCK",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out := renderFixture(t, tc.answers)
			if !strings.Contains(out, tc.wantHead) {
				t.Errorf("output missing verdict header %q\n%s", tc.wantHead, out)
			}
		})
	}
}

// TestRenderOutputShape asserts the full report shape: the verdict header, the
// full untruncated table, an Items to fix section listing the blocking finding,
// and an Important to consider section listing the non-blocking note.
func TestRenderOutputShape(t *testing.T) {
	requirePython3(t)

	answers := `[` +
		`{"id":"F1","status":"CONFIDENT","answer":"Fully implemented.","evidence":"x.go:1","note":"a minor doc nit"},` +
		`{"id":"G1","status":"FLAW_FOUND","answer":"A test is missing.","evidence":"y_test.go:9"}]`
	out := renderFixture(t, answers)

	mustContain := []string{
		"## qa-review — PR #42: ⛔ BLOCK",                        // verdict header
		"| ID | Topic | Result | Question | Answer |",           // table header
		"| F1 | fixed | ✅ CONFIDENT | Implements the issue? |",  // full, untruncated row
		"| G1 | tests | ❌ FLAW_FOUND | Covered by tests? |",     // full, untruncated row
		"### Items to fix",                                      // blocking section
		"- **G1 · FLAW_FOUND** — A test is missing.",            // the blocking finding + evidence
		"### Important to consider",                             // non-blocking section
		"- **F1** — a minor doc nit",                            // the note field
	}
	for _, want := range mustContain {
		if !strings.Contains(out, want) {
			t.Errorf("report missing %q\n---\n%s", want, out)
		}
	}

	// A blocking finding's answer must appear untruncated in the Items to fix
	// section, not just in the table.
	fix := out[strings.Index(out, "### Items to fix"):]
	if !strings.Contains(fix, "A test is missing.") {
		t.Errorf("Items to fix section did not carry the finding text:\n%s", fix)
	}

	// The verdict is a header emoji, never a trailing machine marker — deriving
	// QA-VERDICT is #1715's job.
	if strings.Contains(out, "QA-VERDICT") {
		t.Errorf("render_report.py must not emit a QA-VERDICT marker:\n%s", out)
	}
}

// ---------------------------------------------------------------------------
// questioner.repair_json() — invalid-escape repair without double-escaping.
// ---------------------------------------------------------------------------

// TestRepairJSON drives questioner.repair_json via `python3 -c`, importing the
// module, so the repair path is exercised with no live model.
func TestRepairJSON(t *testing.T) {
	requirePython3(t)

	// The script prints, in order:
	//   1. the value parsed from a payload with an invalid \s escape (repaired),
	//   2. whether an already-valid \\ pair is left unchanged by the repair,
	//   3. the value parsed from that already-valid \\ pair.
	prog := `
import json, questioner
invalid = r'{"q":"match \s here"}'
print(json.loads(questioner.repair_json(invalid))["q"])
valid = '{"q":"back\\\\slash"}'   # JSON source contains a literal \\ pair
print("UNCHANGED" if questioner.repair_json(valid) == valid else "MUTATED")
print(json.loads(questioner.repair_json(valid))["q"])
`
	stdout, stderr, code := runPython(t, "", "-c", prog)
	if code != 0 {
		t.Fatalf("repair_json probe exit=%d stderr=%s", code, stderr)
	}
	lines := strings.Split(strings.TrimRight(stdout, "\n"), "\n")
	if len(lines) != 3 {
		t.Fatalf("expected 3 output lines, got %d: %q", len(lines), stdout)
	}
	if lines[0] != `match \s here` {
		t.Errorf("invalid \\s escape not repaired: got %q", lines[0])
	}
	if lines[1] != "UNCHANGED" {
		t.Errorf("a valid \\\\ pair was double-escaped by repair_json: got %q", lines[1])
	}
	if lines[2] != `back\slash` {
		t.Errorf("valid \\\\ pair parsed wrong: got %q", lines[2])
	}
}

// ---------------------------------------------------------------------------
// adjudicator.parse_items_to_fix() + the BLOCK rule.
// ---------------------------------------------------------------------------

// A rendered qa-review comment with two blocking findings and one non-blocking
// note, matching render_report.py's output shape.
const priorComment = "## qa-review — PR #1710: ⛔ BLOCK\n" +
	"_subtitle_\n\n" +
	"| ID | Topic | Result | Question | Answer |\n" +
	"|----|----|----|----|----|\n" +
	"| F2 | fixed | ❌ FLAW_FOUND | q | a |\n\n" +
	"### Items to fix\n" +
	"- **F2 · FLAW_FOUND** — Docs still list only H100.\n" +
	"  _docs/guide/latency-models.md:42-47_\n" +
	"- **G5 · CANNOT_ANSWER** — Not enough evidence.\n\n" +
	"### Important to consider\n" +
	"- **F1** — should also list H200.\n"

func TestAdjudicatorParseItemsToFix(t *testing.T) {
	requirePython3(t)

	comment := writeFixture(t, "comment.md", priorComment)
	prog := `
import json, sys, adjudicator
body = open(sys.argv[1], encoding="utf-8").read()
json.dump(adjudicator.parse_items_to_fix(body), sys.stdout)
`
	stdout, stderr, code := runPython(t, "", "-c", prog, comment)
	if code != 0 {
		t.Fatalf("parse_items_to_fix probe exit=%d stderr=%s", code, stderr)
	}

	var items []struct {
		ID   string `json:"id"`
		Was  string `json:"was"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal([]byte(stdout), &items); err != nil {
		t.Fatalf("parse_items_to_fix output is not JSON: %v (%s)", err, stdout)
	}
	// Exactly the two blocking findings — never the Important-to-consider note.
	if len(items) != 2 {
		t.Fatalf("want 2 findings, got %d: %+v", len(items), items)
	}
	if items[0].ID != "F2" || items[0].Was != "FLAW_FOUND" {
		t.Errorf("finding 0 = %+v, want F2/FLAW_FOUND", items[0])
	}
	if items[1].ID != "G5" || items[1].Was != "CANNOT_ANSWER" {
		t.Errorf("finding 1 = %+v, want G5/CANNOT_ANSWER", items[1])
	}
	if !strings.Contains(items[0].Text, "Docs still list only H100") {
		t.Errorf("finding 0 text lost: %q", items[0].Text)
	}
	for _, it := range items {
		if it.ID == "F1" {
			t.Errorf("parse_items_to_fix captured a non-blocking note (F1): %+v", it)
		}
	}
}

// TestAdjudicatorBlockRule pins the aggregate rule: BLOCK iff any prior finding
// is STILL_OPEN OR left un-adjudicated. render() returns (report, verdict);
// the probe prints the verdict.
func TestAdjudicatorBlockRule(t *testing.T) {
	requirePython3(t)

	// Two prior findings, F2 and G5.
	items := `[{"id":"F2","was":"FLAW_FOUND","text":"t"},{"id":"G5","was":"CANNOT_ANSWER","text":"t"}]`

	cases := []struct {
		name     string
		verdicts string
		want     string
	}{
		{
			name:     "all-cleared-passes",
			verdicts: `[{"id":"F2","verdict":"RESOLVED","rationale":"fixed"},{"id":"G5","verdict":"WAIVED_DEFERRED","rationale":"deferred"}]`,
			want:     "PASS",
		},
		{
			name:     "one-still-open-blocks",
			verdicts: `[{"id":"F2","verdict":"RESOLVED","rationale":"fixed"},{"id":"G5","verdict":"STILL_OPEN","rationale":"nope"}]`,
			want:     "BLOCK",
		},
		{
			name:     "un-adjudicated-blocks",
			verdicts: `[{"id":"F2","verdict":"RESOLVED","rationale":"fixed"}]`, // G5 missing
			want:     "BLOCK",
		},
		{
			name:     "unknown-verdict-blocks",
			verdicts: `[{"id":"F2","verdict":"RESOLVED","rationale":"x"},{"id":"G5","verdict":"MAYBE","rationale":"?"}]`,
			want:     "BLOCK",
		},
	}
	prog := `
import json, sys, adjudicator
items = json.loads(sys.argv[1])
verdicts = json.loads(sys.argv[2])
_, agg = adjudicator.render(items, verdicts, "1710", "azure/gpt-5.6-sol")
print(agg)
`
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stdout, stderr, code := runPython(t, "", "-c", prog, items, tc.verdicts)
			if code != 0 {
				t.Fatalf("render probe exit=%d stderr=%s", code, stderr)
			}
			if got := strings.TrimSpace(stdout); got != tc.want {
				t.Errorf("aggregate verdict = %q, want %q", got, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// --no-exec seam — tools_for() drops `go` while keeping every read-only tool.
// ---------------------------------------------------------------------------

// TestNoExecSeam asserts, for both answerer.py and adjudicator.py, that
// tools_for(no_exec=True) excludes the code-executing `go` tool from BOTH the
// implementation map and the advertised schema, while tools_for(no_exec=False)
// includes it — and that no other tool is dropped. Model-free and deterministic.
func TestNoExecSeam(t *testing.T) {
	requirePython3(t)

	// For a module, print four sorted lines: exec-on impl names, exec-on schema
	// names, no-exec impl names, no-exec schema names.
	prog := `
import json, importlib, sys
m = importlib.import_module(sys.argv[1])
def names(no_exec):
    impl, schema = m.tools_for(no_exec)
    return sorted(impl.keys()), sorted(t["function"]["name"] for t in schema)
on_impl, on_schema = names(False)
off_impl, off_schema = names(True)
json.dump({"on_impl": on_impl, "on_schema": on_schema,
           "off_impl": off_impl, "off_schema": off_schema}, sys.stdout)
`
	for _, mod := range []string{"answerer", "adjudicator"} {
		t.Run(mod, func(t *testing.T) {
			stdout, stderr, code := runPython(t, "", "-c", prog, mod)
			if code != 0 {
				t.Fatalf("tools_for probe exit=%d stderr=%s", code, stderr)
			}
			var got struct {
				OnImpl    []string `json:"on_impl"`
				OnSchema  []string `json:"on_schema"`
				OffImpl   []string `json:"off_impl"`
				OffSchema []string `json:"off_schema"`
			}
			if err := json.Unmarshal([]byte(stdout), &got); err != nil {
				t.Fatalf("tools_for output not JSON: %v (%s)", err, stdout)
			}

			has := func(ss []string, s string) bool {
				for _, x := range ss {
					if x == s {
						return true
					}
				}
				return false
			}

			// go present with the flag absent (verbatim prototype behavior).
			if !has(got.OnImpl, "go") || !has(got.OnSchema, "go") {
				t.Errorf("%s: go missing with --no-exec absent: impl=%v schema=%v", mod, got.OnImpl, got.OnSchema)
			}
			// go dropped from BOTH under --no-exec.
			if has(got.OffImpl, "go") || has(got.OffSchema, "go") {
				t.Errorf("%s: --no-exec did not drop go: impl=%v schema=%v", mod, got.OffImpl, got.OffSchema)
			}
			// Every read-only tool survives the drop.
			readonly := []string{"read_file", "grep", "list_dir"}
			if mod == "adjudicator" {
				readonly = append(readonly, "gh_issue", "pr_diff")
			}
			for _, tool := range readonly {
				if !has(got.OffImpl, tool) || !has(got.OffSchema, tool) {
					t.Errorf("%s: --no-exec dropped read-only tool %q: impl=%v schema=%v", mod, tool, got.OffImpl, got.OffSchema)
				}
			}
			// The ONLY difference between the two sets is `go`.
			if len(got.OnImpl) != len(got.OffImpl)+1 || len(got.OnSchema) != len(got.OffSchema)+1 {
				t.Errorf("%s: --no-exec changed more than just go: on_impl=%v off_impl=%v", mod, got.OnImpl, got.OffImpl)
			}
		})
	}
}
