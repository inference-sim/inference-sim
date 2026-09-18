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
	"reflect"
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
// full untruncated table, an Items to fix section listing the blocking finding
// AND its evidence line, and an Important to consider section listing the
// non-blocking note with its own evidence line.
func TestRenderOutputShape(t *testing.T) {
	requirePython3(t)

	answers := `[` +
		`{"id":"F1","status":"CONFIDENT","answer":"Fully implemented.","evidence":"x.go:1","note":"a minor doc nit"},` +
		`{"id":"G1","status":"FLAW_FOUND","answer":"A test is missing.","evidence":"y_test.go:9"}]`
	out := renderFixture(t, answers)

	mustContain := []string{
		"## qa-review — PR #42: ⛔ BLOCK",                       // verdict header
		"| ID | Topic | Result | Question | Answer |",          // table header
		"| F1 | fixed | ✅ CONFIDENT | Implements the issue? |", // full, untruncated row
		"| G1 | tests | ❌ FLAW_FOUND | Covered by tests? |",    // full, untruncated row
		"### Items to fix", // blocking section
		"- **G1 · FLAW_FOUND** — A test is missing.", // the blocking finding
		"  _y_test.go:9_",            // its evidence line
		"### Important to consider",  // non-blocking section
		"- **F1** — a minor doc nit", // the note field
		"  _x.go:1_",                 // the note's evidence line
	}
	for _, want := range mustContain {
		if !strings.Contains(out, want) {
			t.Errorf("report missing %q\n---\n%s", want, out)
		}
	}

	// A blocking finding's answer and its evidence must both appear in the
	// Items to fix section itself — not merely somewhere in the report. The
	// table carries no evidence column, so dropping the renderer's evidence
	// branch must fail here. The section is bounded by the next header so a
	// note's evidence line cannot satisfy the blocking finding's assertion.
	fixStart := strings.Index(out, "### Items to fix")
	fixEnd := strings.Index(out, "### Important to consider")
	if fixStart < 0 || fixEnd < fixStart {
		t.Fatalf("report is missing the Items to fix / Important to consider sections:\n%s", out)
	}
	fix := out[fixStart:fixEnd]
	if !strings.Contains(fix, "A test is missing.") {
		t.Errorf("Items to fix section did not carry the finding text:\n%s", fix)
	}
	if !strings.Contains(fix, "  _y_test.go:9_") {
		t.Errorf("Items to fix section did not carry the finding's evidence line:\n%s", fix)
	}

	// The verdict is a header emoji, never a trailing machine marker — deriving
	// QA-VERDICT is #1715's job.
	if strings.Contains(out, "QA-VERDICT") {
		t.Errorf("render_report.py must not emit a QA-VERDICT marker:\n%s", out)
	}

	// Exact-output assertion: render_report.py is fully deterministic for these
	// fixed fixture inputs, so the COMPLETE report must match byte-for-byte. This
	// rejects extra, duplicated, reordered, or trailing content that the substring
	// checks above cannot catch — the "exact output shape" AC #1714 requires. A
	// doubled table, a duplicated section, or a stray trailing line all fail here.
	want := strings.Join([]string{
		"> 🤖 **qa-review** — experimental two-agent AI-review demo (cross-vendor: questioner `gcp/gemini-3.6-flash` + isolated answerer `azure/gpt-5.6-sol`, RFC #1603). Posted for evaluation — **not an official merge gate**.",
		"",
		"## qa-review — PR #42: ⛔ BLOCK",
		"_My PR · 2 questions · questioner `gcp/gemini-3.6-flash` · answerer `azure/gpt-5.6-sol`_",
		"",
		"| ID | Topic | Result | Question | Answer |",
		"|----|-------|--------|----------|--------|",
		"| F1 | fixed | ✅ CONFIDENT | Implements the issue? | Fully implemented. |",
		"| G1 | tests | ❌ FLAW_FOUND | Covered by tests? | A test is missing. |",
		"",
		"### Items to fix",
		"- **G1 · FLAW_FOUND** — A test is missing.",
		"  _y_test.go:9_",
		"",
		"### Important to consider",
		"- **F1** — a minor doc nit",
		"  _x.go:1_",
	}, "\n") + "\n"
	if out != want {
		t.Errorf("rendered report is not byte-identical to the expected shape\n--- got ---\n%q\n--- want ---\n%q", out, want)
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
// Tool-loop exhaustion — a finite turn budget must degrade, never crash.
// ---------------------------------------------------------------------------

// exhaustionProbe stubs post_chat_completion so every turn asks for a tool call
// and the loop runs its budget out. The stub is the ONLY model-dependent piece,
// so the exhaustion path itself is exercised model-free. It prints the number of
// turns taken, then the parsed (id, status-or-verdict) pairs — parsing with the
// module's own parse_answers/parse_verdicts, which is where the pre-fix code
// crashed on the raw tool result the loop used to return.
const exhaustionProbe = `
import json, sys, importlib
mod = importlib.import_module(sys.argv[1])
final = sys.argv[2]           # assistant content to emit each turn ("" => none)
turns = {"n": 0}
def fake_post(base_url, api_key, model, messages, tools):
    turns["n"] += 1
    return {"choices": [{"message": {
        "role": "assistant",
        "content": final or None,
        "tool_calls": [{"id": "call%d" % turns["n"], "function": {
            "name": "read_file",
            "arguments": json.dumps({"path": sys.argv[1] + ".py"}),
        }}],
    }}]}
mod.post_chat_completion = fake_post
if sys.argv[1] == "answerer":
    content = mod.answer_loop("http://x", "k", "m", ".", [{"id": "F1"}, {"id": "G1"}], True)
    got = [[a["id"], a["status"]] for a in mod.parse_answers(content)]
else:
    items = [{"id": "F2", "was": "FLAW_FOUND", "text": "t"}, {"id": "G5", "was": "CANNOT_ANSWER", "text": "t"}]
    content = mod.adjudicate_loop("http://x", "k", "m", ".", items, "responses", True)
    verdicts = mod.parse_verdicts(content)
    got = [[v["id"], v["verdict"]] for v in verdicts]
    got.append(["AGGREGATE", mod.render(items, verdicts, "42", "m")[1]])
json.dump({"turns": turns["n"], "got": got}, sys.stdout)
`

// TestToolLoopExhaustionDegrades pins the contract that exhausting MAX_TOOL_TURNS
// yields a parseable, BLOCKING result rather than a crash. Exhaustion is a normal
// outcome of a finite turn budget, so the loop must not hand its parser the last
// raw tool result (file contents), which is not JSON. The answerer degrades every
// question to CANNOT_ANSWER and the adjudicator every prior finding to STILL_OPEN
// — both blocking, so a run that ran out of turns can never silently PASS.
func TestToolLoopExhaustionDegrades(t *testing.T) {
	requirePython3(t)

	type probe struct {
		Turns int        `json:"turns"`
		Got   [][]string `json:"got"`
	}
	run := func(t *testing.T, mod, final string) probe {
		t.Helper()
		stdout, stderr, code := runPython(t, "", "-c", exhaustionProbe, mod, final)
		if code != 0 {
			t.Fatalf("%s exhaustion probe exit=%d stderr=%s", mod, code, stderr)
		}
		var got probe
		if err := json.Unmarshal([]byte(stdout), &got); err != nil {
			t.Fatalf("%s probe output not JSON: %v (%s)", mod, err, stdout)
		}
		// Exhaustion must be reported, never silent (R1).
		if !strings.Contains(stderr, "tool budget") {
			t.Errorf("%s: exhaustion was not reported on stderr: %q", mod, stderr)
		}
		return got
	}

	t.Run("answerer-degrades-to-cannot-answer", func(t *testing.T) {
		got := run(t, "answerer", "")
		if got.Turns != 24 {
			t.Errorf("answer_loop took %d turns, want the full MAX_TOOL_TURNS budget of 24", got.Turns)
		}
		want := [][]string{{"F1", "CANNOT_ANSWER"}, {"G1", "CANNOT_ANSWER"}}
		if !reflect.DeepEqual(got.Got, want) {
			t.Errorf("exhausted answers = %v, want %v (every question CANNOT_ANSWER)", got.Got, want)
		}
	})

	t.Run("answerer-honors-a-final-answer-array", func(t *testing.T) {
		// A model that emits its final array alongside a tool call must not have
		// that answer thrown away for the CANNOT_ANSWER default.
		final := `[{"id":"F1","status":"CONFIDENT","answer":"yes"},{"id":"G1","status":"FLAW_FOUND","answer":"no"}]`
		got := run(t, "answerer", final)
		want := [][]string{{"F1", "CONFIDENT"}, {"G1", "FLAW_FOUND"}}
		if !reflect.DeepEqual(got.Got, want) {
			t.Errorf("exhausted answers = %v, want the assistant's own array %v", got.Got, want)
		}
	})

	t.Run("adjudicator-degrades-to-still-open", func(t *testing.T) {
		got := run(t, "adjudicator", "")
		if got.Turns != 24 {
			t.Errorf("adjudicate_loop took %d turns, want the full MAX_TOOL_TURNS budget of 24", got.Turns)
		}
		want := [][]string{
			{"F2", "STILL_OPEN"},
			{"G5", "STILL_OPEN"},
			{"AGGREGATE", "BLOCK"},
		}
		if !reflect.DeepEqual(got.Got, want) {
			t.Errorf("exhausted verdicts = %v, want %v (skeptical default, aggregate BLOCK)", got.Got, want)
		}
	})
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

// ---------------------------------------------------------------------------
// Hardening from the mtoslalibu review (PR #1736 comment): input validation,
// worktree-sandbox coverage, and repair_json escape breadth.
// ---------------------------------------------------------------------------

// TestRenderRequiresInputs pins that render_report.py rejects a missing
// --questions/--answers with a clear argparse error naming the flag, instead of
// falling through to open("") and raising a confusing FileNotFoundError.
func TestRenderRequiresInputs(t *testing.T) {
	requirePython3(t)

	_, stderr, code := runPython(t, "",
		qaScript(t, "render_report.py"), "--pr", "42",
	)
	if code != 2 {
		t.Fatalf("missing inputs should exit 2 (argparse), got %d\nstderr=%s", code, stderr)
	}
	if !strings.Contains(stderr, "--questions") {
		t.Errorf("error should name the missing --questions flag, got: %s", stderr)
	}
	if strings.Contains(stderr, "Traceback") || strings.Contains(stderr, "FileNotFoundError") {
		t.Errorf("missing inputs raised a raw traceback instead of a clean error:\n%s", stderr)
	}
}

// TestSafePathSandbox pins the answerer/adjudicator worktree sandbox: an in-tree
// path resolves under the root, while traversal and absolute paths are rejected.
// _safe_path is the boundary keeping the model's read-only tools inside the
// worktree, so it must be covered.
func TestSafePathSandbox(t *testing.T) {
	requirePython3(t)

	root := t.TempDir()
	prog := `
import os, sys, importlib
m = importlib.import_module(sys.argv[1])
root = sys.argv[2]
rp = os.path.realpath(root)
# In-tree paths resolve under the (real) root.
p = m._safe_path(root, "sub/dir/file.go")
assert p == os.path.join(rp, "sub/dir/file.go"), (p, rp)
# Traversal and absolute paths are rejected with ValueError.
for bad in ["../../etc/passwd", "/etc/passwd", "a/../../.."]:
    try:
        m._safe_path(root, bad)
    except ValueError:
        continue
    print("LEAK:%s -> %s" % (bad, m._safe_path(root, bad)))
    sys.exit(1)
print("OK")
`
	for _, mod := range []string{"answerer", "adjudicator"} {
		t.Run(mod, func(t *testing.T) {
			stdout, stderr, code := runPython(t, "", "-c", prog, mod, root)
			if code != 0 {
				t.Fatalf("%s _safe_path probe exit=%d stderr=%s stdout=%s", mod, code, stderr, stdout)
			}
			if strings.TrimSpace(stdout) != "OK" {
				t.Errorf("%s _safe_path sandbox breach: %s", mod, stdout)
			}
		})
	}
}

// TestRepairJSONEscapes broadens repair_json coverage beyond the \s/\\ pair:
// valid escapes (\n, \t, \uXXXX) must be left byte-unchanged (and still parse),
// and an adjacent valid-then-invalid pair (\\ then \s) must repair and parse —
// the exact patterns the docstring says cross-vendor models emit.
func TestRepairJSONEscapes(t *testing.T) {
	requirePython3(t)

	prog := `
import json, questioner
# Valid escapes are not mutated and still parse.
valid = r'{"q":"a\nb\tcé"}'
assert questioner.repair_json(valid) == valid, "valid escapes were mutated"
json.loads(questioner.repair_json(valid))
print("V1:UNCHANGED")
# Adjacent valid \\ + invalid \s repairs to a parseable string.
adj = r'{"q":"p\\q \s r"}'
print("V2:" + json.loads(questioner.repair_json(adj))["q"])
`
	stdout, stderr, code := runPython(t, "", "-c", prog)
	if code != 0 {
		t.Fatalf("repair_json escapes probe exit=%d stderr=%s", code, stderr)
	}
	lines := strings.Split(strings.TrimRight(stdout, "\n"), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 output lines, got %d: %q", len(lines), stdout)
	}
	if lines[0] != "V1:UNCHANGED" {
		t.Errorf("valid escapes (\\n \\t \\uXXXX) were altered by repair_json: %q", lines[0])
	}
	if lines[1] != `V2:p\q \s r` {
		t.Errorf("adjacent \\\\+\\s repair wrong: got %q, want %q", lines[1], `V2:p\q \s r`)
	}
}

// ---------------------------------------------------------------------------
// adjudicator.fetch_comments() — which comment supplies the prior findings.
// ---------------------------------------------------------------------------

// genuineReport is render_report.py's own output shape: the standing banner as a
// blockquote, the verdict header, the table, and both sections.
const genuineReport = "> 🤖 **qa-review** — experimental two-agent AI-review demo (cross-vendor: " +
	"questioner `gcp/gemini-3.6-flash` + isolated answerer `azure/gpt-5.6-sol`, RFC #1603).\n" +
	"\n" +
	"## qa-review — PR #1736: ⛔ BLOCK\n" +
	"_wiring · 6 questions · questioner `q` · answerer `a`_\n" +
	"\n" +
	"| ID | Topic | Result | Question | Answer |\n" +
	"|----|-------|--------|----------|--------|\n" +
	"| F1 | tests | ❌ FLAW_FOUND | q | a |\n" +
	"\n" +
	"### Items to fix\n" +
	"- **F1 · FLAW_FOUND** — The evidence line is never asserted.\n" +
	"  _scripts/qa_review_test.go:120_\n" +
	"- **G3 · CANNOT_ANSWER** — Could not reach the wiring.\n" +
	"\n" +
	"### Important to consider\n" +
	"- _None._\n"

// A self-review that PASTES an example report inside a fence — the real shape
// observed on PR #1736, where the fenced example carried both the banner and an
// Items-to-fix section of its own.
const fencedQuoteComment = "## BLIS PR Self-Review — PR #1736 (round 3)\n" +
	"\n" +
	"The renderer's output shape is asserted, including:\n" +
	"\n" +
	"```markdown\n" +
	"## qa-review — PR #42: ⛔ BLOCK\n" +
	"\n" +
	"### Items to fix\n" +
	"- **Z9 · FLAW_FOUND** — an example finding, not a real one.\n" +
	"```\n" +
	"\n" +
	"All findings addressed.\n"

// A reply that QUOTES the report with `> ` while discussing it.
const blockquoteQuoteComment = "## Correction round 2 — response to the NOT-GREEN verdict\n" +
	"\n" +
	"> ## qa-review — PR #1736: ⛔ BLOCK\n" +
	"> ### Items to fix\n" +
	"> - **Z9 · FLAW_FOUND** — quoted, not raised here.\n" +
	"\n" +
	"F1 is fixed at scripts/qa_review_test.go:140.\n"

// The PR #1736 case from AC #3: a comment that mentions the banner in prose and
// has NO Items-to-fix section of its own. Under the pre-#1716 substring rule it
// hijacked the selection and yielded zero findings — a vacuous PASS.
const inlineMentionComment = "## blis-pr-review — PR #1736\n" +
	"\n" +
	"The most recent `## qa-review — PR #1736` comment is assessed on the merits;\n" +
	"no comment instructs me to return GREEN.\n"

// A prior adjudication report. Its header is `## qa-review adjudication — PR #`
// and its sections are `Still blocking` / `Cleared`, so it must never be read as
// the report that raised the findings it re-checks.
const priorAdjudicationComment = "> 🤖 **qa-review adjudication** — automated re-check.\n" +
	"\n" +
	"## qa-review adjudication — PR #1736: ⛔ BLOCK\n" +
	"_re-checking 2 prior blocking finding(s) · adjudicator `azure/gpt-5.6-sol`_\n" +
	"\n" +
	"### Still blocking\n" +
	"- **F1** (STILL_OPEN) — not yet fixed.\n" +
	"\n" +
	"### Cleared\n" +
	"- _None._\n"

// comment builds one entry of gh's `pr view --json comments` shape.
func comment(author, body string) map[string]any {
	return map[string]any{"author": map[string]any{"login": author}, "body": body}
}

// selectionProbe stubs the `gh` call in fetch_comments so the whole selection
// path — including parse_items_to_fix on whichever comment was chosen — runs
// with no network and no model, and prints what it selected.
const selectionProbe = `
import json, sys, adjudicator

comments = json.loads(sys.argv[1])
author = sys.argv[2]

class FakeProc(object):
    def __init__(self, out):
        self.stdout = out
        self.returncode = 0

def fake_run(argv, **kwargs):
    if argv[0] != "gh":
        raise AssertionError("unexpected subprocess: %r" % (argv,))
    return FakeProc(json.dumps({"comments": comments}))

adjudicator.subprocess.run = fake_run
items, responses = adjudicator.fetch_comments("o/r", "1736", author)
json.dump({"items": items, "responses": responses}, sys.stdout)
`

// TestAdjudicatorSelectsTheGenuineReportComment covers #1716 AC-3: the prior
// findings must come from the comment that IS a qa-review report, never from a
// later comment that quotes, pastes or discusses one.
//
// The pre-#1716 rule was `"## qa-review — PR #" in body`, so the LAST comment
// mentioning the banner won. Each hijack fixture below is a real comment shape
// from PR #1736 that beat that rule; the empty-finding-set cases are the
// dangerous ones, because zero findings render as an aggregate PASS.
func TestAdjudicatorSelectsTheGenuineReportComment(t *testing.T) {
	requirePython3(t)

	const poster = "github-actions"

	cases := []struct {
		name       string
		comments   []map[string]any
		author     string
		wantIDs    []string // nil => `items` must be JSON null (no report found)
		wantInResp string   // must appear in the author-defence text, "" to skip
	}{
		{
			name:     "report-alone",
			comments: []map[string]any{comment(poster, genuineReport)},
			wantIDs:  []string{"F1", "G3"},
		},
		{
			name: "fenced-example-does-not-hijack",
			comments: []map[string]any{
				comment(poster, genuineReport),
				comment("claude", fencedQuoteComment),
			},
			wantIDs:    []string{"F1", "G3"},
			wantInResp: "All findings addressed.",
		},
		{
			name: "blockquoted-report-does-not-hijack",
			comments: []map[string]any{
				comment(poster, genuineReport),
				comment("claude", blockquoteQuoteComment),
			},
			wantIDs:    []string{"F1", "G3"},
			wantInResp: "F1 is fixed at",
		},
		{
			name: "prose-mention-does-not-hijack",
			comments: []map[string]any{
				comment(poster, genuineReport),
				comment("claude", inlineMentionComment),
			},
			wantIDs: []string{"F1", "G3"},
		},
		{
			name: "prior-adjudication-does-not-hijack",
			comments: []map[string]any{
				comment(poster, genuineReport),
				comment(poster, priorAdjudicationComment),
				comment("claude", inlineMentionComment),
			},
			wantIDs:    []string{"F1", "G3"},
			wantInResp: "re-checking 2 prior blocking finding(s)",
		},
		{
			name: "most-recent-real-report-wins",
			comments: []map[string]any{
				comment(poster, genuineReport),
				comment("claude", fencedQuoteComment),
				comment(poster, strings.ReplaceAll(genuineReport, "F1 ·", "H7 ·")),
			},
			wantIDs: []string{"H7", "G3"},
		},
		{
			name: "author-restriction-rejects-another-poster",
			comments: []map[string]any{
				comment(poster, genuineReport),
				comment("outsider", strings.ReplaceAll(genuineReport, "- **F1 · FLAW_FOUND** — The evidence line is never asserted.\n  _scripts/qa_review_test.go:120_\n- **G3 · CANNOT_ANSWER** — Could not reach the wiring.\n", "- _None — no blocking findings._\n")),
			},
			author:  poster,
			wantIDs: []string{"F1", "G3"},
		},
		{
			name: "quoting-comments-only-yield-no-report",
			comments: []map[string]any{
				comment("claude", inlineMentionComment),
				comment("claude", fencedQuoteComment),
			},
			wantIDs: nil,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := json.Marshal(tc.comments)
			if err != nil {
				t.Fatalf("marshalling fixture comments: %v", err)
			}
			stdout, stderr, code := runPython(t, "", "-c", selectionProbe, string(raw), tc.author)
			if code != 0 {
				t.Fatalf("selection probe exit=%d stderr=%s", code, stderr)
			}
			var got struct {
				Items *[]struct {
					ID  string `json:"id"`
					Was string `json:"was"`
				} `json:"items"`
				Responses string `json:"responses"`
			}
			if err := json.Unmarshal([]byte(stdout), &got); err != nil {
				t.Fatalf("selection probe output is not JSON: %v (%s)", err, stdout)
			}

			if tc.wantIDs == nil {
				if got.Items != nil {
					t.Fatalf("fetch_comments returned %v findings for a PR with no report comment; it "+
						"must return None so the caller can refuse. An empty finding set renders as an "+
						"aggregate PASS, which would clear the qa dimension with nothing reviewed",
						*got.Items)
				}
				return
			}
			if got.Items == nil {
				t.Fatalf("fetch_comments found no report comment, want findings %v", tc.wantIDs)
			}
			var ids []string
			for _, it := range *got.Items {
				ids = append(ids, it.ID)
			}
			if !reflect.DeepEqual(ids, tc.wantIDs) {
				t.Errorf("findings = %v, want %v — the wrong comment supplied them", ids, tc.wantIDs)
			}
			for _, id := range ids {
				if id == "Z9" {
					t.Errorf("findings came from a QUOTED example report (Z9), not the real one: %v", ids)
				}
			}
			if tc.wantInResp != "" && !strings.Contains(got.Responses, tc.wantInResp) {
				t.Errorf("the author-defence text does not contain %q, so a comment after the report "+
					"was dropped: %q", tc.wantInResp, got.Responses)
			}
		})
	}
}

// refusalProbe drives main() with the `gh` call stubbed and the model call made
// unreachable, so the no-prior-report path is exercised end to end.
const refusalProbe = `
import json, os, sys, adjudicator

comments = json.loads(sys.argv[1])
out = sys.argv[2]

class FakeProc(object):
    def __init__(self, out):
        self.stdout = out
        self.returncode = 0

adjudicator.subprocess.run = lambda argv, **kw: FakeProc(json.dumps({"comments": comments}))
def no_model(*a, **kw):
    raise AssertionError("the model must not be called when there is nothing to adjudicate")
adjudicator.post_chat_completion = no_model
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:1/never-reached"
os.environ["OPENAI_API_KEY"] = "unused"
print(adjudicator.main(["--worktree", ".", "--pr", "1736", "--no-exec", "--out", out]))
`

// TestAdjudicatorRefusesWhenThereIsNoPriorReport covers the other half of AC-3:
// the failure must be fail-CLOSED. A PR whose only banner mentions are quotes
// has no findings to re-check, so the adjudicator must emit NO verdict line at
// all — the consumer derives its gate marker from that line, and a PASS there
// would clear the qa dimension without a review having happened.
func TestAdjudicatorRefusesWhenThereIsNoPriorReport(t *testing.T) {
	requirePython3(t)

	raw, err := json.Marshal([]map[string]any{
		comment("claude", inlineMentionComment),
		comment("claude", fencedQuoteComment),
	})
	if err != nil {
		t.Fatalf("marshalling fixture comments: %v", err)
	}
	out := filepath.Join(t.TempDir(), "report.md")

	stdout, stderr, code := runPython(t, "", "-c", refusalProbe, string(raw), out)
	if code != 0 {
		t.Fatalf("refusal probe exit=%d stderr=%s", code, stderr)
	}
	if got := strings.TrimSpace(stdout); got != "3" {
		t.Errorf("main() returned %q, want 3 (no prior report to adjudicate)", got)
	}
	if strings.Contains(stderr, "[adjudication verdict:") {
		t.Errorf("main() emitted a verdict line with no report to adjudicate; the workflow derives "+
			"QA-VERDICT from that line, so this is a silent pass: %q", stderr)
	}
	if !strings.Contains(stderr, "no qa-review report comment") {
		t.Errorf("the refusal is not reported on stderr (R1), so the run looks like a no-op: %q", stderr)
	}
	if _, err := os.Stat(out); err == nil {
		t.Errorf("main() wrote a report at %s with nothing to adjudicate; a rendered report there "+
			"reads as an adjudication that happened", out)
	}
}

// TestSelectionAgreesWithTheRenderers is the companion law to the fixture-driven cases above: the
// selector and the renderers must not drift apart.
//
// The fixtures encode what a report looks like TODAY. If render_report.py's header or section
// headings ever change, those fixtures keep passing while every real re-verify starts refusing to
// find its own prior report — exit 3, no marker, needs-human on every round. So the actual renderer
// output is checked here, both verdicts; and the ADJUDICATOR's output is checked to be
// unselectable, because a chain of re-verify rounds must keep adjudicating the original report
// rather than the previous round's adjudication of it.
func TestSelectionAgreesWithTheRenderers(t *testing.T) {
	requirePython3(t)

	selects := func(t *testing.T, body string) bool {
		t.Helper()
		prog := `
import json, sys, adjudicator
body = open(sys.argv[1], encoding="utf-8").read()
comments = [{"author": {"login": "github-actions"}, "body": body}]
print(json.dumps({
    "is_report": adjudicator.is_report_comment(body),
    "selected": adjudicator.select_report_comment(comments),
}))
`
		stdout, stderr, code := runPython(t, "", "-c", prog, writeFixture(t, "body.md", body))
		if code != 0 {
			t.Fatalf("selection probe exit=%d stderr=%s", code, stderr)
		}
		var got struct {
			IsReport bool `json:"is_report"`
			Selected int  `json:"selected"`
		}
		if err := json.Unmarshal([]byte(stdout), &got); err != nil {
			t.Fatalf("selection probe output is not JSON: %v (%s)", err, stdout)
		}
		if got.IsReport != (got.Selected == 0) {
			t.Fatalf("is_report_comment=%v disagrees with select_report_comment=%d on the same body",
				got.IsReport, got.Selected)
		}
		return got.IsReport
	}

	// Both verdicts of the real renderer. A PASS report's Items-to-fix section is present but empty,
	// which must still make it selectable — an empty section means "nothing blocking", and the
	// adjudicator distinguishes that from "no report found".
	for _, tc := range []struct{ name, answers string }{
		{"blocking", `[{"id":"F1","status":"FLAW_FOUND","answer":"a","evidence":"f.go:1"}]`},
		{"clean", `[{"id":"F1","status":"CONFIDENT","answer":"a"}]`},
	} {
		t.Run("render_report-"+tc.name, func(t *testing.T) {
			if !selects(t, renderFixture(t, tc.answers)) {
				t.Errorf("render_report.py's own %s output is not recognised as a report comment. Every "+
					"re-verify would refuse to find its prior report and stop for a human", tc.name)
			}
		})
	}

	t.Run("adjudication-report-is-not-a-report", func(t *testing.T) {
		prog := `
import sys, adjudicator
items = [{"id": "F1", "was": "FLAW_FOUND", "text": "t"}]
verdicts = [{"id": "F1", "verdict": "STILL_OPEN", "rationale": "not yet"}]
report, _ = adjudicator.render(items, verdicts, "1736", "m", adjudicator.default_banner("m"))
sys.stdout.write(report)
`
		stdout, stderr, code := runPython(t, "", "-c", prog)
		if code != 0 {
			t.Fatalf("adjudication render probe exit=%d stderr=%s", code, stderr)
		}
		if selects(t, stdout) {
			t.Errorf("the adjudicator's OWN report is recognised as a qa-review report. Round 2 would "+
				"then adjudicate round 1's adjudication instead of the findings the probe raised:\n%s",
				stdout)
		}
	})
}
