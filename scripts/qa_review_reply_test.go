package scripts_test

// #1895: an empty or unparseable model reply is a TRANSIENT failure, retried with the shared
// bounded backoff in _http.py like a gateway error, and an oversized questioner input is capped
// deterministically before the call. On exhaustion the scripts still fail closed (no output, a
// non-zero exit, so the gate reads MISSING), but stderr names the cause and the input size rather
// than leaving a bare JSONDecodeError traceback.
//
// These probes fake `_http.send` — one level BELOW post_chat_completion — so the real retry loop,
// the real reply check and each script's real main()/tool loop run; only the network is fake.

import (
	"encoding/json"
	"strings"
	"testing"
)

// replyProbe runs one qa-review script against a scripted sequence of completion CONTENTS.
//
// argv[1]: module name ("questioner", "answerer" or "adjudicator").
// argv[2]: JSON array of content strings, one per transport attempt. A null element means the
// message has no content key at all. Attempts past the end reuse the last element.
// argv[3]: optional path to a diff file (questioner only).
//
// stdout: {"rc": exit code, "out": captured stdout, "attempts": N, "prompts": [user turn per attempt]}.
const replyProbe = `
import importlib, io, json, os, sys
http = importlib.import_module("_http")
http.MAX_ATTEMPTS = 3
http.sleep = lambda s: None
name = sys.argv[1]
contents = json.loads(sys.argv[2])
diff_file = sys.argv[3] if len(sys.argv) > 3 else ""
mod = importlib.import_module(name)
os.environ["OPENAI_BASE_URL"] = "http://proxy"
os.environ["OPENAI_API_KEY"] = "k"
prompts = []
def fake_send(req, timeout):
    body = json.loads(req.data.decode("utf-8"))
    prompts.append(body["messages"][1]["content"])
    c = contents[min(len(prompts) - 1, len(contents) - 1)]
    msg = {"role": "assistant"}
    if c is not None:
        msg["content"] = c
    return json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}]})
http.send = fake_send
real_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    if name == "questioner":
        argv = ["--title", "T", "--body", "B"]
        if diff_file:
            argv += ["--diff-file", diff_file]
        else:
            argv += ["--diff", "D"]
        rc = mod.main(argv)
    elif name == "answerer":
        rc = mod.main(["--worktree", ".", "--no-exec",
                       "--questions", json.dumps([{"id": "F1", "topic": "fixed", "question": "q"}])])
    else:
        # fetch_comments reads the trusted-comment filter; feed it one genuine report with F2.
        report = ("## qa-review \u2014 PR #7: \u26d4 BLOCK\n\n### Items to fix\n"
                  "- **F2 \u00b7 FLAW_FOUND** \u2014 t\n")
        class P(object):
            def __init__(self, out):
                self.returncode, self.stdout, self.stderr = 0, out, ""
        mod.subprocess.run = lambda argv, **kw: P(json.dumps({"comments": [
            {"source": "conversation", "author": {"login": "github-actions[bot]"},
             "label": "automation", "body": report, "createdAt": "2026-01-01T00:00:00Z"}]}))
        os.environ.pop("QA_REPORT_AUTHOR", None)
        rc = mod.main(["--worktree", ".", "--pr", "7", "--no-exec"])
except SystemExit as exc:
    rc = exc.code
out = sys.stdout.getvalue()
sys.stdout = real_stdout
json.dump({"rc": rc, "out": out, "attempts": len(prompts), "prompts": prompts}, sys.stdout)
`

type replyProbeResult struct {
	RC       int      `json:"rc"`
	Out      string   `json:"out"`
	Attempts int      `json:"attempts"`
	Prompts  []string `json:"prompts"`
}

func runReplyProbe(t *testing.T, mod string, contents []any, extra ...string) (replyProbeResult, string) {
	t.Helper()
	raw, err := json.Marshal(contents)
	if err != nil {
		t.Fatalf("marshalling contents: %v", err)
	}
	args := append([]string{"-c", replyProbe, mod, string(raw)}, extra...)
	stdout, stderr, code := runPython(t, "", args...)
	if code != 0 {
		t.Fatalf("%s reply probe crashed: exit=%d stderr=%s stdout=%s", mod, code, stderr, stdout)
	}
	var got replyProbeResult
	if err := json.Unmarshal([]byte(stdout), &got); err != nil {
		t.Fatalf("%s reply probe output is not JSON: %v (%s)", mod, err, stdout)
	}
	return got, stderr
}

// validReply is a well-formed final reply for each script.
var validReply = map[string]string{
	"questioner":  `{"questions":[{"topic":"tests","question":"Is the retry covered?"}]}`,
	"answerer":    `[{"id":"F1","status":"CONFIDENT","answer":"a","evidence":"e"}]`,
	"adjudicator": `[{"id":"F2","was":"FLAW_FOUND","verdict":"RESOLVED","rationale":"r"}]`,
}

// TestUnusableReplyIsRetriedThenSucceeds is #1895's acceptance test: the #1892 incident was an HTTP
// 200 whose completion was EMPTY. That is not an HTTP error, so the transport retry never saw it and
// the questioner crashed in json.loads. The same reply shape reaches the answerer's and adjudicator's
// final-answer parse, so all three are covered. Every unusable shape — empty, no content key, prose,
// a JSON value of the wrong type — must be retried, and the retry must re-send the same input.
func TestUnusableReplyIsRetriedThenSucceeds(t *testing.T) {
	requirePython3(t)

	unusable := []struct {
		name    string
		content any
	}{
		{"empty-content", ""},
		{"whitespace-content", "  \n"},
		{"no-content-key", nil},
		{"prose", "Sorry, I cannot help with that."},
		{"wrong-json-type", `"just a string"`},
	}
	// Shapes that parse as the right container but carry malformed elements.
	elementShape := map[string][]any{
		"questioner": {`{"questions":["x"]}`, `{"questions":[{"question":5}]}`, `{}`},
	}
	for mod, shapes := range elementShape {
		for i, c := range shapes {
			unusable := c
			t.Run(mod+"/element-shape-"+itoa(i), func(t *testing.T) {
				got, stderr := runReplyProbe(t, mod, []any{unusable, validReply[mod]})
				if got.RC != 0 || got.Attempts != 2 {
					t.Errorf("%s reply %v: rc=%d attempts=%d, want a retry then success; stderr=%s",
						mod, unusable, got.RC, got.Attempts, stderr)
				}
			})
		}
	}
	for _, mod := range []string{"questioner", "answerer", "adjudicator"} {
		for _, u := range unusable {
			t.Run(mod+"/"+u.name, func(t *testing.T) {
				got, stderr := runReplyProbe(t, mod, []any{u.content, validReply[mod]})
				if got.RC != 0 {
					t.Fatalf("%s exited %d after an unusable reply followed by a valid one, want 0 (the "+
						"unusable reply must be retried, not fatal): stderr=%s", mod, got.RC, stderr)
				}
				if got.Attempts != 2 {
					t.Errorf("%s made %d attempts, want 2 (one unusable, one retry)", mod, got.Attempts)
				}
				if len(got.Prompts) == 2 && got.Prompts[0] != got.Prompts[1] {
					t.Errorf("%s retried with a different input; the retry must re-send the same request", mod)
				}
				if !strings.Contains(stderr, "unparseable reply") {
					t.Errorf("%s retried silently; stderr must name the unparseable reply (R1): %q", mod, stderr)
				}
				switch mod {
				case "questioner":
					if !strings.Contains(got.Out, "Is the retry covered?") {
						t.Errorf("questioner output lacks the generated question after the retry: %s", got.Out)
					}
				case "answerer":
					if !strings.Contains(got.Out, `"CONFIDENT"`) {
						t.Errorf("answerer output lacks the answer after the retry: %s", got.Out)
					}
				case "adjudicator":
					if !strings.Contains(got.Out, "RESOLVED") {
						t.Errorf("adjudicator output lacks the verdict after the retry: %s", got.Out)
					}
				}
			})
		}
	}
}

// TestUnusableReplyExhaustionFailsClosedAndNamesTheCause: when every attempt is unusable the script
// must still fail closed — non-zero exit, nothing on stdout that a caller could parse as a result —
// but stderr must say WHY, with the attempt count and the input size, instead of a bare traceback
// that reads as MISSING with no cause (the blis-schemas#100 case was three empty replies in a row).
func TestUnusableReplyExhaustionFailsClosedAndNamesTheCause(t *testing.T) {
	requirePython3(t)
	for _, mod := range []string{"questioner", "answerer", "adjudicator"} {
		t.Run(mod, func(t *testing.T) {
			got, stderr := runReplyProbe(t, mod, []any{""})
			if got.RC == 0 {
				t.Fatalf("%s exited 0 with only unusable replies; it must fail closed", mod)
			}
			if strings.TrimSpace(got.Out) != "" {
				t.Errorf("%s wrote %q to stdout after failing; a caller could read that as a result", mod, got.Out)
			}
			if got.Attempts != 3 {
				t.Errorf("%s made %d attempts, want the whole budget of 3", mod, got.Attempts)
			}
			if !strings.Contains(stderr, "unparseable reply after 3 attempts (input ") ||
				!strings.Contains(stderr, " bytes)") {
				t.Errorf("%s exhaustion message does not name the cause, attempts and input size: %q", mod, stderr)
			}
			if strings.Contains(stderr, "Traceback") {
				t.Errorf("%s crashed with a traceback instead of a named, fail-closed exit: %q", mod, stderr)
			}
		})
	}
}

// questionerCapProbe calls questioner.cap_inputs directly and prints what it returns.
const questionerCapProbe = `
import json, sys, questioner
body = sys.argv[1] * int(sys.argv[2])
diff = sys.argv[3] * int(sys.argv[4])
b, d, notes = questioner.cap_inputs(body, diff, int(sys.argv[5]))
json.dump({"body": b, "diff": d, "notes": notes}, sys.stdout)
`

// TestQuestionerCapsOversizedInput pins the second #1895 trigger: an oversized input. Under the
// threshold the input is byte-identical (the cap is invisible to every normal PR); over it, the body
// and the diff are each cut to a bounded size, at a line boundary, with an explicit marker the model
// can see, and the cut is deterministic (same input, same prompt).
func TestQuestionerCapsOversizedInput(t *testing.T) {
	requirePython3(t)

	run := func(bodyUnit string, bodyN int, diffUnit string, diffN, limit int) (string, string, []string) {
		t.Helper()
		stdout, stderr, code := runPython(t, "", "-c", questionerCapProbe,
			bodyUnit, itoa(bodyN), diffUnit, itoa(diffN), itoa(limit))
		if code != 0 {
			t.Fatalf("cap probe exit=%d stderr=%s", code, stderr)
		}
		var got struct {
			Body  string   `json:"body"`
			Diff  string   `json:"diff"`
			Notes []string `json:"notes"`
		}
		if err := json.Unmarshal([]byte(stdout), &got); err != nil {
			t.Fatalf("cap probe output is not JSON: %v (%s)", err, stdout)
		}
		return got.Body, got.Diff, got.Notes
	}

	t.Run("small-input-is-untouched", func(t *testing.T) {
		b, d, notes := run("body line\n", 10, "+diff line\n", 10, 4096)
		if b != strings.Repeat("body line\n", 10) || d != strings.Repeat("+diff line\n", 10) {
			t.Errorf("an input under the threshold was changed")
		}
		if len(notes) != 0 {
			t.Errorf("an input under the threshold was reported as capped: %v", notes)
		}
	})

	t.Run("oversized-input-is-capped-at-a-line-boundary", func(t *testing.T) {
		const limit = 4096
		b, d, notes := run("body line\n", 2000, "+diff line\n", 2000, limit)
		if total := len(b) + len(d); total > limit {
			t.Errorf("capped body+diff is %d bytes, want <= %d", total, limit)
		}
		for name, s := range map[string]string{"body": b, "diff": d} {
			if !strings.Contains(s, "omitted by the qa-review questioner input cap") {
				t.Errorf("the capped %s carries no marker, so the model cannot tell it is partial", name)
			}
		}
		if !strings.HasPrefix(d, "+diff line\n") || strings.Contains(d, "+diff li\n") {
			t.Errorf("the diff was not cut at a line boundary: %q", d[:min(len(d), 80)])
		}
		if len(notes) != 2 {
			t.Errorf("want one stderr note per capped section, got %v", notes)
		}
		b2, d2, _ := run("body line\n", 2000, "+diff line\n", 2000, limit)
		if b != b2 || d != d2 {
			t.Errorf("capping is not deterministic")
		}
	})

	t.Run("a-tiny-configured-cap-is-raised-to-the-minimum", func(t *testing.T) {
		b, d, notes := run("body line\n", 2000, "+diff line\n", 2000, 10)
		// Raised to the minimum, the cut still keeps real content, not just the two markers.
		if total := len(b) + len(d); total > 4096 || total < 3000 {
			t.Errorf("capped body+diff is %d bytes, want between 3000 and the 4096-byte minimum", total)
		}
		if len(notes) == 0 || !strings.Contains(notes[0], "below the minimum") {
			t.Errorf("raising the cap was not reported: %v", notes)
		}
	})

		t.Run("a-small-body-leaves-its-share-to-the-diff", func(t *testing.T) {
		const limit = 4096
		b, d, _ := run("b\n", 5, "+diff line\n", 2000, limit)
		if b != strings.Repeat("b\n", 5) {
			t.Errorf("a body under its share was changed")
		}
		if len(d) < limit-len(b)-200 {
			t.Errorf("the diff got %d bytes; the unused body share must go to the diff", len(d))
		}
	})
}

// TestQuestionerMainLogsTheCap: main() applies the cap from the default threshold before the call
// and says so on stderr, naming the original and capped sizes.
func TestQuestionerMainLogsTheCap(t *testing.T) {
	requirePython3(t)
	diff := writeFixture(t, "big.diff", strings.Repeat("+a line of an enormous diff\n", 40000)) // ~1.1 MB
	got, stderr := runReplyProbe(t, "questioner", []any{validReply["questioner"]}, diff)
	if got.RC != 0 {
		t.Fatalf("questioner exit=%d stderr=%s", got.RC, stderr)
	}
	if len(got.Prompts) != 1 {
		t.Fatalf("want one call, got %d", len(got.Prompts))
	}
	if n := len(got.Prompts[0]); n > 300*1024 {
		t.Errorf("the questioner sent a %d-byte prompt; the input cap was not applied", n)
	}
	if !strings.Contains(stderr, "input cap") || !strings.Contains(stderr, "diff") {
		t.Errorf("the cap was applied silently; stderr must say so: %q", stderr)
	}
}

func itoa(n int) string {
	b, _ := json.Marshal(n)
	return string(b)
}
