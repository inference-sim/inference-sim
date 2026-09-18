#!/usr/bin/env python3
"""qa-review answerer — cross-vendor answerer (agentic, read-only).

Runs an OpenAI-compatible function-calling loop with read-only tools sandboxed
to ``--worktree``. The answerer investigates the actual PR-head code to answer
each probing question, and runs on a DIFFERENT model family than the questioner
(RFC #1603). To judge the fixed policy questions (does this PR fully implement
the issue it closes? etc.) it also has the read-only ``gh_issue`` (the closing
issue's acceptance criteria) and ``pr_diff`` (the base→head diff) tools — the
same ones the adjudicator uses — so completeness is verifiable in round 0
(#1792), while behavior is still checked against the ACTUAL code.

Tools:
  read_file(path[, start, end])   read a worktree file (optionally a line range)
  grep(pattern[, path])           ripgrep-style search of the worktree
  list_dir(path)                  list a worktree directory
  go(subcommand)                  run ``go build``/``go test`` (dropped by --no-exec)
  gh_issue(number)                read a GitHub issue + comments (acceptance criteria)
  pr_diff(number)                 read the PR base→head diff

read_file/grep/list_dir/go are confined to ``--worktree``; gh_issue/pr_diff are
read-only ``gh`` network calls against ``QA_REPO`` (mirroring the adjudicator).
The PR number and the closing issue number are given to the model (``--pr`` /
``--issue``) so it knows which to query.

Per-question contract: status in {CONFIDENT, CANNOT_ANSWER, FLAW_FOUND} plus
answer, evidence (file:line / repro), and an optional non-blocking note.

Env:
  OPENAI_BASE_URL      LiteLLM proxy base URL (required)
  OPENAI_API_KEY       proxy key; falls back to LITELLM_KEY
  QA_ANSWERER_MODEL    default azure/gpt-5.6-sol
  QA_REPO              owner/repo for gh_issue/pr_diff (default inference-sim/inference-sim)

stdout: [{id,status,answer,evidence,note}]
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request

DEFAULT_MODEL = "azure/gpt-5.6-sol"
MAX_TOOL_TURNS = 24

STATUSES = ("CONFIDENT", "CANNOT_ANSWER", "FLAW_FOUND")

SYSTEM_PROMPT = """You are an isolated, skeptical answerer in a two-agent \
cross-vendor PR review for a discrete-event LLM-inference simulator (BLIS). A \
different model generated the questions; you must answer them by investigating \
the ACTUAL code at the PR head, using the read-only tools provided. To judge the \
fixed policy questions (whether the PR fully implements the issue it closes, \
documentation currency, stale comments) you also have the read-only gh_issue tool \
(the closing issue's acceptance criteria + comments) and pr_diff tool (the \
base→head diff): consult them for F1/F2/F3, but do NOT answer from the diff alone \
when the surrounding code decides the answer — verify against the real code. Never \
trust the PR description over the code.

For each question:
  - Use the tools to gather concrete evidence (file:line ranges, exact values, \
a reproduction).
  - Decide a status:
      CONFIDENT      you verified the answer against the code.
      CANNOT_ANSWER  you could not gather enough evidence to decide.
      FLAW_FOUND     you found a real defect, regression, or unmet contract.
  - Cite evidence as file:line references or a concrete repro.
  - Optionally add a non-blocking `note` for a lesser observation.

CANNOT_ANSWER and FLAW_FOUND are BLOCKING; use them only with justification.

When finished with ALL questions, return ONLY a JSON array:
[{"id": "...", "status": "...", "answer": "...", "evidence": "...", \
"note": "..."}]
Do not wrap it in markdown fences."""


# ---------------------------------------------------------------------------
# Read-only, worktree-sandboxed tool implementations.
# ---------------------------------------------------------------------------

def _safe_path(worktree, path):
    """Resolve path inside the worktree; reject traversal outside it."""
    root = os.path.realpath(worktree)
    target = os.path.realpath(os.path.join(root, path))
    if target != root and not target.startswith(root + os.sep):
        raise ValueError("path escapes the worktree sandbox: %s" % path)
    return target


def tool_read_file(worktree, path, start=None, end=None):
    target = _safe_path(worktree, path)
    with open(target, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    if start is None and end is None:
        body = "".join(lines)
    else:
        s = max(1, int(start or 1))
        e = min(len(lines), int(end or len(lines)))
        body = "".join("%d\t%s" % (i, lines[i - 1]) for i in range(s, e + 1))
    return body


def tool_grep(worktree, pattern, path=None):
    root = os.path.realpath(worktree)
    target = _safe_path(worktree, path) if path else root
    proc = subprocess.run(
        ["grep", "-rnE", "--", pattern, target],
        capture_output=True,
        text=True,
        cwd=root,
    )
    # grep exits 0 with matches, 1 for no-match (no output), >=2 on error. Map
    # no-match to an explicit marker so the model cannot confuse "found nothing"
    # with "the search failed".
    if proc.returncode == 0:
        return proc.stdout
    if proc.returncode == 1:
        return "(no matches)"
    return proc.stderr or "grep failed (exit %d)" % proc.returncode


def tool_list_dir(worktree, path="."):
    target = _safe_path(worktree, path)
    return "\n".join(sorted(os.listdir(target)))


def tool_go(worktree, subcommand):
    """Run a guarded ``go build``/``go test`` against the worktree.

    This tool EXECUTES the code under review; it is dropped by --no-exec so the
    answerer can run on a self-hosted runner without compiling PR-head code.
    """
    allowed = {
        "build": ["go", "build", "./..."],
        "test": ["go", "test", "./..."],
        "vet": ["go", "vet", "./..."],
    }
    argv = allowed.get(subcommand.strip())
    if argv is None:
        return "unsupported go subcommand: %s (allowed: build, test, vet)" % subcommand
    proc = subprocess.run(
        argv, capture_output=True, text=True, cwd=os.path.realpath(worktree)
    )
    return (proc.stdout + proc.stderr)[-8000:]


# gh_issue/pr_diff let the round-0 answerer verify issue-completeness (F1) that the
# worktree code alone cannot decide (#1792). They are read-only gh NETWORK calls, not
# PR-code execution, so they stay under --no-exec (only the code-executing `go` tool is
# dropped). They are DUPLICATED verbatim in adjudicator.py rather than hoisted into a
# shared module: each vendored qa-review script stays self-contained (matching the
# prototype's per-script layout), so the two are kept identical by hand — if you edit one,
# edit both. A shared read-only-gh helper is a possible follow-up.
#
# gh_issue uses `--json ... -q`, NOT the default `--comments` view: the pretty view
# fetches Projects-classic data (repository.issue.projectCards), which this repo's GitHub
# has DEPRECATED, so `gh issue view --comments` exits non-zero with only a deprecation
# notice and never returns the acceptance criteria — the exact failure that would defeat
# the round-0 completeness check. A non-zero gh exit is surfaced as an explicit failure
# marker so the model treats it as missing evidence (leading to CANNOT_ANSWER), never
# mistakes an error string for the issue body or the diff.
_GH_ISSUE_JQ = (
    r'"#\(.number) \(.title)\n\n\(.body)\n\n--- comments ---\n"'
    r' + ([.comments[] | "@\(.author.login): \(.body)"] | join("\n\n"))'
)


def tool_gh_issue(worktree, number):
    """Read-only: fetch a GitHub issue's acceptance criteria + comments (#1792)."""
    repo = os.environ.get("QA_REPO", "inference-sim/inference-sim")
    proc = subprocess.run(
        [
            "gh", "issue", "view", str(number), "--repo", repo,
            "--json", "number,title,body,comments", "-q", _GH_ISSUE_JQ,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return "gh_issue failed (exit %d): %s" % (proc.returncode, proc.stderr.strip()[:2000])
    return proc.stdout[:12000]


def tool_pr_diff(worktree, number):
    """Read-only: fetch the PR base→head diff (#1792)."""
    repo = os.environ.get("QA_REPO", "inference-sim/inference-sim")
    proc = subprocess.run(
        ["gh", "pr", "diff", str(number), "--repo", repo],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return "pr_diff failed (exit %d): %s" % (proc.returncode, proc.stderr.strip()[:2000])
    return proc.stdout[:16000]


# TOOLS_IMPL maps a tool name to its implementation. TOOLS_SCHEMA is the
# OpenAI-compatible function schema advertised to the model. The `go` tool is
# registered in BOTH; the read-only gh_issue/pr_diff tools mirror the adjudicator
# (#1792) and, unlike `go`, are KEPT under --no-exec.
TOOLS_IMPL = {
    "read_file": tool_read_file,
    "grep": tool_grep,
    "list_dir": tool_list_dir,
    "go": tool_go,
    "gh_issue": tool_gh_issue,
    "pr_diff": tool_pr_diff,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file in the worktree, optionally a line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search the worktree with an extended regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List a directory in the worktree.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "go",
            "description": "Run 'go build ./...', 'go test ./...', or 'go vet ./...'.",
            "parameters": {
                "type": "object",
                "properties": {"subcommand": {"type": "string"}},
                "required": ["subcommand"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gh_issue",
            "description": "Read a GitHub issue (acceptance criteria) with comments.",
            "parameters": {
                "type": "object",
                "properties": {"number": {"type": "string"}},
                "required": ["number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pr_diff",
            "description": "Read the PR diff.",
            "parameters": {
                "type": "object",
                "properties": {"number": {"type": "string"}},
                "required": ["number"],
            },
        },
    },
]


def tools_for(no_exec):
    """Return (impl, schema) for the answerer, dropping the code-executing
    `go` tool when no_exec is True while KEEPING the read-only
    read_file/grep/list_dir and gh_issue/pr_diff tools intact.

    Flag absent (no_exec False) => the `go` tool is present in both the
    implementation map and the advertised schema => verbatim prototype
    behavior. This selector is the seam #1715 uses to honor the self-hosted
    runner's no-execution invariant (issue #1714 carve-out 2): gh_issue/pr_diff
    are read-only network calls, not PR-code execution, so they survive the drop
    (#1792), exactly as they do in the adjudicator.
    """
    if no_exec:
        impl = {k: v for k, v in TOOLS_IMPL.items() if k != "go"}
        schema = [t for t in TOOLS_SCHEMA if t["function"]["name"] != "go"]
        return impl, schema
    return dict(TOOLS_IMPL), list(TOOLS_SCHEMA)


def post_chat_completion(base_url, api_key, model, messages, tools):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + api_key)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_tool(impl, worktree, name, arguments):
    fn = impl.get(name)
    if fn is None:
        return "tool not available: %s" % name
    try:
        return fn(worktree, **arguments)
    except Exception as exc:  # surface the error to the model, do not crash
        # Also log to stderr: returning the error only to the model leaves an
        # operator (or CI) blind to a tool that is failing every call.
        sys.stderr.write("qa-review answerer: tool error (%s): %s\n" % (name, exc))
        return "tool error (%s): %s" % (name, exc)


def last_assistant_content(messages):
    """Content of the most recent assistant message ("" when there is none).

    The loop appends a tool RESULT after every assistant turn, so the trailing
    message on exhaustion is normally raw tool output (file contents), never a
    final answer. Only an assistant message can carry one.
    """
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content") or ""
    return ""


def exhausted_answers(questions):
    """CANNOT_ANSWER for every question — the degraded result on exhaustion.

    CANNOT_ANSWER is the contract's "could not gather enough evidence" status
    and is BLOCKING, so a run that ran out of turns can never silently PASS.
    """
    reason = (
        "the answerer exhausted its %d-turn tool budget before returning a "
        "final answer" % MAX_TOOL_TURNS
    )
    return [
        {
            "id": q.get("id", "") if isinstance(q, dict) else str(q),
            "status": "CANNOT_ANSWER",
            "answer": reason,
            "evidence": "",
        }
        for q in questions
    ]


def answerer_user_message(questions, pr="", issue=""):
    """Assemble the answerer's user turn: a short preamble naming the PR and the
    closing issue so the model knows which numbers to pass to the pr_diff /
    gh_issue tools when judging the fixed policy questions, then the questions.

    Both numbers are optional. With neither known the preamble is omitted and the
    turn is byte-identical to the pre-#1792 questions-only prompt (the tools are
    still advertised, so a caller can surface the numbers another way)."""
    hints = []
    if pr:
        hints.append(
            "the PR under review is #%s (use pr_diff(number=%s) to read the diff)" % (pr, pr)
        )
    if issue:
        hints.append(
            "the issue it closes is #%s (use gh_issue(number=%s) for its acceptance criteria)"
            % (issue, issue)
        )
    parts = []
    if hints:
        parts.append(
            "Context for the fixed policy questions (F1/F2/F3): "
            + "; ".join(hints)
            + ". Always verify behavior against the actual code with read_file/grep/list_dir."
        )
    parts.append(
        "Answer these questions by investigating the code:\n"
        + json.dumps(questions, indent=2)
    )
    return "\n\n".join(parts)


def answer_loop(base_url, api_key, model, worktree, questions, no_exec, pr="", issue=""):
    impl, schema = tools_for(no_exec)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": answerer_user_message(questions, pr, issue)},
    ]
    for _ in range(MAX_TOOL_TURNS):
        resp = post_chat_completion(base_url, api_key, model, messages, schema)
        msg = resp["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return msg.get("content", "")
        messages.append(msg)
        for call in tool_calls:
            name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                # Keep the loop alive on malformed model output, but do not let
                # it pass silently — an operator must be able to see the model
                # emitted unparseable tool arguments.
                sys.stderr.write(
                    "qa-review answerer: tool %r had unparseable arguments: %r\n"
                    % (name, call["function"].get("arguments"))
                )
                arguments = {}
            result = run_tool(impl, worktree, name, arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": str(result)[:12000],
                }
            )
    # Tool budget exhausted. Exhaustion is a normal outcome of a finite budget,
    # not an error, so the loop must still hand back something parse_answers()
    # accepts — returning the trailing message would hand json.loads a raw tool
    # result and crash. Honor a final answer array the assistant already
    # produced alongside its tool calls; otherwise degrade to CANNOT_ANSWER.
    sys.stderr.write(
        "qa-review answerer: tool budget (%d turns) exhausted\n" % MAX_TOOL_TURNS
    )
    try:
        answers = parse_answers(last_assistant_content(messages))
    except (ValueError, IndexError):
        answers = None
    if not isinstance(answers, list) or not answers:
        answers = exhausted_answers(questions)
    return json.dumps(answers)


def parse_answers(content):
    """Parse the final answer array, tolerating markdown fences."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[: -3]
    return json.loads(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description="qa-review answerer")
    parser.add_argument("--worktree", required=True, help="read-only PR-head worktree")
    parser.add_argument("--questions", default="", help="questions JSON (inline)")
    parser.add_argument("--questions-file", default="", help="questions JSON (file)")
    parser.add_argument("--pr", default="", help="PR number (surfaced for the pr_diff tool)")
    parser.add_argument(
        "--issue", default="", help="closing issue number (surfaced for the gh_issue tool)"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("QA_ANSWERER_MODEL", DEFAULT_MODEL),
        help="answerer model (default from QA_ANSWERER_MODEL)",
    )
    parser.add_argument(
        "--no-exec",
        action="store_true",
        help="drop the code-executing 'go' tool (self-hosted-runner safe)",
    )
    args = parser.parse_args(argv)

    base_url = os.environ.get("OPENAI_BASE_URL", "")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LITELLM_KEY", "")
    if not base_url:
        sys.stderr.write("OPENAI_BASE_URL is required\n")
        return 2
    if not api_key:
        sys.stderr.write("OPENAI_API_KEY (or LITELLM_KEY) is required\n")
        return 2

    if args.questions:
        raw = args.questions
    elif args.questions_file:
        with open(args.questions_file, "r", encoding="utf-8") as fh:
            raw = fh.read()
    else:
        raw = sys.stdin.read()
    parsed = json.loads(raw)
    questions = parsed["questions"] if isinstance(parsed, dict) else parsed

    content = answer_loop(
        base_url,
        api_key,
        args.model,
        args.worktree,
        questions,
        args.no_exec,
        pr=args.pr,
        issue=args.issue,
    )
    answers = parse_answers(content)

    json.dump(answers, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
