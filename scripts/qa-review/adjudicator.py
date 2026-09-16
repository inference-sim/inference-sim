#!/usr/bin/env python3
"""qa-review adjudicator — author-defence re-check (used by #1716).

Reads the most recent qa-review comment's "Items to fix" (parse_items_to_fix)
and every later comment (the PR author's responses), then adjudicates each
prior blocking finding against the author's defence and the current code.

Per-finding verdict:
  RESOLVED               the finding is fixed in the current code.
  WAIVED_JUSTIFICATION   the author gave a valid reason it is not a defect.
  WAIVED_DEFERRED        legitimately deferred to a tracked follow-up.
  STILL_OPEN             not resolved / not validly waived (the default).

Skeptical default-block: an unrecognized or missing per-finding verdict counts
as STILL_OPEN. Ambiguity-pushback: on a genuine acceptance-criterion
interpretation fork, return STILL_OPEN and ask the author to pin the
interpretation rather than guessing.

Aggregate verdict: BLOCK iff ANY finding is STILL_OPEN or left un-adjudicated;
emitted on STDERR as "[adjudication verdict: PASS|BLOCK]" (not a PR marker —
#1716 derives the gate marker).

Env:
  OPENAI_BASE_URL       LiteLLM proxy base URL (required)
  OPENAI_API_KEY        proxy key; falls back to LITELLM_KEY
  QA_ADJUDICATOR_MODEL  default azure/gpt-5.6-sol
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request

DEFAULT_MODEL = "azure/gpt-5.6-sol"
MAX_TOOL_TURNS = 24

# The four per-finding verdicts. Only STILL_OPEN blocks; any value not in this
# set is treated as STILL_OPEN (skeptical default).
CLEARED_VERDICTS = {"RESOLVED", "WAIVED_JUSTIFICATION", "WAIVED_DEFERRED"}
VERDICT_EMOJI = {
    "RESOLVED": "✅ RESOLVED",
    "WAIVED_JUSTIFICATION": "✅ WAIVED_JUSTIFICATION",
    "WAIVED_DEFERRED": "✅ WAIVED_DEFERRED",
    "STILL_OPEN": "⛔ STILL_OPEN",
}

SYSTEM_PROMPT = """You are a skeptical adjudicator in a two-agent cross-vendor \
PR review for a discrete-event LLM-inference simulator (BLIS). A prior review \
raised blocking findings; the PR author has since responded and pushed \
changes. For EACH prior blocking finding, decide whether it is now cleared, \
using only the read-only tools provided (the current worktree, the closing \
issue, and the PR diff).

Verdicts:
  RESOLVED              the finding is fixed in the current code (cite it).
  WAIVED_JUSTIFICATION  the author gave a valid reason it was never a defect.
  WAIVED_DEFERRED       legitimately deferred to a tracked follow-up issue.
  STILL_OPEN            not resolved and not validly waived. THIS IS THE \
DEFAULT — choose it unless the evidence clearly clears the finding.

If clearing a finding depends on a genuine fork in interpreting the \
acceptance criteria, do NOT guess: return STILL_OPEN and ask the author to \
pin the interpretation.

Return ONLY a JSON array:
[{"id": "...", "was": "...", "verdict": "...", "rationale": "..."}]
Do not wrap it in markdown fences."""


# ---------------------------------------------------------------------------
# "Items to fix" parsing.
# ---------------------------------------------------------------------------

# A blocking bullet rendered by render_report.py looks like:
#   - **F2 · FLAW_FOUND** — <text>
_ITEM_RE = re.compile(r"^\s*-\s+\*\*(?P<id>[A-Za-z]\d+)\s*·\s*(?P<was>[A-Z_]+)\*\*\s*—\s*(?P<text>.*)$")


def parse_items_to_fix(comment_body):
    """Extract (id, was, text) findings from a rendered qa-review comment's
    "Items to fix" section. Returns [] when the section is empty/absent."""
    items = []
    in_section = False
    for line in comment_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            in_section = stripped.lower().startswith("### items to fix")
            continue
        if not in_section:
            continue
        m = _ITEM_RE.match(line)
        if m:
            items.append(
                {"id": m.group("id"), "was": m.group("was"), "text": m.group("text").strip()}
            )
    return items


# ---------------------------------------------------------------------------
# Read-only tools (worktree + gh). `go` executes code; --no-exec drops it.
# ---------------------------------------------------------------------------

def _safe_path(worktree, path):
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
        return "".join(lines)
    s = max(1, int(start or 1))
    e = min(len(lines), int(end or len(lines)))
    return "".join("%d\t%s" % (i, lines[i - 1]) for i in range(s, e + 1))


def tool_grep(worktree, pattern, path=None):
    root = os.path.realpath(worktree)
    target = _safe_path(worktree, path) if path else root
    proc = subprocess.run(
        ["grep", "-rnE", "--", pattern, target], capture_output=True, text=True, cwd=root
    )
    return proc.stdout or proc.stderr


def tool_list_dir(worktree, path="."):
    return "\n".join(sorted(os.listdir(_safe_path(worktree, path))))


def tool_go(worktree, subcommand):
    """Guarded ``go build``/``go test`` — EXECUTES the code under review.
    Dropped by --no-exec."""
    allowed = {
        "build": ["go", "build", "./..."],
        "test": ["go", "test", "./..."],
        "vet": ["go", "vet", "./..."],
    }
    argv = allowed.get(subcommand.strip())
    if argv is None:
        return "unsupported go subcommand: %s (allowed: build, test, vet)" % subcommand
    proc = subprocess.run(argv, capture_output=True, text=True, cwd=os.path.realpath(worktree))
    return (proc.stdout + proc.stderr)[-8000:]


def tool_gh_issue(worktree, number):
    """Read-only: fetch a GitHub issue body/comments (the acceptance criteria)."""
    repo = os.environ.get("QA_REPO", "inference-sim/inference-sim")
    proc = subprocess.run(
        ["gh", "issue", "view", str(number), "--repo", repo, "--comments"],
        capture_output=True,
        text=True,
    )
    return (proc.stdout + proc.stderr)[:12000]


def tool_pr_diff(worktree, number):
    """Read-only: fetch the PR diff."""
    repo = os.environ.get("QA_REPO", "inference-sim/inference-sim")
    proc = subprocess.run(
        ["gh", "pr", "diff", str(number), "--repo", repo],
        capture_output=True,
        text=True,
    )
    return (proc.stdout + proc.stderr)[:16000]


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
                "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
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
    """Return (impl, schema) for the adjudicator, dropping the code-executing
    `go` tool when no_exec is True while keeping the read-only gh_issue/pr_diff
    tools intact.

    Flag absent (no_exec False) => the `go` tool is present => verbatim
    prototype behavior (issue #1714 carve-out 2)."""
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
    except Exception as exc:
        return "tool error (%s): %s" % (name, exc)


def adjudicate_loop(base_url, api_key, model, worktree, items, responses, no_exec):
    impl, schema = tools_for(no_exec)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Prior blocking findings:\n"
            + json.dumps(items, indent=2)
            + "\n\nThe author's later responses:\n"
            + responses,
        },
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
                arguments = {}
            result = run_tool(impl, worktree, name, arguments)
            messages.append(
                {"role": "tool", "tool_call_id": call.get("id", ""), "content": str(result)[:12000]}
            )
    return messages[-1].get("content", "") if messages else ""


def parse_verdicts(content):
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
    return json.loads(text)


def render(items, verdicts, pr, amodel, banner=None):
    """Render the adjudication comment and compute the aggregate verdict.

    BLOCK iff any prior finding is STILL_OPEN OR left un-adjudicated (an id in
    `items` with no verdict blocks). Returns (report, aggregate_verdict)."""
    by_id = {v.get("id"): v for v in verdicts}

    aggregate = "PASS"
    for item in items:
        v = by_id.get(item["id"])
        if v is None or v.get("verdict") not in CLEARED_VERDICTS:
            aggregate = "BLOCK"
            break

    emoji = "⛔" if aggregate == "BLOCK" else "✅"
    lines = []
    if banner:
        lines.append(banner)
        lines.append("")
    lines.append("## qa-review adjudication — PR #%s: %s %s" % (pr, emoji, aggregate))
    lines.append(
        "_re-checking %d prior blocking finding(s) · adjudicator `%s`_" % (len(items), amodel)
    )
    lines.append("")

    lines.append("| Prior finding | Was | Verdict | Rationale |")
    lines.append("|----|----|----|----|")
    for item in items:
        v = by_id.get(item["id"], {})
        verdict = v.get("verdict", "STILL_OPEN")
        label = VERDICT_EMOJI.get(verdict, "⛔ STILL_OPEN")
        rationale = (v.get("rationale") or "").replace("|", "\\|").replace("\n", " ").strip()
        lines.append("| %s | %s | %s | %s |" % (item["id"], item.get("was", ""), label, rationale))
    lines.append("")

    still = [
        item
        for item in items
        if by_id.get(item["id"], {}).get("verdict") not in CLEARED_VERDICTS
    ]
    lines.append("### Still blocking")
    if still:
        for item in still:
            v = by_id.get(item["id"], {})
            lines.append(
                "- **%s** (%s) — %s"
                % (item["id"], v.get("verdict", "STILL_OPEN"), (v.get("rationale") or "").strip())
            )
    else:
        lines.append("- _None — all prior findings resolved or validly waived._")
    lines.append("")

    cleared = [
        item for item in items if by_id.get(item["id"], {}).get("verdict") in CLEARED_VERDICTS
    ]
    lines.append("### Cleared")
    if cleared:
        for item in cleared:
            v = by_id[item["id"]]
            lines.append(
                "- **%s** (%s) — %s"
                % (item["id"], VERDICT_EMOJI.get(v["verdict"], v["verdict"]), (v.get("rationale") or "").strip())
            )
    else:
        lines.append("- _None._")

    return "\n".join(lines) + "\n", aggregate


def default_banner(amodel):
    return (
        "> 🤖 **qa-review adjudication** — automated re-check of the prior "
        "blocking findings against the PR author's responses (adjudicator `%s`, "
        "RFC #1603). No maintainer approval required — the author defends, the "
        "gate judges. **Not an official merge gate.**" % amodel
    )


def fetch_comments(repo, pr):
    """Return (items_to_fix, later_author_responses) from the PR's comments.

    The most recent qa-review comment supplies the prior blocking findings;
    every comment after it is treated as the author's defence."""
    proc = subprocess.run(
        ["gh", "pr", "view", str(pr), "--repo", repo, "--json", "comments"],
        capture_output=True,
        text=True,
        check=True,
    )
    comments = json.loads(proc.stdout).get("comments", [])
    last_qa = -1
    for i, c in enumerate(comments):
        if "## qa-review — PR #" in c.get("body", ""):
            last_qa = i
    if last_qa < 0:
        return [], ""
    items = parse_items_to_fix(comments[last_qa]["body"])
    responses = "\n\n".join(c.get("body", "") for c in comments[last_qa + 1 :])
    return items, responses


def main(argv=None):
    parser = argparse.ArgumentParser(description="qa-review adjudicator")
    parser.add_argument("--worktree", required=True, help="read-only PR-head worktree")
    parser.add_argument("--pr", required=True, help="PR number")
    parser.add_argument("--repo", default=os.environ.get("QA_REPO", "inference-sim/inference-sim"))
    parser.add_argument("--title", default="", help="PR title")
    parser.add_argument(
        "--model",
        default=os.environ.get("QA_ADJUDICATOR_MODEL", DEFAULT_MODEL),
        help="adjudicator model (default from QA_ADJUDICATOR_MODEL)",
    )
    parser.add_argument("--out", default="", help="write the report here (else stdout)")
    parser.add_argument("--post-to-pr", action="store_true", help="post as a PR comment")
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

    items, responses = fetch_comments(args.repo, args.pr)
    if not items:
        sys.stderr.write("[adjudication verdict: PASS]\n")
        report, _ = render([], [], args.pr, args.model, default_banner(args.model))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(report)
        else:
            sys.stdout.write(report)
        return 0

    content = adjudicate_loop(
        base_url, api_key, args.model, args.worktree, items, responses, args.no_exec
    )
    verdicts = parse_verdicts(content)

    banner = default_banner(args.model)
    report, aggregate = render(items, verdicts, args.pr, args.model, banner)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)
    else:
        sys.stdout.write(report)

    if args.post_to_pr:
        subprocess.run(
            ["gh", "pr", "comment", str(args.pr), "--repo", args.repo, "--body", report],
            check=True,
        )

    sys.stderr.write("[adjudication verdict: %s]\n" % aggregate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
