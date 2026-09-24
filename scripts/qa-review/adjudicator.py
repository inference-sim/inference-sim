#!/usr/bin/env python3
"""qa-review adjudicator — author-defence re-check (used by #1716).

Reads the most recent qa-review REPORT comment's "Items to fix"
(select_report_comment + parse_items_to_fix) and every later comment (the PR
author's responses), then adjudicates each prior blocking finding against the
author's defence and the current code.

Source-comment selection is structural, not a substring search: a comment
qualifies only if it IS a rendered report — an unquoted, unfenced
"## qa-review — PR #" heading AND an "### Items to fix" section — and, when
--report-author/QA_REPORT_AUTHOR is set, only if that login posted it. Keying
on the bare banner substring let a later comment that merely QUOTED it (a
self-review discussing the findings, or one pasting an example report inside
``` fences) hijack the selection; with no Items-to-fix section of its own that
comment yielded zero findings and a vacuous PASS — fail-OPEN on the one signal
the delivery gate treats as fail-closed (observed on PR #1736).

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
#1716 derives the gate marker from that line).

Exit codes: 0 a verdict was emitted, 2 missing proxy configuration, 3 no prior
qa-review report comment to adjudicate (deliberately NOT a PASS: there is
nothing to re-check, so a verdict would be vacuous).

Env:
  OPENAI_BASE_URL       LiteLLM proxy base URL (required)
  OPENAI_API_KEY        proxy key; falls back to LITELLM_KEY
  QA_ADJUDICATOR_MODEL  default azure/gpt-5.6-sol
  QA_REPORT_AUTHOR      restrict the prior-report search to this comment
                        author login (empty = any author)
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

# render_report.py emits this exact bullet when a report has no blocking findings;
# it is the ONLY non-finding line that legitimately fills the Items-to-fix section
# (a genuine PASS report). Matched tolerant of whitespace/rendering.
_NO_FINDINGS_RE = re.compile(r"^\s*-\s+_None\b.*no blocking findings", re.IGNORECASE)


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
    # 0 = matches, 1 = no-match (no output), >=2 = error. Distinguish no-match
    # from failure so the model can tell "found nothing" from "search broke".
    if proc.returncode == 0:
        return proc.stdout
    if proc.returncode == 1:
        return "(no matches)"
    return proc.stderr or "grep failed (exit %d)" % proc.returncode


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


# gh_issue/pr_diff are read-only gh NETWORK calls, not PR-code execution, so they stay
# under --no-exec. They are kept IDENTICAL to answerer.py's copies (#1792 gave the answerer
# the same tools); each vendored qa-review script stays self-contained rather than sharing a
# module, so the two are kept in sync by hand — if you edit one, edit both.
#
# gh_issue reads the issue HEAD (number/title/body) via `--json ... -q`, NOT the default
# `--comments` view: the pretty view fetches Projects-classic data
# (repository.issue.projectCards), which this repo's GitHub has DEPRECATED, so
# `gh issue view --comments` exits non-zero with only a deprecation notice and never returns
# the acceptance criteria. A non-zero gh exit is surfaced as an explicit failure marker so
# the model treats it as missing evidence, never mistakes an error string for data.
#
# The issue COMMENTS are read through scripts/deliver-trusted-comments.sh (#1806), NOT
# inline: this repository is public, so a stranger's comment on the closing issue is a
# prompt-injection surface into this LLM. The filter returns only comments whose author
# holds write access (plus this repo's automation); a read failure surfaces as an UNREAD
# marker rather than an empty thread, so a failed read is never mistaken for "no comments".
_GH_ISSUE_HEAD_JQ = r'"#\(.number) \(.title)\n\n\(.body)"'

# scripts/deliver-trusted-comments.sh, resolved relative to THIS file (scripts/qa-review/)
# so it is found regardless of the caller's working directory.
_TRUSTED_COMMENTS_SH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
    "deliver-trusted-comments.sh",
)


def _trusted_comment_lines(repo, mode, number):
    """Return the write-access-filtered comments as `@login: body` blocks (#1806).

    `mode` is "--issue" or "--pr". On ANY read failure the filter exits non-zero (its first
    line is COMMENT-READ-FAILED); this returns that marker so the caller renders the channel
    as UNREAD rather than as an empty thread — a stranger's comment must never reach the
    model, and a failed read must never look like "no comments"."""
    proc = subprocess.run(
        ["bash", _TRUSTED_COMMENTS_SH, "--json", mode, str(number)],
        capture_output=True,
        text=True,
        env=dict(os.environ, GH_REPO=repo),
    )
    if proc.returncode != 0:
        return "COMMENT-READ-FAILED: the comment channel could not be read (%s)" % (
            proc.stderr.strip()[:500] or "no detail"
        )
    try:
        comments = json.loads(proc.stdout).get("comments", [])
    except json.JSONDecodeError:
        return "COMMENT-READ-FAILED: the comment filter did not return JSON"
    return "\n\n".join(
        "@%s: %s" % ((c.get("author") or {}).get("login", ""), c.get("body", ""))
        for c in comments
    )


def tool_gh_issue(worktree, number):
    """Read-only: a GitHub issue's acceptance criteria + WRITE-ACCESS-FILTERED comments
    (#1792, #1806)."""
    repo = os.environ.get("QA_REPO", "inference-sim/inference-sim")
    proc = subprocess.run(
        [
            "gh", "issue", "view", str(number), "--repo", repo,
            "--json", "number,title,body", "-q", _GH_ISSUE_HEAD_JQ,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return "gh_issue failed (exit %d): %s" % (proc.returncode, proc.stderr.strip()[:2000])
    comments = _trusted_comment_lines(repo, "--issue", number)
    return (proc.stdout + "\n\n--- comments ---\n" + comments)[:12000]


def tool_pr_diff(worktree, number):
    """Read-only: fetch the PR base→head diff."""
    repo = os.environ.get("QA_REPO", "inference-sim/inference-sim")
    proc = subprocess.run(
        ["gh", "pr", "diff", str(number), "--repo", repo],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return "pr_diff failed (exit %d): %s" % (proc.returncode, proc.stderr.strip()[:2000])
    return proc.stdout[:16000]


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
        # Also log to stderr so a tool failing every call is visible to CI.
        sys.stderr.write("qa-review adjudicator: tool error (%s): %s\n" % (name, exc))
        return "tool error (%s): %s" % (name, exc)


def last_assistant_content(messages):
    """Content of the most recent assistant message ("" when there is none).

    The loop appends a tool RESULT after every assistant turn, so the trailing
    message on exhaustion is normally raw tool output (file contents), never a
    final verdict array. Only an assistant message can carry one.
    """
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content") or ""
    return ""


def exhausted_verdicts(items):
    """STILL_OPEN for every prior finding — the degraded result on exhaustion.

    STILL_OPEN is the documented skeptical default, so running out of turns
    blocks rather than clearing a finding it never actually adjudicated.
    """
    rationale = (
        "the adjudicator exhausted its %d-turn tool budget before returning a "
        "verdict; STILL_OPEN is the skeptical default" % MAX_TOOL_TURNS
    )
    return [
        {
            "id": item.get("id", ""),
            "was": item.get("was", ""),
            "verdict": "STILL_OPEN",
            "rationale": rationale,
        }
        for item in items
    ]


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
                sys.stderr.write(
                    "qa-review adjudicator: tool %r had unparseable arguments: %r\n"
                    % (name, call["function"].get("arguments"))
                )
                arguments = {}
            result = run_tool(impl, worktree, name, arguments)
            messages.append(
                {"role": "tool", "tool_call_id": call.get("id", ""), "content": str(result)[:12000]}
            )
    # Tool budget exhausted. Exhaustion is a normal outcome of a finite budget,
    # not an error, so the loop must still hand back something parse_verdicts()
    # accepts — returning the trailing message would hand json.loads a raw tool
    # result and crash. Honor a final verdict array the assistant already
    # produced alongside its tool calls; otherwise degrade to STILL_OPEN.
    sys.stderr.write(
        "qa-review adjudicator: tool budget (%d turns) exhausted\n" % MAX_TOOL_TURNS
    )
    try:
        verdicts = parse_verdicts(last_assistant_content(messages))
    except (ValueError, IndexError):
        verdicts = None
    if not isinstance(verdicts, list) or not verdicts:
        verdicts = exhausted_verdicts(items)
    return json.dumps(verdicts)


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


# ---------------------------------------------------------------------------
# Source-comment selection (#1716).
# ---------------------------------------------------------------------------

# render_report.py's own two structural landmarks: the top-level verdict header
# and the blocking-findings section. Both are emitted on every report, PASS or
# BLOCK, so requiring both identifies a report without assuming its verdict.
_REPORT_HEADING = "## qa-review — pr #"
_ITEMS_HEADING = "### items to fix"


def _leading_cols(raw):
    """Leading indentation of `raw` in columns, tabs expanded to a stop of 4
    (CommonMark). Distinguishes a >=4-column indented code block from ordinary
    text without counting a tab as a single column."""
    cols = 0
    for ch in raw:
        if ch == " ":
            cols += 1
        elif ch == "\t":
            cols += 4 - (cols % 4)
        else:
            break
    return cols


def significant_lines(body):
    """Yield `body`'s lines, stripped, with HTML comments, fenced code blocks,
    blockquotes, and indented code blocks dropped.

    Those are how a comment QUOTES or HIDES a report it is discussing rather
    than being one — an example report pasted inside ``` fences, indented four
    columns as a code block, quoted with `> `, or concealed inside an
    `<!-- ... -->` HTML comment that renders invisibly to a human yet still
    carries the structural landmarks (#1716 G1/G6). Dropping all of them is what
    stops a discussion of the findings — or a deliberately hidden report shape —
    from being mistaken for the report that raised them. A genuine report
    contains none of them (render_report.py emits its landmarks unfenced,
    unindented and unquoted, and only its optional banner is a blockquote), so
    nothing a report needs is lost.

    HTML comments are removed span-wise (multi-line, and an unclosed `<!--`
    through end-of-body), and each span is replaced by the newlines it spanned —
    or a single newline when it spanned none — so the fragments on either side,
    whether same-line (`## qa-<!--x-->review`) or across a line break, can never
    be fused into a synthetic landmark line (#1716 G1). Removal can therefore only
    DELETE landmarks or leave blank lines behind — never synthesise one."""
    body = re.sub(
        r"<!--.*?(?:-->|$)",
        lambda m: "\n" * max(1, m.group(0).count("\n")),
        body,
        flags=re.DOTALL,
    )
    fence_char = ""  # "" when not in a fence; otherwise the fence char "`" or "~"
    fence_len = 0
    for raw in body.splitlines():
        cols = _leading_cols(raw)
        stripped = raw.lstrip(" \t")
        # A fence marker is a run of >=3 of the same char (` or ~) at <=3 columns
        # of indentation (CommonMark; 4+ columns is indented code, not a fence).
        marker_char, marker_len = "", 0
        if cols <= 3 and stripped[:1] in ("`", "~"):
            ch = stripped[0]
            run = len(stripped) - len(stripped.lstrip(ch))
            if run >= 3:
                marker_char, marker_len = ch, run
        if fence_char:
            # Inside a fence: only a genuine CLOSING fence ends it — the SAME char, a
            # run at least as long as the opener, and nothing but whitespace after it.
            # A shorter run, a different char, or a trailing info string does NOT close
            # the fence, so ``` cannot close ````, and ~~~ cannot close ``` (the #1716
            # G1 bug: a mismatched marker toggled the block off early and exposed a
            # quoted report heading as if it were a real one).
            if (marker_char == fence_char and marker_len >= fence_len
                    and stripped[marker_len:].strip() == ""):
                fence_char, fence_len = "", 0
            continue
        if marker_char:
            # Opening a new fence (an info string after the run is allowed).
            fence_char, fence_len = marker_char, marker_len
            continue
        # A line indented >=4 columns is a CommonMark indented code block, not a
        # structural landmark — drop it so an indented copy of the headings cannot
        # pose as a genuine report (#1716 G6). render_report.py's landmarks sit at
        # column 0.
        if cols >= 4:
            continue
        line = raw.strip()
        if line.startswith(">"):
            continue
        yield line


def is_report_comment(body):
    """True when `body` IS a rendered qa-review report rather than a comment
    that quotes or discusses one.

    Requires the verdict header, the Items-to-fix heading, AND a non-degenerate
    Items-to-fix section — at least one line under it that is a finding bullet
    (`- **ID · STATUS**`) or render_report.py's explicit "no blocking findings"
    sentinel. A report SHAPE whose Items-to-fix section is empty parses to zero
    findings, which the adjudicator would otherwise clear as a vacuous aggregate
    PASS; requiring content keeps a genuine PASS report (which carries the
    sentinel) selectable while rejecting an empty shell that a newer comment
    could use to supersede a real report's findings (#1716 G2)."""
    has_heading = False
    in_items = False
    items_has_content = False
    for line in significant_lines(body):
        lowered = line.lower()
        if lowered.startswith(_REPORT_HEADING):
            has_heading = True
            in_items = False
        elif lowered.startswith("### "):
            # Any next section heading closes the Items-to-fix section; only the
            # Items-to-fix heading (re)opens it.
            in_items = lowered.startswith(_ITEMS_HEADING)
        elif in_items and (_ITEM_RE.match(line) or _NO_FINDINGS_RE.match(line)):
            # Agree with parse_items_to_fix: only a real finding bullet (_ITEM_RE) or
            # render_report's "no blocking findings" sentinel counts as content. A bare
            # "- ..." line parses to ZERO findings, so accepting it here would let a
            # malformed-bullet report shell supersede a real report and clear as a
            # vacuous aggregate PASS (#1716 G2).
            items_has_content = True
    return has_heading and items_has_content


def select_report_comment(comments, report_author=""):
    """Index of the most recent genuine qa-review report comment, or -1.

    A comment qualifies iff is_report_comment() accepts its body and, when
    `report_author` is given, that login posted it. The author restriction is
    strict on purpose: this runs against a PUBLIC repository, so without it any
    commenter could post a report-shaped comment with an empty Items-to-fix
    section and clear every outstanding finding. It is empty by default so the
    tool stays usable by hand, where the report's poster is whoever ran it."""
    chosen = -1
    for i, c in enumerate(comments):
        if not is_report_comment(c.get("body") or ""):
            continue
        if report_author and (c.get("author") or {}).get("login", "") != report_author:
            continue
        chosen = i
    return chosen


def fetch_comments(repo, pr, report_author=""):
    """Return (items_to_fix, later_author_responses) from the PR's comments.

    The most recent genuine qa-review REPORT comment supplies the prior
    blocking findings; every comment after it is treated as the author's
    defence. `items` is None — distinct from an empty list, which is a real
    report with no blocking findings — when no report comment was found at all,
    so a caller can refuse rather than adjudicate nothing.

    Comments are read through scripts/deliver-trusted-comments.sh (#1806), so a stranger's
    comment on this PUBLIC PR never enters the adjudication prompt. The filter also returns
    PR reviews and inline review comments; only CONVERSATION comments carry the qa-review
    report and the author's later defence, so the others are ignored here — preserving this
    selection's pre-#1806 behaviour. A read failure exits non-zero (check=True), which the
    caller already treats as 'no adjudicable evidence' rather than a clean pass."""
    proc = subprocess.run(
        ["bash", _TRUSTED_COMMENTS_SH, "--json", "--pr", str(pr)],
        capture_output=True,
        text=True,
        env=dict(os.environ, GH_REPO=repo),
        check=True,
    )
    comments = [
        c for c in json.loads(proc.stdout).get("comments", [])
        if c.get("source") == "conversation"
    ]
    last_qa = select_report_comment(comments, report_author)
    if last_qa < 0:
        return None, ""
    items = parse_items_to_fix(comments[last_qa].get("body") or "")
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
    parser.add_argument(
        "--report-author",
        default=os.environ.get("QA_REPORT_AUTHOR", ""),
        help="only adjudicate a prior report posted by this comment author login "
        "(empty = any author; see select_report_comment)",
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

    items, responses = fetch_comments(args.repo, args.pr, args.report_author)
    if items is None:
        # NOT a PASS. There is no prior report to re-check, so any verdict would
        # be vacuous — and the consumer that turns this into a gate signal reads
        # the verdict line, so emitting one here would clear the qa dimension
        # without anything having been reviewed.
        sys.stderr.write(
            "no qa-review report comment%s was found on #%s, so there are no prior "
            "findings to adjudicate; refusing to emit a verdict\n"
            % ((" from '%s'" % args.report_author) if args.report_author else "", args.pr)
        )
        return 3
    if not items:
        # A real report whose Items-to-fix section is empty: it found nothing
        # blocking, so there is genuinely nothing left open. That is a PASS on
        # the strength of a review that ran — unlike the `items is None` case
        # above, where no review was found at all.
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
