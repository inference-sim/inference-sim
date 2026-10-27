#!/usr/bin/env python3
"""qa-review questioner — cross-vendor probing-question generator.

Emits three FIXED policy questions (F1..F3) verbatim plus model-generated,
topic-seeded questions (G1..) covering eight review topics. One POST to the
LiteLLM proxy's OpenAI-compatible /chat/completions surface.

The questioner deliberately runs on a DIFFERENT model family than the
code-writer and the primary reviewer (RFC #1603): a decorrelated second opinion
catches what a single-vendor review misses.

Env:
  OPENAI_BASE_URL       LiteLLM proxy base URL (required)
  OPENAI_API_KEY        proxy key; falls back to LITELLM_KEY
  QA_QUESTIONER_MODEL   default gcp/gemini-3.6-flash
  QA_QUESTIONER_MAX_INPUT_BYTES
                        cap on PR body + diff bytes sent to the model (default
                        262144 = 256 KiB; see MAX_INPUT_BYTES)
  QA_HTTP_*             transport timeout/retry knobs — see _http.py

An empty or unparseable completion is retried with _http.py's bounded backoff
(#1895). When every attempt is unusable the script exits 1 with nothing on
stdout and a stderr line "unparseable reply after N attempts (input M bytes)",
so the gate still reads MISSING, now with a named cause.

stdout: {"model", "questions":[{id,topic,question}]}
"""

import argparse
import json
import os
import re
import sys

# The one canonical LLM transport, shared with answerer.py and adjudicator.py:
# a bounded retry with backoff around each completion so a transient gateway
# error no longer crashes the review (#1833). Imported as a module attribute so
# tests can still monkeypatch post_chat_completion.
from _http import post_chat_completion  # noqa: F401 — re-exported call target
from _http import UnparseableReply, _env_number

# The eight review topics the generated (G) questions are seeded across.
TOPICS = [
    "correctness",
    "invariants",
    "rules",
    "parity",
    "tests",
    "rationale",
    "scope",
    "errors",
]

# The three fixed policy questions, asked verbatim of every PR. They encode the
# repository's standing review policy (issue implementation completeness,
# documentation currency, stale comments) so the model cannot omit them.
FIXED_QUESTIONS = [
    {
        "id": "F1",
        "topic": "fixed",
        "question": "Does this PR fully implement the issue it closes, without unnecessary additions?",
    },
    {
        "id": "F2",
        "topic": "fixed",
        "question": "Has all documentation been updated to reflect the changes in this PR?",
    },
    {
        "id": "F3",
        "topic": "fixed",
        "question": "Are there any stale code comments left by this change?",
    },
]

DEFAULT_MODEL = "gcp/gemini-3.6-flash"

SYSTEM_PROMPT = """You are a skeptical, senior reviewer generating probing \
questions about a GitHub pull request for a discrete-event LLM-inference \
simulator (BLIS). You do NOT answer the questions; a separate isolated \
answerer will investigate the actual code to answer them.

Generate sharp, specific, evidence-seeking questions that a rigorous reviewer \
would ask to decide whether this PR is correct and complete. Seed questions \
across these review topics: correctness, invariants, rules, parity, tests, \
rationale, scope, errors. Prefer concrete questions that reference specific \
mechanisms, values, files, or claims from the diff over generic ones.

Return ONLY a JSON object of the form:
{"questions": [{"topic": "<one of the topics>", "question": "<the question>"}]}
Do not wrap it in markdown fences or add prose."""


# Cap on the PR body + diff bytes sent to the questioner (#1895). Under the cap the
# input is passed through byte-identical, so it only ever touches outliers.
#
# Why 256 KiB: at ~4 bytes/token that is ~64k tokens of input, well inside the
# default model's context window, so the cap is not about fitting. It is about the
# reply: the questioner only needs the shape of the change to ask good questions
# (the answerer reads the actual code), while a very long input is where empty or
# unparseable replies were observed (blis-schemas#100: three in a row on an input
# padded with long reply comments, clean on the first try once trimmed). 256 KiB
# still passes ~95% of recent merged PRs untouched (of the last 32 merged PRs here,
# only the two largest diffs, at 274 KB and 446 KB, exceed it). The exhaustion
# message reports the request size, so this can be retuned from evidence.
MAX_INPUT_BYTES = _env_number("QA_QUESTIONER_MAX_INPUT_BYTES", 256 * 1024, int)

# When capping, the PR description gets at most this fraction of the budget. A
# description is prose about the change; the diff IS the change, so when something
# must go it is the long tail of the description first. Any share the description
# does not use goes to the diff.
BODY_SHARE = 1.0 / 8.0

_CAP_MARKER = "\n[... %d bytes omitted by the qa-review questioner input cap ...]\n"
# Room reserved for the marker inside each section's budget, so the capped
# body + diff never exceeds the limit (the marker's digits are bounded by len()).
_CAP_MARKER_RESERVE = len(_CAP_MARKER % 10 ** 12)


def _cap_text(text, budget):
    """`text` cut to at most `budget` UTF-8 bytes at a line boundary, plus a marker.

    Deterministic: the same input and budget always give the same output. The cut
    keeps the head (a diff's file headers and first hunks; a body's summary) and
    backs off to the last newline so no line is split, unless the head has none.
    """
    raw = text.encode("utf-8")
    if len(raw) <= budget:
        return text
    keep = max(0, budget - _CAP_MARKER_RESERVE)
    head = raw[:keep].decode("utf-8", "ignore")
    nl = head.rfind("\n")
    if nl >= 0:
        head = head[: nl + 1]
    omitted = len(raw) - len(head.encode("utf-8"))
    return head.rstrip("\n") + (_CAP_MARKER % omitted)


def cap_inputs(body, diff, limit=None):
    """Cap PR body + diff to `limit` bytes; return (body, diff, notes).

    Under the limit both are returned unchanged and `notes` is empty. Over it, the
    body is capped to its share (BODY_SHARE of the limit) and the diff gets the rest,
    each cut by _cap_text. `notes` has one line per capped section for stderr.
    """
    limit = MAX_INPUT_BYTES if limit is None else limit
    body_bytes = len(body.encode("utf-8"))
    diff_bytes = len(diff.encode("utf-8"))
    if body_bytes + diff_bytes <= limit:
        return body, diff, []
    body_budget = min(body_bytes, int(limit * BODY_SHARE))
    new_body = _cap_text(body, body_budget)
    diff_budget = limit - len(new_body.encode("utf-8"))
    new_diff = _cap_text(diff, diff_budget)
    notes = []
    for name, old, new in (("body", body, new_body), ("diff", diff, new_diff)):
        if new != old:
            notes.append(
                "qa-review questioner: input cap %d bytes: PR %s cut from %d to %d bytes"
                % (limit, name, len(old.encode("utf-8")), len(new.encode("utf-8")))
            )
    return new_body, new_diff, notes


def validate_completion(resp):
    """Raise ValueError when the completion carries no usable questions object (#1895).

    Passed to post_chat_completion as `validate`, so an empty or unparseable reply
    is retried like a gateway error instead of crashing the script."""
    content = resp["choices"][0]["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("the completion has no content")
    generated = parse_generated(content)
    if not isinstance(generated, dict) or not isinstance(generated.get("questions", []), list):
        raise ValueError("the completion is JSON but not a {\"questions\": [...]} object")
    return generated


def build_user_prompt(title, body, diff):
    """Assemble the single user turn describing the PR under review."""
    parts = []
    if title:
        parts.append("PR title:\n" + title)
    if body:
        parts.append("PR description:\n" + body)
    if diff:
        parts.append("PR diff:\n" + diff)
    parts.append(
        "Generate topic-seeded probing questions per the system instructions. "
        "Return only the JSON object."
    )
    return "\n\n".join(parts)


def repair_json(payload):
    """Repair invalid backslash escapes in a model-emitted JSON string.

    Cross-vendor models frequently emit regex/shell fragments like ``\\s`` or
    ``\\d`` inside JSON string values. Those are invalid JSON escapes and make
    ``json.loads`` raise. This escapes any backslash that is not already part of
    a valid JSON escape sequence (``\\" \\\\ \\/ \\b \\f \\n \\r \\t`` or a
    ``\\uXXXX`` unicode escape), leaving well-formed escapes untouched so an
    already-valid ``\\\\`` pair is never double-escaped.

    Extracted verbatim from the inline repair that used to live in ``main()``
    so it can be unit-tested without a live model (issue #1714 carve-out 1).
    """

    def _fix(m):
        seq = m.group(0)
        # A valid escape sequence is consumed whole and passed through, so its
        # trailing character is never re-examined; a lone backslash is doubled.
        return seq if len(seq) > 1 else "\\\\"

    return re.sub(r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})|\\', _fix, payload)


def parse_generated(payload):
    """Parse a model JSON payload, repairing invalid escapes on first failure."""
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return json.loads(repair_json(payload))


def read_arg_or_file(value, path):
    """Return an inline value, else the contents of path, else empty string."""
    if value:
        return value
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    return ""


def main(argv=None):
    parser = argparse.ArgumentParser(description="qa-review questioner")
    parser.add_argument("--title", default="", help="PR title")
    parser.add_argument("--body", default="", help="PR description (inline)")
    parser.add_argument("--body-file", default="", help="PR description (file)")
    parser.add_argument("--diff", default="", help="PR diff (inline)")
    parser.add_argument("--diff-file", default="", help="PR diff (file)")
    parser.add_argument(
        "--model",
        default=os.environ.get("QA_QUESTIONER_MODEL", DEFAULT_MODEL),
        help="questioner model (default from QA_QUESTIONER_MODEL)",
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

    body = read_arg_or_file(args.body, args.body_file)
    diff = read_arg_or_file(args.diff, args.diff_file)
    body, diff, notes = cap_inputs(body, diff)
    for note in notes:
        sys.stderr.write(note + "\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(args.title, body, diff)},
    ]
    try:
        resp = post_chat_completion(
            base_url, api_key, args.model, messages, validate=validate_completion
        )
    except UnparseableReply as exc:
        # Fail closed (no stdout, non-zero exit => the gate reads MISSING), but name why.
        sys.stderr.write("qa-review questioner: %s\n" % exc)
        return 1
    generated = validate_completion(resp)

    questions = list(FIXED_QUESTIONS)
    n = 0
    for item in generated.get("questions", []):
        text = item.get("question", "").strip()
        if not text:
            # Skip empty generated questions: emitting one would waste an
            # answerer turn and could produce a spurious BLOCK.
            continue
        n += 1
        topic = item.get("topic", "")
        if topic not in TOPICS:
            topic = TOPICS[(n - 1) % len(TOPICS)]
        questions.append({"id": "G%d" % n, "topic": topic, "question": text})

    json.dump({"model": args.model, "questions": questions}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
