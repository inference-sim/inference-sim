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

stdout: {"model", "questions":[{id,topic,question}]}
"""

import argparse
import json
import os
import re
import sys
import urllib.request

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


def post_chat_completion(base_url, api_key, model, messages):
    """One POST to the OpenAI-compatible /chat/completions endpoint."""
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps({"model": model, "messages": messages}).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer " + api_key)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(args.title, body, diff)},
    ]
    resp = post_chat_completion(base_url, api_key, args.model, messages)
    content = resp["choices"][0]["message"]["content"]
    generated = parse_generated(content)

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
