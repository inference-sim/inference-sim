#!/usr/bin/env python3
"""qa-review report renderer (+ optional PR posting).

Combines the questioner's questions with the answerer's answers into a single
Markdown report and (optionally) posts it as a PR comment.

Verdict: BLOCK iff any answer status in {FLAW_FOUND, CANNOT_ANSWER}, else PASS.

Output shape (and nothing else):
  - a one-line verdict HEADER: "## qa-review — PR #N: <emoji> PASS|BLOCK"
  - a subtitle line
  - the full untruncated "ID | Topic | Result | Question | Answer" table
  - an "Items to fix" section (blocking findings + evidence)
  - an "Important to consider" section (non-blocking `note` fields)

The verdict is a header emoji, NOT a trailing machine marker. Deriving a
"QA-VERDICT: PASS|BLOCK" gate marker from this output is #1715's job.
"""

import argparse
import json
import os
import subprocess
import sys

# Status -> (is-blocking, table-cell label with emoji).
STATUS_LABEL = {
    "CONFIDENT": (False, "✅ CONFIDENT"),
    "FLAW_FOUND": (True, "❌ FLAW_FOUND"),
    "CANNOT_ANSWER": (True, "⚠️ CANNOT_ANSWER"),
}

BLOCKING_STATUSES = {"FLAW_FOUND", "CANNOT_ANSWER"}


def default_banner(qmodel, amodel):
    """Build the standing experimental banner from the actual models used, so
    it cannot drift from the run that produced the report."""
    return (
        "> 🤖 **qa-review** — experimental two-agent AI-review demo "
        "(cross-vendor: questioner `%s` + isolated answerer `%s`, RFC #1603). "
        "Posted for evaluation — **not an official merge gate**." % (qmodel, amodel)
    )


def _escape_cell(text):
    """Keep a Markdown table cell on one row: pipes escaped, newlines flattened."""
    return (text or "").replace("|", "\\|").replace("\n", " ").strip()


def verdict_for(answers):
    """PASS unless any answer carries a blocking status."""
    for a in answers:
        if a.get("status") in BLOCKING_STATUSES:
            return "BLOCK"
    return "PASS"


def render(questions, answers, pr, title, qmodel, amodel, banner=None):
    by_id = {q["id"]: q for q in questions}
    verdict = verdict_for(answers)
    emoji = "⛔" if verdict == "BLOCK" else "✅"

    lines = []
    if banner:
        lines.append(banner)
        lines.append("")

    lines.append("## qa-review — PR #%s: %s %s" % (pr, emoji, verdict))
    subtitle = "_%s · %d questions · questioner `%s` · answerer `%s`_" % (
        title,
        len(answers),
        qmodel,
        amodel,
    )
    lines.append(subtitle)
    lines.append("")

    lines.append("| ID | Topic | Result | Question | Answer |")
    lines.append("|----|-------|--------|----------|--------|")
    for a in answers:
        qid = a.get("id", "")
        q = by_id.get(qid, {})
        _, label = STATUS_LABEL.get(a.get("status", ""), (True, a.get("status", "")))
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (
                qid,
                _escape_cell(q.get("topic", "")),
                label,
                _escape_cell(q.get("question", "")),
                _escape_cell(a.get("answer", "")),
            )
        )
    lines.append("")

    # Items to fix — every blocking finding, with its evidence.
    lines.append("### Items to fix")
    blocking = [a for a in answers if a.get("status") in BLOCKING_STATUSES]
    if blocking:
        for a in blocking:
            lines.append(
                "- **%s · %s** — %s"
                % (a.get("id", ""), a.get("status", ""), (a.get("answer") or "").strip())
            )
            if a.get("evidence"):
                lines.append("  _%s_" % a["evidence"].strip())
    else:
        lines.append("- _None — no blocking findings._")
    lines.append("")

    # Important to consider — non-blocking notes.
    lines.append("### Important to consider")
    notes = [a for a in answers if (a.get("note") or "").strip()]
    if notes:
        for a in notes:
            lines.append("- **%s** — %s" % (a.get("id", ""), a["note"].strip()))
            if a.get("evidence"):
                lines.append("  _%s_" % a["evidence"].strip())
    else:
        lines.append("- _None._")

    return "\n".join(lines) + "\n"


def post_to_pr(repo, pr, body):
    subprocess.run(
        ["gh", "pr", "comment", str(pr), "--repo", repo, "--body", body],
        check=True,
    )


def _load(inline, path):
    if inline:
        return json.loads(inline)
    with open(path, "r", encoding="utf-8") as fh:
        return json.loads(fh.read())


def main(argv=None):
    parser = argparse.ArgumentParser(description="qa-review report renderer")
    parser.add_argument("--questions", default="", help="questions JSON file")
    parser.add_argument("--answers", default="", help="answers JSON file")
    parser.add_argument("--questions-inline", default="", help="questions JSON (inline)")
    parser.add_argument("--answers-inline", default="", help="answers JSON (inline)")
    parser.add_argument("--pr", required=True, help="PR number")
    parser.add_argument("--title", default="", help="PR title (subtitle)")
    parser.add_argument("--qmodel", default="", help="questioner model")
    parser.add_argument("--amodel", default="", help="answerer model")
    parser.add_argument("--out", default="", help="write report here (else stdout)")
    parser.add_argument("--post-to-pr", action="store_true", help="post as a PR comment")
    parser.add_argument("--repo", default=os.environ.get("QA_REPO", "inference-sim/inference-sim"))
    parser.add_argument("--banner", default="", help="override the default banner")
    args = parser.parse_args(argv)
    # Validate inputs up front. Without this, omitting --questions/--answers
    # falls through to open("") and raises a confusing
    # "FileNotFoundError: [Errno 2] ... ''" instead of naming the missing flag.
    if not args.questions and not args.questions_inline:
        parser.error("provide --questions or --questions-inline")
    if not args.answers and not args.answers_inline:
        parser.error("provide --answers or --answers-inline")

    qdoc = _load(args.questions_inline, args.questions)
    questions = qdoc["questions"] if isinstance(qdoc, dict) else qdoc
    qmodel = args.qmodel or (qdoc.get("model", "") if isinstance(qdoc, dict) else "")
    answers = _load(args.answers_inline, args.answers)

    banner = args.banner or default_banner(qmodel, args.amodel)
    report = render(questions, answers, args.pr, args.title, qmodel, args.amodel, banner)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)
    else:
        sys.stdout.write(report)

    if args.post_to_pr:
        post_to_pr(args.repo, args.pr, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
