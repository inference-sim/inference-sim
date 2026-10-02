#!/usr/bin/env python3
"""assemble_comment.py — combine the three reviewers' outputs into ONE PR comment.

A reviewer that failed or produced nothing is reported as "did not complete",
never silently dropped (R1): a missing section must read as "we don't know", not
as "nothing to say". The combined body is advisory and says so.
"""
import argparse
import os


def _section(title: str, path: str) -> str:
    if path and os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            body = fh.read().strip()
        if body:
            return f"### {title}\n\n{body}\n"
    return f"### {title}\n\n_did not complete — see the workflow run logs._\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr", required=True)
    ap.add_argument("--archon", default="")
    ap.add_argument("--qa", default="")
    ap.add_argument("--blis", default="")
    ap.add_argument("--head-sha", default="", help="the commit SHA that was reviewed")
    ap.add_argument("--stale", action="store_true",
                    help="the PR head moved since the review was captured")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Name the reviewed commit so a reader knows exactly what was assessed — the
    # worktree, archon and diff are all pinned to this SHA.
    reviewed = f" · reviewed at `{args.head_sha}`" if args.head_sha else ""

    # STALE: the head moved (or couldn't be confirmed) during the review. We do
    # NOT publish the reviewer sections — a reviewer (notably qa's answerer, whose
    # pr_diff tool reads the live PR) may have mixed the moved head with the
    # captured tree, so presenting findings "at <sha>" would be a false claim.
    # Publish only a re-run notice. (Suppress, don't annotate — per review.)
    if args.stale:
        body = "\n".join([
            f"## /pr-review — PR #{args.pr}{reviewed}",
            "",
            f"> ⚠️ The PR head moved during this review (or could not be confirmed), so the "
            f"reviewers' inputs may mix `{args.head_sha}` with a newer head. Findings are "
            f"**withheld** rather than shown under a SHA they may not reflect. "
            f"**Re-run `/pr-review`** on the current head.",
        ]).rstrip() + "\n"
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(body)
        return 0

    parts = [
        f"## /pr-review — PR #{args.pr}{reviewed}",
        "",
        "_Automated, **advisory** review (archon + blis-pr-review + qa-review). "
        "It gates nothing and merges nothing; a human maintainer decides._",
        "",
        _section("Architecture (archon)", args.archon),
        _section("Correctness (blis-pr-review)", args.blis),
        _section("Cross-vendor (qa-review)", args.qa),
    ]
    body = "\n".join(parts).rstrip() + "\n"

    # GitHub rejects an issue comment over 65,536 chars. archon alone can approach
    # 60 KB, so a legitimate combined review can exceed the limit and the post
    # would fail. Truncate with a notice rather than lose the whole comment.
    LIMIT = 65000
    if len(body) > LIMIT:
        notice = "\n\n_… truncated — the combined review exceeded GitHub's comment limit; see the workflow run logs for the full output._\n"
        body = body[: LIMIT - len(notice)] + notice

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
