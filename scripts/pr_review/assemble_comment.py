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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    parts = [
        f"## /pr-review — PR #{args.pr}",
        "",
        "_Automated, **advisory** review (archon + blis-pr-review + qa-review). "
        "It gates nothing and merges nothing; a human maintainer decides._",
        "",
        _section("Architecture (archon)", args.archon),
        _section("Correctness (blis-pr-review)", args.blis),
        _section("Cross-vendor (qa-review)", args.qa),
    ]
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts).rstrip() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
