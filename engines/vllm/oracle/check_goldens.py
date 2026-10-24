# Original to blis-vllm.
"""Fail if a commit changes what a golden trace asserts without naming a reviewer.

A golden trace (core/sched/testdata/golden/*.json) is vLLM's own output for a scenario, and its
decisions (`steps` and `finished`) are what blis-vllm is checked against. A commit that changes
the decisions of an existing golden, or deletes one, must carry a `golden-reviewed: <reviewer>`
trailer (docs/testing.md). Adding a golden for a new scenario, or refreshing only its
`provenance` block, needs none, and this check verifies that such a commit changed nothing else.

Usage: python3 -I oracle/check_goldens.py <base> <head>
"""
import json
import subprocess
import sys

GOLDEN_DIR = "core/sched/testdata/golden/"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def decisions(rev: str, path: str):
    raw = git("show", f"{rev}:{path}")
    trace = json.loads(raw)
    return trace.get("steps"), trace.get("finished")


def main(base: str, head: str) -> int:
    if not base or set(base) == {"0"}:
        # A first push: check every commit from the root.
        base = git("rev-list", "--max-parents=0", head).split()[-1]
    bad = 0
    for c in git("rev-list", "--reverse", "--no-merges", f"{base}..{head}").split():
        needs_review = []
        for line in git("diff-tree", "--no-commit-id", "-r", "--name-status", c).splitlines():
            status, path = line.split("\t", 1)
            if not path.startswith(GOLDEN_DIR):
                continue
            if status == "D":
                needs_review.append(f"{path} deleted")
            elif status == "M" and decisions(c + "^", path) != decisions(c, path):
                needs_review.append(f"{path} decisions changed")
        if not needs_review:
            continue
        trailer = git("log", "-1", "--format=%(trailers:key=golden-reviewed,valueonly)", c).strip()
        if not trailer:
            subject = git("log", "-1", "--format=%h %s", c).strip()
            for n in needs_review:
                print(f"{subject}: {n} without a golden-reviewed: trailer")
            bad = 1
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
