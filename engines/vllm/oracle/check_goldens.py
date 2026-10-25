# Original to blis-vllm.
"""Fail if a commit changes what a golden trace asserts without naming a reviewer.

A golden trace is vLLM's own output for a scenario, and what it records beside its
`provenance` block is what blis-vllm is checked against. Every golden directory holds them:
the scheduler's (core/sched), the EngineCore's and the DP loop's (core/engine), the NIXL
connector's (core/connector/nixl) and the metric series' (core/metrics); and core/kv's block
hashes (GOLDEN_FILES), which are vLLM's own hash_block_tokens output. A commit that changes
the recorded content of an existing golden, or deletes one, must carry a
`golden-reviewed: <reviewer>` trailer (docs/testing.md). Adding a golden for a new scenario, or
refreshing only its `provenance` block, needs none, and this check verifies that such a commit
changed nothing else. A golden directory this list does not name fails the check, so a new
tier's goldens cannot go unguarded. A golden whose provenance names a blis-vllm commit outside
head's history (one nobody can fetch, a "+dirty" tree or "unknown") fails too: its provenance
must name a commit anyone can check out.

Usage: python3 -I oracle/check_goldens.py <base> <head>
"""
import json
import re
import subprocess
import sys

GOLDEN_DIRS = (
    "core/sched/testdata/golden/",
    "core/engine/testdata/golden/",
    "core/engine/testdata/golden-dp/",
    "core/connector/nixl/testdata/golden/",
    "core/metrics/testdata/golden/",
)

# Golden files outside a golden directory: vLLM's own output, guarded as the directories are.
# core/kv's block hashes are vLLM's hash_block_tokens over each case (oracle/gen_block_hashes.py).
GOLDEN_FILES = (
    "core/kv/testdata/block_hashes.json",
)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def decisions(rev: str, path: str):
    """Everything a golden records except its provenance, which a refresh may change."""
    trace = json.loads(git("show", f"{rev}:{path}"))
    trace.pop("provenance", None)
    return trace


def is_golden(path: str) -> bool:
    return path in GOLDEN_FILES or (path.endswith(".json") and path.startswith(GOLDEN_DIRS))


def unguarded(rev: str) -> list[str]:
    """Golden directories at rev that GOLDEN_DIRS does not name."""
    dirs = set()
    for path in git("ls-tree", "-r", "--name-only", rev).splitlines():
        parts = path.split("/")
        for i, part in enumerate(parts[:-1]):
            if part.startswith("golden") and i > 0 and parts[i - 1] == "testdata":
                dirs.add("/".join(parts[: i + 1]) + "/")
    return sorted(d for d in dirs if d not in GOLDEN_DIRS)


def unreachable_commits(head: str) -> list[str]:
    """Goldens at head whose provenance names a blis_vllm_commit that is not a full commit id
    in head's history."""
    out = []
    reachable: dict[str, bool] = {}
    for path in git("ls-tree", "-r", "--name-only", head).splitlines():
        if not is_golden(path):
            continue
        commit = json.loads(git("show", f"{head}:{path}")).get("provenance", {}).get("blis_vllm_commit")
        if commit is None:
            continue
        if commit not in reachable:
            reachable[commit] = bool(re.fullmatch(r"[0-9a-f]{40}", commit)) and subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, head], capture_output=True).returncode == 0
        if not reachable[commit]:
            out.append(f"{path}: records blis_vllm_commit {commit}, which is not a commit in {head}'s history "
                       "(refresh it from the oracle-traces workflow run on main)")
    return out


def main(base: str, head: str) -> int:
    bad = 0
    for d in unguarded(head):
        print(f"{d}: a golden directory oracle/check_goldens.py does not guard; add it to GOLDEN_DIRS")
        bad = 1
    for problem in unreachable_commits(head):
        print(problem)
        bad = 1
    if not base or set(base) == {"0"}:
        # A first push: check every commit from the root.
        base = git("rev-list", "--max-parents=0", head).split()[-1]
    for c in git("rev-list", "--reverse", "--no-merges", f"{base}..{head}").split():
        needs_review = []
        for line in git("diff-tree", "--no-commit-id", "-r", "-M", "--name-status", c).splitlines():
            status, *paths = line.split("\t")
            path = paths[0]
            if not is_golden(path):
                continue
            if status == "D":
                needs_review.append(f"{path} deleted")
            elif status.startswith("R"):
                needs_review.append(f"{path} renamed to {paths[1]}")
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
