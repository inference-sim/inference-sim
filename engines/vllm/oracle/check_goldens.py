# Original to blis-vllm.
"""Fail if a commit changes what a golden trace asserts without naming a reviewer.

A golden trace is vLLM's own output for a scenario, and what it records beside its
`provenance` block is what blis-vllm is checked against. Every golden directory holds them:
the scheduler's (core/sched), the EngineCore's and the DP loop's (core/engine), the NIXL
connector's (core/connector/nixl), the OffloadingConnector's (core/connector/offloading) and the metric series' (core/metrics); and core/kv's block
hashes (GOLDEN_FILES), which are vLLM's own hash_block_tokens output. A commit that changes
the recorded content of an existing golden, or deletes one, must carry a
`golden-reviewed:` trailer for each trace it changes, which names the trace (its path or file name)
and gives the reason (design FID-3, section 7.7): `golden-reviewed: <trace>: <reason>`. A trailer that
names no changed trace, or gives no reason, does not count; a person's name is neither. Adding a golden for a new scenario (in a
golden file, appending a case), or refreshing only its `provenance` block, needs none, and this
check verifies that such a commit changed nothing else. A golden directory this list does not name fails the check, so a new
tier's goldens cannot go unguarded. A golden whose provenance names a blis-vllm commit outside
head's history (one nobody can fetch, a "+dirty" tree or "unknown") fails too: its provenance
must name a commit anyone can check out. A golden added in the range base..head must
also have been produced on the platform CI runs (CI_PLATFORM) and from the oracle environment the
repository commits (its requirements_lock_sha256 is the hash of oracle/requirements.lock at head),
so a trace recorded elsewhere cannot enter unnoticed; the goldens already committed before this
check existed are not held to it.

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
    "core/engine/testdata/golden-dp-async/",
    "core/connector/nixl/testdata/golden/",
    "core/connector/offloading/testdata/golden/",
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


def decisions_changed(old_rev: str, new_rev: str, path: str) -> bool:
    old, new = decisions(old_rev, path), decisions(new_rev, path)
    if path in GOLDEN_FILES:
        # Each case is one golden: appending cases adds goldens; changing or removing one does not.
        old_cases, new_cases = old.pop("cases", []), new.pop("cases", [])
        return old != new or new_cases[: len(old_cases)] != old_cases
    return old != new


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


def trailer_covers(trailers: list[str], path: str, change: str) -> bool:
    """Whether one of the commit's golden-reviewed trailers names this trace and gives a reason.

    A trailer is `<trace>: <reason>`. The trace is the golden's path or its file name; a renamed
    golden is named by either path. The reason is what follows the first colon after the trace and
    must not be empty. A trailer that is only a name, or names another trace, covers nothing.
    """
    names = {path, path.rsplit("/", 1)[-1]}
    m = re.search(r" renamed to (\S+)$", change)
    if m:
        names |= {m.group(1), m.group(1).rsplit("/", 1)[-1]}
    for t in trailers:
        trace, sep, reason = t.partition(": ")
        if sep and trace.strip() in names and reason.strip():
            return True
    return False


# The platform the oracle-traces workflow and the oracle jobs run on (ubuntu-latest, x86_64).
CI_PLATFORM = "linux-x86_64"


def foreign_provenance(base: str, head: str) -> list[str]:
    """Goldens added in base..head whose provenance names a platform other than CI's,
    or a committed-lock hash other than head's oracle/requirements.lock."""
    import hashlib
    lock = hashlib.sha256(git("show", f"{head}:oracle/requirements.lock").encode()).hexdigest()
    out = []
    for line in git("diff", "--name-status", "--diff-filter=A", f"{base}..{head}").splitlines():
        status, path = line.split("\t", 1)
        if not is_golden(path) or path in GOLDEN_FILES:
            continue
        prov = json.loads(git("show", f"{head}:{path}")).get("provenance", {})
        if prov.get("platform") != CI_PLATFORM:
            out.append(f"{path}: recorded on platform {prov.get('platform')!r}, not CI's {CI_PLATFORM!r} "
                       "(record it with the oracle-traces workflow)")
        if prov.get("requirements_lock_sha256") != lock:
            out.append(f"{path}: records requirements_lock_sha256 {prov.get('requirements_lock_sha256')!r}, "
                       f"not the committed oracle/requirements.lock's {lock!r}")
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
    for problem in foreign_provenance(base, head):
        print(problem)
        bad = 1
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
            elif status == "M" and decisions_changed(c + "^", c, path):
                needs_review.append(f"{path} decisions changed")
        if not needs_review:
            continue
        trailers = [t.strip() for t in git("log", "-1", "--format=%(trailers:key=golden-reviewed,valueonly)", c)
                    .splitlines() if t.strip()]
        subject = git("log", "-1", "--format=%h %s", c).strip()
        for n in needs_review:
            if not trailer_covers(trailers, n.split(" ", 1)[0], n):
                print(f"{subject}: {n} without a golden-reviewed: trailer that names the trace and gives the reason")
                bad = 1
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
