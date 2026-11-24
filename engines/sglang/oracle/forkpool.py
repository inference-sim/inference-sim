# Original to blis-sglang.
"""Run an oracle's cases in forked children, several at once, with the answers in case order.

Each case runs in a process of its own, forked from a parent that has imported nothing of SGLang,
so a case's answer depends neither on the other cases nor on how many run at once. Up to jobs()
children run at a time; each writes its answer to a file of its own, which the parent reads once
that child has exited, so a large answer never blocks a child on a full pipe.

jobs() is ORACLE_JOBS when set (a positive integer), else cpus(). A workflow step that runs several
parts side by side gives each part its share of cpus(), which `python3 -I oracle/forkpool.py` prints.

cpus() is the CPUs this process may use: the fewest of its CPU affinity and its cgroup's CPU quota
(cgroup v2 cpu.max, or v1 cpu.cfs_quota_us over cpu.cfs_period_us), rounded up. In a Kubernetes pod
a CPU limit is a quota, not a CPU set, so os.cpu_count() and nproc report the node's CPUs: a
4-CPU runner pod on a 64-CPU node would fork 64 SGLang children at once and run out of memory.

Usage (from a harness): answers = forkpool.run(cases, child), where child(case) returns a dict that
json can write. A child that raises, or exits without an answer, gives
{"name": case's name, "harness_error": why} at its position instead.
"""
import json
import os
import sys
import tempfile
import traceback


def _quota(cgroup: str):
    """The cgroup's CPU quota in CPUs, rounded up, or None when it states none."""
    try:
        with open(os.path.join(cgroup, "cpu.max")) as f:
            quota, period = f.read().split()[:2]
        if quota != "max":
            return -(-int(quota) // int(period))
        return None
    except (OSError, ValueError):
        pass
    try:
        with open(os.path.join(cgroup, "cpu", "cpu.cfs_quota_us")) as f:
            quota = int(f.read())
        with open(os.path.join(cgroup, "cpu", "cpu.cfs_period_us")) as f:
            period = int(f.read())
        if quota > 0 and period > 0:
            return -(-quota // period)
    except (OSError, ValueError):
        pass
    return None


def cpus(cgroup: str = "/sys/fs/cgroup") -> int:
    """The CPUs this process may use: the fewest of its affinity and its cgroup's quota."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    q = _quota(cgroup)
    return max(1, min(n, q) if q else n)


def jobs() -> int:
    """How many children run at once: ORACLE_JOBS, else cpus()."""
    v = os.environ.get("ORACLE_JOBS", "")
    if not v:
        return cpus()
    if not v.isdigit() or int(v) < 1:
        raise SystemExit("ORACLE_JOBS=%r: want a positive integer" % v)
    return int(v)


def _child(case, child, path) -> None:
    """The forked child: run one case, write its answer, and exit without returning."""
    code = 0
    try:
        res = child(case)
    except BaseException as e:  # a harness failure, not a refusal
        res = {"name": case.get("name"), "harness_error": "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc())}
    try:
        with open(path, "w") as f:
            json.dump(res, f)
    except BaseException:
        code = 1
    os._exit(code)


def run(cases, child, n=None):
    """child(case) for every case, each in a forked child, at most n (jobs()) at once; the answers
    in the order of cases."""
    n = n or jobs()
    answers = [None] * len(cases)
    with tempfile.TemporaryDirectory(prefix="oracle-forkpool-") as tmp:
        running = {}
        nxt = 0
        while nxt < len(cases) or running:
            while nxt < len(cases) and len(running) < n:
                path = os.path.join(tmp, "%d.json" % nxt)
                sys.stdout.flush()
                sys.stderr.flush()
                pid = os.fork()
                if pid == 0:
                    _child(cases[nxt], child, path)
                running[pid] = (nxt, path)
                nxt += 1
            pid, _ = os.wait()
            if pid not in running:
                continue
            i, path = running.pop(pid)
            try:
                with open(path) as f:
                    answers[i] = json.load(f)
            except (OSError, ValueError):
                answers[i] = {"name": cases[i].get("name"), "harness_error": "no answer"}
    return answers


if __name__ == "__main__":
    print(cpus())
