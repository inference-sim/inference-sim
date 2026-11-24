# Original to blis-sglang.
"""Run an oracle's cases in forked children, several at once, with the answers in case order.

Each case runs in a process of its own, forked from a parent that has imported nothing of SGLang,
so a case's answer depends neither on the other cases nor on how many run at once. Up to jobs()
children run at a time; each writes its answer to a file of its own, which the parent reads once
that child has exited, so a large answer never blocks a child on a full pipe.

jobs() is ORACLE_JOBS when set (a positive integer), else the CPU count. A workflow step that runs
several parts side by side gives each part its share of the CPUs.

Usage (from a harness): answers = forkpool.run(cases, child), where child(case) returns a dict that
json can write. A child that raises, or exits without an answer, gives
{"name": case's name, "harness_error": why} at its position instead.
"""
import json
import os
import sys
import tempfile
import traceback


def jobs() -> int:
    """How many children run at once: ORACLE_JOBS, else the CPU count."""
    v = os.environ.get("ORACLE_JOBS", "")
    if not v:
        return os.cpu_count() or 1
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
