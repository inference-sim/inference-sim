# Original to blis-sglang.
"""Tier-2 scheduler runner with read coverage (T1, issue #2350): oracle/run_sched.py, and for a scenario
that names a field to probe, a record of which of the field's reads the run executed.

A scenario without `inertness` is run exactly as oracle/run_sched.py runs it. One with

    "inertness": {"field": NAME, "sites": [{"at": "path:line", "to": last line}, ...],
                  "consumers": [{"at": "path:line", "expr": TEXT}, ...], "stream": BOOL}

adds NAME to the ServerArgs the scenario may set (its `server_args` gives the probed value), runs the
scheduler under a line tracer on the files that hold `sites` (the field's reads, from
`fields/gen_classes.py --reads`) and `consumers` (the reviewed lines that use a value a read stored),
and records in the trace, under config.inertness:

- "executed": the sites the run executed, by their first line;
- "consumers": the consumers it executed.

`stream` makes every request a streaming one (the harness sends none; the request is copied with stream
set, no object is written into), so the code that sends a streamed token is run. Nothing else differs from oracle/run_sched.py; the replaced seams are the ones
oracle/mocks.json lists.

Usage: python -I oracle/run_inertness.py <sglang checkout> <scenario.json> [out.json]
Environment: as oracle/run_scheduler.py.
"""
import json
import os
import sys

import msgspec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_sched  # noqa: E402

ROOT = None  # the sglang checkout, set by main() before the scheduler's process is forked
INERT = {}  # the probe's Coverage ("cover") and config.inertness ("report") in this process


class Coverage:
    def __init__(self, root: str, sites: list, consumers: list):
        self.root = os.path.abspath(root)
        self.roots = {self.root, os.path.realpath(root)}
        self.sites = [(s["at"], s["to"]) for s in sites]
        self.consumers = [c["at"] for c in consumers]
        self.files, self.lines, self.executed, self.consumed = {}, {}, set(), set()
        for kind, at, to in [("site", a, t) for a, t in self.sites] + [("consumer", a, int(a.rsplit(":", 1)[1])) for a in self.consumers]:
            rel, _, first = at.rpartition(":")
            for root in self.roots:
                path = os.path.join(root, rel)
                self.files[path] = rel
                for line in range(int(first), to + 1):
                    self.lines.setdefault((path, line), []).append((kind, at))

    def start(self) -> None:
        import threading

        sys.settrace(self.global_trace)
        threading.settrace(self.global_trace)

    def global_trace(self, frame, event, arg):
        return self.local_trace if frame.f_code.co_filename in self.files else None

    def local_trace(self, frame, event, arg):
        if event == "line":
            for kind, at in self.lines.get((frame.f_code.co_filename, frame.f_lineno), ()):
                (self.executed if kind == "site" else self.consumed).add(at)
        return self.local_trace

    def report(self) -> dict:
        return {"executed": sorted(self.executed), "consumers": sorted(self.consumed)}


def setup(scenario: dict) -> None:
    spec = scenario.get("inertness")
    if spec is None:
        return
    field = spec["field"]
    cover = Coverage(ROOT, spec["sites"], spec.get("consumers", []))
    INERT["cover"] = cover
    INERT["report"] = {}
    cover.start()


def config(s, cfg: dict) -> dict:
    cover = INERT.get("cover")
    if cover is not None:
        INERT["report"].update(cover.report())
        cfg["inertness"] = INERT["report"]
    return cfg


def extend_run(base):
    class InertRun(base):
        def begin_step(self):
            reqs = super().begin_step()
            spec = self.scenario.get("inertness")
            if spec is not None:
                if spec.get("stream"):
                    reqs = [msgspec.structs.replace(r, stream=True) if hasattr(r, "sampling_params") else r for r in reqs]
                INERT["report"].update(INERT["cover"].report())
            return reqs

    return InertRun


def main() -> int:
    global ROOT
    if len(sys.argv) >= 3:
        ROOT = sys.argv[1]
        with open(sys.argv[2]) as f:
            field = (json.load(f).get("inertness") or {}).get("field")
        if field:
            run_sched.HOOKS.update(server_args=(field,))
    run_sched.HOOKS.update(inputs=("oracle/run_inertness.py",), run=extend_run, config=config, setup=setup)
    return run_sched.main()


if __name__ == "__main__":
    sys.exit(main())
