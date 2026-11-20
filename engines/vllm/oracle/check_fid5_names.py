"""Fail if the committed recorded interface traces are not exactly the ones the offload scenarios produce.

oracle/run_kv_interfaces.py writes recorded-<scenario>.fid5.json for each scenario of
oracle/scenarios/offloading and oracle/scenarios/multiconnector, and oracle/drive_kv_interfaces.py writes
direct-<sequence>.fid5.json for each sequence of oracle/scenarios-fid5. The oracle-scenarios step compares them
with core/connector/offloading/testdata/golden-fid5, but it needs vLLM, so an item that adds a scenario
without its trace first fails there, after the merge. This reads names only: a scenario with no committed
trace, or a trace of the two prefixes with no scenario, fails here, in ci-light. It does not read a trace's
content. A golden directory with a <dir>.pending.json beside it waits for an issue and is not checked.

Usage: python3 -I oracle/check_fid5_names.py [<repo root>]
"""
import os
import sys

GOLDEN = "core/connector/offloading/testdata/golden-fid5"
SOURCES = (("oracle/scenarios/offloading", "recorded-"), ("oracle/scenarios/multiconnector", "recorded-"),
           ("oracle/scenarios-fid5", "direct-"))


def problems(root: str) -> list:
    golden = os.path.join(root, GOLDEN)
    if os.path.exists(golden + ".pending.json"):
        return []
    have = {f for f in os.listdir(golden) if f.endswith(".fid5.json")} if os.path.isdir(golden) else set()
    want = set()
    for d, prefix in SOURCES:
        for f in sorted(os.listdir(os.path.join(root, d))):
            if f.endswith(".json") and not f.endswith(".trace.json"):
                want.add(prefix + f[:-5] + ".fid5.json")
    out = ["%s has no committed trace %s/%s (record it with oracle-traces or the oracle-tier2 artifact)" % (
        n, GOLDEN, n) for n in sorted(want - have)]
    out += ["%s/%s has no scenario that produces it" % (GOLDEN, n) for n in sorted(have - want)]
    return out


def main(argv: list) -> int:
    bad = problems(argv[0] if argv else ".")
    for b in bad:
        print(b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
