# Original to blis-vllm.
"""Refresh committed golden traces' provenance from fresh traces vLLM produced in CI.

A golden's decisions are vLLM's output for its scenario. Its `provenance` block records where
that output came from. When the environment a golden records must be replaced by CI's own --
the committed lock CI installs, CI's platform, the mocks' current version -- the workflow
oracle-traces re-runs vLLM on every scenario and uploads the traces. This copies each fresh
trace's provenance into its golden, and only that: a golden whose decisions differ in any field
from the fresh trace is left unchanged and reported, and the script fails. check_goldens.py
accepts a provenance-only change without a reviewer's trailer, and refuses anything else.

Usage: python3 -I oracle/refresh_provenance.py <golden dir> <fresh dir> <fresh suffix>
  e.g. core/sched/testdata/golden oracle-traces/oracle/out/scenarios .trace.json
"""
import json
import os
import sys


def main(golden_dir: str, fresh_dir: str, suffix: str) -> int:
    names = sorted(f for f in os.listdir(golden_dir) if f.endswith(".json"))
    if not names:
        print(f"{golden_dir}: no goldens")
        return 1
    bad = refreshed = 0
    for name in names:
        stem = name[:-len(".metrics.json")] if name.endswith(".metrics.json") else name[:-len(".json")]
        fresh_path = os.path.join(fresh_dir, stem + suffix)
        if not os.path.exists(fresh_path):
            print(f"{name}: no fresh trace at {fresh_path}")
            bad += 1
            continue
        gpath = os.path.join(golden_dir, name)
        with open(gpath) as f:
            golden = json.load(f)
        with open(fresh_path) as f:
            fresh = json.load(f)
        g = {k: v for k, v in golden.items() if k != "provenance"}
        n = {k: v for k, v in fresh.items() if k != "provenance"}
        if g != n:
            keys = sorted(k for k in set(g) | set(n) if g.get(k) != n.get(k))
            print(f"{name}: the fresh trace's decisions differ ({keys}); provenance not refreshed")
            bad += 1
            continue
        if golden["provenance"]["scenario_sha256"] != fresh["provenance"]["scenario_sha256"]:
            print(f"{name}: the fresh trace ran on a different scenario file")
            bad += 1
            continue
        golden["provenance"] = fresh["provenance"]
        with open(gpath, "w") as f:
            json.dump(golden, f, indent=1, sort_keys=True)
            f.write("\n")
        refreshed += 1
    print(f"{golden_dir}: {refreshed} provenance blocks refreshed, {bad} refused")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
