# Original to blis-vllm.
"""Fail if a committed golden trace is not what vLLM produces now.

The goldens under core/sched/testdata/golden were produced once, by the oracle on a
developer's machine. CI re-runs vLLM v0.32.0 (the pinned release wheel) on every scenario and
this compares each fresh trace with its golden, decision by decision (`steps` and `finished`,
and `kv_pool`, the pool vLLM sized, where a scenario states its KV memory),
so a golden passes only while the pinned vLLM release still produces it.

Usage: python3 -I oracle/compare_goldens.py <golden dir> <fresh dir>
(<fresh dir> holds the <name>.trace.json files of run_scheduler.py's directory mode.)
"""
import json
import os
import sys


def first_difference(golden: dict, fresh: dict) -> str:
    gs, fs = golden["steps"], fresh["steps"]
    for i in range(max(len(gs), len(fs))):
        if i >= len(gs):
            return f"step {i}: only vLLM's fresh trace has it"
        if i >= len(fs):
            return f"step {i}: only the golden has it"
        if gs[i] != fs[i]:
            keys = sorted(k for k in set(gs[i]) | set(fs[i]) if gs[i].get(k) != fs[i].get(k))
            return f"step {i}: fields {keys} differ"
    if golden["finished"] != fresh["finished"]:
        return "the finished records differ"
    if golden.get("kv_pool") != fresh.get("kv_pool"):
        return "the KV pools vLLM sized differ"
    return ""


def main(golden_dir: str, fresh_dir: str) -> int:
    names = sorted(f[:-5] for f in os.listdir(golden_dir) if f.endswith(".json"))
    if not names:
        # An empty golden directory compares nothing, which must not read as a pass.
        print(f"{golden_dir}: no goldens to compare")
        return 1
    bad = 0
    for name in names:
        fresh_path = os.path.join(fresh_dir, name + ".trace.json")
        if not os.path.exists(fresh_path):
            print(f"{name}: no fresh trace from vLLM")
            bad += 1
            continue
        with open(os.path.join(golden_dir, name + ".json")) as f:
            golden = json.load(f)
        with open(fresh_path) as f:
            fresh = json.load(f)
        if golden["provenance"]["scenario_sha256"] != fresh["provenance"]["scenario_sha256"]:
            print(f"{name}: the golden and vLLM's fresh trace ran on different scenario files")
            bad += 1
            continue
        diff = first_difference(golden, fresh)
        if diff:
            print(f"{name}: {diff}")
            bad += 1
    print(f"{len(names)} goldens, {bad} differ from vLLM's fresh traces")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
