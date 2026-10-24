# Original to blis-vllm.
"""Fail if a committed tier-3 golden trace is not what vLLM produces now.

CI re-runs vLLM v0.32.0's EngineCore (oracle/run_enginecore.py) and its DP engine-core loop
(oracle/run_dp.py) on every tier-3 scenario and compares each fresh trace with its golden:
every batch or routing decision, every step and the finished records, so a golden passes only
while the pinned release still produces it.

Usage: python3 -I oracle/compare_goldens_t3.py <golden dir> <fresh dir>
"""
import json
import os
import sys


def main(golden_dir: str, fresh_dir: str) -> int:
    names = sorted(f[:-5] for f in os.listdir(golden_dir) if f.endswith(".json"))
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
        for key in ("batches", "routed", "steps", "finished"):
            if key in golden and golden.get(key) != fresh.get(key):
                print(f"{name}: {key} differ from vLLM's fresh trace")
                bad += 1
                break
    print(f"{len(names)} tier-3 goldens, {bad} differ from vLLM's fresh traces")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
