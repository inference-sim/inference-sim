# Original to blis-vllm.
"""Fail if a committed metric golden is not what vLLM's own Prometheus logger produces now.

The tier-5 goldens under core/metrics/testdata/golden record vLLM's metric series for a
tier-2 scenario (oracle/run_metrics.py): every vllm:* sample after the last step (`series`),
and the samples of the step-decided families after every step (`per_step`). CI re-runs the
pinned vLLM release on the same scenarios, and this compares each fresh series with its
golden, so a golden passes only while the pinned release still produces it. A golden is
compared on what it records beside its provenance (`steps`, `per_step`, `series`); the
scenario file must be the one the golden was made from.

Usage: python3 -I oracle/compare_goldens_metrics.py <golden dir> <fresh dir>
(<fresh dir> holds the <name>.metrics.json files of run_metrics.py's directory mode.)
"""
import json
import os
import sys


def first_difference(golden: dict, fresh: dict) -> str:
    if golden["steps"] != fresh["steps"]:
        return f"steps: golden {golden['steps']}, vLLM now {fresh['steps']}"
    gp, fp = golden["per_step"], fresh["per_step"]
    for i in range(max(len(gp), len(fp))):
        if i >= len(gp) or i >= len(fp):
            return f"per_step {i}: present on one side only"
        if gp[i] != fp[i]:
            keys = sorted(k for k in set(gp[i]) | set(fp[i]) if gp[i].get(k) != fp[i].get(k))
            return f"per_step {i}: samples {keys[:5]} differ"
    gs, fs = golden["series"], fresh["series"]
    if gs != fs:
        keys = sorted(k for k in set(gs) | set(fs) if gs.get(k) != fs.get(k))
        return f"series: samples {keys[:5]} differ"
    return ""


def main(golden_dir: str, fresh_dir: str) -> int:
    names = sorted(f[: -len(".metrics.json")] for f in os.listdir(golden_dir) if f.endswith(".metrics.json"))
    if not names:
        # An empty golden directory compares nothing, which must not read as a pass.
        print(f"{golden_dir}: no metric goldens to compare")
        return 1
    bad = 0
    for name in names:
        fresh_path = os.path.join(fresh_dir, name + ".metrics.json")
        if not os.path.exists(fresh_path):
            print(f"{name}: no fresh series from vLLM")
            bad += 1
            continue
        with open(os.path.join(golden_dir, name + ".metrics.json")) as f:
            golden = json.load(f)
        with open(fresh_path) as f:
            fresh = json.load(f)
        if golden["provenance"]["scenario_sha256"] != fresh["provenance"]["scenario_sha256"]:
            print(f"{name}: the golden and vLLM's fresh series ran on different scenario files")
            bad += 1
            continue
        diff = first_difference(golden, fresh)
        if diff:
            print(f"{name}: {diff}")
            bad += 1
    print(f"{len(names)} metric goldens, {bad} differ from vLLM's fresh series")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
