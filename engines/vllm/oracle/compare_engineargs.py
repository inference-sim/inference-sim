# Original to blis-vllm.
"""Fail if the committed EngineArgs trace is not what vLLM produces now.

core/vllm/testdata/engineargs.json holds vLLM v0.32.0's own EngineArgs(...).create_engine_config()
answers for oracle/engineargs-cases.json (oracle/run_engineargs.py). CI runs the same cases and
this compares each case's acceptance, refusal message and resolved values, so the trace the Go
test reads passes only while the pinned release still gives it. Provenance is left out of the
comparison, as it is for the other goldens.

Usage: python3 -I oracle/compare_engineargs.py <committed trace> <fresh trace>
"""
import json
import sys


def main(committed: str, fresh: str) -> int:
    with open(committed) as f:
        a = json.load(f)
    with open(fresh) as f:
        b = json.load(f)
    if a["provenance"]["scenario_sha256"] != b["provenance"]["scenario_sha256"]:
        print("the committed trace and the fresh one ran on different case files")
        return 1
    bad = 0
    ca, cb = a["cases"], b["cases"]
    if len(ca) != len(cb):
        print(f"{len(ca)} committed cases, {len(cb)} fresh")
        return 1
    for x, y in zip(ca, cb):
        if x != y:
            keys = sorted(k for k in set(x) | set(y) if x.get(k) != y.get(k))
            print(f"{x['name']}: fields {keys} differ")
            bad += 1
    print(f"{len(ca)} cases, {bad} differ from vLLM's fresh answers")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
