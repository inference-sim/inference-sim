# Original to blis-vllm.
"""Fail if a committed adapter-cache golden is not what vLLM produces now.

core/lora/testdata/golden holds the traces oracle/run_lora_cache.py made from vLLM's own
LRUCacheWorkerLoRAManager. CI re-runs the pinned release on every scenario and this compares each
fresh trace with its golden, batch by batch: the adapters requested, every load, eviction,
activation and deactivation in order, the error vLLM raised, the two caches' LRU order and the GPU
slots after the batch.

Usage: python3 -I oracle/compare_goldens_lora.py <golden dir> <fresh dir>
(<fresh dir> holds the <name>.trace.json files of run_lora_cache.py's directory mode.)
"""
import json
import os
import sys

FIELDS = ("op", "result", "requested", "events", "error", "registered", "active", "slots")


def first_difference(golden: dict, fresh: dict) -> str:
    if golden["config"] != fresh["config"] or golden["adapter_ids"] != fresh["adapter_ids"]:
        return "the configuration or the adapter numbering differs"
    gs, fs = golden["steps"], fresh["steps"]
    for i in range(max(len(gs), len(fs))):
        if i >= len(gs):
            return f"step {i}: only vLLM's fresh trace has it"
        if i >= len(fs):
            return f"step {i}: only the golden has it"
        bad = [k for k in FIELDS if gs[i].get(k) != fs[i].get(k)]
        if bad:
            return f"step {i}: fields {bad} differ"
    return ""


def main(golden_dir: str, fresh_dir: str) -> int:
    names = sorted(f[:-5] for f in os.listdir(golden_dir) if f.endswith(".json"))
    if not names:
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
    print(f"{len(names) - bad} of {len(names)} adapter-cache goldens are vLLM's own output")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
