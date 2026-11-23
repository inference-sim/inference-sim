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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import goldengz  # noqa: E402  (oracle/goldengz.py; -I leaves the script directory off sys.path)


def comparable(step: dict) -> dict:
    """A step with each tier's block file names cut to their last three components. The first
    component of a name is the storage directory vLLM derives from the model's path (and a hash of
    it), so it records where the run's checkout was; the rest (the block-hash directories and the
    file) is the tier's decision. core/vllm's own comparison of these traces does the same."""
    off = step.get("offload")
    if not isinstance(off, dict) or "tier_files" not in off:
        return step
    files = [sorted("/".join(n.split("/")[-3:]) for n in tier) for tier in off["tier_files"]]
    return {**step, "offload": {**off, "tier_files": files}}


def first_difference(golden: dict, fresh: dict) -> str:
    gs, fs = [comparable(s) for s in golden["steps"]], [comparable(s) for s in fresh["steps"]]
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
    for key in ("groups_error", "groups", "swa_layout"):
        if golden.get(key) != fresh.get(key):
            return f"the {key} vLLM gave differ"
    return ""


def main(golden_dir: str, fresh_dir: str) -> int:
    names = sorted(f[:-5] for f in goldengz.listdir(golden_dir) if f.endswith(".json"))
    if not names:
        # An empty golden directory compares nothing, which must not read as a pass.
        print(f"{golden_dir}: no goldens to compare")
        return 1
    bad = 0
    derived = 0
    for name in names:
        fresh_path = os.path.join(fresh_dir, name + ".trace.json")
        if not os.path.exists(fresh_path):
            print(f"{name}: no fresh trace from vLLM")
            bad += 1
            continue
        golden = goldengz.load(os.path.join(golden_dir, name + ".json"))
        with open(fresh_path) as f:
            fresh = json.load(f)
        if golden["provenance"]["scenario_sha256"] != fresh["provenance"]["scenario_sha256"]:
            print(f"{name}: the golden and vLLM's fresh trace ran on different scenario files")
            bad += 1
            continue
        diff = first_difference(golden, fresh)
        prov = golden["provenance"]
        if "derived_from_source" in prov:
            # A trace derived from the pinned engine source (oracle_run false), for a rule the
            # recorded release does not have: it must say so, and the release's own trace must
            # differ from it. If they agree, the derivation is stale and the recording replaces it.
            if prov.get("oracle_run") is not False or not prov["derived_from_source"]:
                print(f"{name}: a derived golden must state derived_from_source and record oracle_run false")
                bad += 1
            elif not diff:
                print(f"{name}: derived from source, but the oracle's trace is the same; replace it with the recording")
                bad += 1
            else:
                derived += 1
            continue
        if diff:
            print(f"{name}: {diff}")
            bad += 1
    print(f"{len(names)} goldens, {derived} derived from source (the oracle's release differs, as stated), {bad} differ from vLLM's fresh traces")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
