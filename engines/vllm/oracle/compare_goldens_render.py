# Original to blis-vllm.
"""Fail if a committed render-endpoint golden is not what vLLM's own code decides now.

core/vllm/testdata/golden-render holds the traces oracle/run_render.py made from the pinned
vLLM's register_scale_out_api_routers: whether the server answers the router's render calls, and
the status and message of the refusal when it does not. CI re-runs the pinned release on every
scenario and this compares each fresh trace with its golden.

Usage: python3 -I oracle/compare_goldens_render.py <golden dir> <fresh dir>
(<fresh dir> holds the <name>.trace.json files of run_render.py's directory mode.)
"""
import json
import os
import sys

FIELDS = ("config", "field_default", "dedicated_mode", "serves_render", "render_status",
          "render_error_type", "render_error_message")


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
        differ = [k for k in FIELDS if golden.get(k) != fresh.get(k)]
        if differ:
            print(f"{name}: fields {differ} differ")
            bad += 1
    print(f"{len(names) - bad} of {len(names)} render goldens are vLLM's own output")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
