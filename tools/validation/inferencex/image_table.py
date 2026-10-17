"""Per-image resolution table: for every vLLM image the harness replayed, how each version-dependent
behaviour BLIS models was decided (from the run's log / recipe, or that version's own default), and the
named MISMATCHes. Reads work/points/*/*/result.json; writes results/image_resolution.json and prints a
markdown table for HARNESS.md.

Usage: uv run --no-project python -I image_table.py
"""

import collections
import json
from pathlib import Path
import sys as _s
_s.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PX  # noqa: E402

HERE = Path(__file__).resolve().parent
FIELDS = ["engine.gpu_memory_utilization", "engine.max_num_seqs", "engine.max_num_batched_tokens", "engine.block_size",
          "engine.max_model_len", "engine.cudagraph_capture_sizes", "engine.async_scheduling", "engine.frontend",
          "engine.swa_bounded_replay", "acceptance"]


def main():
    by = collections.defaultdict(lambda: {"points": 0, "vllm_version": set(), "fields": collections.defaultdict(collections.Counter),
                                          "values": collections.defaultdict(collections.Counter), "model_runner": collections.Counter(),
                                          "a2a": collections.Counter(), "mismatches": collections.Counter()})
    for p in (PX.ROOT / "work" / "points").glob("*/*/result.json"):
        r = json.loads(p.read_text())
        res = r.get("resolved")
        if not res:
            continue
        img = res.get("engine_image") or "?"
        e = by[img]
        e["points"] += 1
        e["vllm_version"].add(res.get("vllm_version"))
        for f in FIELDS:
            s = res["sources"].get(f)
            if s:
                e["fields"][f][s] += 1
                v = (res.get("engine") or {}).get(f.split(".", 1)[-1])
                if v is not None and not isinstance(v, (list, dict)):
                    e["values"][f][str(v)] += 1
        e["model_runner"][res.get("model_runner_image")] += 1
        e["a2a"][res.get("a2a_resolved")] += 1
        for m in r.get("mismatches", []):
            e["mismatches"][m["item"]] += 1
    out = {img: {"points": e["points"], "vllm_version": sorted(x for x in e["vllm_version"] if x),
                 "fields": {f: dict(c) for f, c in e["fields"].items()},
                 "values": {f: dict(c) for f, c in e["values"].items()},
                 "model_runner": dict(e["model_runner"]), "all2all": dict(e["a2a"]), "mismatches": dict(e["mismatches"])}
           for img, e in sorted(by.items())}
    (PX.ROOT / "results" / "image_resolution.json").write_text(json.dumps(out, indent=1))
    print("| image | vLLM | pts | GMU | max_num_seqs | runner | swa_bounded_replay | async | frontend |")
    print("|---|---|---|---|---|---|---|---|---|")
    for img, e in out.items():
        def show(f):
            vals = e["values"].get(f, {})
            src = e["fields"].get(f, {})
            s = "; ".join(sorted(set(x.split(" (")[0] for x in src)))
            return (",".join(sorted(vals)) or "-") + (f" ({s})" if s else "")
        print(f"| {img} | {','.join(e['vllm_version'])} | {e['points']} | {show('engine.gpu_memory_utilization')} | "
              f"{show('engine.max_num_seqs')} | {','.join(map(str, e['model_runner']))} | {show('engine.swa_bounded_replay')} | "
              f"{show('engine.async_scheduling')} | {show('engine.frontend')} |")


if __name__ == "__main__":
    main()
