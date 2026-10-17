"""Gross KV-pool deviations among r7 scored points, clustered by setup (KV_METRICS.md sec 6).
Gross: |BLIS/measured - 1| > 0.30 (outside 0.7-1.3) or ratio outside [0.5, 2].
Usage: uv run --no-project python -I kvpool_triage.py -> kvm/kvpool.json + table"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as PX  # noqa: E402
C = PX.ROOT
import cov  # noqa: E402


def main():
    rows = []
    for f in (C / "work-r7" / "points").glob("*/*/result.json"):
        x = json.loads(f.read_text())
        if x.get("status") != "scored":
            continue
        P = (x.get("predicted") or {}).get("kv_cache_pool_tokens")
        slid = (x.get("source") or {}).get("server_log_id")
        rs = x.get("resolved") or {}
        eng = rs.get("engine") or {}
        for mr in x.get("measured_rows", []):
            m = (mr.get("measured") or {}).get("kv_cache_pool_tokens")
            if not P or not m or not P.get("value") or not m.get("value"):
                continue
            ratio = P["value"] / m["value"]
            rows.append({"point_id": x["point_id"], "set": x["set"], "model": x["model_slug"], "gpu": x["gpu"],
                         "conc": x["conc"], "scenario": x["scenario"], "units_differ": cov.kv_units_differ(slid),
                         "ratio": ratio, "blis": P["value"], "meas": m["value"], "server_log_id": slid,
                         "tp": rs.get("tp"), "dp": rs.get("dp"), "ep": rs.get("enable_expert_parallel"),
                         "ver": rs.get("vllm_version"), "mml": eng.get("max_model_len"),
                         "gmu": eng.get("gpu_memory_utilization"), "bs": eng.get("block_size"),
                         "dtype": eng.get("cache_dtype"), "spec": rs.get("spec_method"),
                         "quant": rs.get("quantization_log"), "mism": [mm["item"] for mm in x.get("mismatches", [])]})
            break
    (C / "kvm" / "kvpool.json").write_text(json.dumps(rows) + "\n")
    gross = [r for r in rows if not (0.7 <= r["ratio"] <= 1.3)]
    print(f"pool compared: {len(rows)} points; gross (ratio outside 0.7-1.3): {len(gross)}; "
          f"of which units_differ (DSV4-Pro class): {sum(r['units_differ'] for r in gross)}")
    cl = collections.defaultdict(list)
    for r in gross:
        cl[(r["model"], r["gpu"], f"tp{r['tp']}dp{r['dp']}ep{int(bool(r['ep']))}", r["ver"], r["dtype"], r["units_differ"])].append(r)
    print("| model | gpu | layout | vLLM | kv dtype | units_differ | n | ratio min..max | mml | gmu | block |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in sorted(cl.items(), key=lambda kv: -len(kv[1])):
        rr = [r["ratio"] for r in v]
        print(f"| {' | '.join(str(a) for a in k)} | {len(v)} | {min(rr):.2f}..{max(rr):.2f} | "
              f"{sorted({r['mml'] for r in v})} | {sorted({r['gmu'] for r in v}, key=str)} | {sorted({r['bs'] for r in v}, key=str)} |")


if __name__ == "__main__":
    main()
