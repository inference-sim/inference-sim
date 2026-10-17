"""Before/after on the same scored points: re-simulate each point's exact scenario and workload with another
BLIS binary / registry (the coverage branch heads), score both with the same scorer, and tabulate the error
change. "before" is the point's existing result (pins.r6). Writes <out>.json.

Usage: uv run --no-project python -I ab.py --select a2a=allgather_reducescatter --set HELD \
         --blis ../snapshots/blis-kcov@3d96b81 --registry ../snapshots/blis-registry@2e7070c --out ../results/heldout/agrs_ab
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cov  # noqa: E402
import dist_agg  # noqa: E402
import distance as DIST  # noqa: E402

KEYS = ["p50_ttft_ms", "median_ttft_ms", "p50_itl_ms", "median_tpot_ms", "p90_intvty", "median_intvty",
        "output_tput_tps", "output_tput_per_gpu", "tput_per_gpu", "gpu_cache_hit_rate_whole_run"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", required=True)
    ap.add_argument("--set", default="HELD")
    ap.add_argument("--blis", required=True)
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    k, v = a.select.split("=", 1)
    pts = []
    for f in sorted((cov.WORK / "points").glob("*/*/result.json")):
        r = json.loads(f.read_text())
        if r["set"] == a.set and r["status"] == "scored" and str((r.get("resolved") or {}).get("a2a_resolved")) == v:
            pts.append((f.parent, r))
    items, meta = [], []
    for pdir, r in pts:
        bdir = pdir / "ab"
        bdir.mkdir(exist_ok=True)
        for y in pdir.glob("cov-*.yaml"):
            (bdir / y.name).write_text(y.read_text())
        cmd = r["command"].split()
        cmd[0] = str(Path(a.blis).resolve())
        cmd[cmd.index("--registry") + 1] = str(Path(a.registry).resolve())
        cmd[cmd.index("--scenarios") + 1] = str(bdir)
        cmd[cmd.index("--metrics-path") + 1] = str(bdir / "metrics.json")
        items.append({"cmd": cmd, "cwd": str(bdir), "timeout": 3000, "stderr": str(bdir / "stderr.log"), "stamp": " ".join(cmd),
                      "pdir": str(bdir)})
        meta.append((pdir, r, bdir))
    bf = Path(a.out + ".batch.json")
    bf.write_text(json.dumps(items))
    subprocess.run(cov.PX.heavy([sys.executable, "-I", str(HERE / "batch_exec.py"), str(bf)]), check=False)
    rows = []
    for pdir, r, bdir in meta:
        mp = bdir / "metrics.json"
        if not mp.exists():
            rows.append({"point_id": r["point_id"], "error": (bdir / "stderr.log").read_text()[-300:]})
            continue
        inv_ids = {r["run_id"]}
        cov.load_inv(inv_ids)
        x = {"scenario": r["scenario"]}
        meas = cov.db_rows(r["source"]["db_rows"])
        d = dict(r["resolved"])
        after = cov.score(x, d, json.loads(mp.read_text()), meas, None, cov.INVMAP[r["run_id"]])
        e0 = r["measured_rows"][0]["signed_err_pct"]
        e1 = after["measured_rows"][0]["signed_err_pct"]
        d0, d1 = dist_agg.values(DIST.of_result(r, pdir)), dist_agg.values(after.get("distances"))
        rows.append({"point_id": r["point_id"], "run_id": r["run_id"], "conc": r["conc"], "gpu": r["gpu"],
                     "model": d.get("model"), "layout": f"tp{d.get('tp')}-dp{d.get('dp')}", "backend": d.get("a2a_resolved"),
                     "err_before": {m: e0.get(m) for m in KEYS if m in e0}, "err_after": {m: e1.get(m) for m in KEYS if m in e1},
                     "dist_before": d0, "dist_after": d1})
    ok = [x for x in rows if "dist_before" in x]
    keys = sorted({k for x in ok for k in x["dist_before"]} & {k for x in ok for k in x["dist_after"]})
    dsum = {k: {"n": len(v), "before_mean": sum(b for b, _ in v) / len(v), "after_mean": sum(c for _, c in v) / len(v)}
            for k in keys for v in [[(x["dist_before"][k], x["dist_after"][k]) for x in ok
                                     if k in x["dist_before"] and k in x["dist_after"]]] if v}
    doc = {"distances": dsum, "select": a.select, "set": a.set, "before": "pins.r6 (bin/blis-r6, registry 78017af)",
           "after": {"blis": a.blis, "blis_sha256": cov.sha_file(Path(a.blis)), "registry": a.registry}, "points": rows}
    Path(a.out + ".json").write_text(json.dumps(doc, indent=1))
    print(len(rows), "points ->", a.out + ".json")


if __name__ == "__main__":
    main()
