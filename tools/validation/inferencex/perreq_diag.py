"""FIT-only per-request diagnostics on the one-to-one join (PER_REQUEST.md section 0, D1-D3).

Reads FIT agentic results only (refuses any point whose set is not FIT). For each point: join
verification, distribution distances on the joined set (all pairs, matched in-flight, both solo), and
per-request residuals r = log(BLIS / measured) for TTFT and ITL binned by ISL, prior context (cached
prefix proxy), new tokens, session position and in-flight at arrival. Run-clustered summaries: each
run contributes its own median per bin; across runs we report the median of run medians and a
bootstrap CI over runs.

Usage: uv run --no-project python -I perreq_diag.py <points.jsonl | work dir> <out.json> [--build NAME]
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as PX  # noqa: E402
import perreq as P  # noqa: E402

COV = PX.ROOT
RT = PX.API / "request-timeline"

BINS = {
    "isl": [0, 20e3, 50e3, 100e3, 200e3, 400e3, 2e6],
    "prev_ctx": [0, 1, 20e3, 50e3, 100e3, 200e3, 400e3, 2e6],
    "new_tok": [0, 1e3, 4e3, 16e3, 64e3, 2e6],
    "inflight": [0, 1, 2, 4, 8, 16, 32, 1e9],
}


def load_results(src: Path):
    if src.is_dir():
        for f in sorted(src.glob("points/*/*/result.json")):
            yield json.loads(f.read_text()), src.parent
    else:
        for ln in src.open():
            yield json.loads(ln), COV


def binlab(edges, x):
    for lo, hi in zip(edges, edges[1:]):
        if lo <= x < hi:
            return f"{int(lo)}-{int(hi)}"
    return None


def feats(m, b):
    new = max(0, m["isl"] - m.get("prev_ctx", 0))
    return {"isl": m["isl"], "prev_ctx": m.get("prev_ctx", 0), "new_tok": new, "inflight": m.get("inflight", 0)}


def point_diag(res, root):
    mm, bb, _ = P.load_point(res, root, RT)
    j = P.join(mm, bb)
    pairs = j["pairs"]
    match = [(m, b) for m, b in pairs if abs(m["inflight"] - b["inflight"]) <= 1]
    solo = [(m, b) for m, b in pairs if m["inflight"] == 0 and b["inflight"] == 0]
    out = {"point_id": res["point_id"], "run_id": res["run_id"], "conc": res["conc"], "model": res["model_slug"],
           "gpu": res["gpu"], "verify": j["verify"],
           "inflight_mean": {"measured": _mean([m["inflight"] for m, _ in pairs]),
                             "blis": _mean([b["inflight"] for _, b in pairs])},
           "dist": {"all": P.pair_distances(pairs), "matched_inflight": P.pair_distances(match),
                    "both_solo": P.pair_distances(solo)},
           "n": {"all": len(pairs), "matched": len(match), "solo": len(solo)}}
    rows = []
    for m, b in match:
        f = feats(m, b)
        rows.append({**f, "first": m.get("first_turn", False), "kind": m.get("kind"),
                     "r_ttft": P.log_ratio(m, b, "ttft_ms"), "r_itl": P.log_ratio(m, b, "itl_ms"),
                     "ttft_m": m["ttft_ms"], "ttft_b": b["ttft_ms"], "front_m": m.get("front_ms"),
                     "sched_b": b.get("sched_ms"), "itl_m": m.get("itl_ms"), "itl_b": b.get("itl_ms")})
    out["rows"] = rows
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def clustered(points, key, metric):
    """Median of per-run medians of `metric` per bin of `key`, with a bootstrap CI over runs."""
    per_run = {}
    for p in points:
        for r in p["rows"]:
            v = r.get(metric)
            if v is None:
                continue
            lab = (("first" if r["first"] else "later") if key == "position" else binlab(BINS[key], r[key]))
            per_run.setdefault(lab, {}).setdefault(p["run_id"], []).append(v)
    out = {}
    rng = random.Random(0)
    for lab, runs in sorted(per_run.items(), key=lambda kv: str(kv[0])):
        meds = [statistics.median(v) for v in runs.values()]
        bs = sorted(statistics.median(rng.choices(meds, k=len(meds))) for _ in range(500))
        out[lab] = {"runs": len(meds), "n": sum(len(v) for v in runs.values()),
                    "median_pct": (math.exp(statistics.median(meds)) - 1) * 100,
                    "ci_pct": [(math.exp(bs[12]) - 1) * 100, (math.exp(bs[487]) - 1) * 100]}
    return out


def ttft_slope(points):
    """Per run: OLS of (BLIS - measured) TTFT ms on prior context (k tokens) over matched pairs with
    fewer than 4k new tokens; a context-proportional cost BLIS lacks shows as a NEGATIVE slope.
    Returns per-run slopes and the run-clustered median with a bootstrap CI."""
    per = {}
    for p in points:
        xs, ys = [], []
        for r in p["rows"]:
            if r["new_tok"] < 4e3 and r["prev_ctx"] > 0:
                xs.append(r["prev_ctx"] / 1e3)
                ys.append(r["ttft_b"] - r["ttft_m"])
        if len(xs) >= 20 and max(xs) - min(xs) > 20:
            mx, my = statistics.mean(xs), statistics.mean(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            per.setdefault(p["run_id"], []).append(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx)
    slopes = {k: statistics.median(v) for k, v in per.items()}
    v = list(slopes.values())
    if not v:
        return {"runs": 0}
    rng = random.Random(1)
    bs = sorted(statistics.median(rng.choices(v, k=len(v))) for _ in range(500))
    return {"runs": len(v), "median_ms_per_1k": statistics.median(v), "ci": [bs[12], bs[487]], "per_run": slopes}


def run_weighted(points, scope, metric, field="w1_norm"):
    by = {}
    for p in points:
        d = (p["dist"][scope] or {}).get(metric)
        if d and d.get(field) is not None:
            by.setdefault(p["run_id"], []).append(abs(d[field]))
    meds = [statistics.mean(v) for v in by.values()]
    return {"runs": len(meds), "mean": statistics.mean(meds) if meds else None}


def main():
    src, outp = Path(sys.argv[1]), Path(sys.argv[2])
    pts = []
    for res, root in load_results(src):
        if res.get("scenario") != "agentic_traces" or res.get("status") != "scored":
            continue
        if res.get("set") != "FIT":
            raise SystemExit(f"refusing non-FIT point {res['point_id']} ({res.get('set')})")
        rid = res["source"]["db_rows"][0]
        if not (RT / f"rt_{rid}.body").exists():
            continue
        pts.append(point_diag(res, root))
    summ = {"points": len(pts), "runs": len({p["run_id"] for p in pts})}
    for metric in ("r_ttft", "r_itl"):
        summ[metric] = {k: clustered(pts, k, metric) for k in ("isl", "prev_ctx", "new_tok", "inflight", "position")}
    summ["ttft_context_slope"] = ttft_slope(pts)
    summ["distance"] = {s: {k: {f: run_weighted(pts, s, k, f) for f in ("w1_norm", "ks", "q50_err_pct", "q90_err_pct",
                                                                       "q99_err_pct")}
                            for k in P.METRICS} for s in ("all", "matched_inflight", "both_solo")}
    out = {"summary": summ, "points": [{k: v for k, v in p.items() if k != "rows"} for p in pts]}
    outp.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "ttft_context_slope"} | {
        "ttft_context_slope": {k: v for k, v in summ["ttft_context_slope"].items() if k != "per_run"}}, indent=1)[:6000])


if __name__ == "__main__":
    main()
