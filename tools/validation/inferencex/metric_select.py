"""Metric selection for fitting (RELEASE_V1 item 31), FIT points only. Reads the seed-repeat and coefficient-
perturbation variants of FIT points (a variants dir laid out <point_id>/<tag>/metrics.json, built by
dist_variants.py) and the stored FIT results, and writes per metric x distance:
  signal   between-config variance of the value / mean within-config seed variance (BLIS seeds 42-44 agentic,
           0-2 single-turn), plus the measured-side repeat spread where a FIT config was run twice
  (agentic: dp = 1 points only)
  sens     per perturbed term: median over points of |value(term) - value(base)| / seed SD (a z score)
  indep    Spearman correlation across FIT points between metrics (base seed)
Usage: uv run --no-project python -I metric_select.py <variants dir> <out.json>
"""

from __future__ import annotations

import collections
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as PX  # noqa: E402
import cov  # noqa: E402
import dist_agg  # noqa: E402
import distance as DIST  # noqa: E402
import metrics as M  # noqa: E402

WORK = PX.ROOT / "work"
MAPE_KEYS_AG = ["p50_ttft_ms", "p90_ttft_ms", "p50_itl_ms", "p90_itl_ms", "p50_e2el_ms", "p90_intvty",
                "output_tput_tps", "mean_output_tokens"]
MAPE_KEYS_ST = ["median_ttft_ms", "p90_ttft_ms", "median_tpot_ms", "p90_tpot_ms", "median_e2el_ms", "p99_tpot_ms",
                "median_intvty", "tput_per_gpu"]


def fit_points():
    out = {}
    for f in (WORK / "points").glob("*/*/result.json"):
        r = json.loads(f.read_text())
        if r["set"] == "FIT" and r["status"] == "scored":
            out[r["point_id"]] = (f.parent, r)
    return out


def score_variant(r: dict, m: dict) -> dict:
    """{metric.distance: value} plus {'err.<key>': signed %} for a variant's metrics.json of point r."""
    mr = r["measured_rows"][0]
    meas = {k: v["value"] for k, v in mr["measured"].items()}
    hc = r["b2_host_completion_ms"]
    rows = M.blis_rows(m, hc, True)
    if r["scenario"] == "single_turn":
        tp = int(r["resolved"].get("tp") or 1)
        pred = M.single_turn_blis(m, rows, tp)
        keys = MAPE_KEYS_ST
        err = {k: M.signed_pct(pred.get(k), meas.get(k)) for k in keys if k in meas}
        fake = {"measured_rows": [{"signed_err_pct": {k: M.signed_pct(pred.get(k), meas.get(k)) for k in meas}}]}
        dist = DIST.single_turn_quantiles(fake)
    else:
        pred = M.request_metrics(rows)
        keys = MAPE_KEYS_AG
        err = {k: M.signed_pct(pred.get(k), meas.get(k)) for k in keys if k in meas}
        dist = DIST.point_distances(m, hc, mr["row_id"], {"total_requests_completed": meas.get("requests")},
                                    int(r["resolved"].get("dp") or 1))
    out = dist_agg.values(dist)
    out.update({f"err.{k}": abs(v) for k, v in err.items() if v is not None})
    return out


def spearman(a, b):
    def rank(x):
        o = sorted(range(len(x)), key=lambda i: x[i])
        rk = [0.0] * len(x)
        for r, i in enumerate(o):
            rk[i] = r
        return rk
    if len(a) < 4:
        return None
    ra, rb = rank(a), rank(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else None


def main():
    vdir, outp = Path(sys.argv[1]), Path(sys.argv[2])
    pts = fit_points()
    V = collections.defaultdict(dict)  # pid -> tag -> values
    for pid, (pdir, r) in pts.items():
        d = vdir / pid
        if not d.exists():
            continue
        for t in d.iterdir():
            mp = t / "metrics.json"
            if mp.exists():
                try:
                    V[pid][t.name] = score_variant(r, json.loads(mp.read_text()))
                except Exception as e:  # recorded, never silent
                    V[pid][t.name] = {"_error": f"{type(e).__name__}: {e}"[:200]}
    seeds = {"single_turn": ["seed0", "seed1", "seed2"], "agentic_traces": ["seed42", "seed43", "seed44"]}
    base_tag = {"single_turn": "seed0", "agentic_traces": "seed42"}
    doc = {"points": {}, "metrics": {}, "sensitivity": {}, "measured_repeat": {}, "correlation": {}}
    keys = sorted({k for pv in V.values() for tv in pv.values() for k in tv if not k.startswith("_")})
    for sc in ("single_turn", "agentic_traces"):
        # dp > 1 agentic points are excluded: BLIS is ~10x slow there (a separate DP-attention mechanism) and their
        # server series are a named mismatch, so they would dominate both variances.
        P = [p for p in V if pts[p][1]["scenario"] == sc and int(pts[p][1]["resolved"].get("dp") or 1) == 1]
        doc["points"][sc] = {p: {"run_id": pts[p][1]["run_id"], "conc": pts[p][1]["conc"],
                                 "model": pts[p][1]["resolved"].get("model"), "gpu": pts[p][1]["gpu"],
                                 "tags": sorted(V[p])} for p in P}
        for k in keys:
            within, base_vals = [], []
            for p in P:
                xs = [V[p][s][k] for s in seeds[sc] if s in V[p] and k in V[p][s]]
                if len(xs) >= 2:
                    within.append(statistics.variance(xs))
                b = V[p].get(base_tag[sc], {}).get(k)
                if b is not None:
                    base_vals.append(b)
            if len(base_vals) < 3 or not within:
                continue
            wv = statistics.fmean(within)
            bv = statistics.variance(base_vals)
            sd = math.sqrt(wv) if wv > 0 else None
            sens, dabs = {}, {}
            for p in P:
                b = V[p].get(base_tag[sc], {}).get(k)
                psd = math.sqrt(statistics.variance([V[p][s][k] for s in seeds[sc] if s in V[p] and k in V[p][s]])) \
                    if sum(1 for s in seeds[sc] if s in V[p] and k in V[p][s]) >= 2 else None
                for t, tv in V[p].items():
                    if t.startswith("seed") or k not in tv or b is None:
                        continue
                    noise = max(psd if psd is not None else (sd or 0.0), 1e-9 * max(1.0, abs(b)))
                    sens.setdefault(t, []).append(abs(tv[k] - b) / noise)
                    dabs.setdefault(t, []).append(abs(tv[k] - b))
            doc["metrics"].setdefault(sc, {})[k] = {
                "n_points": len(base_vals), "between_var": bv, "within_seed_var": wv,
                "snr": bv / wv if wv > 0 else (float("inf") if bv > 0 else None),
                "base_mean": statistics.fmean(base_vals), "seed_sd_mean": statistics.fmean(math.sqrt(x) for x in within),
                "sd_basis": "per-point seed SD where the point has >= 2 seeds, else the pooled within-seed SD",
                "sens_median_z": {t: statistics.median(z) for t, z in sorted(sens.items())},
                "sens_max_z": {t: max(z) for t, z in sorted(sens.items())},
                "sens_median_abs_delta": {t: statistics.median(z) for t, z in sorted(dabs.items())}}
        # correlation across points (base seed), distances and |errors|
        ks_ = [k for k in keys if all(k in V[p].get(base_tag[sc], {}) for p in P)]
        doc["correlation"][sc] = {a: {b: spearman([V[p][base_tag[sc]][a] for p in P], [V[p][base_tag[sc]][b] for p in P])
                                      for b in ks_} for a in ks_}
    # measured-side repeats in FIT: same config_id and conc scored twice (different InferenceX runs)
    inv = {}
    for ln in cov.INV.open():
        x = json.loads(ln)
        inv[x["run_id"]] = x.get("config_id")
    grp = collections.defaultdict(list)
    for pid, (pdir, r) in pts.items():
        if r["scenario"] == "agentic_traces":
            grp[(inv.get(r["run_id"]), r["conc"])].append(pid)
    for g, pids in grp.items():
        if len(pids) >= 2:
            vals = [dist_agg.values(DIST.of_result(pts[p][1], pts[p][0])) for p in pids]
            doc["measured_repeat"][f"config {g[0]} c{g[1]}"] = {
                "points": pids, "spread": {k: [v.get(k) for v in vals] for k in sorted(vals[0]) if all(k in v for v in vals)}}
    outp.write_text(json.dumps(doc, indent=1, default=str))
    print("wrote", outp, {sc: len(v) for sc, v in doc["metrics"].items()})


if __name__ == "__main__":
    main()
