"""Held-out error report at the freeze: scored HELD points are a seeded stratified sample of the planned HELD
points (cov.py plan, HELD_SEED). For each metric: MAPE and signed median, estimated per stratum and combined with
the planned stratum sizes as weights (post-stratified), with a 95% bootstrap CI that resamples RUNS (clusters)
within each stratum. Per-factor tables only where a level has n >= 10 scored points; below that, counts only.
Resolved and unresolvable builds are reported separately.

Writes results/heldout/heldout_report.json. Usage: uv run --no-project python -I heldout_report.py
"""

import collections
import json
import sys as _sys
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = PX.ROOT / "work"
OUT = PX.ROOT / "results" / "heldout" / "heldout_report.json"
METRICS = ["p50_ttft_ms", "median_ttft_ms", "p90_ttft_ms", "p50_itl_ms", "p90_itl_ms", "median_tpot_ms", "p90_tpot_ms",
           "p50_intvty", "p90_intvty", "median_intvty", "p50_e2el_ms", "median_e2el_ms", "output_tput_tps",
           "output_tput_per_gpu", "tput_per_gpu", "gpu_cache_hit_rate_whole_run", "kv_cache_pool_tokens"]
FACTORS = ["model", "gpu", "layout", "pd", "offload", "spec", "frontend", "scenario", "conc_bin"]
B = 1000
SEED = 20261017


_sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PX  # noqa: E402
import dist_agg  # noqa: E402
import distance as DIST  # noqa: E402


def _drop_kv(r):
    """kv_cache_pool_tokens is not compared where the log's KV size is not token capacity (cov.kv_units_differ)."""
    import cov
    if cov.kv_units_differ((r.get("source") or {}).get("server_log_id"), r.get("model_slug")):
        for mr in r.get("measured_rows", []):
            mr["signed_err_pct"].pop("kv_cache_pool_tokens", None)
        r.setdefault("mismatches", []).append({"item": "KV pool units", "measured": "hybrid/compressed cache: logged size "
                                               "inconsistent with its max-concurrency line", "blis": "token capacity",
                                               "removable": False})
    return r


def main():
    plan = [json.loads(ln) for ln in (WORK / "plan.jsonl").open() if '"HELD"' in ln]
    planned = collections.Counter(x["stratum"] for x in plan)
    scored = []
    status = collections.Counter()
    for x in plan:
        f = WORK / "points" / x["run_id"] / f"c{x['conc']}_{x['point_id']}" / "result.json"
        if not f.exists():
            status["not reached"] += 1
            continue
        r = _drop_kv(json.loads(f.read_text()))
        status[r["status"]] += 1
        if r["status"] != "scored":
            continue
        e = r["measured_rows"][0]["signed_err_pct"]
        unres = bool(((r.get("resolved") or {}).get("build") or {}).get("flag"))
        s = x["stratum"].split("|")
        scored.append({"stratum": x["stratum"], "run": r["run_id"], "err": e, "unresolvable": unres,
                       "anchor": x["model"] == "deepseek-v4.1-flash",
                       "f": dict(zip(FACTORS, s)), "dist": DIST.of_result(r, f.parent)})
    reached = collections.Counter(x["stratum"] for x in plan if (WORK / "points" / x["run_id"] / f"c{x['conc']}_{x['point_id']}" / "result.json").exists())
    sc = collections.Counter(p["stratum"] for p in scored)
    doc = {"held_planned": len(plan), "status": dict(status), "held_scored": len(scored), "bootstrap": {"B": B, "seed": SEED,
           "cluster": "run_id within stratum"},
           "sampling": {"strata_planned": len(planned), "strata_reached": len(reached), "strata_scored": len(sc),
                        "per_stratum": {k: {"planned": planned[k], "reached": reached.get(k, 0), "scored": sc.get(k, 0),
                                            "fraction_reached": reached.get(k, 0) / planned[k]} for k in sorted(planned)}},
           "statement": f"estimated over {len(scored)} of {len(plan)} held-out points, stratified (seed 20261017); "
                        f"strata scored {len(sc)} of {len(planned)}"}
    for name, sel in (("all", scored), ("resolved_builds", [p for p in scored if not p["unresolvable"]]),
                      ("unresolvable_builds", [p for p in scored if p["unresolvable"]]),
                      ("dsv41_anchors", [p for p in scored if p["anchor"]])):
        doc[name] = {"n": len(sel), "metrics": {m: estimate(sel, m, planned) for m in METRICS}}
    doc["by_factor"] = {}
    res = [p for p in scored if not p["unresolvable"]]
    for fk in FACTORS:
        lv = collections.defaultdict(list)
        for p in res:
            lv[p["f"][fk]].append(p)
        doc["by_factor"][fk] = {k: ({"n": len(v), "metrics": {m: estimate(v, m, planned) for m in METRICS}} if len(v) >= 10
                                    else {"n": len(v), "note": "n < 10: count only"}) for k, v in sorted(lv.items())}
    doc["distances"] = {name: dist_agg.summarize(sel, planned, FACTORS) for name, sel in
                        (("resolved_builds", res), ("dsv41_anchors", [p for p in scored if p["anchor"]]))}
    OUT.write_text(json.dumps(doc, indent=1))
    print(doc["statement"])
    for m in ("median_ttft_ms", "p50_ttft_ms", "median_tpot_ms", "p50_itl_ms", "p90_intvty", "tput_per_gpu", "output_tput_tps"):
        e = doc["resolved_builds"]["metrics"][m]
        if e:
            print(f"  {m}: n={e['n']} MAPE {e['mape']:.1f}% [{e['mape_ci95'][0]:.1f}, {e['mape_ci95'][1]:.1f}]  signed med {e['signed_median']:+.1f}%")


def estimate(pts, m, planned):
    v = [(p["stratum"], p["run"], p["err"][m]) for p in pts if p["err"].get(m) is not None]
    if not v:
        return None
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for s, r, e in v:
        by[s][r].append(e)

    def ps_mape(groups):
        num = den = 0.0
        for s, runs in groups.items():
            xs = [abs(e) for es in runs.values() for e in es]
            if xs:
                num += planned[s] * statistics.mean(xs)
                den += planned[s]
        return num / den

    point = ps_mape(by)
    rng = random.Random(SEED)
    boots = []
    for _ in range(B):
        g = {}
        for s, runs in by.items():
            keys = list(runs)
            g[s] = {i: runs[rng.choice(keys)] for i in range(len(keys))}
        boots.append(ps_mape(g))
    boots.sort()
    single = sum(1 for runs in by.values() if len(runs) == 1)
    return {"n": len(v), "runs": len({r for _, r, _ in v}), "strata": len(by), "strata_single_run": single,
            "ci_note": "run-clustered bootstrap within strata; a stratum with one scored run adds no variance, so the CI "
                       "understates uncertainty in proportion to strata_single_run/strata",
            "mape": point,
            "mape_ci95": [boots[int(0.025 * B)], boots[int(0.975 * B) - 1]],
            "mape_unweighted": statistics.mean(abs(e) for _, _, e in v), "signed_median": statistics.median(e for _, _, e in v)}


if __name__ == "__main__":
    main()
