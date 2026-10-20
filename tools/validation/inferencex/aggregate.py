"""Collect scored points into one table per run and per-factor aggregates.

FIT  -> results/fit/{runs/<run_id>.json, points.jsonl, aggregate.json, blocked.jsonl}
HELD -> results/heldout/... (the same layout). The fits agent must not read results/heldout until it
has frozen a fit (PARITY_PLAN.md D); nothing under results/fit summarises held-out points.

Usage: uv run --no-project python -I aggregate.py
"""

from __future__ import annotations

import collections
import json
import sys as _sys
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = PX.ROOT / "work"
RESULTS = PX.ROOT / "results"
HEAD = ["median_ttft_ms", "p90_ttft_ms", "median_tpot_ms", "p90_tpot_ms", "p50_itl_ms", "p90_itl_ms",
        "p50_intvty", "p90_intvty", "median_intvty", "output_tput_tps", "output_tput_per_gpu", "tput_per_gpu",
        "median_e2el_ms", "p50_e2el_ms", "gpu_cache_hit_rate_whole_run", "kv_cache_pool_tokens"]


def conc_bin(c):
    b = 1
    while b < c:
        b *= 2
    return f"<= {b}"


def factors(r):
    res = r.get("resolved") or {}
    sp = res.get("spec_method") or "none"
    return {"model": res.get("model", r["model_slug"]), "gpu": r["gpu"],
            "layout": f"tp{res.get('tp')}-dp{res.get('dp')}-ep{int(bool(res.get('enable_expert_parallel')))}",
            "pd": "colocated", "offload": "cpu" if res.get("offload") else "off", "spec": sp,
            "frontend": (res.get("engine") or {}).get("frontend"), "scenario": r["scenario"],
            "conc_bin": conc_bin(r["conc"]), "vllm_version": res.get("vllm_version"),
            "build_status": "unresolvable build" if (res.get("build") or {}).get("flag") else
                            (res.get("build") or {}).get("status", "not tabled (pre-table result)")}


ANCHOR_CONFIGS = {2560, 2570, 2640}  # partition named subset dsv41_hopper_anchors


def tag_of(p, inv_cfg):
    if p["set"] == "FIT":
        return "fit"
    return "anchor" if inv_cfg.get(p["run_id"]) in ANCHOR_CONFIGS else "held-out"


_sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PX  # noqa: E402


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
    pts = [_drop_kv(json.loads(p.read_text())) for p in sorted((WORK / "points").glob("*/*/result.json"))]
    inv_cfg = {}
    for ln in PX.INV.open():
        r = json.loads(ln)
        inv_cfg[r["run_id"]] = r.get("config_id")
    for p in pts:
        p["tag"] = tag_of(p, inv_cfg)
    for s, sub in (("FIT", "fit"), ("HELD", "heldout")):
        out = RESULTS / sub
        (out / "runs").mkdir(parents=True, exist_ok=True)
        mine = [p for p in pts if p["set"] == s]
        scored = [p for p in mine if p["status"] == "scored"]
        with (out / "points.jsonl").open("w") as f:
            for p in scored:
                p["factors"] = factors(p)
                f.write(json.dumps(p, default=str) + "\n")
        with (out / "blocked.jsonl").open("w") as f:
            for p in mine:
                if p["status"] != "scored":
                    f.write(json.dumps({k: p.get(k) for k in ("point_id", "run_id", "conc", "status", "tag", "blocked_reason",
                                                               "error", "model_slug", "gpu", "scenario")}) + "\n")
        byrun = collections.defaultdict(list)
        for p in scored:
            byrun[p["run_id"]].append(p)
        for rid, ps in byrun.items():
            ps.sort(key=lambda p: p["conc"])
            tab = []
            for p in ps:
                for mr in p["measured_rows"]:
                    tab.append({"conc": p["conc"], "point_id": p["point_id"], "row_id": mr["row_id"],
                                "measured": {k: v["value"] for k, v in mr["measured"].items()},
                                "blis": {k: p["predicted"][k]["value"] for k in mr["measured"] if k in p["predicted"]},
                                "signed_err_pct": mr["signed_err_pct"]})
            (out / "runs" / f"{rid}.json").write_text(json.dumps({
                "run_id": rid, "set": s, "source": ps[0]["source"], "factors": factors(ps[0]),
                "mismatches": ps[0]["mismatches"], "unexpressed_metrics": ps[0]["unexpressed_metrics"],
                "pins": ps[0]["pins"], "resolved": ps[0]["resolved"], "table": tab}, indent=1, default=str))
        agg = {"set": s, "points_scored": len(scored), "points_blocked": sum(p["status"] == "blocked" for p in mine),
               "points_error": sum(p["status"] == "error" for p in mine),
               "blocked_reasons": collections.Counter((p.get("blocked_reason") or p.get("error") or "")[:140]
                                                      for p in mine if p["status"] != "scored").most_common(40),
               "overall": summarize(scored),
               "overall_resolved_builds": summarize([p for p in scored if p["factors"]["build_status"] != "unresolvable build"]),
               "overall_unresolvable_builds": summarize([p for p in scored if p["factors"]["build_status"] == "unresolvable build"]),
               "rescored_after_behaviour_table": sum((WORK / "points" / p["run_id"] / f"c{p['conc']}_{p['point_id']}" / "rescored").exists()
                                                     for p in mine),
               "by_factor": {}}
        for p in scored:
            p["factors"]["tag"] = p["tag"]
        for fk in ("build_status", "tag", "model", "gpu", "layout", "pd", "offload", "spec", "frontend", "scenario", "conc_bin",
                   "vllm_version"):
            groups = collections.defaultdict(list)
            for p in scored:
                groups[str(p["factors"][fk])].append(p)
            agg["by_factor"][fk] = {lv: summarize(g) for lv, g in sorted(groups.items())}
        agg["distances"] = distance_section(s, scored)
        (out / "aggregate.json").write_text(json.dumps(agg, indent=1))
        print(sub, len(scored), "scored", agg["points_blocked"], "blocked", agg["points_error"], "errors")


DIST_FACTORS = ["model", "gpu", "layout", "spec", "frontend", "scenario", "conc_bin", "tag"]


def planned_strata(set_name: str) -> dict:
    """Planned points per stratum (cov.stratum) for the set: the post-stratification weights."""
    import cov
    plan = [json.loads(ln) for ln in (WORK / "plan.jsonl").open()]
    plan = [x for x in plan if x["set"] == set_name]
    if not all("stratum" in x for x in plan):
        if not cov.INVLITE:
            for ln in cov.INV.open():
                r = json.loads(ln)
                cov.INVLITE[r["run_id"]] = {k: r.get(k) for k in ("parallel", "offload", "spec_method", "frontend")}
        for x in plan:
            x.setdefault("stratum", "|".join(map(str, cov.stratum(x))))
    return collections.Counter(x["stratum"] for x in plan), {x["point_id"]: x["stratum"] for x in plan}


def distance_section(set_name: str, scored: list[dict], work: Path | None = None) -> dict:
    """Post-stratified distribution distances (dist_agg) with run-clustered bootstrap CIs, overall and per factor."""
    import dist_agg
    import distance as DIST
    work = work or WORK
    planned, pstr = planned_strata(set_name)
    pts = []
    for p in scored:
        pdir = work / "points" / p["run_id"] / f"c{p['conc']}_{p['point_id']}"
        f = dict(p.get("factors") or factors(p))
        f.setdefault("tag", p.get("tag"))
        pts.append({"stratum": pstr.get(p["point_id"], "unplanned"), "run": p["run_id"], "f": f,
                    "dist": DIST.of_result(p, pdir)})
    doc = dist_agg.summarize(pts, planned, DIST_FACTORS)
    doc["definition"] = DIST.__doc__.strip().splitlines()[0]
    doc["weights"] = f"planned {set_name} points per stratum (cov.stratum)"
    return doc


def distances_only():
    """Add the 'distances' section to the existing aggregate.json files over exactly the points their points.jsonl
    lists (the published population; nothing else is rewritten)."""
    for s, sub in (("FIT", "fit"), ("HELD", "heldout")):
        out = RESULTS / sub
        scored = [json.loads(ln) for ln in (out / "points.jsonl").open()]
        agg = json.loads((out / "aggregate.json").read_text())
        agg["distances"] = distance_section(s, scored)
        (out / "aggregate.json").write_text(json.dumps(agg, indent=1))
        d = agg["distances"]
        print(sub, d["points"], "points", d["points_with_distances"], "with distances", d["mismatches"])


def summarize(ps):
    errs = collections.defaultdict(list)
    for p in ps:
        for mr in p["measured_rows"]:
            for k, v in mr["signed_err_pct"].items():
                if v is not None:
                    errs[k].append(v)
    return {"points": len(ps), "metrics": {k: {"n": len(v), "mape": statistics.mean(abs(x) for x in v),
                                              "signed_median": statistics.median(v)}
                                          for k, v in sorted(errs.items()) if k in HEAD or len(errs) < 60}}


if __name__ == "__main__":
    distances_only() if "--distances-only" in _sys.argv else main()
