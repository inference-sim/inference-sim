"""r6 vs r7 on the pre-registered HELD subsample (r7cmp/subsample.json): same points, same scorer.
r6 = results/heldout/points.jsonl (blis-r6, registry 78017af, catalog a2a8e47); r7 = work-r7/points/*/result.json.
A point counts only if BOTH builds scored it. Per metric: MAPE (mean |signed err|) and signed median, per factor
level where n >= 10, plus ALL and ANCHORS. kv_cache_pool_tokens dropped where cov.kv_units_differ (as heldout_report).
Writes r7cmp/compare.json and prints the table. Usage: uv run --no-project python -I compare.py"""
import collections, json, statistics as S, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cov  # noqa: E402
import paths as PX  # noqa: E402
C = PX.ROOT
SUB = json.loads((C / "r7cmp" / "subsample.json").read_text())
IDS, ANCH = set(SUB["sample"]), set(SUB["anchors"])
METRICS = ["median_ttft_ms", "p90_ttft_ms", "median_tpot_ms", "p90_tpot_ms", "p50_itl_ms", "median_intvty",
           "output_tput_tps", "output_tput_per_gpu", "median_e2el_ms", "kv_cache_pool_tokens"]
FACTORS = ["model", "gpu", "layout", "pd", "offload", "spec", "frontend", "scenario", "conc_bin"]


def errs(r):
    kvbad = cov.kv_units_differ((r.get("source") or {}).get("server_log_id"))
    out = collections.defaultdict(list)
    for mr in r.get("measured_rows", []):
        for k, v in (mr.get("signed_err_pct") or {}).items():
            if v is not None and not (k == "kv_cache_pool_tokens" and kvbad):
                out[k].append(v)
    return {k: S.fmean(v) for k, v in out.items()}


r6 = {}
for l in (C / "results/heldout/points.jsonl").open():
    x = json.loads(l)
    if x["point_id"] in IDS | ANCH and x.get("status") == "scored":
        r6[x["point_id"]] = x
r7 = {}
for f in (C / "work-r7" / "points").glob("*/*/result.json"):
    x = json.loads(f.read_text())
    if x["point_id"] in IDS | ANCH and x.get("status") == "scored":
        r7[x["point_id"]] = x
both = sorted(set(r6) & set(r7))
status7 = collections.Counter(json.loads(f.read_text()).get("status") for f in (C / "work-r7" / "points").glob("*/*/result.json"))
E6 = {p: errs(r6[p]) for p in both}; E7 = {p: errs(r7[p]) for p in both}


def summ(pids, m):
    a = [(E6[p][m], E7[p][m]) for p in pids if m in E6[p] and m in E7[p]]
    if not a:
        return None
    return {"n": len(a), "mape": [S.fmean(abs(x) for x, _ in a), S.fmean(abs(y) for _, y in a)],
            "median": [S.median(x for x, _ in a), S.median(y for _, y in a)]}


res = {"planned": {"sample": len(IDS), "anchors": len(ANCH)}, "r7_status": dict(status7),
       "both_scored": {"sample": len([p for p in both if p in IDS]), "anchors": len([p for p in both if p in ANCH])},
       "ALL": {m: summ([p for p in both if p in IDS], m) for m in METRICS},
       "ANCHORS": {m: summ([p for p in both if p in ANCH], m) for m in METRICS}, "factors": {}}
for f in FACTORS:
    lv = collections.defaultdict(list)
    for p in both:
        if p in IDS:
            lv[str(r6[p]["factors"].get(f))].append(p)
    res["factors"][f] = {k: {m: summ(v, m) for m in METRICS} for k, v in sorted(lv.items()) if len(v) >= 10}
# Distribution distances (distance.py), the same points scored by both builds, before (r6) and after (r7).
import dist_agg  # noqa: E402
import distance as DIST  # noqa: E402


PSTR = {json.loads(l)["point_id"]: json.loads(l).get("stratum") for l in (C / "work-r7" / "plan.jsonl").open()
        if '"HELD"' in l}


def dpts(src, work, pids):
    out = []
    for p in pids:
        x = src[p]
        pdir = C / work / "points" / x["run_id"] / f"c{x['conc']}_{p}"
        out.append({"stratum": PSTR.get(p) or "unplanned", "run": x["run_id"],
                    "f": {k: str(r6[p]["factors"].get(k)) for k in FACTORS}, "dist": DIST.of_result(x, pdir)})
    return out


def dist_ba(pids, factors=()):
    """Before and after over the SAME points: each distance key is kept on a point only where both builds have it
    (a point scored by a cov.py started before distances existed has none until dist_backfill.py runs)."""
    b, a = dpts(r6, "work", pids), dpts(r7, "work-r7", pids)
    for x, y in zip(b, a):
        vb, va = dist_agg.values(x["dist"]), dist_agg.values(y["dist"])
        keep = set(vb) & set(va)
        x["dist"] = y["dist"] = None
        x["vals"], y["vals"] = {k: vb[k] for k in keep}, {k: va[k] for k in keep}
    before = dist_agg.summarize(b, None, list(factors), b=200)
    after = dist_agg.summarize(a, None, list(factors), b=200)
    return {"before_r6": before, "after_r7": after, "paired": "per distance key, points where both builds have it"}


res["distances"] = {"ALL": dist_ba([p for p in both if p in IDS], FACTORS),
                    "ANCHORS": dist_ba([p for p in both if p in ANCH])}
(C / "r7cmp" / "compare.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: res[k] for k in ("planned", "r7_status", "both_scored")}))
P = ["median_ttft_ms", "median_tpot_ms", "output_tput_tps", "kv_cache_pool_tokens"]
def row(name, d):
    cells = []
    for m in P:
        s = d.get(m)
        cells.append("--" if not s else f"{s['mape'][0]:.1f} -> {s['mape'][1]:.1f} (n {s['n']})")
    return f"| {name} | " + " | ".join(cells) + " |"
print("| set / level | " + " | ".join(f"{m} MAPE r6 -> r7" for m in P) + " |\n|---|" + "---|" * len(P))
print(row("ALL (25% sample)", res["ALL"]))
print(row("DSV4.1 anchors", res["ANCHORS"]))
for f, lvs in res["factors"].items():
    for k, d in lvs.items():
        print(row(f"{f}={k}", d))


DP = [("ttft_ms", "q50_abs_err_pct"), ("ttft_ms", "w1_norm"), ("tpot_ms", "w1_norm"), ("e2el_ms", "w1_norm"),
      ("tpot_ms", "ks"), ("num_running", "w1_norm")]


def drow(name, d):
    cells = []
    for m, k in DP:
        b, a = (d["before_r6"]["overall"].get(f"{m}.{k}"), d["after_r7"]["overall"].get(f"{m}.{k}"))
        cells.append("--" if not (b and a) else f"{b['mean']:.3f} -> {a['mean']:.3f} (n {a['n']})")
    return f"| {name} | " + " | ".join(cells) + " |"


print("\nDistribution distance, r6 -> r7 (mean over points; single-turn rows carry quantile errors only)")
print("| set | " + " | ".join(f"{m} {k}" for m, k in DP) + " |\n|---|" + "---|" * len(DP))
print(drow("ALL (sample)", res["distances"]["ALL"]))
print(drow("DSV4.1 anchors", res["distances"]["ANCHORS"]))
