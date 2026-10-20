"""Re-classify every planned point (FIT + HELD) under the parity rules (blockers.py; PARITY_FIXES.md), and report
the r7cmp HELD subsample before/after on the comparable set. No point is re-simulated: every rule is a blocker or a
scoring-side change (the DSV4-Pro pool block for block), so it is decided from result.json, metrics.json and the
run's server log facts (logfacts.py -> work-r7/logfacts.json) and DB rows (work-r7/row_isl_osl.json).

Point source: work-r7/points (the r7 HELD subsample) where present, else work/points (r6, all of FIT and HELD).
Writes work-r7/parity_reclass.json. Usage: uv run --no-project python -I parity_reclass.py"""
import collections
import json
import statistics as S
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blockers as BLK  # noqa: E402
import paths as PX  # noqa: E402
C = PX.ROOT
import cov  # noqa: E402
import dist_agg  # noqa: E402
import distance as DIST  # noqa: E402

FACTS = json.loads((C / "work-r7" / "logfacts.json").read_text())
ROWS = json.loads((C / "work-r7" / "row_isl_osl.json").read_text())
CQ = cov.catalog_quant(cov.DEFAULT_CATALOG)
INV = {}
for ln in cov.INV.open():
    r = json.loads(ln)
    INV[r["run_id"]] = r["parallel"]["decode"].get("tp")


def pdir(work, x):
    return C / work / "points" / x["run_id"] / f"c{x['conc']}_{x['point_id']}"


def classify(r):
    slid = (r.get("source") or {}).get("server_log_id")
    facts = FACTS.get(str(slid)) if slid else None
    row = ROWS.get(str(r["measured_rows"][0]["row_id"])) or [None, None, None]
    gpus = int(INV.get(r["run_id"]) or r["resolved"]["tp"])
    return BLK.parity_blockers(r, facts, CQ.get(r["resolved"]["model"]), row[1], gpus), facts


def pool_block_err(r, facts, pdir_):
    """DSV4-Pro, dp 1: BLIS blocks / vLLM blocks - 1 (%), or None when the derivation does not reproduce the log."""
    if r["model_slug"] != "deepseek-v4-pro" or (r["resolved"].get("dp") or 1) != 1:
        return None
    e = r["resolved"]["engine"]
    vb = BLK.dsv4_pro_pool_blocks(facts, e["max_model_len"], e["max_num_batched_tokens"])
    P = (r.get("predicted") or {}).get("kv_cache_pool_tokens")
    if not vb or not P:
        return None
    return (P["value"] // e["block_size"] / vb - 1) * 100


def main():
    plan = [json.loads(ln) for ln in (C / "work" / "plan.jsonl").open()]
    out, cnt, cls, overlap = [], collections.Counter(), collections.Counter(), collections.Counter()
    by_model = collections.defaultdict(collections.Counter)
    for x in plan:
        src = "work-r7" if (pdir("work-r7", x) / "result.json").exists() else "work"
        f = pdir(src, x) / "result.json"
        if not f.exists():
            cnt[(x["set"], "not reached")] += 1
            continue
        r = json.loads(f.read_text())
        if r["status"] != "scored":
            cnt[(x["set"], r["status"])] += 1
            continue
        pb, facts = classify(r)
        names = sorted({b["name"] for b in pb})
        cnt[(x["set"], "scored before")] += 1
        cnt[(x["set"], "scoreable after" if not pb else "parity-blocked")] += 1
        for n in names:
            cls[(x["set"], n)] += 1
            by_model[n][(r["model_slug"], r["gpu"], f"c{r['conc']}" if n == "offered load not reached" else "")] += 1
        if len(names) > 1:
            overlap[tuple(names)] += 1
        out.append({"point_id": x["point_id"], "set": x["set"], "src": src, "model": r["model_slug"], "gpu": r["gpu"],
                    "conc": r["conc"], "scenario": r["scenario"], "blockers": pb,
                    "pool_block_err_pct": pool_block_err(r, facts, f.parent),
                    "kv_tokens_compared_old": "kv_cache_pool_tokens" in r["measured_rows"][0]["measured"]
                    and not cov.kv_units_differ((r.get("source") or {}).get("server_log_id")),
                    "kv_tokens_compared_new": "kv_cache_pool_tokens" in r["measured_rows"][0]["measured"]
                    and not cov.kv_units_differ((r.get("source") or {}).get("server_log_id"), r["model_slug"])})
    (C / "work-r7" / "parity_reclass.json").write_text(json.dumps(out) + "\n")
    print("## Points by set and status (before = scored; after = scored and no parity blocker)")
    for k in sorted(cnt):
        print(f"  {k[0]:4} {k[1]:18} {cnt[k]}")
    print("## Per-class counts (a point can carry more than one)")
    for k in sorted(cls):
        print(f"  {k[0]:4} {k[1]:34} {cls[k]}")
    print("  overlaps:", dict(overlap))
    for n, c in by_model.items():
        print(f"## {n}: by (model, gpu[, conc])")
        for k, v in c.most_common(12):
            print(f"  {v:5} {k}")
    dsv = [o for o in out if o["model"] == "deepseek-v4-pro"]
    print("## DSV4-Pro pool guard: points whose log pool was compared as tokens, old guard -> new guard:",
          sum(o["kv_tokens_compared_old"] for o in dsv), "->", sum(o["kv_tokens_compared_new"] for o in dsv))
    be = [o["pool_block_err_pct"] for o in dsv if o["pool_block_err_pct"] is not None]
    print(f"   scored block for block: {len(be)} of {len(dsv)} DSV4-Pro scored points; signed err median "
          f"{S.median(be):+.1f}%, |err| median {S.median(abs(v) for v in be):.1f}%" if be else "   none")
    cmp_subsample(out)


METRICS = ["median_ttft_ms", "median_tpot_ms", "median_e2el_ms", "output_tput_per_gpu", "kv_cache_pool_tokens"]
DKEYS = ["ttft_ms.q50_abs_err_pct", "tpot_ms.q50_abs_err_pct", "ttft_ms.w1_norm", "tpot_ms.w1_norm", "e2el_ms.w1_norm",
         "tpot_ms.ks"]


def cmp_subsample(out):
    sub = json.loads((C / "r7cmp" / "subsample.json").read_text())
    ids = set(sub["sample"])
    by = {o["point_id"]: o for o in out if o["src"] == "work-r7" and o["point_id"] in ids}
    plan = {json.loads(ln)["point_id"]: json.loads(ln) for ln in (C / "work-r7" / "plan.jsonl").open()}
    rs = {}
    for p in by:
        x = plan[p]
        rs[p] = (json.loads((pdir("work-r7", x) / "result.json").read_text()), pdir("work-r7", x))
    before = sorted(by)
    after = [p for p in before if not by[p]["blockers"]]

    def errs(p, guard_new):
        r = rs[p][0]
        e = dict(r["measured_rows"][0]["signed_err_pct"])
        bad = cov.kv_units_differ((r.get("source") or {}).get("server_log_id"), r["model_slug"] if guard_new else None)
        if bad:
            e.pop("kv_cache_pool_tokens", None)
        if guard_new and by[p]["pool_block_err_pct"] is not None:
            e["kv_cache_pool_blocks"] = by[p]["pool_block_err_pct"]
        return e

    def mape(pids, m, guard_new):
        v = [abs(errs(p, guard_new)[m]) for p in pids if errs(p, guard_new).get(m) is not None]
        return (S.fmean(v), len(v)) if v else (None, 0)

    print(f"## r7cmp HELD subsample (pre-registered, seed {sub['seed']}): r7-scored {len(before)}; "
          f"comparable after the parity rules {len(after)}")
    print("| metric | MAPE before (n) | MAPE after (n) |\n|---|---|---|")
    res = {"before_n": len(before), "after_n": len(after), "mape": {}, "dist": {}}
    for m in METRICS + ["kv_cache_pool_blocks"]:
        b, a = mape(before, m, False), mape(after, m, True)
        res["mape"][m] = {"before": b, "after": a}
        f = lambda t: "--" if t[0] is None else f"{t[0]:.1f} ({t[1]})"
        print(f"| {m} | {f(b)} | {f(a)} |")
    pts = {p: dist_agg.values(DIST.of_result(rs[p][0], rs[p][1])) for p in before}
    print("| distance (mean over points) | before (n) | after (n) |\n|---|---|---|")
    for k in DKEYS:
        b = [pts[p][k] for p in before if pts[p].get(k) is not None]
        a = [pts[p][k] for p in after if pts[p].get(k) is not None]
        res["dist"][k] = {"before": [S.fmean(b) if b else None, len(b)], "after": [S.fmean(a) if a else None, len(a)]}
        g = lambda v: "--" if not v else f"{S.fmean(v):.3f} ({len(v)})"
        print(f"| {k} | {g(b)} | {g(a)} |")
    (C / "work-r7" / "parity_r7cmp.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
