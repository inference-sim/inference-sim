"""KV-binding split of scored points and the r6 -> r7 latency change per group (KV_METRICS.md sec 5).

Binding (measured, inside the measured window where one is known):
  kv_p90 >= 0.9, OR (waiting > 0 in an interval whose KV usage >= 0.9), OR any preemption in the log.
Non-binding: a measured signal exists and none of the above holds. Points without a measured signal: "unknown".
r6 = results/heldout/points.jsonl, r7 = work-r7/points/*/result.json; a point counts only when both scored it.
kv_cache_pool_tokens is excluded where cov.kv_units_differ (DSV4-Pro logged size is not token capacity).
Usage: uv run --no-project python -I kvbind.py   -> kvm/kvbind.json + table on stdout
"""
import collections
import json
import statistics as S
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as PX  # noqa: E402
C = PX.ROOT
import cov  # noqa: E402

LAT = ["median_ttft_ms", "p90_ttft_ms", "median_tpot_ms", "p90_tpot_ms", "median_e2el_ms", "output_tput_tps",
       "mean_ttft_ms", "mean_tpot_ms", "p50_ttft_ms", "p50_tpot_ms"]


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo = int(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


def within(pts, win):
    return [v for t, v in pts if win is None or win[0] <= t <= win[1]]


def measured_signal(m):
    """(kv list, running, waiting, preemptions) over the measured window."""
    if m.get("series"):
        s = m["series"]
        w = s.get("window_s")
        kv, run, wait = within(s["kv"], w), within(s["running"], w), within(s["waiting"], w)
        kvt = {t: v for t, v in s["kv"]}
        wait_kv = [(v, kvt.get(t)) for t, v in s["waiting"] if (w is None or w[0] <= t <= w[1])]
        return kv, run, wait, wait_kv, 0
    lg = m.get("log") or {}
    if lg.get("stat_lines"):
        return lg["kv"], lg["running"], lg["waiting"], list(zip(lg["waiting"], lg["kv"])), lg.get("preemptions", 0)
    return None


def binding(m):
    sig = measured_signal(m)
    if not sig or not sig[0]:
        return "unknown", {}
    kv, run, wait, wait_kv, pre = sig
    p90 = q(kv, 0.9)
    kvlim = any(w and w > 0 and k is not None and k >= 0.9 for w, k in wait_kv)
    why = {"kv_p90": p90, "kv_limited_wait": kvlim, "preemptions": pre, "wait_max": max(wait) if wait else None}
    return ("binding" if (p90 >= 0.9 or kvlim or pre) else "nonbinding"), why


def errs(r):
    kvbad = cov.kv_units_differ((r.get("source") or {}).get("server_log_id"), r.get("model_slug"))
    out = collections.defaultdict(list)
    for mr in r.get("measured_rows", []):
        for k, v in (mr.get("signed_err_pct") or {}).items():
            if v is not None and not (k == "kv_cache_pool_tokens" and kvbad):
                out[k].append(v)
    return {k: S.fmean(v) for k, v in out.items()}


def main():
    meas = {json.loads(l)["point_id"]: json.loads(l) for l in (C / "kvm" / "measured.jsonl").open()}
    r6 = {}
    for l in (C / "results/heldout/points.jsonl").open():
        x = json.loads(l)
        if x.get("status") == "scored":
            r6[x["point_id"]] = x
    r7 = {}
    for f in (C / "work-r7" / "points").glob("*/*/result.json"):
        x = json.loads(f.read_text())
        if x.get("status") == "scored":
            r7[x["point_id"]] = x
    rows = []
    for p, x in r7.items():
        g, why = binding(meas.get(p, {}))
        e7 = errs(x)
        e6 = errs(r6[p]) if p in r6 else {}
        rows.append({"point_id": p, "set": x["set"], "model": x["model_slug"], "gpu": x["gpu"], "conc": x["conc"],
                     "scenario": x["scenario"], "group": g, "why": why, "e6": e6, "e7": e7, "in_r6": p in r6})
    (C / "kvm" / "kvbind.json").write_text(json.dumps(rows) + "\n")
    print("| group | n (r7) | n (both) | metric | MAPE r6 -> r7 | signed median r6 -> r7 |\n|---|---|---|---|---|---|")
    for g in ("binding", "nonbinding", "unknown"):
        rs = [r for r in rows if r["group"] == g]
        both = [r for r in rs if r["in_r6"]]
        for m in ("median_ttft_ms", "median_tpot_ms", "p90_ttft_ms", "median_e2el_ms", "output_tput_tps",
                  "kv_cache_pool_tokens"):
            a = [(r["e6"][m], r["e7"][m]) for r in both if m in r["e6"] and m in r["e7"]]
            if not a:
                continue
            print(f"| {g} | {len(rs)} | {len(a)} | {m} | {S.fmean(abs(u) for u, _ in a):.1f} -> "
                  f"{S.fmean(abs(v) for _, v in a):.1f} | {S.median(u for u, _ in a):+.1f} -> {S.median(v for _, v in a):+.1f} |")
    print("\nbinding points by model/gpu:", collections.Counter((r["model"], r["gpu"]) for r in rows if r["group"] == "binding"))
    bp = [r for r in rows if r["group"] == "binding" and r["in_r6"] and "kv_cache_pool_tokens" in r["e6"]]
    print("\nbinding points: r6 pool signed err (BLIS vs measured) and r6 -> r7 TTFT/TPOT signed err")
    for r in sorted(bp, key=lambda r: (r["model"], r["gpu"], r["conc"])):
        e6, e7 = r["e6"], r["e7"]
        f = lambda d, k: f"{d[k]:+.0f}" if k in d else "--"  # noqa: E731
        print(f"  {r['model']:22s} {r['gpu']:5s} c{r['conc']:<4d} pool {f(e6,'kv_cache_pool_tokens')} -> "
              f"{f(e7,'kv_cache_pool_tokens')} | ttft {f(e6,'median_ttft_ms')} -> {f(e7,'median_ttft_ms')} | "
              f"tpot {f(e6,'median_tpot_ms')} -> {f(e7,'median_tpot_ms')} | kv_p90 {r['why'].get('kv_p90'):.2f} "
              f"wait_max {r['why'].get('wait_max')}")


if __name__ == "__main__":
    main()
