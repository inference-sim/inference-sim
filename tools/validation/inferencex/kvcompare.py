"""BLIS vs InferenceX server-state comparison, apples to apples (KV_METRICS.md sec 3-4).

Inputs: kvm/measured.jsonl (kvmeas.py) and, per point, a BLIS metrics.json carrying `server_series`
(inference-sim dsv41/kv-metrics, flag --server-series-interval-s). Matching rules, ASSERTED per point:
- agentic (series source): 1 s bins; both sides restricted to the aiperf profiling window (measured: request
  timeline profiling start..end relative to the series start; BLIS: first profile arrival .. + measured_window_s);
  KV usage = mean over engines (InferenceX averageAcrossEngines), running/waiting = sum over engines, counters
  summed over engines then divided (hit rate = sum hits / sum queries over the window).
- single_turn (log source): vLLM's 10 s stat lines report the gauges at the interval's last step, so BLIS is binned at
  10 s and its *_last point samples are used; only active intervals (any tokens or requests) on both sides.
Per series: time-mean, p50, p90 on each side, the level error (BLIS - measured, in points for fractions, % for counts),
W1 (in the series' own units) and KS. GROSS flags (diagnostic only, never fitted):
  kv usage |dmean| > 0.30 or ratio of means outside [0.5, 2] or W1 > 0.30; hit rate |d| > 0.2;
  pool ratio outside [0.7, 1.4]; waiting: measured p90 > 0 while BLIS p90 == 0 (or the reverse).
Usage: uv run --no-project python -I kvcompare.py [--blis-root DIR]  -> kvm/compare.jsonl, kvm/compare.md
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as S
from pathlib import Path
import sys as _s
_s.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PX  # noqa: E402

HERE = Path(__file__).resolve().parent
C = PX.ROOT


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo = int(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


def w1(a, b):
    """1-Wasserstein distance between two empirical distributions (quantile coupling on 200 points)."""
    if not a or not b:
        return None
    n = 200
    return S.fmean(abs(q(a, (i + 0.5) / n) - q(b, (i + 0.5) / n)) for i in range(n))


def ks(a, b):
    if not a or not b:
        return None
    a, b = sorted(a), sorted(b)
    import bisect
    pts = sorted(set(a) | set(b))
    return max(abs(bisect.bisect_right(a, x) / len(a) - bisect.bisect_right(b, x) / len(b)) for x in pts)


def summ(xs):
    return {"mean": S.fmean(xs), "p50": q(xs, 0.5), "p90": q(xs, 0.9), "n": len(xs)} if xs else None


def blis_series(m: dict, interval: float, scenario: str):
    ss = m.get("server_series")
    if not ss:
        return None
    assert abs(ss["interval_s"] - interval) < 1e-9, (ss["interval_s"], interval)
    engines = ss["engines"]
    nb = max(e["bins"] for e in engines)
    if scenario == "agentic_traces":
        rq = [r for r in m.get("requests", []) if r.get("phase") == "profile"]
        start = min(r["arrived_at"] for r in rq)  # seconds (metrics.json requests[].arrived_at)
        win = (start, start + m["measured_window_s"])
        kvk, rk, wk = "kv_cache_usage", "num_requests_running", "num_requests_waiting"
    else:
        win = None
        kvk, rk, wk = "kv_cache_usage_last", "num_requests_running_last", "num_requests_waiting_last"
    out = collections.defaultdict(list)
    tot = collections.Counter()
    for k in range(nb):
        t0 = k * interval
        if win and not (win[0] <= t0 and t0 + interval <= win[1] + interval):
            continue
        g = lambda key: [e[key][k] for e in engines if k < e["bins"]]  # noqa: E731
        run, wait, kv = sum(g(rk)), sum(g(wk)), g(kvk)
        pt, gt = sum(g("prompt_tokens")), sum(g("generation_tokens"))
        if win is None and run + wait + pt + gt == 0:
            continue
        out["kv"].append(S.fmean(kv))
        out["running"].append(run)
        out["waiting"].append(wait)
        out["prompt_tps"].append(pt / interval)
        out["gen_tps"].append(gt / interval)
        for c in ("prefix_cache_queries", "prefix_cache_hits", "prompt_tokens", "prompt_tokens_local_compute",
                  "prompt_tokens_local_cache_hit", "prompt_tokens_external_kv_transfer"):
            tot[c] += sum(g(c))
    out = dict(out)
    out["hit_rate_window"] = tot["prefix_cache_hits"] / tot["prefix_cache_queries"] if tot["prefix_cache_queries"] else None
    pt = tot["prompt_tokens"]
    out["src_share"] = {s: tot["prompt_tokens_" + s] / pt for s in ("local_compute", "local_cache_hit", "external_kv_transfer")} if pt else None
    out["engines"] = len(engines)
    return out


def meas_series(r: dict):
    if r.get("series"):
        s = r["series"]
        w = s["window_s"]
        inw = lambda pts: [v for t, v in pts if w[0] <= t <= w[1]]  # noqa: E731
        out = {k: inw(s[k]) for k in ("kv", "running", "waiting", "prompt_tps", "gen_tps")}
        hits = sum(inw(s["hits_tps"]))
        hr = {t: v for t, v in s["hit"]}
        queries = sum(h / hr[t] for t, h in s["hits_tps"] if w[0] <= t <= w[1] and hr.get(t))
        out["hit_rate_window"] = hits / queries if queries else None
        src = {k: sum(inw(v)) for k, v in s["by_source"].items()}
        tot = sum(src.values())
        out["src_share"] = {k: v / tot for k, v in src.items()} if tot else None
        out["engines"] = max(1, len(s["kv_by_engine"]))
        out["interval"], out["source"] = 1.0, "trace-server-metrics"
        return out
    lg = r.get("log") or {}
    if lg.get("stat_lines"):
        out = {k: lg[k] for k in ("kv", "running", "waiting", "prompt_tps", "gen_tps")}
        out["hit_rate_window"] = None  # the log's per-interval rate has no query weights
        out["src_share"] = None
        out["engines"] = lg["engines"]
        out["interval"], out["source"] = 10.0, "server-log stat lines"
        return out
    return None


def compare(pid, meas, blis, pool_ratio):
    assert meas["engines"] == blis["engines"] or meas["source"] == "server-log stat lines", (pid, meas["engines"], blis["engines"])
    res = {"point_id": pid, "source": meas["source"], "interval_s": meas["interval"], "pool_ratio": pool_ratio,
           "series": {}, "gross": []}
    for k in ("kv", "running", "waiting", "prompt_tps", "gen_tps"):
        a, b = meas.get(k) or [], blis.get(k) or []
        sa, sb = summ(a), summ(b)
        if not sa or not sb:
            continue
        res["series"][k] = {"meas": sa, "blis": sb, "w1": w1(a, b), "ks": ks(a, b)}
    kv = res["series"].get("kv")
    if kv:
        dm = kv["blis"]["mean"] - kv["meas"]["mean"]
        rat = kv["blis"]["mean"] / kv["meas"]["mean"] if kv["meas"]["mean"] > 0 else None
        if abs(dm) > 0.30 or (rat is not None and kv["meas"]["mean"] > 0.05 and not 0.5 <= rat <= 2) or kv["w1"] > 0.30:
            res["gross"].append(f"kv_usage mean {kv['meas']['mean']:.2f} -> BLIS {kv['blis']['mean']:.2f} (W1 {kv['w1']:.2f})")
    wq = res["series"].get("waiting")
    if wq and ((wq["meas"]["p90"] > 0) != (wq["blis"]["p90"] > 0)):
        res["gross"].append(f"waiting p90 {wq['meas']['p90']:.1f} vs BLIS {wq['blis']['p90']:.1f}")
    if meas.get("hit_rate_window") is not None and blis.get("hit_rate_window") is not None:
        d = blis["hit_rate_window"] - meas["hit_rate_window"]
        res["hit_rate"] = {"meas": meas["hit_rate_window"], "blis": blis["hit_rate_window"]}
        if abs(d) > 0.2:
            res["gross"].append(f"prefix hit rate {meas['hit_rate_window']:.2f} vs BLIS {blis['hit_rate_window']:.2f}")
    if meas.get("src_share") and blis.get("src_share"):
        res["src_share"] = {"meas": meas["src_share"], "blis": blis["src_share"]}
    if pool_ratio is not None and not 0.7 <= pool_ratio <= 1.4:
        res["gross"].append(f"pool ratio {pool_ratio:.2f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blis-root", default=str(C / "kvm" / "runs"))
    a = ap.parse_args()
    pools = {r["point_id"]: r["ratio"] for r in json.loads((C / "kvm" / "kvpool.json").read_text())}
    out = []
    for l in (C / "kvm" / "measured.jsonl").open():
        r = json.loads(l)
        mp = Path(a.blis_root) / r["point_id"] / "metrics.json"
        if not mp.exists():
            continue
        ms = meas_series(r)
        if not ms:
            continue
        bs = blis_series(json.loads(mp.read_text()), ms["interval"], r["scenario"])
        if not bs:
            continue
        c = compare(r["point_id"], ms, bs, pools.get(r["point_id"]))
        c.update({"set": r["set"], "scenario": r["scenario"], "run_id": r["run_id"]})
        out.append(c)
    with (C / "kvm" / "compare.jsonl").open("w") as f:
        for c in out:
            f.write(json.dumps(c) + "\n")
    print(len(out), "points compared;", sum(bool(c["gross"]) for c in out), "with a gross flag")


if __name__ == "__main__":
    main()
