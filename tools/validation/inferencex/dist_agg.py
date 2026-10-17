"""Aggregate distribution distances (distance.py) over points: post-stratified mean of each distance with a
run-clustered bootstrap CI within strata, overall and per factor level (n >= 10). Quantile errors enter as
absolute values (mean |q err|, like MAPE). Shared by aggregate.py, heldout_report.py, ab.py and r7cmp/compare.py.
"""

from __future__ import annotations

import collections
import random
import statistics

import distance as DIST

B = 400
SEED = 20261017
MIN_N = 10


def values(dist: dict | None) -> dict:
    """{'ttft_ms.w1_norm': v, 'ttft_ms.q90_abs_err_pct': |v|, ...} for one point."""
    out = {}
    for k, v in DIST.flat(dist).items():
        if k.endswith("_err_pct"):
            out[k.replace("_err_pct", "_abs_err_pct")] = abs(v)
        else:
            out[k] = v
    return out


def estimate(rows: list[tuple[str, str, float]], planned: dict | None = None, b: int = B) -> dict | None:
    """rows: (stratum, run_id, value). planned: stratum -> weight (planned points); None weights strata by n."""
    if not rows:
        return None
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for s, r, v in rows:
        by[s][r].append(v)
    wt = {s: (planned or {}).get(s) or sum(len(x) for x in runs.values()) for s, runs in by.items()}

    def ps(groups):
        num = den = 0.0
        for s, runs in groups.items():
            xs = [v for vs in runs.values() for v in vs]
            num += wt[s] * statistics.fmean(xs)
            den += wt[s]
        return num / den

    rng = random.Random(SEED)
    boots = []
    for _ in range(b):
        g = {}
        for s, runs in by.items():
            keys = list(runs)
            g[s] = {i: runs[rng.choice(keys)] for i in range(len(keys))}
        boots.append(ps(g))
    boots.sort()
    return {"n": len(rows), "runs": len({r for _, r, _ in rows}), "strata": len(by),
            "strata_single_run": sum(1 for runs in by.values() if len(runs) == 1), "mean": ps(by),
            "ci95": [boots[int(0.025 * b)], boots[int(0.975 * b) - 1]],
            "median": statistics.median(v for _, _, v in rows)}


def summarize(pts: list[dict], planned: dict | None = None, factors: list[str] | None = None,
              b: int = B) -> dict:
    """pts: {'stratum', 'run', 'dist' (distance.py dict), 'f' (factor -> level)}."""
    def table(sel):
        acc = collections.defaultdict(list)
        for p in sel:
            for k, v in (p["vals"] if "vals" in p else values(p["dist"])).items():
                if v is not None:
                    acc[k].append((p["stratum"], p["run"], v))
        return {k: estimate(v, planned, b) for k, v in sorted(acc.items())}

    mm = collections.Counter(m["item"] for p in pts for m in ((p["dist"] or {}).get("mismatches") or []))
    out = {"points": len(pts), "points_with_distances": sum(bool(p["vals"] if "vals" in p else values(p["dist"])) for p in pts),
           "mismatches": dict(mm), "bootstrap": {"B": b, "seed": SEED, "cluster": "run_id within stratum"},
           "overall": table(pts), "by_factor": {}}
    for fk in factors or []:
        lv = collections.defaultdict(list)
        for p in pts:
            lv[str(p["f"].get(fk))].append(p)
        out["by_factor"][fk] = {k: (table(v) if len(v) >= MIN_N else {"n": len(v), "note": "n < 10: count only"})
                                for k, v in sorted(lv.items())}
    return out
