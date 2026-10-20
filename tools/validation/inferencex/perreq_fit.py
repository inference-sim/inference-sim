"""Distribution-distance fit of a per-request TTFT term on FIT solo pairs (PER_REQUEST.md section 3).

Model (post-hoc, valid only where the request ran alone on both sides, so an added admission cost
cannot change any other request): TTFT_blis'(c0, c1) = TTFT_blis + c0 + c1 * prior_ctx/1k (ms).
c0 is a per-request constant, c1 the context-proportional admission cost (ms per 1k cached tokens).
Bounds: 0 <= c0 <= 300 ms, 0 <= c1 <= 3 ms/1k (grid 10 ms x 0.1 ms/1k).
Loss per run: W1(meas, blis') / median(meas) + mean |q_p err| (p50/p90/p99, fraction); runs weighted
equally within a config group (model, gpu, layout), groups equally. Grid search (bounded, two
parameters), run-bootstrap CI, a parameter-recovery check (synthetic truth = BLIS + known (c0, c1)),
and the bootstrap correlation of (c0, c1) as the identifiability check.

FIT points only (refuses others). Usage: perreq_fit.py <points.jsonl | work dir> <out.json>
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import perreq as P  # noqa: E402
import perreq_diag as D  # noqa: E402

C0 = list(range(0, 310, 10))
C1 = [round(0.1 * i, 1) for i in range(31)]


def run_loss(meas, blis):
    d = P.distances(meas, blis)
    return d["w1_norm"] + statistics.mean(abs(d[f"q{p}_err_pct"]) / 100 for p in (50, 90, 99))


def collect(src):
    runs = {}
    for res, root in D.load_results(Path(src)):
        if res.get("scenario") != "agentic_traces" or res.get("status") != "scored":
            continue
        if res.get("set") != "FIT":
            raise SystemExit(f"refusing non-FIT point {res['point_id']}")
        if not (D.RT / f"rt_{res['source']['db_rows'][0]}.body").exists():
            continue
        p = D.point_diag(res, root)
        rows = [(r["ttft_m"], r["ttft_b"], r["prev_ctx"] / 1e3) for r in p["rows"] if r["inflight"] == 0]
        if len(rows) >= 20:
            r = res["resolved"]
            g = (res["model_slug"], res["gpu"], f"tp{r.get('tp')}dp{r.get('dp')}")
            runs.setdefault(res["run_id"], {"group": g, "rows": []})["rows"] += rows
    return runs


def weights(runs, ids):
    gcount = {}
    for i in ids:
        gcount[runs[i]["group"]] = gcount.get(runs[i]["group"], 0) + 1
    ng = len(gcount)
    return {i: 1.0 / (gcount[runs[i]["group"]] * ng) for i in ids}


def total(runs, ids, c0, c1, truth=None):
    w = weights(runs, ids)
    s = 0.0
    for i in ids:
        rows = runs[i]["rows"]
        meas = [m for m, _, _ in rows] if truth is None else [b + truth[0] + truth[1] * x for _, b, x in rows]
        s += w[i] * run_loss(meas, [b + c0 + c1 * x for _, b, x in rows])
    return s


def argmin(runs, ids, truth=None):
    return min(((total(runs, ids, a, b, truth), a, b) for a in C0 for b in C1))


def main():
    runs = collect(sys.argv[1])
    ids = sorted(runs)
    base = total(runs, ids, 0, 0)
    best = argmin(runs, ids)
    only_c1 = min((total(runs, ids, 0, b), b) for b in C1)
    only_c0 = min((total(runs, ids, a, 0), a) for a in C0)
    rng = random.Random(20261017)
    boots = []
    for _ in range(40):
        bi = [rng.choice(ids) for _ in ids]
        # duplicate run ids are fine for the loss: weight follows the resample
        boots.append(argmin(runs, bi)[1:])
    c0s, c1s = [b[0] for b in boots], [b[1] for b in boots]
    corr = statistics.correlation(c0s, c1s) if len(set(c0s)) > 1 and len(set(c1s)) > 1 else None
    recov = {}
    for t in ((50, 0.5), (100, 1.0), (0, 2.0)):
        recov[str(t)] = argmin(runs, ids, truth=t)[1:]
    per_group = {}
    for g in sorted({runs[i]["group"] for i in ids}):
        gi = [i for i in ids if runs[i]["group"] == g]
        per_group["|".join(g)] = {"runs": len(gi), "pairs": sum(len(runs[i]["rows"]) for i in gi),
                                  "loss0": total(runs, gi, 0, 0), "argmin": argmin(runs, gi)}
    out = {"runs": len(ids), "pairs": sum(len(runs[i]["rows"]) for i in ids), "loss_baseline": base,
           "argmin": {"loss": best[0], "c0_ms": best[1], "c1_ms_per_1k": best[2]},
           "c1_only": {"loss": only_c1[0], "c1": only_c1[1]}, "c0_only": {"loss": only_c0[0], "c0": only_c0[1]},
           "bootstrap_runs": {"c0_p5_p95": [sorted(c0s)[2], sorted(c0s)[37]], "c1_p5_p95": [sorted(c1s)[2], sorted(c1s)[37]],
                              "corr_c0_c1": corr},
           "parameter_recovery": recov, "per_group": per_group}
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
