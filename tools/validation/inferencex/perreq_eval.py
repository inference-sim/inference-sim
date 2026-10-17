"""HELD evaluation of the frozen per-request TTFT candidates (PER_REQUEST.md section 0 gates).

Run ONLY after results/perreq/frozen_candidates.json is written; it asserts the file's sha256 is the
one recorded in PER_REQUEST.md. For each HELD agentic point with a request timeline:
- per-request (joined, both-solo pairs): run-level W1/median + tail (q50/q90/q99) of TTFT before and
  after each candidate; median |log ratio| per factor bin (ISL, prior context, new tokens, position);
- aggregate: p50 / p90 TTFT over the point's BLIS profile population, with the shift applied to the
  requests that ran alone in BLIS (in-flight 0), against the measured row's aiperf p50 / p90.
  APPROXIMATION: a post-hoc shift does not move later arrivals of the same lane; a solo request has no
  co-runner, so only its own lane's next arrival is delayed (by <= the shift).
Usage: perreq_eval.py <points.jsonl | work dir> <out.json> <expected frozen sha256>
"""

from __future__ import annotations

import hashlib
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
import perreq_diag as D  # noqa: E402

FROZEN = PX.ROOT / "results" / "perreq" / "frozen_candidates.json"


def shift_for(cands, name, model):
    c = cands[name]
    if "c0_ms" in c:
        return c["c0_ms"], c["c1_ms_per_1k"]
    e = c.get(model) or c["_fallback"]
    return e["c0_ms"], e["c1_ms_per_1k"]


def loss(meas, blis):
    d = P.distances(meas, blis)
    return {"w1": d["w1_norm"], "tail": statistics.mean(abs(d[f"q{p}_err_pct"]) / 100 for p in (50, 90, 99)),
            "ks": d["ks"]}


def boot_ci(vals, seed=0):
    rng = random.Random(seed)
    bs = sorted(statistics.mean(rng.choices(vals, k=len(vals))) for _ in range(1000))
    return [bs[25], bs[974]]


def main():
    src, outp, want = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    got = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    assert got == want, f"frozen candidates changed: {got} != {want}"
    cands = json.loads(FROZEN.read_text())["candidates"]
    names = ["baseline"] + sorted(cands)
    runs, factor = {}, {}
    agg = []
    for res, root in D.load_results(src):
        if res.get("scenario") != "agentic_traces" or res.get("status") != "scored" or res.get("set") != "HELD":
            continue
        if not (D.RT / f"rt_{res['source']['db_rows'][0]}.body").exists():
            continue
        mm, bb, m = P.load_point(res, root, D.RT)
        j = P.join(mm, bb)
        solo = [(a, b) for a, b in j["pairs"] if a["inflight"] == 0 and b["inflight"] == 0]
        model = res["model_slug"]
        meas_row = (res.get("measured_rows") or [{}])[0].get("measured", {})
        pop = [r for r in bb["profile"] if not r["cancelled"]]
        ent = {"point": res["point_id"], "run": res["run_id"], "conc": res["conc"], "model": model, "gpu": res["gpu"],
               "verify": j["verify"], "solo_pairs": len(solo)}
        for nm in names:
            c0, c1 = (0, 0) if nm == "baseline" else shift_for(cands, nm, model)
            sh = [b["ttft_ms"] + c0 + c1 * b["prev_ctx"] / 1e3 for _, b in solo]
            if len(solo) >= 20:
                runs.setdefault(res["run_id"], {"model": model, "gpu": res["gpu"]}).setdefault(nm, []).append(
                    loss([a["ttft_ms"] for a, _ in solo], sh))
                for a, b in solo:
                    r = math.log((b["ttft_ms"] + c0 + c1 * b["prev_ctx"] / 1e3) / a["ttft_ms"])
                    f = D.feats(a, b)
                    for key in ("isl", "prev_ctx", "new_tok"):
                        lab = D.binlab(D.BINS[key], f[key])
                        factor.setdefault((nm, key, lab), {}).setdefault(res["run_id"], []).append(abs(r))
                    factor.setdefault((nm, "position", "first" if a.get("first_turn") else "later"), {}).setdefault(
                        res["run_id"], []).append(abs(r))
                    factor.setdefault((nm, "model", model), {}).setdefault(res["run_id"], []).append(abs(r))
            popv = [r["ttft_ms"] + ((c0 + c1 * r.get("prev_ctx", 0) / 1e3) if r.get("inflight") == 0 else 0) for r in pop]
            for q, key in ((0.5, "p50_ttft_ms"), (0.9, "p90_ttft_ms")):
                mv = (meas_row.get(key) or {}).get("value")
                if mv:
                    ent.setdefault("agg", {}).setdefault(nm, {})[key] = (P.quantile(popv, q) / mv - 1) * 100
        agg.append(ent)
    # run-level summaries
    summ = {"runs": len(runs), "points": len(agg)}
    for nm in names:
        per = [statistics.mean(x["w1"] + x["tail"] for x in v[nm]) for v in runs.values() if nm in v]
        summ[f"perreq_loss_{nm}"] = {"mean": statistics.mean(per) if per else None, "runs": len(per)}
        if nm != "baseline":
            dif = [statistics.mean(x["w1"] + x["tail"] for x in v[nm]) - statistics.mean(x["w1"] + x["tail"] for x in v["baseline"])
                   for v in runs.values() if nm in v]
            summ[f"perreq_loss_change_{nm}"] = {"mean": statistics.mean(dif), "ci95_runs": boot_ci(dif),
                                               "runs_worse": sum(d > 0 for d in dif), "runs": len(dif)}
        for key in ("p50_ttft_ms", "p90_ttft_ms"):
            vals = [abs(e["agg"][nm][key]) for e in agg if key in (e.get("agg") or {}).get(nm, {})]
            summ[f"agg_mape_{key}_{nm}"] = {"mean": statistics.mean(vals) if vals else None, "n": len(vals)}
    fac = {}
    for (nm, key, lab), byrun in factor.items():
        meds = [statistics.median(v) for v in byrun.values()]
        fac.setdefault(f"{key}:{lab}", {})[nm] = {"runs": len(meds), "median_abs_log": statistics.median(meds)}
    out = {"summary": summ, "per_factor": fac, "per_run": {k: {n: v[n] for n in names if n in v} for k, v in runs.items()},
           "points": agg}
    outp.write_text(json.dumps(out, indent=1))
    print(json.dumps(summ, indent=1))
    for k in sorted(fac):
        print(k, "  ".join(f"{n}: {v['median_abs_log']:.3f} (r{v['runs']})" for n, v in fac[k].items()))


if __name__ == "__main__":
    main()
