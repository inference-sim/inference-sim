"""A/B of the block-size parity fix: the same points scored before (A) and after (B).

Per point: aggregate signed error for p50/p90 TTFT, p50 ITL, output throughput (the harness's own
measured_rows vs predicted); distribution distances (distances.json from dist_backfill: TTFT and ITL
W1/median and KS); and the per-request joined both-solo TTFT loss (W1/median + mean |q50,q90,q99|).
Control: points whose scenario is unchanged must have byte-identical metrics.json.

Usage: blockfix_ab.py <out.json> A_dir:B_dir ... [--control A_dir:B_dir ...]
Each pair names one point directory scored before and after.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as PX  # noqa: E402
import perreq as P  # noqa: E402

RT = PX.API / "request-timeline"
AGG = ("p50_ttft_ms", "p90_ttft_ms", "p50_itl_ms", "output_tput_tps")


def agg_err(res):
    meas = res["measured_rows"][0]["measured"]
    out = {}
    for k in AGG:
        m, p = (meas.get(k) or {}).get("value"), (res["predicted"].get(k) or {}).get("value")
        if m and p is not None:
            out[k] = (p / m - 1) * 100
    return out


def dist(pdir: Path):
    f = pdir / "distances.json"
    if not f.exists():
        return {}
    d = json.loads(f.read_text())
    flat = {}

    def walk(x, pre=""):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{pre}{k}.")
        elif isinstance(x, (int, float)):
            flat[pre[:-1]] = x
    walk(d)
    return {k: v for k, v in flat.items() if any(s in k for s in ("ttft", "itl", "tpot")) and
            any(s in k for s in ("w1", "ks"))}


def perreq_loss(pdir: Path, res):
    rt = json.loads((RT / f"rt_{res['source']['db_rows'][0]}.body").read_text())
    m = json.loads((pdir / "metrics.json").read_text())
    j = P.join(P.measured_rows(rt), P.blis_rows(m, res.get("b2_host_completion_ms", 0.777)))
    solo = [(a, b) for a, b in j["pairs"] if a["inflight"] == 0 and b["inflight"] == 0]
    if len(solo) < 10:
        return None
    d = P.distances([a["ttft_ms"] for a, _ in solo], [b["ttft_ms"] for _, b in solo])
    return {"n": len(solo), "w1": d["w1_norm"], "ks": d["ks"],
            "tail": statistics.mean(abs(d[f"q{p}_err_pct"]) / 100 for p in (50, 90, 99)),
            "loss": d["w1_norm"] + statistics.mean(abs(d[f"q{p}_err_pct"]) / 100 for p in (50, 90, 99))}


def side(pdir: Path):
    res = json.loads((pdir / "result.json").read_text())
    return {"agg": agg_err(res), "dist": dist(pdir), "perreq": perreq_loss(pdir, res),
            "block_size": res["resolved"]["engine"]["block_size"], "set": res["set"], "run": res["run_id"],
            "conc": res["conc"]}


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = Path(sys.argv[1])
    args = sys.argv[2:]
    ctl = args[args.index("--control") + 1:] if "--control" in args else []
    pairs = args[:args.index("--control")] if "--control" in args else args
    res = {"affected": [], "control": []}
    for pr in pairs:
        a, b = (Path(x) for x in pr.split(":"))
        res["affected"].append({"point": a.name, "A": side(a), "B": side(b)})
    for pr in ctl:
        a, b = (Path(x) for x in pr.split(":"))
        res["control"].append({"point": a.name, "identical": sha(a / "metrics.json") == sha(b / "metrics.json")})
    out.write_text(json.dumps(res, indent=1))
    for e in res["affected"]:
        A, B = e["A"], e["B"]
        print(f"{e['point']} {A['set']} c{A['conc']} block {A['block_size']}->{B['block_size']}")
        for k in AGG:
            if k in A["agg"]:
                print(f"   {k}: {A['agg'][k]:+.1f}% -> {B['agg'][k]:+.1f}%")
        if A["perreq"] and B["perreq"]:
            print(f"   per-request solo TTFT loss {A['perreq']['loss']:.3f} -> {B['perreq']['loss']:.3f} "
                  f"(W1 {A['perreq']['w1']:.3f}->{B['perreq']['w1']:.3f}, KS {A['perreq']['ks']:.3f}->{B['perreq']['ks']:.3f}, "
                  f"n {A['perreq']['n']}->{B['perreq']['n']})")
        for k in sorted(A["dist"]):
            if k in B["dist"]:
                print(f"   {k}: {A['dist'][k]:.3f} -> {B['dist'][k]:.3f}")
    if res["control"]:
        print(f"control identical: {sum(c['identical'] for c in res['control'])}/{len(res['control'])}")


if __name__ == "__main__":
    main()
