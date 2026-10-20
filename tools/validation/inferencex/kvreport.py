"""Summarise kvm/compare.jsonl (KV_METRICS.md sec 4): level errors, distances, gross clusters, binding subsets.
Usage: uv run --no-project python -I kvreport.py -> stdout (copied into kvm/kvreport.out)"""
import collections
import json
import statistics as S
from pathlib import Path
import sys as _s
_s.path.insert(0, str(Path(__file__).resolve().parent))
import paths as PX  # noqa: E402

HERE = Path(__file__).resolve().parent
C = PX.ROOT


def main():
    cmp_ = [json.loads(l) for l in (C / "kvm" / "compare.jsonl").open()]
    meta = {}
    for f in (C / "work-r7" / "points").glob("*/*/result.json"):
        x = json.loads(f.read_text())
        rs = x.get("resolved") or {}
        meta[x["point_id"]] = (x["model_slug"], x["gpu"], f"tp{rs.get('tp')}dp{rs.get('dp')}ep{int(bool(rs.get('enable_expert_parallel')))}",
                               "offload" if rs.get("offload") else "no-offload", str(rs.get("vllm_version"))[:12], x["conc"])
    bind = {r["point_id"]: r["group"] for r in json.loads((C / "kvm" / "kvbind.json").read_text())}
    par = {r["point_id"]: r["classes"] for r in json.loads((C / "kvm" / "parity.json").read_text())}
    print(f"points compared: {len(cmp_)} (agentic {sum(c['scenario'] == 'agentic_traces' for c in cmp_)}, "
          f"single-turn {sum(c['scenario'] != 'agentic_traces' for c in cmp_)}); gross: {sum(bool(c['gross']) for c in cmp_)}")
    for scen in ("agentic_traces", "single_turn"):
        for grp in ("all", "binding", "nonbinding"):
            cs = [c for c in cmp_ if c["scenario"] == scen and (grp == "all" or bind.get(c["point_id"]) == grp)]
            if not cs:
                continue
            print(f"\n### {scen} / {grp} (n {len(cs)})")
            print("| series | meas mean (med over pts) | BLIS mean | signed level err of time-mean (median) | |err| p50 / p90 over pts | p90 err (median) | W1 (median) | KS (median) |")
            print("|---|---|---|---|---|---|---|---|")
            for k in ("kv", "running", "waiting", "prompt_tps", "gen_tps"):
                v = [c["series"][k] for c in cs if k in c["series"]]
                if not v:
                    continue
                frac = k == "kv"
                def err(s, q):
                    a, b = s["meas"][q], s["blis"][q]
                    return (b - a) * (100 if frac else 1) if frac else ((b - a) / a * 100 if a else None)
                e = [err(s, "mean") for s in v if err(s, "mean") is not None]
                e90 = [err(s, "p90") for s in v if err(s, "p90") is not None]
                ae = sorted(abs(x) for x in e)
                unit = "pt" if frac else "%"
                print(f"| {k} | {S.median(s['meas']['mean'] for s in v):.3g} | {S.median(s['blis']['mean'] for s in v):.3g} | "
                      f"{S.median(e):+.1f}{unit} | {ae[len(ae) // 2]:.1f} / {ae[int(len(ae) * 0.9)]:.1f}{unit} | "
                      f"{S.median(e90) if e90 else float('nan'):+.1f}{unit} | {S.median(s['w1'] for s in v):.3g} | {S.median(s['ks'] for s in v):.2f} |")
            hr = [c["hit_rate"] for c in cs if c.get("hit_rate")]
            if hr:
                print(f"| prefix hit (window) | {S.median(h['meas'] for h in hr):.3f} | {S.median(h['blis'] for h in hr):.3f} | "
                      f"{S.median((h['blis'] - h['meas']) * 100 for h in hr):+.1f}pt | | | | |")
            ss = [c["src_share"] for c in cs if c.get("src_share")]
            if ss:
                for s in ("local_compute", "local_cache_hit", "external_kv_transfer"):
                    a = [x["meas"].get(s, 0) for x in ss]
                    b = [x["blis"].get(s, 0) for x in ss]
                    print(f"| prompt share {s} | {S.median(a):.3f} | {S.median(b):.3f} | {S.median(y - x for x, y in zip(a, b)) * 100:+.1f}pt | | | | |")
    cl = collections.defaultdict(list)
    for c in cmp_:
        for g in c["gross"]:
            kind = g.split(" ")[0]
            cl[(kind,) + meta.get(c["point_id"], ("?",) * 6)[:5]].append((c["point_id"], g, par.get(c["point_id"])))
    print("\n### gross-deviation clusters (by flag, model, GPU, layout, offload, version)")
    print("| flag | model | gpu | layout | offload | vLLM | n | parity classes | example |\n|---|---|---|---|---|---|---|---|---|")
    for k, v in sorted(cl.items(), key=lambda kv: -len(kv[1]))[:30]:
        pc = collections.Counter(",".join(p or ["-"]) for _, _, p in v)
        print(f"| {' | '.join(k)} | {len(v)} | {dict(pc)} | {v[0][1]} |")


if __name__ == "__main__":
    main()
