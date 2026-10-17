"""Parity classification of gross deviations among r7 scored points (KV_METRICS.md sec 6), and the latency MAPE
before / after excluding the parity classes.

Gross: KV pool ratio (BLIS / measured) outside 0.7-1.3 (pool off > 30%), or offered load not reached
(Little's-law implied measured concurrency = output_tput * E2E / OSL < 0.5 x nominal), or (from kvm/compare.jsonl
when present) the KV-usage W1 > 0.30.
Classes: (a) setup parity, (b) workload parity, (c) harness definition/units, (d) model or catalog.
Usage: uv run --no-project python -I parity.py -> kvm/parity.json + tables"""
import collections
import json
import statistics as S
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths as PX  # noqa: E402
C = PX.ROOT
import cov  # noqa: E402

LAT = ["median_ttft_ms", "median_tpot_ms", "p90_ttft_ms", "median_e2el_ms"]


def ckpt_by_log(ids):
    q = ("select id, (regexp_match(server_log, 'Initializing a V1 LLM engine \\([^)]*\\) with config: "
         "model=''([^'']+)'''))[1] from server_logs where id in (%s)" % ",".join(map(str, ids)))
    out = subprocess.run(["psql", "-d", PX.PGDATABASE, "-At", "-c", q], capture_output=True, text=True, check=True).stdout
    return {int(l.split("|")[0]): l.split("|")[1] for l in out.splitlines() if "|" in l}


def classify(x, pool, little, kvw1):
    model = x["model_slug"]
    mism = {m["item"]: m for m in x.get("mismatches", [])}
    spec = (x.get("resolved") or {}).get("spec_method")
    why = []
    if pool is not None and not 0.7 <= pool <= 1.3:
        if model == "deepseek-v4-pro":
            why.append(("c", "DSV4-Pro: logged 'GPU KV cache size' counts compressed-cache tokens, not "
                             "BLIS kv_capacity_tokens (units); compare blocks or exclude"))
        elif "served checkpoint" in mism:
            why.append(("a", f"served checkpoint {mism['served checkpoint']['measured']} priced as catalog "
                             f"{mism['served checkpoint']['blis']} (weights bytes differ, so does the pool)"))
        elif spec == "dspark":
            why.append(("d", "DSpark drafter weights/KV not charged against the pool (pool over-predicted)"))
        else:
            why.append(("d", "pool sizing term (activation / non-torch reserve on a small KV remainder)"))
    if little is not None and little < 0.5:
        why.append(("b", f"offered load not reached: implied concurrency {little:.2f} x nominal (client-limited)"))
    if kvw1 is not None and kvw1 > 0.30:
        why.append(("d?", f"KV usage W1 {kvw1:.2f}"))
    return why


def main():
    little = {r["point_id"]: r["ratio"] for r in json.loads((C / "kvm" / "little.json").read_text())}
    pools = {r["point_id"]: r["ratio"] for r in json.loads((C / "kvm" / "kvpool.json").read_text())}
    kvw1 = {}
    cp = C / "kvm" / "compare.jsonl"
    if cp.exists():
        for l in cp.open():
            c = json.loads(l)
            if "kv" in c["series"]:
                kvw1[c["point_id"]] = c["series"]["kv"]["w1"]
    rows = []
    for f in (C / "work-r7" / "points").glob("*/*/result.json"):
        x = json.loads(f.read_text())
        if x.get("status") != "scored":
            continue
        p = x["point_id"]
        why = classify(x, pools.get(p), little.get(p), kvw1.get(p))
        e = x["measured_rows"][0]["signed_err_pct"]
        rows.append({"point_id": p, "set": x["set"], "model": x["model_slug"], "gpu": x["gpu"], "conc": x["conc"],
                     "tp": (x.get("resolved") or {}).get("tp"), "scenario": x["scenario"], "why": why,
                     "classes": sorted({c for c, _ in why}), "err": {k: e.get(k) for k in LAT}})
    (C / "kvm" / "parity.json").write_text(json.dumps(rows) + "\n")
    cnt = collections.Counter()
    ex = collections.defaultdict(list)
    for r in rows:
        for c, w in r["why"]:
            key = (c, w.split(":")[0] if c != "a" else "served checkpoint")
            cnt[key] += 1
            if len(ex[key]) < 3:
                ex[key].append(f"{r['model']} {r['gpu']} tp{r['tp']} c{r['conc']} ({r['set']})")
    print("| class | cause | points | examples |\n|---|---|---|---|")
    for k, v in sorted(cnt.items()):
        print(f"| {k[0]} | {k[1][:90]} | {v} | {'; '.join(ex[k])} |")
    print("\nlatency MAPE (|signed err| mean, %), all scored points vs after excluding parity classes (a)(b)(c):")
    keep = [r for r in rows if not set(r["classes"]) & {"a", "b", "c"}]
    print("| set | metric | all (n) | parity-clean (n) | excluded (n) |\n|---|---|---|---|---|")
    for st in ("HELD", "FIT"):
        for m in LAT:
            def mape(rs):
                v = [abs(r["err"][m]) for r in rs if r["set"] == st and r["err"].get(m) is not None]
                return f"{S.fmean(v):.1f} ({len(v)})" if v else "--"
            exc = [r for r in rows if set(r["classes"]) & {"a", "b", "c"}]
            if mape(rows) != "--":
                print(f"| {st} | {m} | {mape(rows)} | {mape(keep)} | {mape(exc)} |")
    print("\nby model (HELD median_tpot_ms MAPE all -> parity-clean):")
    for mo in sorted({r["model"] for r in rows}):
        a = [abs(r["err"]["median_tpot_ms"]) for r in rows if r["model"] == mo and r["err"].get("median_tpot_ms") is not None]
        b = [abs(r["err"]["median_tpot_ms"]) for r in keep if r["model"] == mo and r["err"].get("median_tpot_ms") is not None]
        if a:
            print(f"  {mo}: {S.fmean(a):.1f} (n {len(a)}) -> {S.fmean(b) if b else float('nan'):.1f} (n {len(b)})")


if __name__ == "__main__":
    main()
