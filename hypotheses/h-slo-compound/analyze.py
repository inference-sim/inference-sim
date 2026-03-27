#!/usr/bin/env python3
"""Analyze Iteration 5: Mixed-SLO compound experiment."""
import json, os, sys

RD = sys.argv[1] if len(sys.argv) > 1 else "hypotheses/h-slo-compound/results"
SEEDS = [42, 123, 7777]
RATES = [200, 300, 400]
POLICIES = [
    "rr-baseline", "static-default", "adaptive-ortho",
    "static-slopri", "adaptive-slopri", "static-sjf",
]

def parse(fp):
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        c = f.read().strip()
    if not c:
        return None
    instances = []
    in_json = False
    cur = []
    for line in c.split("\n"):
        if line.strip() == "{":
            in_json = True
            cur = [line]
        elif in_json:
            cur.append(line)
            if line.strip() == "}":
                in_json = False
                try:
                    instances.append(json.loads("\n".join(cur)))
                except:
                    pass
    if not instances:
        return None
    p99 = [i["ttft_p99_ms"] for i in instances if i.get("ttft_p99_ms", 0) > 0]
    mn = [i["ttft_mean_ms"] for i in instances if i.get("ttft_mean_ms", 0) > 0]
    comp = sum(i.get("completed_requests", 0) for i in instances)
    q = sum(i.get("still_queued", 0) for i in instances)
    return {
        "p99": max(p99) if p99 else 0,
        "mean": sum(mn) / len(mn) if mn else 0,
        "comp": comp,
        "queued": q,
    }


print("=" * 90)
print("Iteration 5: Mixed-SLO Compound Experiment")
print("=" * 90)

for rate in RATES:
    print(f"\n--- RATE {rate} ---")
    fmt = "  {:<22} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("Policy", "TTFT p99", "TTFT mean", "Completed", "Queued"))
    print("  " + "-" * 62)
    rr_p99 = None
    for pol in POLICIES:
        vals = []
        for seed in SEEDS:
            r = parse(os.path.join(RD, f"{pol}_rate{rate}_seed{seed}.json"))
            if r:
                vals.append(r)
        if vals:
            ap99 = sum(v["p99"] for v in vals) / len(vals)
            amn = sum(v["mean"] for v in vals) / len(vals)
            acomp = sum(v["comp"] for v in vals) // len(vals)
            aq = sum(v["queued"] for v in vals) // len(vals)
            if pol == "rr-baseline":
                rr_p99 = ap99
            tag = ""
            if rr_p99 and rr_p99 > 0 and pol != "rr-baseline":
                imp = (rr_p99 - ap99) / rr_p99 * 100
                tag = f"  ({imp:+.1f}% vs RR)"
            print(f"  {pol:<22} {ap99:>9.2f}ms {amn:>9.2f}ms {acomp:>10d} {aq:>10d}{tag}")
        else:
            print(f"  {pol:<22} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

# Best strategy per rate
print(f"\n{'=' * 60}")
print("BEST STRATEGY PER RATE (TTFT p99)")
print("=" * 60)
for rate in RATES:
    best_pol = None
    best_p99 = float("inf")
    for pol in POLICIES:
        vals = []
        for seed in SEEDS:
            r = parse(os.path.join(RD, f"{pol}_rate{rate}_seed{seed}.json"))
            if r:
                vals.append(r["p99"])
        if vals:
            avg = sum(vals) / len(vals)
            if avg < best_p99:
                best_p99 = avg
                best_pol = pol
    print(f"  Rate {rate}: {best_pol} ({best_p99:.2f}ms)")
