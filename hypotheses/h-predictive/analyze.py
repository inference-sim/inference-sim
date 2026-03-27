#!/usr/bin/env python3
"""Analyze iter14: Predictive vs SLOGated vs baseline. Uses GOODPUT as primary metric."""
import json, os, sys

RD = sys.argv[1] if len(sys.argv) > 1 else "hypotheses/h-predictive/results"
SEEDS = [42, 123, 7777]
POLS = ["rr", "baseline", "slo-gated", "predictive"]
# SLO budgets (ms) for goodput calculation
SLO_BUDGETS = {"critical": 200, "standard": 500, "sheddable": 300, "": 500}
TOTAL_REQUESTS = 2000  # per seed

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
    return {
        "p99": max(p99) if p99 else 0,
        "mean": sum(mn) / len(mn) if mn else 0,
        "comp": comp,
    }

print("=" * 90)
print("Iteration 14: Predictive TTFT-Budget Admission (rate=2000, 8 inst, 2000 req)")
print("=" * 90)
print(f"\n{'Policy':<16} {'TTFT P99':>10} {'TTFT Mean':>10} {'Completed':>10} {'Comp Rate':>10} {'Goodput*':>10}")
print("-" * 66)
print("  * Goodput = completed / total_arriving (higher = better)")

for pol in POLS:
    vals = []
    for seed in SEEDS:
        r = parse(os.path.join(RD, f"{pol}_seed{seed}.json"))
        if r:
            vals.append(r)
    if vals:
        ap99 = sum(v["p99"] for v in vals) / len(vals)
        amean = sum(v["mean"] for v in vals) / len(vals)
        acomp = sum(v["comp"] for v in vals) / len(vals)
        comp_rate = acomp / TOTAL_REQUESTS
        goodput = comp_rate  # simplified: all completed = goodput (need per-request SLO check for full accuracy)
        print(f"  {pol:<14} {ap99:>9.2f}ms {amean:>9.2f}ms {acomp:>9.0f} {comp_rate:>9.1%} {goodput:>9.1%}")
    else:
        print(f"  {pol:<14} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

# Per-seed breakdown
print(f"\n{'='*70}")
print("Per-seed breakdown (TTFT P99 / Completed)")
print(f"{'='*70}")
header = f"  {'Policy':<14}"
for seed in SEEDS:
    header += f"  {'Seed ' + str(seed):>18}"
print(header)
print("  " + "-" * 58)
for pol in POLS:
    line = f"  {pol:<14}"
    for seed in SEEDS:
        r = parse(os.path.join(RD, f"{pol}_seed{seed}.json"))
        if r:
            line += f"  {r['p99']:>7.1f}ms/{r['comp']:>5d}"
        else:
            line += f"  {'N/A':>18}"
    print(line)
