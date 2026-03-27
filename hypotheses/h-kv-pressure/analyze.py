#!/usr/bin/env python3
"""Analyze Iteration 6: KV pressure experiment."""
import json, os, sys

RD = sys.argv[1] if len(sys.argv) > 1 else "hypotheses/h-kv-pressure/results"
SEEDS = [42, 123, 7777]
KV_LEVELS = [132139, 5000, 2000, 1500]
POLICIES = ["rr", "static-default", "kv-heavy", "kv-pressure", "compound"]

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
    dropped = sum(i.get("dropped_unservable", 0) for i in instances)
    preempt = sum(i.get("preemption_count", 0) for i in instances)
    return {
        "p99": max(p99) if p99 else 0,
        "mean": sum(mn) / len(mn) if mn else 0,
        "comp": comp,
        "q": q,
        "dropped": dropped,
        "preempt": preempt,
    }


print("=" * 95)
print("Iteration 6: KV Pressure Experiment (4 instances, mixed-SLO)")
print("=" * 95)

for kv in KV_LEVELS:
    print(f"\n--- KV Blocks: {kv} {'(normal)' if kv > 100000 else '(pressure)' if kv < 3000 else ''} ---")
    fmt = "  {:<18} {:>10} {:>10} {:>6} {:>6} {:>8} {:>8}"
    print(fmt.format("Policy", "TTFT p99", "TTFT mean", "Done", "Queue", "Dropped", "Preempt"))
    print("  " + "-" * 68)
    rr_p99 = None
    for pol in POLICIES:
        vals = []
        for seed in SEEDS:
            r = parse(os.path.join(RD, f"{pol}_kv{kv}_seed{seed}.json"))
            if r:
                vals.append(r)
        if vals:
            ap99 = sum(v["p99"] for v in vals) / len(vals)
            amn = sum(v["mean"] for v in vals) / len(vals)
            acomp = sum(v["comp"] for v in vals) // len(vals)
            aq = sum(v["q"] for v in vals) // len(vals)
            adrop = sum(v["dropped"] for v in vals) // len(vals)
            apre = sum(v["preempt"] for v in vals) // len(vals)
            if pol == "rr":
                rr_p99 = ap99
            tag = ""
            if rr_p99 and rr_p99 > 0 and pol != "rr":
                imp = (rr_p99 - ap99) / rr_p99 * 100
                tag = f"  ({imp:+.1f}%)"
            print(f"  {pol:<18} {ap99:>9.2f}ms {amn:>9.2f}ms {acomp:>6d} {aq:>6d} {adrop:>8d} {apre:>8d}{tag}")
        else:
            print(f"  {pol:<18} {'N/A':>10} {'N/A':>10} {'N/A':>6} {'N/A':>6} {'N/A':>8} {'N/A':>8}")

# Cross-KV comparison for each policy
print(f"\n{'=' * 70}")
print("CROSS-KV COMPARISON: TTFT p99 by KV level")
print("=" * 70)
header = f"  {'Policy':<18}"
for kv in KV_LEVELS:
    header += f" {'KV=' + str(kv):>12}"
print(header)
print("  " + "-" * 66)
for pol in POLICIES:
    line = f"  {pol:<18}"
    for kv in KV_LEVELS:
        vals = []
        for seed in SEEDS:
            r = parse(os.path.join(RD, f"{pol}_kv{kv}_seed{seed}.json"))
            if r:
                vals.append(r["p99"])
        if vals:
            avg = sum(vals) / len(vals)
            line += f" {avg:>11.2f}ms"
        else:
            line += f" {'N/A':>12}"
    print(line)
