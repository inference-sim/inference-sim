#!/usr/bin/env python3
"""Analyze iter16: Precise vs approximate KV routing under varying KV pressure and staleness."""
import json, os, sys

RD = sys.argv[1] if len(sys.argv) > 1 else "hypotheses/h-precise-kv/results"
SEEDS = [42, 123, 7777]
KV_LEVELS = [132139, 5000, 2000]
INTERVALS = [0, 10, 100]
POLICIES = ["rr", "pa3qd2kv2", "pa3qd2", "pa4qd3"]

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
    comp = sum(i.get("completed_requests", 0) for i in instances)
    pre = sum(i.get("preemption_count", 0) for i in instances)
    return {
        "p99": max(p99) if p99 else 0,
        "comp": comp,
        "preempt": pre,
    }

def avg(vals, key):
    vs = [v[key] for v in vals]
    return sum(vs) / len(vs) if vs else 0

print("=" * 90)
print("Iteration 16: Precise vs Approximate KV Routing")
print("=" * 90)

# Table: KV level × Staleness for each policy
for pol in POLICIES:
    print(f"\n--- Policy: {pol} ---")
    header = f"  {'KV Blocks':<12}"
    for interval in INTERVALS:
        header += f" {'int=' + str(interval) + 'ms':>14}"
    print(header)
    print("  " + "-" * 54)

    for kv in KV_LEVELS:
        line = f"  {kv:<12}"
        for interval in INTERVALS:
            vals = []
            for seed in SEEDS:
                r = parse(os.path.join(RD, f"{pol}_kv{kv}_int{interval}_seed{seed}.json"))
                if r:
                    vals.append(r)
            if vals:
                ap = avg(vals, "p99")
                line += f" {ap:>13.2f}ms"
            else:
                line += f" {'N/A':>14}"
        print(line)

# Cross-policy comparison at each KV level (interval=0 = fresh)
print(f"\n{'=' * 70}")
print("CROSS-POLICY COMPARISON (interval=0, fresh snapshots)")
print("=" * 70)
for kv in KV_LEVELS:
    print(f"\n--- KV={kv} ---")
    rr_p99 = None
    for pol in POLICIES:
        vals = []
        for seed in SEEDS:
            r = parse(os.path.join(RD, f"{pol}_kv{kv}_int0_seed{seed}.json"))
            if r:
                vals.append(r)
        if vals:
            ap = avg(vals, "p99")
            ac = avg(vals, "comp")
            apre = avg(vals, "preempt")
            if pol == "rr":
                rr_p99 = ap
            tag = ""
            if rr_p99 and rr_p99 > 0 and pol != "rr":
                tag = f"  ({(rr_p99 - ap) / rr_p99 * 100:+.1f}% vs RR)"
            print(f"  {pol:<16} P99: {ap:>9.2f}ms  comp: {ac:>5.0f}  preempt: {apre:>5.0f}{tag}")
        else:
            print(f"  {pol:<16} N/A")

# Staleness degradation: how much does interval=100 degrade vs interval=0?
print(f"\n{'=' * 70}")
print("STALENESS DEGRADATION: interval=100ms vs interval=0ms (% TTFT increase)")
print("=" * 70)
header = f"  {'Policy':<16}"
for kv in KV_LEVELS:
    header += f" {'KV=' + str(kv):>14}"
print(header)
print("  " + "-" * 58)
for pol in POLICIES:
    line = f"  {pol:<16}"
    for kv in KV_LEVELS:
        vals0 = [parse(os.path.join(RD, f"{pol}_kv{kv}_int0_seed{seed}.json")) for seed in SEEDS]
        vals100 = [parse(os.path.join(RD, f"{pol}_kv{kv}_int100_seed{seed}.json")) for seed in SEEDS]
        vals0 = [v for v in vals0 if v]
        vals100 = [v for v in vals100 if v]
        if vals0 and vals100:
            p0 = avg(vals0, "p99")
            p100 = avg(vals100, "p99")
            if p0 > 0:
                deg = (p100 - p0) / p0 * 100
                line += f" {deg:>+13.1f}%"
            else:
                line += f" {'N/A':>14}"
        else:
            line += f" {'N/A':>14}"
    print(line)
