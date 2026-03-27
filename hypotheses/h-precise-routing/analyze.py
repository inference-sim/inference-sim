#!/usr/bin/env python3
"""Analyze Iter20: Precise vs Approximate KV Routing."""
import json, os, sys

RD = sys.argv[1] if len(sys.argv) > 1 else "hypotheses/h-precise-routing/results"
SEEDS = [42, 123, 7777]

def parse(fp):
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        c = f.read().strip()
    if not c or c.startswith("ERROR") or c.startswith("TIMEOUT"):
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
    comp = max(i.get("completed_requests", 0) for i in instances)
    pre = sum(i.get("preemption_count", 0) for i in instances)
    cache_hr = [i.get("cache_hit_rate", 0) for i in instances if i.get("cache_hit_rate", 0) > 0]
    return {
        "p99": max(p99) if p99 else 0,
        "mean": sum(mn) / len(mn) if mn else 0,
        "comp": comp,
        "preempt": pre,
        "cache_hr": sum(cache_hr) / len(cache_hr) if cache_hr else 0,
    }

def avg(vals, key):
    vs = [v[key] for v in vals]
    return sum(vs) / len(vs) if vs else 0

def collect(tag):
    vals = []
    for seed in SEEDS:
        r = parse(os.path.join(RD, f"{tag}_seed{seed}.json"))
        if r:
            vals.append(r)
    return vals

def print_row(pol, vals, rr_p99=None, approx_p99=None):
    if vals:
        ap = avg(vals, "p99")
        am = avg(vals, "mean")
        ac = avg(vals, "comp")
        apre = avg(vals, "preempt")
        achr = avg(vals, "cache_hr")
        rr_tag = ""
        if rr_p99 and rr_p99 > 0 and "rr" not in pol:
            rr_tag = f"  ({(rr_p99 - ap) / rr_p99 * 100:+.1f}% vs RR)"
        prec_tag = ""
        if approx_p99 and approx_p99 > 0 and "precise" in pol:
            prec_tag = f"  ({(approx_p99 - ap) / approx_p99 * 100:+.1f}% vs approx)"
        print(f"  {pol:<18} P99: {ap:>9.2f}ms  mean: {am:>9.2f}ms  comp: {ac:>5.0f}  preempt: {apre:>5.0f}  CHR: {achr:.3f}{rr_tag}{prec_tag}")
    else:
        print(f"  {pol:<18} N/A")

# === Main results ===
print("=" * 90)
print("Iter 20: Precise KV Routing — Approximate vs Precise under KV pressure")
print("  (8 instances, rate=400, 10s horizon)")
print("=" * 90)

for kvblocks in [5000, 2000, 1000]:
    for ngroups in [4, 10, 20]:
        tag_base = f"kv{kvblocks}_g{ngroups}"
        print(f"\n--- KV={kvblocks}, {ngroups} prefix groups ---")

        rr_vals = collect(f"rr_{tag_base}")
        approx_vals = collect(f"approx_{tag_base}")
        precise_vals = collect(f"precise_{tag_base}")

        rr_p99 = avg(rr_vals, "p99") if rr_vals else None
        approx_p99 = avg(approx_vals, "p99") if approx_vals else None

        print_row("rr", rr_vals)
        print_row("approx (pa:4,qd:3)", approx_vals, rr_p99)
        print_row("precise (pa:4,qd:3)", precise_vals, rr_p99, approx_p99)

# === Overload ===
print(f"\n{'=' * 90}")
print("Overload: rate=800, KV=2000, 20 prefix groups")
print("=" * 90)
rr_vals = collect("rr_overload")
approx_vals = collect("approx_overload")
precise_vals = collect("precise_overload")
rr_p99 = avg(rr_vals, "p99") if rr_vals else None
approx_p99 = avg(approx_vals, "p99") if approx_vals else None
print_row("rr", rr_vals)
print_row("approx (pa:4,qd:3)", approx_vals, rr_p99)
print_row("precise (pa:4,qd:3)", precise_vals, rr_p99, approx_p99)

# === Summary table ===
print(f"\n{'=' * 90}")
print("SUMMARY: Precise vs Approximate TTFT P99 Improvement")
print("=" * 90)
print(f"  {'Config':<35} {'Approx P99':>12} {'Precise P99':>12} {'Δ Precise':>12} {'Precise/RR':>12}")
print(f"  {'-'*83}")

configs = []
for kvblocks in [5000, 2000, 1000]:
    for ngroups in [4, 10, 20]:
        tag_base = f"kv{kvblocks}_g{ngroups}"
        configs.append((f"kv={kvblocks}, {ngroups}g", tag_base))
configs.append(("overload (kv=2000, 20g)", "overload"))

for label, tag_base in configs:
    rr_vals = collect(f"rr_{tag_base}")
    approx_vals = collect(f"approx_{tag_base}")
    precise_vals = collect(f"precise_{tag_base}")
    if approx_vals and precise_vals:
        ap = avg(approx_vals, "p99")
        pp = avg(precise_vals, "p99")
        rp = avg(rr_vals, "p99") if rr_vals else 0
        imp = (ap - pp) / ap * 100 if ap > 0 else 0
        rr_imp = (rp - pp) / rp * 100 if rp > 0 else 0
        ratio = ap / pp if pp > 0 else 0
        print(f"  {label:<35} {ap:>11.2f}ms {pp:>11.2f}ms {imp:>+11.1f}% {rr_imp:>+11.1f}%")
