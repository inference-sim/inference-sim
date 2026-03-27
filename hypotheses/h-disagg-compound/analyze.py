#!/usr/bin/env python3
"""Analyze Iter22: Combined Disaggregation + Compound Strategy."""
import json, os, sys

RD = sys.argv[1] if len(sys.argv) > 1 else "hypotheses/h-disagg-compound/results"
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
    e2e_p99 = [i["e2e_p99_ms"] for i in instances if i.get("e2e_p99_ms", 0) > 0]
    e2e_mn = [i["e2e_mean_ms"] for i in instances if i.get("e2e_mean_ms", 0) > 0]
    comp = max((i.get("completed_requests", 0) for i in instances), default=0)
    pre = sum(i.get("preemption_count", 0) for i in instances)
    return {
        "p99": max(p99) if p99 else 0,
        "mean": sum(mn) / len(mn) if mn else 0,
        "e2e_p99": max(e2e_p99) if e2e_p99 else 0,
        "e2e_mean": sum(e2e_mn) / len(e2e_mn) if e2e_mn else 0,
        "comp": comp, "preempt": pre,
    }

def avg(vals, key):
    vs = [v[key] for v in vals]
    return sum(vs) / len(vs) if vs else 0

def collect(tag):
    return [v for v in (parse(os.path.join(RD, f"{tag}_seed{s}.json")) for s in SEEDS) if v]

def prow(label, vals, ref_p99=None):
    if not vals:
        print(f"  {label:<35} N/A")
        return
    p = avg(vals, "p99")
    m = avg(vals, "mean")
    e = avg(vals, "e2e_p99")
    c = avg(vals, "comp")
    tag = ""
    if ref_p99 and ref_p99 > 0:
        tag = f"  ({(ref_p99 - p) / ref_p99 * 100:+.1f}%)"
    print(f"  {label:<35} TTFT P99: {p:>9.2f}ms  mean: {m:>8.2f}ms  E2E P99: {e:>9.2f}ms  comp: {c:>5.0f}{tag}")

# === Section 1: Prefill pool routing ===
print("=" * 100)
print("Section 1: Prefill Pool Routing (4 inst, rate=400, output=1)")
print("=" * 100)
for tag, label in [("pfx_rr", "RR"), ("pfx_ll", "Least-Loaded"),
                    ("pfx_pa4qd3", "PA:4,QD:3"), ("pfx_pa1qd4", "PA:1,QD:4")]:
    prow(label, collect(tag))

# === Section 2: KV migration cost ===
print(f"\n{'=' * 100}")
print("Section 2: KV Migration Cost Sensitivity (rate=200)")
print("=" * 100)
print(f"  {'Migration':<12} {'Prefill TTFT P99':>18} {'Decode E2E P99':>18} {'Combined P99':>18}")
print(f"  {'-' * 66}")
for ms in [0, 1, 5, 10, 50]:
    pfx = collect(f"migrate{ms}_prefill")
    dec = collect(f"migrate{ms}_decode")
    if pfx and dec:
        pp = avg(pfx, "p99")
        de = avg(dec, "e2e_p99")
        combined = pp + de
        print(f"  {f'{ms}ms':<12} {pp:>17.2f}ms {de:>17.2f}ms {combined:>17.2f}ms")

# === Section 3: Load crossover ===
print(f"\n{'=' * 100}")
print("Section 3: Load Crossover (shared-8 vs disagg P:2/D:6)")
print("=" * 100)
print(f"  {'Rate':<8} {'Shared TTFT P99':>16} {'Disagg TTFT P99':>16} {'Shared E2E P99':>16} {'Disagg E2E P99':>16} {'TTFT Speedup':>12}")
print(f"  {'-' * 84}")
for rate in [50, 100, 200, 300, 400]:
    sh = collect(f"xover_shared_r{rate}")
    pfx = collect(f"xover_pfx_r{rate}")
    dec = collect(f"xover_dec_r{rate}")
    if sh and pfx:
        sp = avg(sh, "p99")
        pp = avg(pfx, "p99")
        se = avg(sh, "e2e_p99")
        de = avg(dec, "e2e_p99") if dec else 0
        disagg_e2e = pp + de
        ratio = sp / pp if pp > 0 else 0
        print(f"  {rate:<8} {sp:>15.2f}ms {pp:>15.2f}ms {se:>15.2f}ms {disagg_e2e:>15.2f}ms {ratio:>11.1f}x")

# === Section 4: Compound ===
print(f"\n{'=' * 100}")
print("Section 4: Compound Disaggregation (rate=400)")
print("=" * 100)
sh = collect("compound_shared")
pfx = collect("compound_pfx")
dec = collect("compound_dec")
prow("Shared (8, compound)", sh)
if pfx:
    pp = avg(pfx, "p99")
    print(f"  {'Disagg prefill (2, PA)':<35} TTFT P99: {pp:>9.2f}ms")
if dec:
    de = avg(dec, "e2e_p99")
    print(f"  {'Disagg decode (6, SLO+QD)':<35} E2E P99:  {de:>9.2f}ms")
if sh and pfx and dec:
    sp = avg(sh, "p99")
    pp = avg(pfx, "p99")
    de = avg(dec, "e2e_p99")
    ratio = sp / pp if pp > 0 else 0
    print(f"  >>> Disagg TTFT speedup: {ratio:.1f}x | Combined E2E: {pp+de:.2f}ms vs shared {avg(sh, 'e2e_p99'):.2f}ms")

# === Summary ===
print(f"\n{'=' * 100}")
print("SUMMARY")
print("=" * 100)
print("""
Key findings:
1. Prefill pool routing: [see Section 1 for which policy wins]
2. KV migration cost: [see Section 2 for break-even point]
3. Load crossover: [see Section 3 for where disagg stops helping]
4. Compound: [see Section 4 for combined benefit]
""")
