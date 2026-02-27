#!/usr/bin/env python3
"""Analyze Iter21: Prefill-Decode Disaggregation."""
import json, os, sys

RD = sys.argv[1] if len(sys.argv) > 1 else "hypotheses/h-disaggregation/results"
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
    comp = max(i.get("completed_requests", 0) for i in instances)
    pre = sum(i.get("preemption_count", 0) for i in instances)
    return {
        "p99": max(p99) if p99 else 0,
        "mean": sum(mn) / len(mn) if mn else 0,
        "e2e_p99": max(e2e_p99) if e2e_p99 else 0,
        "e2e_mean": sum(e2e_mn) / len(e2e_mn) if e2e_mn else 0,
        "comp": comp,
        "preempt": pre,
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

# === Sweep 1: Rate sensitivity ===
print("=" * 95)
print("Iter 21: Prefill-Decode Disaggregation — Rate Sensitivity")
print("  Shared: 8 inst, full lifecycle | Disaggregated: 4 prefill + 4 decode")
print("=" * 95)

for rate in [100, 200, 400]:
    print(f"\n--- Rate={rate} ---")
    shared = collect(f"shared_r{rate}")
    rr = collect(f"rr_r{rate}")
    prefill = collect(f"prefill_r{rate}")
    decode = collect(f"decode_r{rate}")

    if shared:
        sp = avg(shared, "p99")
        sm = avg(shared, "mean")
        sc = avg(shared, "comp")
        se = avg(shared, "e2e_p99")
        print(f"  {'shared (8 inst)':<30} TTFT P99: {sp:>9.2f}ms  mean: {sm:>9.2f}ms  E2E P99: {se:>9.2f}ms  comp: {sc:>5.0f}")
    if rr:
        rp = avg(rr, "p99")
        rm = avg(rr, "mean")
        rc = avg(rr, "comp")
        re = avg(rr, "e2e_p99")
        print(f"  {'RR (8 inst)':<30} TTFT P99: {rp:>9.2f}ms  mean: {rm:>9.2f}ms  E2E P99: {re:>9.2f}ms  comp: {rc:>5.0f}")
    if prefill:
        pp = avg(prefill, "p99")
        pm = avg(prefill, "mean")
        pc = avg(prefill, "comp")
        pe = avg(prefill, "e2e_p99")
        print(f"  {'disagg-prefill (4 inst)':<30} TTFT P99: {pp:>9.2f}ms  mean: {pm:>9.2f}ms  E2E P99: {pe:>9.2f}ms  comp: {pc:>5.0f}")
    if decode:
        dp = avg(decode, "p99")
        dm = avg(decode, "mean")
        dc = avg(decode, "comp")
        de = avg(decode, "e2e_p99")
        print(f"  {'disagg-decode (4 inst)':<30} TTFT P99: {dp:>9.2f}ms  mean: {dm:>9.2f}ms  E2E P99: {de:>9.2f}ms  comp: {dc:>5.0f}")

    # Key comparison: disaggregated prefill TTFT vs shared TTFT
    if shared and prefill:
        sp = avg(shared, "p99")
        pp = avg(prefill, "p99")
        if sp > 0:
            imp = (sp - pp) / sp * 100
            ratio = sp / pp if pp > 0 else float('inf')
            print(f"  >>> TTFT P99 improvement: {imp:+.1f}% ({ratio:.1f}x speedup)")

# === Sweep 2: Instance split ratios ===
print(f"\n{'=' * 95}")
print("Instance Split Ratios (rate=200, total=8)")
print("=" * 95)
print(f"  {'Split':<20} {'Prefill TTFT P99':>18} {'Decode E2E P99':>18} {'Combined':>18}")
print(f"  {'-'*74}")

for pfx_inst in [2, 4, 6]:
    dec_inst = 8 - pfx_inst
    tag_pfx = f"split_p{pfx_inst}d{dec_inst}_prefill"
    tag_dec = f"split_p{pfx_inst}d{dec_inst}_decode"
    pfx_vals = collect(tag_pfx)
    dec_vals = collect(tag_dec)
    if pfx_vals and dec_vals:
        pp = avg(pfx_vals, "p99")
        de = avg(dec_vals, "e2e_p99")
        combined = pp + de  # approximate: TTFT from prefill + E2E from decode
        print(f"  {f'P:{pfx_inst} D:{dec_inst}':<20} {pp:>17.2f}ms {de:>17.2f}ms {combined:>17.2f}ms")

shared_200 = collect("shared_r200")
if shared_200:
    se = avg(shared_200, "e2e_p99")
    sp = avg(shared_200, "p99")
    print(f"  {'Shared (8)':<20} {sp:>17.2f}ms {'(included)':>18} {se:>17.2f}ms")

# === Sweep 3: KV pressure interaction ===
print(f"\n{'=' * 95}")
print("KV Pressure Interaction (rate=200)")
print("=" * 95)

for kvblocks in [5000, 2000]:
    print(f"\n--- KV={kvblocks} ---")
    shared = collect(f"kv{kvblocks}_shared")
    prefill = collect(f"kv{kvblocks}_prefill")
    decode = collect(f"kv{kvblocks}_decode")

    if shared:
        sp = avg(shared, "p99")
        spre = avg(shared, "preempt")
        print(f"  {'shared (8 inst)':<30} TTFT P99: {sp:>9.2f}ms  preempt: {spre:>5.0f}")
    if prefill:
        pp = avg(prefill, "p99")
        ppre = avg(prefill, "preempt")
        print(f"  {'disagg-prefill (4 inst)':<30} TTFT P99: {pp:>9.2f}ms  preempt: {ppre:>5.0f}")
    if shared and prefill:
        sp = avg(shared, "p99")
        pp = avg(prefill, "p99")
        if sp > 0:
            imp = (sp - pp) / sp * 100
            ratio = sp / pp if pp > 0 else float('inf')
            print(f"  >>> TTFT P99 improvement: {imp:+.1f}% ({ratio:.1f}x)")

# === Summary ===
print(f"\n{'=' * 95}")
print("SUMMARY: Disaggregation TTFT P99 Improvement (prefill TTFT vs shared TTFT)")
print("=" * 95)
print(f"  {'Config':<35} {'Shared TTFT P99':>16} {'Disagg TTFT P99':>16} {'Improvement':>12} {'Speedup':>8}")
print(f"  {'-'*87}")

for rate in [100, 200, 400]:
    shared = collect(f"shared_r{rate}")
    prefill = collect(f"prefill_r{rate}")
    if shared and prefill:
        sp = avg(shared, "p99")
        pp = avg(prefill, "p99")
        imp = (sp - pp) / sp * 100 if sp > 0 else 0
        ratio = sp / pp if pp > 0 else 0
        print(f"  {f'rate={rate}, kv=5000':<35} {sp:>15.2f}ms {pp:>15.2f}ms {imp:>+11.1f}% {ratio:>7.1f}x")

for kvblocks in [5000, 2000]:
    shared = collect(f"kv{kvblocks}_shared")
    prefill = collect(f"kv{kvblocks}_prefill")
    if shared and prefill:
        sp = avg(shared, "p99")
        pp = avg(prefill, "p99")
        imp = (sp - pp) / sp * 100 if sp > 0 else 0
        ratio = sp / pp if pp > 0 else 0
        print(f"  {f'rate=200, kv={kvblocks}':<35} {sp:>15.2f}ms {pp:>15.2f}ms {imp:>+11.1f}% {ratio:>7.1f}x")
