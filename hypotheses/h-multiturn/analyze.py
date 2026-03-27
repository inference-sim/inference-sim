#!/usr/bin/env python3
import json, os, sys

RD = sys.argv[1] if len(sys.argv) > 1 else "hypotheses/h-multiturn/results"
SEEDS = [42, 123, 7777]

def parse(fp):
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        c = f.read().strip()
    if not c:
        return None
    ii = []
    ij = False
    cur = []
    for line in c.split("\n"):
        if line.strip() == "{":
            ij = True
            cur = [line]
        elif ij:
            cur.append(line)
            if line.strip() == "}":
                ij = False
                try:
                    ii.append(json.loads("\n".join(cur)))
                except:
                    pass
    if not ii:
        return None
    p99 = [i["ttft_p99_ms"] for i in ii if i.get("ttft_p99_ms", 0) > 0]
    mn = [i["ttft_mean_ms"] for i in ii if i.get("ttft_mean_ms", 0) > 0]
    comp = sum(i.get("completed_requests", 0) for i in ii)
    return {"p99": max(p99) if p99 else 0, "mean": sum(mn) / len(mn) if mn else 0, "comp": comp}

print("=" * 70)
print("Iter 17: Multi-Session (8 prefix groups × 1024-2048 tokens, 8 inst)")
print("=" * 70)
rr_p99 = None
for pol in ["rr", "pa3qd2", "pa4qd3", "pa3qd2kv2"]:
    vals = []
    for seed in SEEDS:
        r = parse(os.path.join(RD, f"{pol}_seed{seed}.json"))
        if r:
            vals.append(r)
    if vals:
        ap = sum(v["p99"] for v in vals) / len(vals)
        am = sum(v["mean"] for v in vals) / len(vals)
        ac = sum(v["comp"] for v in vals) // len(vals)
        if pol == "rr":
            rr_p99 = ap
        tag = ""
        if rr_p99 and rr_p99 > 0 and pol not in ("rr",):
            tag = f"  ({(rr_p99 - ap) / rr_p99 * 100:+.1f}% vs RR)"
        print(f"  {pol:<16} TTFT P99: {ap:>9.2f}ms  mean: {am:>9.2f}ms  comp: {ac:>5d}{tag}")
    else:
        print(f"  {pol:<16} N/A")
