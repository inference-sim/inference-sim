#!/usr/bin/env python3
import json, os, sys

RD = sys.argv[1] if len(sys.argv) > 1 else "hypotheses/h-admission/results"
SEEDS = [42, 123, 7777]
POLS = ["rr", "baseline", "compound"]

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

print("=" * 80)
print("Iteration 11: SLO-Gated Admission + Priority Cascade")
print("=" * 80)
for rate in [200, 400, 1000, 2000]:
    print(f"\n--- Rate {rate} ---")
    fmt = "  {:<20} {:>10} {:>10} {:>10}"
    print(fmt.format("Policy", "TTFT p99", "TTFT mean", "Completed"))
    print("  " + "-" * 50)
    rr_p99 = None
    for pol in POLS:
        vals = []
        for seed in SEEDS:
            r = parse(os.path.join(RD, f"{pol}_rate{rate}_seed{seed}.json"))
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
                imp = (rr_p99 - ap) / rr_p99 * 100
                tag = f"  ({imp:+.1f}% vs RR)"
            print(f"  {pol:<20} {ap:>9.2f}ms {am:>9.2f}ms {ac:>10d}{tag}")
        else:
            print(f"  {pol:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
