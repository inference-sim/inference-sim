#!/usr/bin/env python3
"""Analyze iteration 4 rate sweep: cost-benefit vs static-default vs RR."""
import json, os, sys

RD = sys.argv[1] if len(sys.argv) > 1 else "hypotheses/h-cost-benefit/results"
SEEDS = [42, 123, 7777]
RATES = [100, 200, 300, 400, 500]
POLICIES = ["cost-benefit", "static-default", "round-robin"]

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
print("Iteration 4: Cost-Benefit Rate Sweep (RAG workload, 8 instances)")
print("=" * 90)

fmt_h = "  {:>6} {:<20} {:>10} {:>10} {:>10} {:>12}"
fmt_r = "  {:>6} {:<20} {:>9.2f}ms {:>9.2f}ms {:>10d}{}"
print(fmt_h.format("Rate", "Policy", "TTFT p99", "TTFT mean", "Completed", "vs RR"))
print("-" * 74)

for rate in RATES:
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
            if pol == "round-robin":
                rr_p99 = ap99
            tag = ""
            if rr_p99 and rr_p99 > 0 and pol != "round-robin":
                imp = (rr_p99 - ap99) / rr_p99 * 100
                tag = f"  {imp:+.1f}%"
            print(fmt_r.format(rate, pol, ap99, amn, acomp, tag))
        else:
            print(f"  {rate:>6} {pol:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
    print()

# Head-to-head
print("=" * 60)
print("HEAD-TO-HEAD: Cost-Benefit vs Static-Default")
print("=" * 60)
fmt2 = "  {:>6} {:>12.2f}ms {:>12.2f}ms {:>+11.1f}%  {}"
print(f"  {'Rate':>6} {'CB p99':>13} {'Static p99':>13} {'CB wins by':>12}  {'Winner'}")
print("  " + "-" * 56)
for rate in RATES:
    cb_vals = [parse(os.path.join(RD, f"cost-benefit_rate{rate}_seed{seed}.json")) for seed in SEEDS]
    sd_vals = [parse(os.path.join(RD, f"static-default_rate{rate}_seed{seed}.json")) for seed in SEEDS]
    cb_vals = [v for v in cb_vals if v]
    sd_vals = [v for v in sd_vals if v]
    if cb_vals and sd_vals:
        cb = sum(v["p99"] for v in cb_vals) / len(cb_vals)
        sd = sum(v["p99"] for v in sd_vals) / len(sd_vals)
        diff = (sd - cb) / sd * 100
        winner = "COST-BENEFIT" if cb < sd else ("STATIC" if sd < cb else "TIE")
        print(fmt2.format(rate, cb, sd, diff, winner))
