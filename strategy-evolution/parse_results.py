#!/usr/bin/env python3
"""Parse BLIS output for key metrics."""
import sys, json, re

content = sys.stdin.read()
for block in content.split("{"):
    block = "{" + block
    try:
        end = block.index("}") + 1
        d = json.loads(block[:end])
        if d.get("instance_id") == "cluster":
            inj = d.get("injected_requests", 0)
            rej = 1000 - inj
            tps = d["tokens_per_sec"]
            ttft = d["ttft_p99_ms"]
            print(f"  TTFT={ttft:.1f} TPS={tps:.0f} rej={rej}")
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

current_slo = None
for line in content.split("\n"):
    line = line.strip()
    if line.endswith(":") and line.rstrip(":") in ("critical", "standard", "sheddable"):
        current_slo = line.rstrip(":")
    elif current_slo and "TTFT:" in line:
        m = re.search(r"p99=([\d.]+)", line)
        if m:
            print(f"  {current_slo}={float(m.group(1))/1000:.1f}")
        current_slo = None
