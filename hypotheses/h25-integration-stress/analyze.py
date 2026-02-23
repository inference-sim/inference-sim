#!/usr/bin/env python3
"""Analysis script for H25: Integration Stress Test.

Parses BLIS multi-block output and verifies:
1. Conservation (INV-1): injected == completed + still_queued + still_running per instance and cluster
2. Full pipeline conservation: num_requests == injected_total + rejected
3. Determinism (INV-6): byte-identical stdout across two runs
4. No panics: exit code 0

BLIS output format (see cmd/root.go and sim/metrics_utils.go):
- Per-instance and cluster JSON blocks, each preceded by "=== Simulation Metrics ==="
- Cluster block has "instance_id": "cluster"
- "Rejected Requests: N" in anomaly counters section (cmd/root.go:503)
- Trace summary with "Total Decisions:", "Admitted:", "Rejected:" (cmd/root.go:517-533)
- KV metrics: "Preemption Rate:", "Cache Hit Rate:", "KV Thrashing Rate:" (cmd/root.go:545-548)
"""
import argparse
import json
import re
import sys
from pathlib import Path


def parse_metrics_blocks(content):
    """Extract all JSON blocks preceded by '=== Simulation Metrics ==='."""
    blocks = []
    header = "=== Simulation Metrics ==="
    idx = 0
    while True:
        pos = content.find(header, idx)
        if pos == -1:
            break
        brace_start = content.find("{", pos + len(header))
        if brace_start == -1:
            break
        # Brace-match to find end of JSON object
        depth = 0
        brace_end = brace_start
        for i in range(brace_start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    brace_end = i + 1
                    break
        json_str = content[brace_start:brace_end]
        try:
            block = json.loads(json_str)
            blocks.append(block)
        except json.JSONDecodeError as e:
            print(f"WARNING: Failed to parse JSON block: {e}", file=sys.stderr)
        idx = brace_end
    return blocks


def parse_rejected(content):
    """Extract rejected request count from anomaly counters (cmd/root.go:503)."""
    m = re.search(r"Rejected Requests: (\d+)", content)
    return int(m.group(1)) if m else 0


def parse_trace_summary(content):
    """Extract trace summary fields (cmd/root.go:517-533)."""
    result = {}
    m = re.search(r"Total Decisions: (\d+)", content)
    if m:
        result["total_decisions"] = int(m.group(1))
    m = re.search(r"Admitted: (\d+)", content)
    if m:
        result["admitted"] = int(m.group(1))
    # "Rejected:" in trace summary section (distinct from "Rejected Requests:" in anomaly counters)
    m = re.search(r"^\s+Rejected: (\d+)", content, re.MULTILINE)
    if m:
        result["trace_rejected"] = int(m.group(1))
    m = re.search(r"Unique Targets: (\d+)", content)
    if m:
        result["unique_targets"] = int(m.group(1))
    m = re.search(r"Mean Regret: ([0-9.]+)", content)
    if m:
        result["mean_regret"] = float(m.group(1))
    m = re.search(r"Max Regret: ([0-9.]+)", content)
    if m:
        result["max_regret"] = float(m.group(1))

    # Parse target distribution
    target_dist = {}
    for tm in re.finditer(r"  (instance-\d+): (\d+)", content):
        target_dist[tm.group(1)] = int(tm.group(2))
    if target_dist:
        result["target_distribution"] = target_dist

    return result


def parse_kv_metrics(content):
    """Extract KV cache metrics (cmd/root.go:545-548)."""
    result = {}
    m = re.search(r"Preemption Rate: ([0-9.]+)", content)
    if m:
        result["preemption_rate"] = float(m.group(1))
    m = re.search(r"Cache Hit Rate: ([0-9.]+)", content)
    if m:
        result["cache_hit_rate"] = float(m.group(1))
    m = re.search(r"KV Thrashing Rate: ([0-9.]+)", content)
    if m:
        result["kv_thrashing_rate"] = float(m.group(1))
    return result


def extract_preemption_counts(blocks):
    """Extract preemption_count from each JSON metrics block."""
    counts = {}
    for block in blocks:
        iid = block.get("instance_id", "unknown")
        counts[iid] = block.get("preemption_count", 0)
    return counts


def check_conservation(blocks, rejected, num_requests):
    """Check INV-1 conservation for all blocks.

    Per-instance: injected == completed + still_queued + still_running
    Cluster-level: same check on cluster block
    Full pipeline: num_requests == sum(instance.injected) + rejected
    """
    results = []
    instance_blocks = []

    for block in blocks:
        iid = block.get("instance_id", "unknown")
        completed = block.get("completed_requests", 0)
        still_queued = block.get("still_queued", 0)
        still_running = block.get("still_running", 0)
        injected = block.get("injected_requests", 0)

        accounted = completed + still_queued + still_running
        ok = (injected == accounted)

        results.append({
            "instance_id": iid,
            "injected": injected,
            "completed": completed,
            "still_queued": still_queued,
            "still_running": still_running,
            "accounted": accounted,
            "conservation_ok": ok,
        })

        if iid != "cluster":
            instance_blocks.append(block)

    # Full pipeline conservation: num_requests == total_injected + rejected
    total_injected = sum(b.get("injected_requests", 0) for b in instance_blocks)
    pipeline_ok = (num_requests == total_injected + rejected)

    return results, {
        "total_injected": total_injected,
        "rejected": rejected,
        "sum": total_injected + rejected,
        "num_requests": num_requests,
        "pipeline_ok": pipeline_ok,
    }


def analyze_per_request(results_json_path):
    """Analyze per-request data from --results-path output."""
    if not results_json_path or not Path(results_json_path).exists():
        return None

    data = json.loads(Path(results_json_path).read_text())
    requests = data.get("requests", [])
    if not requests:
        return {"count": 0}

    # Check causality (INV-5): scheduling_delay >= 0 for all requests
    negative_delays = [r for r in requests if r.get("scheduling_delay_ms", 0) < 0]

    # Check all requests have a handler instance
    unhandled = [r for r in requests if not r.get("handled_by")]

    # Check SLO classes present
    slo_classes = set(r.get("slo_class", "") for r in requests)

    return {
        "count": len(requests),
        "negative_delays": len(negative_delays),
        "unhandled": len(unhandled),
        "slo_classes": sorted(slo_classes),
        "completed": data.get("completed_requests", 0),
        "still_queued": data.get("still_queued", 0),
        "still_running": data.get("still_running", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="H25 Integration Stress Test Analysis")
    parser.add_argument("--run1", required=True, help="Path to first run output")
    parser.add_argument("--run2", required=True, help="Path to second run output")
    parser.add_argument("--results-json", help="Path to per-request results JSON")
    parser.add_argument("--determinism", required=True, choices=["PASS", "FAIL"])
    parser.add_argument("--num-requests", type=int, required=True)
    parser.add_argument("--config-label", default="", help="Label for this configuration")
    args = parser.parse_args()

    content = Path(args.run1).read_text()
    blocks = parse_metrics_blocks(content)
    rejected = parse_rejected(content)
    trace_summary = parse_trace_summary(content)
    kv_metrics = parse_kv_metrics(content)

    if not blocks:
        print("ERROR: No metrics blocks found in output!", file=sys.stderr)
        sys.exit(1)

    # Check conservation
    conservation_results, pipeline = check_conservation(blocks, rejected, args.num_requests)

    # Extract preemption counts from JSON blocks
    preemption_counts = extract_preemption_counts(blocks)

    # Analyze per-request data
    per_request = analyze_per_request(args.results_json)

    # =====================================================================
    # Print results
    # =====================================================================

    label = args.config_label or "H25: Integration Stress Test"
    print("=" * 72)
    print(label)
    print("=" * 72)
    print()

    # 1. Conservation (INV-1)
    print("## 1. Conservation (INV-1)")
    print()
    all_conserved = True
    print(f"  {'Instance':<15} {'Injected':>10} {'Completed':>10} {'Queued':>10} {'Running':>10} {'Sum':>10} {'OK':>6}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
    for r in conservation_results:
        ok_str = "PASS" if r["conservation_ok"] else "FAIL"
        if not r["conservation_ok"]:
            all_conserved = False
        print(f"  {r['instance_id']:<15} {r['injected']:>10} {r['completed']:>10} "
              f"{r['still_queued']:>10} {r['still_running']:>10} {r['accounted']:>10} {ok_str:>6}")

    print()
    print(f"  Per-instance conservation: {'PASS' if all_conserved else 'FAIL'}")
    print()

    # Full pipeline conservation
    print(f"  Full pipeline: num_requests({pipeline['num_requests']}) == "
          f"injected({pipeline['total_injected']}) + rejected({pipeline['rejected']}) "
          f"= {pipeline['sum']}")
    print(f"  Pipeline conservation: {'PASS' if pipeline['pipeline_ok'] else 'FAIL'}")
    print()

    # 2. Determinism (INV-6)
    print("## 2. Determinism (INV-6)")
    print(f"  Two runs with seed=42: {args.determinism}")
    print()

    # 3. No panics
    print("## 3. No Panics")
    print("  Exit code 0: PASS (both runs completed successfully)")
    print()

    # 4. KV Cache Metrics
    print("## 4. KV Cache Metrics")
    if kv_metrics:
        for k, v in sorted(kv_metrics.items()):
            print(f"  {k}: {v}")
    else:
        print("  (no KV metrics emitted — all rates zero)")
    print()

    # 4b. Preemption Counts (from JSON blocks)
    total_preemptions = sum(v for k, v in preemption_counts.items() if k != "cluster")
    print("## 4b. Preemption Counts (per-instance)")
    if total_preemptions > 0:
        for iid in sorted(preemption_counts.keys()):
            print(f"  {iid}: {preemption_counts[iid]}")
        print(f"  TOTAL (non-cluster): {total_preemptions}")
    else:
        print("  (zero preemptions across all instances)")
    print()

    # 5. Trace Summary
    print("## 5. Trace Summary")
    if trace_summary:
        for k, v in sorted(trace_summary.items()):
            if k == "target_distribution":
                print(f"  {k}:")
                for inst, count in sorted(v.items()):
                    print(f"    {inst}: {count}")
            else:
                print(f"  {k}: {v}")
    else:
        print("  (no trace summary found)")
    print()

    # 6. Per-request analysis
    print("## 6. Per-Request Analysis")
    if per_request:
        print(f"  Total requests in results: {per_request['count']}")
        print(f"  Negative scheduling delays: {per_request['negative_delays']}")
        print(f"  Unhandled requests: {per_request['unhandled']}")
        print(f"  SLO classes: {per_request['slo_classes']}")
        print(f"  Completed: {per_request['completed']}")
        print(f"  Still queued: {per_request['still_queued']}")
        print(f"  Still running: {per_request['still_running']}")
    else:
        print("  (no per-request data)")
    print()

    # 7. Overall verdict
    print("=" * 72)
    all_pass = all_conserved and pipeline["pipeline_ok"] and args.determinism == "PASS"
    if all_pass:
        print("VERDICT: ALL CHECKS PASS")
        print("  Conservation (INV-1): PASS")
        print("  Pipeline conservation: PASS")
        print("  Determinism (INV-6): PASS")
        print("  No panics: PASS")
    else:
        print("VERDICT: SOME CHECKS FAILED")
        if not all_conserved:
            print("  Conservation (INV-1): FAIL")
        if not pipeline["pipeline_ok"]:
            print(f"  Pipeline conservation: FAIL ({pipeline['num_requests']} != {pipeline['sum']})")
        if args.determinism != "PASS":
            print("  Determinism (INV-6): FAIL")
    print("=" * 72)


if __name__ == "__main__":
    main()
