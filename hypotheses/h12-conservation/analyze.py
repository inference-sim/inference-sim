#!/usr/bin/env python3
"""Analysis script for H12: Request Conservation Invariant.

Parses BLIS multi-block output and verifies conservation invariants:
  1. Per-instance: injected == completed + still_queued + still_running
  2. Cluster:      injected == completed + still_queued + still_running
  3. Cross-instance: sum(per-instance injected) == cluster injected
  4. Full pipeline: num_requests == injected + rejected (when admission control active)

Usage:
    python3 analyze.py <num_requests> <output_files...>
"""

import json
import re
import sys
from pathlib import Path


def parse_output(filepath, num_requests):
    """Parse multi-block BLIS output into per-instance and cluster metrics."""
    content = Path(filepath).read_text()

    instances = []
    cluster = None

    # Extract all JSON metric blocks
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        block = json.loads(match.group(1))
        if block.get("instance_id") == "cluster":
            cluster = block
        else:
            instances.append(block)

    # Extract rejected count from anomaly counters
    rejected = 0
    rejected_match = re.search(r"Rejected Requests: (\d+)", content)
    if rejected_match:
        rejected = int(rejected_match.group(1))

    # Extract preemption count if present
    preemptions = 0
    preempt_match = re.search(r"Preemptions?: (\d+)", content)
    if preempt_match:
        preemptions = int(preempt_match.group(1))

    return {
        "instances": instances,
        "cluster": cluster,
        "rejected": rejected,
        "preemptions": preemptions,
        "num_requests": num_requests,
    }


def check_conservation(parsed):
    """Check all conservation invariants. Returns (pass_count, fail_count, details)."""
    details = []
    passes = 0
    fails = 0

    instances = parsed["instances"]
    cluster = parsed["cluster"]
    rejected = parsed["rejected"]
    num_requests = parsed["num_requests"]

    # --- Invariant 1: Per-instance conservation ---
    for inst in instances:
        iid = inst["instance_id"]
        injected = inst["injected_requests"]
        completed = inst["completed_requests"]
        queued = inst["still_queued"]
        running = inst["still_running"]
        total = completed + queued + running

        if injected == total:
            passes += 1
        else:
            fails += 1
            details.append(
                f"  FAIL per-instance ({iid}): "
                f"injected={injected} != completed({completed})"
                f"+queued({queued})+running({running})={total}"
            )

    # --- Invariant 2: Cluster conservation ---
    if cluster:
        injected = cluster["injected_requests"]
        completed = cluster["completed_requests"]
        queued = cluster["still_queued"]
        running = cluster["still_running"]
        total = completed + queued + running

        if injected == total:
            passes += 1
        else:
            fails += 1
            details.append(
                f"  FAIL cluster: "
                f"injected={injected} != completed({completed})"
                f"+queued({queued})+running({running})={total}"
            )

    # --- Invariant 3: Cross-instance consistency ---
    if cluster and instances:
        sum_instance_injected = sum(i["injected_requests"] for i in instances)
        cluster_injected = cluster["injected_requests"]

        if sum_instance_injected == cluster_injected:
            passes += 1
        else:
            fails += 1
            details.append(
                f"  FAIL cross-instance: "
                f"sum(per-instance injected)={sum_instance_injected}"
                f" != cluster injected={cluster_injected}"
            )

    # --- Invariant 4: Full pipeline (when admission control rejects) ---
    if cluster:
        cluster_injected = cluster["injected_requests"]
        pipeline_total = cluster_injected + rejected

        if pipeline_total == num_requests:
            passes += 1
        else:
            fails += 1
            details.append(
                f"  FAIL pipeline: "
                f"injected({cluster_injected})+rejected({rejected})"
                f"={pipeline_total} != num_requests({num_requests})"
            )

    return passes, fails, details


def config_label(filename):
    """Extract human-readable label from filename."""
    stem = Path(filename).stem
    labels = {
        "cfg01_baseline": "round-robin + fcfs + always-admit",
        "cfg02_least_loaded": "least-loaded + fcfs + always-admit",
        "cfg03_weighted_qd": "weighted(qd:1) + fcfs + always-admit",
        "cfg04_weighted_full": "weighted(pa:3,qd:2,kv:2) + fcfs",
        "cfg05_sjf": "round-robin + sjf + always-admit",
        "cfg06_priority": "round-robin + priority-fcfs + slo-based",
        "cfg07_token_bucket": "round-robin + fcfs + token-bucket",
        "cfg08_high_rate": "least-loaded + fcfs + rate=2000",
        "cfg09_combined": "weighted + priority-fcfs + token-bucket",
        "cfg10_pathological": "always-busiest + reverse-priority",
    }
    return labels.get(stem, stem)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <num_requests> <output_files...>")
        sys.exit(1)

    num_requests = int(sys.argv[1])
    files = sorted(sys.argv[2:])

    total_passes = 0
    total_fails = 0
    all_details = []

    # Header
    print(
        f"  {'#':<4} {'Configuration':<42} "
        f"{'Inj':>5} {'Comp':>5} {'Q':>3} {'R':>3} "
        f"{'Rej':>5} {'Preempt':>7} "
        f"{'Checks':>6} {'Status':<6}"
    )
    print(f"  {'-'*4} {'-'*42} {'-'*5} {'-'*5} {'-'*3} {'-'*3} {'-'*5} {'-'*7} {'-'*6} {'-'*6}")

    for i, f in enumerate(files, 1):
        parsed = parse_output(f, num_requests)
        passes, fails, details = check_conservation(parsed)

        cluster = parsed["cluster"]
        if cluster:
            injected = cluster["injected_requests"]
            completed = cluster["completed_requests"]
            queued = cluster["still_queued"]
            running = cluster["still_running"]
        else:
            injected = completed = queued = running = 0

        status = "PASS" if fails == 0 else "FAIL"
        check_str = f"{passes}/{passes + fails}"

        print(
            f"  {i:<4} {config_label(f):<42} "
            f"{injected:>5} {completed:>5} {queued:>3} {running:>3} "
            f"{parsed['rejected']:>5} {parsed['preemptions']:>7} "
            f"{check_str:>6} {status:<6}"
        )

        total_passes += passes
        total_fails += fails
        if details:
            all_details.extend([f"  Config {i} ({config_label(f)}):"] + details)

    # Summary
    print()
    if total_fails == 0:
        print(
            f"  RESULT: ALL PASS — {total_passes} invariant checks across "
            f"{len(files)} configurations, zero violations."
        )
        print()
        print("  Conservation invariant (INV-1) holds for all tested policy combinations.")
    else:
        print(
            f"  RESULT: {total_fails} FAILURES "
            f"({total_passes} passed, {total_fails} failed)"
        )
        print()
        print("  Failure details:")
        for d in all_details:
            print(d)
        print()
        print(
            "  Conservation invariant VIOLATED — "
            "this is a critical correctness bug."
        )

    return 0 if total_fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
