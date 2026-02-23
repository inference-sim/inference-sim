#!/usr/bin/env python3
"""Shared analysis helpers for hypothesis experiments.

Import in analyze.py:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
    from analyze_helpers import parse_blis_output, check_for_timeout

Provides:
    parse_blis_output(filepath) -> dict  — parse BLIS output with timeout/error handling
    check_for_timeout(filepath) -> bool  — detect timeout/error sentinel files
"""

import json
import re
import sys
from pathlib import Path


def check_for_timeout(filepath):
    """Check if an output file indicates a timeout or error.

    Returns True if the file is empty, missing, or starts with TIMEOUT/ERROR.
    Prints a warning to stderr when a timeout/error is detected.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"WARNING: output file missing: {filepath}", file=sys.stderr)
        return True
    content = path.read_text().strip()
    if not content:
        print(f"WARNING: output file empty: {filepath}", file=sys.stderr)
        return True
    first_line = content.split("\n")[0]
    if first_line.startswith("TIMEOUT"):
        print(f"WARNING: simulation timed out: {filepath}", file=sys.stderr)
        return True
    if first_line.startswith("ERROR:"):
        print(f"WARNING: simulation error ({first_line}): {filepath}", file=sys.stderr)
        return True
    return False


def parse_blis_output(filepath):
    """Parse BLIS CLI output into a metrics dict.

    Handles timeout/error/empty files gracefully by returning default zeros
    with a 'timed_out' flag.

    Returns dict with keys:
        ttft_mean, ttft_p99, e2e_mean, e2e_p99  — latency metrics (ms)
        throughput                                — responses_per_sec
        completed                                — completed request count
        preemption_rate, cache_hit_rate           — from text summary (float 0-1)
        rejected                                  — rejected request count
        injected, still_queued, still_running     — conservation fields (INV-1)
        preemption_count                          — total preemptions across instances
        timed_out                                 — True if output is missing/empty/TIMEOUT/ERROR

    Output format reference:
        - cmd/root.go: text summary + JSON blocks per instance + cluster
        - sim/metrics_utils.go: MetricsOutput JSON struct
        - Cluster JSON block: "instance_id": "cluster" (assumes no nested objects)
        - KV summary: "Preemption Rate: 0.1750", "Cache Hit Rate: 0.0452"
        - Trace summary: "Rejected Requests: N", "Target Distribution:"
    """
    defaults = {
        "ttft_mean": 0, "ttft_p99": 0,
        "e2e_mean": 0, "e2e_p99": 0,
        "throughput": 0, "completed": 0,
        "preemption_rate": 0.0, "cache_hit_rate": 0.0,
        "rejected": 0,
        # Conservation fields (used by h8, h12, h25, h-overload-kv)
        "injected": 0, "still_queued": 0, "still_running": 0,
        "preemption_count": 0,
        "timed_out": False,
    }

    if check_for_timeout(filepath):
        defaults["timed_out"] = True
        return defaults

    content = Path(filepath).read_text()

    # Extract cluster-level JSON block
    cluster = None
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        try:
            block = json.loads(match.group(1))
            if block.get("instance_id") == "cluster":
                cluster = block
        except json.JSONDecodeError as e:
            print(f"WARNING: failed to parse JSON block in {filepath}: {e}", file=sys.stderr)
            continue

    if cluster is None:
        print(f"WARNING: no cluster JSON block found in {filepath}", file=sys.stderr)
        defaults["timed_out"] = True
        return defaults

    # KV summary metrics (from cmd/root.go text output)
    preemption_rate = 0.0
    m = re.search(r"Preemption Rate: ([0-9.]+)", content)
    if m:
        preemption_rate = float(m.group(1))

    cache_hit_rate = 0.0
    m = re.search(r"Cache Hit Rate: ([0-9.]+)", content)
    if m:
        cache_hit_rate = float(m.group(1))

    rejected = 0
    m = re.search(r"Rejected Requests: (\d+)", content)
    if m:
        rejected = int(m.group(1))

    return {
        "ttft_mean": cluster.get("ttft_mean_ms", 0),
        "ttft_p99": cluster.get("ttft_p99_ms", 0),
        "e2e_mean": cluster.get("e2e_mean_ms", 0),
        "e2e_p99": cluster.get("e2e_p99_ms", 0),
        "throughput": cluster.get("responses_per_sec", 0),
        "completed": cluster.get("completed_requests", 0),
        "preemption_rate": preemption_rate,
        "cache_hit_rate": cache_hit_rate,
        "rejected": rejected,
        # Conservation fields
        "injected": cluster.get("injected_requests", 0),
        "still_queued": cluster.get("still_queued", 0),
        "still_running": cluster.get("still_running", 0),
        "preemption_count": cluster.get("preemption_count", 0),
        "timed_out": False,
    }
