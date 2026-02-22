#!/usr/bin/env python3
"""Analysis script for H-Overload: 10x Overload Robustness.

Parses BLIS multi-block output across rate sweep and verifies:
  1. No panics or non-zero exit codes at any overload level
  2. Conservation invariant (INV-1) holds at every rate level:
     - Cluster: injected == completed + still_queued + still_running
     - Pipeline: admitted + rejected == total_decisions (from trace summary)
  3. Behavioral characterization: queue growth (always-admit) vs rejection (token-bucket)

BLIS output format (see cmd/root.go and sim/metrics_utils.go):
- Per-instance and cluster JSON blocks, each preceded by "=== Simulation Metrics ==="
- Cluster block has "instance_id": "cluster"
- Anomaly counters section: "Rejected Requests: N" (cmd/root.go:501)
- Trace summary: "Total Decisions: N" (cmd/root.go:515)
- Trace summary: "  Admitted: N" (cmd/root.go:516)
- Trace summary: "  Rejected: N" (cmd/root.go:517)
- KV cache section: "Preemption Rate: %.4f" (cmd/root.go:544)
- Exit code marker appended by run.sh: "---EXIT_CODE=N---"
- Panic marker appended by run.sh: "---PANIC_DETECTED---"

Usage:
    python3 analyze.py <max_num_requests> <results_dir>
"""

import json
import os
import re
import sys
from pathlib import Path


def _warn_missing(content, section_header, metric_name, filepath):
    """Warn on stderr if a section header exists but a metric regex didn't match."""
    if section_header in content:
        print(f"WARNING: '{metric_name}' not found in '{filepath}' "
              f"despite '{section_header}' section being present. "
              f"Check regex against cmd/root.go format strings.",
              file=sys.stderr)


def parse_output(filepath):
    """Parse BLIS output -> cluster metrics + trace summary + exit code + panic status."""
    content = Path(filepath).read_text()

    # Extract exit code (appended by run.sh)
    exit_code = -1
    m = re.search(r"---EXIT_CODE=(\d+)---", content)
    if m:
        exit_code = int(m.group(1))

    # Detect panic
    panic_detected = "---PANIC_DETECTED---" in content

    # Extract cluster-level JSON block
    cluster = None
    instances = []
    for match in re.finditer(
        r"=== Simulation Metrics ===\s*\n(\{[^}]+\})", content, re.DOTALL
    ):
        block = json.loads(match.group(1))
        if block.get("instance_id") == "cluster":
            cluster = block
        else:
            instances.append(block)

    # Extract trace summary: Total Decisions, Admitted, Rejected (cmd/root.go:515-517)
    # Format: "Total Decisions: %d"
    total_decisions = 0
    m = re.search(r"Total Decisions: (\d+)", content)
    if m:
        total_decisions = int(m.group(1))
    else:
        _warn_missing(content, "=== Trace Summary ===", "Total Decisions", filepath)

    # Format: "  Admitted: %d"
    trace_admitted = 0
    m = re.search(r"^\s+Admitted:\s+(\d+)", content, re.MULTILINE)
    if m:
        trace_admitted = int(m.group(1))
    else:
        _warn_missing(content, "=== Trace Summary ===", "Admitted", filepath)

    # Format: "  Rejected: %d"
    trace_rejected = 0
    m = re.search(r"^\s+Rejected:\s+(\d+)", content, re.MULTILINE)
    if m:
        trace_rejected = int(m.group(1))
    else:
        _warn_missing(content, "=== Trace Summary ===", "Rejected", filepath)

    # Extract rejected count from anomaly counters (cmd/root.go:501)
    # Format: "Rejected Requests: %d"
    anomaly_rejected = 0
    rej_match = re.search(r"Rejected Requests: (\d+)", content)
    if rej_match:
        anomaly_rejected = int(rej_match.group(1))

    # Extract preemption rate (cmd/root.go:544)
    # Format: "Preemption Rate: %.4f"
    preemption_rate = 0.0
    preempt_match = re.search(r"Preemption Rate: ([0-9.]+)", content)
    if preempt_match:
        preemption_rate = float(preempt_match.group(1))

    return {
        "cluster": cluster,
        "instances": instances,
        "exit_code": exit_code,
        "panic": panic_detected,
        "total_decisions": total_decisions,
        "trace_admitted": trace_admitted,
        "trace_rejected": trace_rejected,
        "anomaly_rejected": anomaly_rejected,
        "preemption_rate": preemption_rate,
    }


def check_conservation(parsed):
    """Check conservation invariants. Returns (passes, fails, details)."""
    passes = 0
    fails = 0
    details = []

    cluster = parsed["cluster"]
    total_decisions = parsed["total_decisions"]
    trace_admitted = parsed["trace_admitted"]
    trace_rejected = parsed["trace_rejected"]
    anomaly_rejected = parsed["anomaly_rejected"]

    if cluster is None:
        # No metrics output (likely panic before completion)
        fails += 1
        details.append("  FAIL: No cluster metrics block found (panic or early exit)")
        return passes, fails, details

    # INV-1a: cluster conservation (injected == completed + queued + running)
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
            f"  FAIL INV-1a: injected={injected} != "
            f"completed({completed})+queued({queued})+running({running})={total}"
        )

    # INV-1b: per-instance conservation
    for inst in parsed["instances"]:
        iid = inst["instance_id"]
        i_inj = inst["injected_requests"]
        i_comp = inst["completed_requests"]
        i_q = inst["still_queued"]
        i_r = inst["still_running"]
        i_total = i_comp + i_q + i_r

        if i_inj == i_total:
            passes += 1
        else:
            fails += 1
            details.append(
                f"  FAIL INV-1b ({iid}): injected={i_inj} != "
                f"completed({i_comp})+queued({i_q})+running({i_r})={i_total}"
            )

    # INV-1c: cross-instance consistency
    if parsed["instances"]:
        sum_inst_inj = sum(i["injected_requests"] for i in parsed["instances"])
        if sum_inst_inj == injected:
            passes += 1
        else:
            fails += 1
            details.append(
                f"  FAIL INV-1c: sum(per-instance)={sum_inst_inj} "
                f"!= cluster={injected}"
            )

    # INV-1d: trace pipeline consistency (admitted + rejected == total_decisions)
    if total_decisions > 0:
        if trace_admitted + trace_rejected == total_decisions:
            passes += 1
        else:
            fails += 1
            details.append(
                f"  FAIL INV-1d: admitted({trace_admitted})+rejected({trace_rejected})"
                f"={trace_admitted + trace_rejected} != total_decisions({total_decisions})"
            )

    # INV-1e: full pipeline conservation (injected + rejected == total generated)
    # total_decisions counts all requests that reached admission. For always-admit,
    # injected should equal total_decisions. For token-bucket, injected + rejected
    # should equal total_decisions. This is the full-pipeline conservation check.
    if total_decisions > 0:
        pipeline_total = injected + trace_rejected
        if pipeline_total == total_decisions:
            passes += 1
        else:
            # Allow off-by-one due to horizon boundary: the last request may be
            # admitted and routed (trace records it), but its instance-level
            # QueuedEvent is pushed past the horizon by the alpha-model queueing
            # delay, so it never enters WaitQ/RunningBatch and isn't counted
            # in injected_requests (= completed + queued + running).
            diff = abs(pipeline_total - total_decisions)
            if diff <= 1:
                passes += 1
                details.append(
                    f"  NOTE INV-1e: injected({injected})+rejected({trace_rejected})"
                    f"={pipeline_total} vs total_decisions({total_decisions}) "
                    f"[off-by-{diff}, horizon boundary edge case]"
                )
            else:
                fails += 1
                details.append(
                    f"  FAIL INV-1e: injected({injected})+rejected({trace_rejected})"
                    f"={pipeline_total} != total_decisions({total_decisions})"
                )

    # INV-1f: rejected consistency between anomaly counter and trace
    if anomaly_rejected > 0 or trace_rejected > 0:
        if anomaly_rejected == trace_rejected:
            passes += 1
        else:
            fails += 1
            details.append(
                f"  FAIL INV-1f: anomaly_rejected({anomaly_rejected}) != "
                f"trace_rejected({trace_rejected})"
            )

    return passes, fails, details


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <max_num_requests> <results_dir>")
        sys.exit(1)

    max_num_requests = int(sys.argv[1])
    results_dir = Path(sys.argv[2])

    # Collect output files, sorted for deterministic ordering
    files = sorted(results_dir.glob("*.txt"))
    if not files:
        print(f"ERROR: No .txt files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Separate by admission policy, filter out stderr files
    always_admit_files = sorted([f for f in files
                                  if f.name.startswith("always_admit_")
                                  and "_stderr" not in f.name])
    token_bucket_files = sorted([f for f in files
                                  if f.name.startswith("token_bucket_")
                                  and "_stderr" not in f.name])

    total_passes = 0
    total_fails = 0
    total_panics = 0
    all_details = []

    all_files = always_admit_files + token_bucket_files

    # ── Part 1: Robustness Check (no panics, no deadlocks) ──────────────────
    print("--- Part 1: Robustness (exit codes and panic detection) ---")
    print()
    print(f"  {'Policy':<15} {'Rate':>6} {'Exit':>5} {'Panic':>6} {'Status':<6}")
    print(f"  {'-'*15} {'-'*6} {'-'*5} {'-'*6} {'-'*6}")

    for f in all_files:
        parsed = parse_output(f)
        policy = "always-admit" if "always_admit" in f.name else "token-bucket"

        # Extract rate multiplier from filename
        mult_match = re.search(r"_(\d+)x", f.name)
        rate_label = f"{mult_match.group(1)}x" if mult_match else "?"

        exit_ok = parsed["exit_code"] == 0
        no_panic = not parsed["panic"]
        status = "PASS" if (exit_ok and no_panic) else "FAIL"

        if not exit_ok or not no_panic:
            total_panics += 1

        print(
            f"  {policy:<15} {rate_label:>6} "
            f"{parsed['exit_code']:>5} {'YES' if parsed['panic'] else 'no':>6} "
            f"{status:<6}"
        )

    print()
    if total_panics == 0:
        print(f"  ROBUSTNESS: ALL PASS -- no panics or non-zero exits across "
              f"{len(all_files)} configurations")
    else:
        print(f"  ROBUSTNESS: {total_panics} FAILURES -- panics or non-zero exits detected")

    print()

    # ── Part 2: Conservation Invariant ───────────────────────────────────────
    print("--- Part 2: Conservation Invariant (INV-1) ---")
    print()
    print(
        f"  {'Policy':<15} {'Rate':>6} "
        f"{'Dec':>5} {'Inj':>5} {'Comp':>5} {'Q':>4} {'R':>4} "
        f"{'Rej':>5} {'Preempt':>8} "
        f"{'Checks':>7} {'Status':<6}"
    )
    print(
        f"  {'-'*15} {'-'*6} "
        f"{'-'*5} {'-'*5} {'-'*5} {'-'*4} {'-'*4} "
        f"{'-'*5} {'-'*8} "
        f"{'-'*7} {'-'*6}"
    )

    for f in all_files:
        parsed = parse_output(f)
        policy = "always-admit" if "always_admit" in f.name else "token-bucket"

        mult_match = re.search(r"_(\d+)x", f.name)
        rate_label = f"{mult_match.group(1)}x" if mult_match else "?"

        passes, fails, details = check_conservation(parsed)
        total_passes += passes
        total_fails += fails

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
            f"  {policy:<15} {rate_label:>6} "
            f"{parsed['total_decisions']:>5} {injected:>5} {completed:>5} {queued:>4} {running:>4} "
            f"{parsed['trace_rejected']:>5} {parsed['preemption_rate']:>8.4f} "
            f"{check_str:>7} {status:<6}"
        )

        if details:
            all_details.extend(
                [f"  {policy} {rate_label}:"] + details
            )

    print()

    # ── Part 3: Behavioral Summary ──────────────────────────────────────────
    print("--- Part 3: Behavioral Summary ---")
    print()

    # Always-admit: expect queue growth as rate increases
    print("  Always-admit behavior (expect queue/running growth under overload):")
    for f in always_admit_files:
        parsed = parse_output(f)
        mult_match = re.search(r"_(\d+)x", f.name)
        rate_label = f"{mult_match.group(1)}x" if mult_match else "?"
        cluster = parsed["cluster"]
        if cluster:
            queued = cluster["still_queued"]
            running = cluster["still_running"]
            completed = cluster["completed_requests"]
            injected = cluster["injected_requests"]
            e2e_mean = cluster.get("e2e_mean_ms", 0)
            incomplete = queued + running
            print(
                f"    {rate_label}: generated={parsed['total_decisions']}, "
                f"completed={completed}, incomplete={incomplete} "
                f"(Q={queued},R={running}), "
                f"e2e_mean={e2e_mean:.1f}ms"
            )
        else:
            print(f"    {rate_label}: NO DATA (panic or early exit)")

    print()

    # Token-bucket: expect rejection growth as rate increases
    print("  Token-bucket behavior (expect rejection growth under overload):")
    for f in token_bucket_files:
        parsed = parse_output(f)
        mult_match = re.search(r"_(\d+)x", f.name)
        rate_label = f"{mult_match.group(1)}x" if mult_match else "?"
        cluster = parsed["cluster"]
        if cluster:
            injected = cluster["injected_requests"]
            completed = cluster["completed_requests"]
            queued = cluster["still_queued"]
            running = cluster["still_running"]
            rejected = parsed["trace_rejected"]
            total = parsed["total_decisions"]
            e2e_mean = cluster.get("e2e_mean_ms", 0)
            reject_pct = (rejected / total * 100) if total > 0 else 0
            print(
                f"    {rate_label}: generated={total}, "
                f"admitted={injected}, rejected={rejected} ({reject_pct:.0f}%), "
                f"completed={completed}, "
                f"e2e_mean={e2e_mean:.1f}ms"
            )
        else:
            print(f"    {rate_label}: NO DATA (panic or early exit)")

    print()

    # ── Overall Summary ─────────────────────────────────────────────────────
    print("--- Overall Summary ---")
    print()

    if total_fails == 0 and total_panics == 0:
        print(
            f"  RESULT: ALL PASS -- {total_passes} invariant checks across "
            f"{len(all_files)} configurations, "
            f"zero panics, zero conservation violations."
        )
        print()
        print("  Conservation invariant (INV-1) holds at all overload levels up to 10x.")
        print("  No undefined behavior (panic, deadlock, silent data loss) observed.")
    else:
        if total_panics > 0:
            print(f"  ROBUSTNESS FAILURES: {total_panics} panics or non-zero exits")
        if total_fails > 0:
            print(
                f"  CONSERVATION FAILURES: {total_fails} "
                f"({total_passes} passed, {total_fails} failed)"
            )
            print()
            print("  Failure details:")
            for d in all_details:
                print(d)
        print()
        print("  OVERLOAD ROBUSTNESS: ISSUES FOUND -- see details above.")

    return 0 if (total_fails == 0 and total_panics == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
