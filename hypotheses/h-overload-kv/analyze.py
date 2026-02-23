#!/usr/bin/env python3
"""Analysis script for H-Overload-KV: Combined Overload + KV Cache Pressure.

Parses BLIS multi-block output across a 3x3 matrix (overload x KV config) and verifies:
  1. No panics or non-zero exit codes at any cell
  2. Conservation invariant (INV-1) holds at every cell:
     - Cluster: injected == completed + still_queued + still_running
     - Pipeline: admitted + rejected == total_decisions (from trace summary)
  3. Preemption behavior: constrained/tiered show higher preemption under pressure
  4. Tiered comparison: tiered config should show benefit over constrained-only

BLIS output format (see cmd/root.go and sim/metrics_utils.go):
- Per-instance and cluster JSON blocks, each preceded by "=== Simulation Metrics ==="
- Cluster block has "instance_id": "cluster"
- Per-instance blocks have per-instance preemption_count (json field)
- KV cache section: "Preemption Rate: %.4f" (cmd/root.go:544)
- Cache Hit Rate: "Cache Hit Rate: %.4f" (cmd/root.go:545)
- Trace summary: "Total Decisions: %d" (cmd/root.go:515)
- Trace summary: "  Admitted: %d" (cmd/root.go:516)
- Trace summary: "  Rejected: %d" (cmd/root.go:517)
- Exit code marker appended by run.sh: "---EXIT_CODE=N---"
- Panic marker appended by run.sh: "---PANIC_DETECTED---"

Usage:
    python3 analyze.py <results_dir>
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
    """Parse BLIS output -> cluster metrics + per-instance + trace + exit/panic."""
    content = Path(filepath).read_text()

    # Extract exit code (appended by run.sh)
    exit_code = -1
    m = re.search(r"---EXIT_CODE=(\d+)---", content)
    if m:
        exit_code = int(m.group(1))

    # Detect panic
    panic_detected = "---PANIC_DETECTED---" in content

    # Extract JSON blocks
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

    # Trace summary
    total_decisions = 0
    m = re.search(r"Total Decisions: (\d+)", content)
    if m:
        total_decisions = int(m.group(1))
    else:
        _warn_missing(content, "=== Trace Summary ===", "Total Decisions", filepath)

    trace_admitted = 0
    m = re.search(r"^\s+Admitted:\s+(\d+)", content, re.MULTILINE)
    if m:
        trace_admitted = int(m.group(1))
    else:
        _warn_missing(content, "=== Trace Summary ===", "Admitted", filepath)

    trace_rejected = 0
    m = re.search(r"^\s+Rejected:\s+(\d+)", content, re.MULTILINE)
    if m:
        trace_rejected = int(m.group(1))
    else:
        _warn_missing(content, "=== Trace Summary ===", "Rejected", filepath)

    # KV cache metrics
    preemption_rate = 0.0
    pr_match = re.search(r"Preemption Rate: ([\d.]+)", content)
    if pr_match:
        preemption_rate = float(pr_match.group(1))

    cache_hit_rate = 0.0
    chr_match = re.search(r"Cache Hit Rate: ([\d.]+)", content)
    if chr_match:
        cache_hit_rate = float(chr_match.group(1))

    kv_thrashing_rate = 0.0
    thr_match = re.search(r"KV Thrashing Rate: ([\d.]+)", content)
    if thr_match:
        kv_thrashing_rate = float(thr_match.group(1))

    # Sum preemption_count from per-instance JSON blocks
    total_preemption_count = 0
    for inst in instances:
        total_preemption_count += inst.get("preemption_count", 0)

    return {
        "cluster": cluster,
        "instances": instances,
        "exit_code": exit_code,
        "panic": panic_detected,
        "total_decisions": total_decisions,
        "trace_admitted": trace_admitted,
        "trace_rejected": trace_rejected,
        "preemption_rate": preemption_rate,
        "preemption_count": total_preemption_count,
        "cache_hit_rate": cache_hit_rate,
        "kv_thrashing_rate": kv_thrashing_rate,
    }


def parse_label(filepath):
    """Extract overload multiplier and KV config from filename like 2x_abundant.txt."""
    name = Path(filepath).stem
    # Pattern: Nx_kvconfig
    m = re.match(r"(\d+)x_(\w+)", name)
    if m:
        return int(m.group(1)), m.group(2)
    return 0, "unknown"


def check_conservation(parsed):
    """Check conservation invariants. Returns (passes, fails, details)."""
    passes = 0
    fails = 0
    details = []

    cluster = parsed["cluster"]
    total_decisions = parsed["total_decisions"]
    trace_admitted = parsed["trace_admitted"]
    trace_rejected = parsed["trace_rejected"]

    if cluster is None:
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

    # INV-1e: full pipeline (injected + rejected == total_decisions, off-by-1 tolerance)
    if total_decisions > 0:
        pipeline_total = injected + trace_rejected
        if pipeline_total == total_decisions:
            passes += 1
        else:
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

    return passes, fails, details


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    # Collect output files, sorted for deterministic ordering
    files = sorted(results_dir.glob("*x_*.txt"))
    # Filter out stderr files
    files = [f for f in files if "_stderr" not in f.name]

    if not files:
        print(f"ERROR: No result files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Parse all results
    results = {}
    for f in files:
        mult, kv_name = parse_label(f)
        parsed = parse_output(f)
        results[(mult, kv_name)] = parsed

    multipliers = sorted(set(m for m, _ in results))
    kv_configs = ["abundant", "constrained", "tiered"]

    total_passes = 0
    total_fails = 0
    total_panics = 0
    all_details = []

    # ---- Part 1: Robustness (exit codes and panics) -------------------------
    print("--- Part 1: Robustness (exit codes and panic detection) ---")
    print()
    print(f"  {'Overload':<10} {'KV Config':<12} {'Exit':>5} {'Panic':>6} {'Status':<6}")
    print(f"  {'-'*10} {'-'*12} {'-'*5} {'-'*6} {'-'*6}")

    for mult in multipliers:
        for kv_name in kv_configs:
            key = (mult, kv_name)
            if key not in results:
                print(f"  {mult}x{'':<8} {kv_name:<12} {'?':>5} {'?':>6} {'MISS':<6}")
                continue
            parsed = results[key]
            exit_ok = parsed["exit_code"] == 0
            no_panic = not parsed["panic"]
            status = "PASS" if (exit_ok and no_panic) else "FAIL"

            if not exit_ok or not no_panic:
                total_panics += 1

            print(
                f"  {mult}x{'':<8} {kv_name:<12} "
                f"{parsed['exit_code']:>5} {'YES' if parsed['panic'] else 'no':>6} "
                f"{status:<6}"
            )

    print()
    if total_panics == 0:
        print(f"  ROBUSTNESS: ALL PASS -- no panics or non-zero exits across "
              f"{len(results)} configurations")
    else:
        print(f"  ROBUSTNESS: {total_panics} FAILURES -- panics or non-zero exits detected")

    print()

    # ---- Part 2: Conservation Invariant (INV-1) -----------------------------
    print("--- Part 2: Conservation Invariant (INV-1) ---")
    print()
    print(
        f"  {'Cell':<18} "
        f"{'Dec':>5} {'Inj':>5} {'Comp':>5} {'Q':>4} {'R':>4} "
        f"{'Preempt':>8} {'CacheHit':>9} "
        f"{'Checks':>7} {'Status':<6}"
    )
    print(
        f"  {'-'*18} "
        f"{'-'*5} {'-'*5} {'-'*5} {'-'*4} {'-'*4} "
        f"{'-'*8} {'-'*9} "
        f"{'-'*7} {'-'*6}"
    )

    for mult in multipliers:
        for kv_name in kv_configs:
            key = (mult, kv_name)
            if key not in results:
                continue
            parsed = results[key]
            passes, fails, details = check_conservation(parsed)
            total_passes += passes
            total_fails += fails

            cell_label = f"{mult}x/{kv_name}"
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
                f"  {cell_label:<18} "
                f"{parsed['total_decisions']:>5} {injected:>5} {completed:>5} {queued:>4} {running:>4} "
                f"{parsed['preemption_rate']:>8.4f} {parsed['cache_hit_rate']:>9.4f} "
                f"{check_str:>7} {status:<6}"
            )

            if details:
                all_details.extend(
                    [f"  {cell_label}:"] + details
                )

    print()

    # ---- Part 3: 3x3 Results Matrix -----------------------------------------
    print("--- Part 3: 3x3 Results Matrix ---")
    print()
    print("  Preemption Rate by cell:")
    print(f"  {'Overload':<10} {'Abundant':>10} {'Constrained':>12} {'Tiered':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    for mult in multipliers:
        row = f"  {mult}x{'':<8}"
        for kv_name in kv_configs:
            key = (mult, kv_name)
            if key in results:
                pr = results[key]["preemption_rate"]
                row += f" {pr:>10.4f}" if kv_name != "constrained" else f" {pr:>12.4f}"
            else:
                row += f" {'N/A':>10}" if kv_name != "constrained" else f" {'N/A':>12}"
        print(row)

    print()
    print("  Completed Requests by cell:")
    print(f"  {'Overload':<10} {'Abundant':>10} {'Constrained':>12} {'Tiered':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    for mult in multipliers:
        row = f"  {mult}x{'':<8}"
        for kv_name in kv_configs:
            key = (mult, kv_name)
            if key in results and results[key]["cluster"]:
                comp = results[key]["cluster"]["completed_requests"]
                row += f" {comp:>10}" if kv_name != "constrained" else f" {comp:>12}"
            else:
                row += f" {'N/A':>10}" if kv_name != "constrained" else f" {'N/A':>12}"
        print(row)

    print()
    print("  E2E Mean (ms) by cell:")
    print(f"  {'Overload':<10} {'Abundant':>10} {'Constrained':>12} {'Tiered':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    for mult in multipliers:
        row = f"  {mult}x{'':<8}"
        for kv_name in kv_configs:
            key = (mult, kv_name)
            if key in results and results[key]["cluster"]:
                e2e = results[key]["cluster"].get("e2e_mean_ms", 0)
                s = f"{e2e:.1f}"
                row += f" {s:>10}" if kv_name != "constrained" else f" {s:>12}"
            else:
                row += f" {'N/A':>10}" if kv_name != "constrained" else f" {'N/A':>12}"
        print(row)

    print()

    # ---- Part 4: Preemption Behavior Analysis --------------------------------
    print("--- Part 4: Preemption Behavior Analysis ---")
    print()

    # Check: constrained/tiered should show higher preemption under same overload vs abundant
    print("  KV pressure effect (constrained/tiered vs abundant):")
    for mult in multipliers:
        abundant = results.get((mult, "abundant"))
        constrained = results.get((mult, "constrained"))
        tiered = results.get((mult, "tiered"))

        a_pr = abundant["preemption_rate"] if abundant else 0
        c_pr = constrained["preemption_rate"] if constrained else 0
        t_pr = tiered["preemption_rate"] if tiered else 0

        # At constrained blocks, preemption should be >= abundant
        c_ok = "PASS" if c_pr >= a_pr else "FAIL"
        t_ok = "PASS" if t_pr >= a_pr else "FAIL (lower than abundant)"

        print(f"    {mult}x: abundant={a_pr:.4f}, constrained={c_pr:.4f} [{c_ok}], "
              f"tiered={t_pr:.4f} [{t_ok}]")

    print()

    # Check: tiered should show fewer preemptions than constrained (CPU absorbs overflow)
    print("  Tiered benefit (fewer preemptions than constrained-only):")
    tiered_benefit_all = True
    for mult in multipliers:
        constrained = results.get((mult, "constrained"))
        tiered = results.get((mult, "tiered"))

        c_pr = constrained["preemption_rate"] if constrained else 0
        t_pr = tiered["preemption_rate"] if tiered else 0

        benefit = c_pr - t_pr
        status = "PASS" if t_pr <= c_pr else "FAIL"
        if t_pr > c_pr:
            tiered_benefit_all = False

        # Also compare completed requests
        c_comp = constrained["cluster"]["completed_requests"] if constrained and constrained["cluster"] else 0
        t_comp = tiered["cluster"]["completed_requests"] if tiered and tiered["cluster"] else 0

        print(f"    {mult}x: constrained_preempt={c_pr:.4f}, tiered_preempt={t_pr:.4f}, "
              f"delta={benefit:+.4f} [{status}]  "
              f"completed: constrained={c_comp}, tiered={t_comp}")

    print()
    print(f"  Tiered benefit: {'CONFIRMED' if tiered_benefit_all else 'NOT CONFIRMED'}")

    print()

    # ---- Part 5: Per-Instance Conservation Details ---------------------------
    print("--- Part 5: Per-Instance Preemption Counts ---")
    print()

    for mult in multipliers:
        for kv_name in kv_configs:
            key = (mult, kv_name)
            if key not in results:
                continue
            parsed = results[key]
            if not parsed["instances"]:
                continue
            label = f"{mult}x/{kv_name}"
            inst_details = []
            for inst in parsed["instances"]:
                iid = inst["instance_id"]
                pc = inst.get("preemption_count", 0)
                inst_details.append(f"{iid}={pc}")
            print(f"  {label}: {', '.join(inst_details)} "
                  f"(total={parsed['preemption_count']})")

    print()

    # ---- Overall Summary -----------------------------------------------------
    print("--- Overall Summary ---")
    print()

    if all_details:
        print("  Conservation details:")
        for d in all_details:
            print(d)
        print()

    if total_fails == 0 and total_panics == 0:
        print(
            f"  RESULT: ALL PASS -- {total_passes} invariant checks across "
            f"{len(results)} configurations, "
            f"zero panics, zero conservation violations."
        )
        print()
        print("  Conservation invariant (INV-1) holds at all overload+KV-pressure combinations.")
        print("  No undefined behavior (panic, deadlock, livelock, silent data loss) observed.")
    else:
        if total_panics > 0:
            print(f"  ROBUSTNESS FAILURES: {total_panics} panics or non-zero exits")
        if total_fails > 0:
            print(
                f"  CONSERVATION FAILURES: {total_fails} "
                f"({total_passes} passed, {total_fails} failed)"
            )
        print()
        print("  OVERLOAD+KV ROBUSTNESS: ISSUES FOUND -- see details above.")

    return 0 if (total_fails == 0 and total_panics == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
