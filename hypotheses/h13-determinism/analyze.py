#!/usr/bin/env python3
"""Analysis script for H13: Determinism Invariant.

Compares pairs of runs (same configuration, same seed) for byte-identical stdout.
Any difference is a non-determinism bug (INV-6 violation).

Usage:
    python3 analyze.py <results_dir>
"""

import sys
from pathlib import Path


def find_run_pairs(results_dir):
    """Find all (run1, run2) pairs in the results directory."""
    files = sorted(Path(results_dir).glob("*.txt"))
    pairs = {}
    for f in files:
        # Filename pattern: cfg01_rr_run1.txt, cfg01_rr_run2.txt
        stem = f.stem
        if "_run1" in stem:
            base = stem.replace("_run1", "")
            partner = f.parent / f"{base}_run2.txt"
            if partner.exists():
                pairs[base] = (f, partner)
        # Skip run2 files — they're found via run1
    return pairs


CONFIG_LABELS = {
    "cfg01_rr": "round-robin + fcfs (simplest)",
    "cfg02_ll": "least-loaded + fcfs (PendingRequests)",
    "cfg03_weighted": "weighted (qd:2,kv:2) (scorer pipeline)",
    "cfg04_prefix": "weighted (pa:3,qd:2,kv:2) (stateful scorer)",
    "cfg05_priority": "least-loaded + priority-fcfs + slo-based",
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    pairs = find_run_pairs(results_dir)

    if not pairs:
        print("  ERROR: No run pairs found in results directory")
        sys.exit(1)

    total_pass = 0
    total_fail = 0
    failures = []

    # Header
    print(
        f"  {'#':<4} {'Configuration':<50} "
        f"{'Run1 Size':>10} {'Run2 Size':>10} {'Status':<6}"
    )
    print(f"  {'-'*4} {'-'*50} {'-'*10} {'-'*10} {'-'*6}")

    for i, (base, (f1, f2)) in enumerate(sorted(pairs.items()), 1):
        content1 = f1.read_text()
        content2 = f2.read_text()

        label = CONFIG_LABELS.get(base, base)
        size1 = len(content1)
        size2 = len(content2)

        if content1 == content2:
            total_pass += 1
            status = "PASS"
        else:
            total_fail += 1
            status = "FAIL"
            # Find first differing line for diagnostics
            lines1 = content1.splitlines()
            lines2 = content2.splitlines()
            diff_line = None
            for j, (l1, l2) in enumerate(zip(lines1, lines2), 1):
                if l1 != l2:
                    diff_line = j
                    break
            if diff_line is None and len(lines1) != len(lines2):
                diff_line = min(len(lines1), len(lines2)) + 1

            failures.append({
                "config": label,
                "diff_line": diff_line,
                "lines1": len(lines1),
                "lines2": len(lines2),
                "size1": size1,
                "size2": size2,
            })

        print(
            f"  {i:<4} {label:<50} "
            f"{size1:>10} {size2:>10} {status:<6}"
        )

    # Summary
    print()
    if total_fail == 0:
        print(
            f"  RESULT: ALL PASS — {total_pass} configuration pairs,"
            f" byte-identical output."
        )
        print()
        print(
            "  Determinism invariant (INV-6) holds for all tested"
            " configurations."
        )
        print(
            "  Same seed produces identical stdout across runs,"
            " including stateful"
        )
        print("  scorers (prefix-affinity with LRU eviction).")
    else:
        print(
            f"  RESULT: {total_fail} FAILURES"
            f" ({total_pass} passed, {total_fail} failed)"
        )
        print()
        print("  Non-determinism detected — INV-6 VIOLATED:")
        for fail in failures:
            print(f"    - {fail['config']}:")
            print(
                f"      First diff at line {fail['diff_line']}"
                f" (run1: {fail['lines1']} lines,"
                f" run2: {fail['lines2']} lines)"
            )
            print(
                f"      Size: run1={fail['size1']} bytes,"
                f" run2={fail['size2']} bytes"
            )
        print()
        print(
            "  Likely causes: unguarded map iteration (R2),"
            " floating-point accumulation"
        )
        print(
            "  order dependency, or stateful scorer with"
            " non-deterministic internal state."
        )

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
