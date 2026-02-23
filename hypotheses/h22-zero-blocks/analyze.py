#!/usr/bin/env python3
"""H22: Zero KV Blocks -- Analyze CLI validation boundary test results.

Reads stderr files and exit codes from the results directory produced by run.sh.
Evaluates each test case against expected behavior and prints a summary table.

Usage: python3 analyze.py <results_dir>
"""

import os
import re
import sys


# Expected behavior per test case:
# Each entry: (description, expect_nonzero_exit, expect_no_panic, expect_fatalf, msg_pattern)
EXPECTATIONS = {
    "zero_kv_blocks": (
        "--total-kv-blocks 0",
        True,   # expect non-zero exit
        True,   # expect no panic
        True,   # expect logrus FATA[]
        r"--total-kv-blocks must be > 0.*got 0",
    ),
    "zero_block_size": (
        "--block-size-in-tokens 0",
        True, True, True,
        r"--block-size-in-tokens must be > 0.*got 0",
    ),
    "negative_kv_blocks": (
        "--total-kv-blocks -1",
        True, True, True,
        r"--total-kv-blocks must be > 0.*got -1",
    ),
    "zero_gpu_with_cpu": (
        "--total-kv-blocks 0 --kv-cpu-blocks 100",
        True, True, True,
        r"--total-kv-blocks must be > 0.*got 0",
    ),
    "negative_cpu_blocks": (
        "--kv-cpu-blocks -1",
        True, True, True,
        r"--kv-cpu-blocks must be >= 0.*got -1",
    ),
    "valid_control": (
        "Valid config (control)",
        False,  # expect exit 0
        True,   # expect no panic
        False,  # expect no FATA[]
        None,   # no message pattern to check
    ),
}

# Order for display
CASE_ORDER = [
    "zero_kv_blocks",
    "zero_block_size",
    "negative_kv_blocks",
    "zero_gpu_with_cpu",
    "negative_cpu_blocks",
    "valid_control",
]


def load_case(results_dir, name):
    """Load a single test case from the results directory."""
    stderr_file = os.path.join(results_dir, f"{name}.stderr")
    exit_code_file = os.path.join(results_dir, f"{name}.exit_code")

    if not os.path.exists(stderr_file):
        return None

    with open(stderr_file) as f:
        stderr = f.read()

    exit_code = 0
    if os.path.exists(exit_code_file):
        with open(exit_code_file) as f:
            exit_code = int(f.read().strip())

    has_panic = bool(re.search(
        r"goroutine [0-9]+|^panic\(|runtime error|runtime\.goexit",
        stderr, re.MULTILINE
    ))

    # logrus uses "FATA[" in TTY mode, "level=fatal" when stderr is redirected
    has_fatalf = bool(re.search(r"FATA\[|level=fatal", stderr))

    # Extract the error message from either format:
    #   TTY:      FATA[0000] --total-kv-blocks must be > 0, got 0
    #   Non-TTY:  time="..." level=fatal msg="--total-kv-blocks must be > 0, got 0"
    fata_match = re.search(r"FATA\[.*?\]\s*(.*)", stderr)
    if not fata_match:
        fata_match = re.search(r'level=fatal\s+msg="([^"]*)"', stderr)
    error_msg = fata_match.group(1).strip() if fata_match else ""

    return {
        "exit_code": exit_code,
        "has_panic": has_panic,
        "has_fatalf": has_fatalf,
        "error_msg": error_msg,
        "stderr_lines": len(stderr.strip().split("\n")) if stderr.strip() else 0,
    }


def evaluate(case, expect):
    """Evaluate a test case. Returns (overall_pass, list_of_check_results).
    Each check result is (pass_bool, description)."""
    desc, expect_nonzero, expect_no_panic, expect_fatalf, msg_pattern = expect
    checks = []

    # Exit code check
    if expect_nonzero:
        ok = case["exit_code"] != 0
        checks.append((ok, f"exit={case['exit_code']} (want != 0)"))
    else:
        ok = case["exit_code"] == 0
        checks.append((ok, f"exit={case['exit_code']} (want 0)"))

    # Panic check
    if expect_no_panic:
        ok = not case["has_panic"]
        checks.append((ok, "no panic" if ok else "PANIC detected"))

    # Fatalf check
    if expect_fatalf:
        ok = case["has_fatalf"]
        checks.append((ok, "logrus fatal present" if ok else "logrus fatal missing"))
    elif not expect_nonzero:
        # Control: should NOT have fatalf
        ok = not case["has_fatalf"]
        checks.append((ok, "no logrus fatal (control)" if ok else "unexpected logrus fatal"))

    # Message pattern check
    if msg_pattern:
        ok = bool(re.search(msg_pattern, case["error_msg"]))
        if ok:
            checks.append((True, "message matches expected pattern"))
        else:
            checks.append((False, f"message mismatch: '{case['error_msg']}'"))

    overall = all(c[0] for c in checks)
    return overall, checks


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]

    print("=" * 78)
    print("  H22: Zero KV Blocks -- Validation Boundary Results")
    print("=" * 78)
    print()

    total_pass = 0
    total_fail = 0
    results_table = []

    for name in CASE_ORDER:
        if name not in EXPECTATIONS:
            continue

        case = load_case(results_dir, name)
        if case is None:
            print(f"  [SKIP] {name}: no results found")
            continue

        expect = EXPECTATIONS[name]
        passed, checks = evaluate(case, expect)

        status = "PASS" if passed else "FAIL"
        if passed:
            total_pass += 1
        else:
            total_fail += 1

        desc = expect[0]
        results_table.append((status, desc, checks, case.get("error_msg", "")))

    # Print detailed results
    for status, desc, checks, msg in results_table:
        print(f"  [{status}] {desc}")
        for ok, detail in checks:
            marker = "+" if ok else "X"
            print(f"         [{marker}] {detail}")
        if msg:
            truncated = msg[:70] + "..." if len(msg) > 70 else msg
            print(f"         msg: {truncated}")
        print()

    # Summary
    print("-" * 78)
    total = total_pass + total_fail
    print(f"  Results: {total_pass}/{total} passed, {total_fail}/{total} failed")
    print()
    if total_fail == 0:
        print("  HYPOTHESIS CONFIRMED")
        print("  All invalid KV configs produce clean CLI errors (logrus.Fatalf).")
        print("  No panics or stack traces reach the user.")
    else:
        print("  HYPOTHESIS REFUTED")
        print("  Some invalid configs not properly validated at CLI boundary.")
    print("-" * 78)

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
