#!/usr/bin/env python3
"""H34: Decode GEMM Time Underestimate Below MFU Grid Minimum -- Analysis.

Analyzes the output of the H34 Go test suite to determine whether
computeTransformerGEMMTimes underestimates decode GEMM time at batch sizes
below the MFU grid minimum (M < 8) due to MFU clamping.

Key analysis:
  1. GEMM time ratio bs=1 / bs=8 across all models
  2. MFU clamping verification (same MFU for all bs < grid min)
  3. Per-GEMM breakdown showing uniform scaling
  4. Regime analysis showing whether GEMM underestimate affects step time
  5. Impact quantification if a constant GEMM floor were applied

Usage:
    python3 analyze.py <output_dir>

Expected files in output_dir:
    raw_output.txt -- Full Go test output
"""

import re
import sys
from pathlib import Path


def parse_key_ratios(lines):
    """Extract H34_KEY_RATIO lines: model, ratio, refuted."""
    results = []
    pattern = re.compile(
        r"H34_KEY_RATIO model=(\S+) ratio_bs1_vs_bs8=([\d.]+) refuted=(\w+)"
    )
    for line in lines:
        m = pattern.search(line)
        if m:
            results.append({
                "model": m.group(1),
                "ratio": float(m.group(2)),
                "refuted": m.group(3) == "true",
            })
    return results


def parse_summary_block(lines):
    """Extract the H34_SUMMARY block with per-model data."""
    results = []
    in_block = False
    for line in lines:
        if "H34_SUMMARY_START" in line:
            in_block = True
            continue
        if "H34_SUMMARY_END" in line:
            break
        if not in_block:
            continue
        # Skip header/separator lines
        if line.startswith("---") or "Model" in line:
            continue
        parts = line.split()
        if len(parts) >= 6:
            try:
                results.append({
                    "model": parts[0],
                    "gemm_bs1_us": float(parts[1]),
                    "gemm_bs8_us": float(parts[2]),
                    "ratio": float(parts[3]),
                    "underest_x": float(parts[4]),
                    "verdict": parts[5],
                })
            except (ValueError, IndexError):
                pass
    return results


def parse_clamping_block(lines):
    """Extract MFU clamping data from H34_CLAMPING block."""
    results = []
    in_block = False
    current_gemm = None
    for line in lines:
        if "H34_CLAMPING_START" in line:
            in_block = True
            continue
        if "H34_CLAMPING_END" in line:
            break
        if not in_block:
            continue
        if line.startswith("---"):
            # Parse GEMM name
            m = re.match(r"--- (\S+)", line)
            if m:
                current_gemm = m.group(1)
            continue
        parts = line.split()
        if len(parts) >= 3 and current_gemm:
            try:
                bs = int(parts[0])
                mfu = float(parts[1])
                status = parts[2]
                results.append({
                    "gemm": current_gemm,
                    "bs": bs,
                    "mfu": mfu,
                    "status": status,
                })
            except (ValueError, IndexError):
                pass
    return results


def parse_regime_block(lines):
    """Extract regime analysis data from H34_REGIME block."""
    results = []
    in_block = False
    for line in lines:
        if "H34_REGIME_START" in line:
            in_block = True
            continue
        if "H34_REGIME_END" in line:
            break
        if not in_block:
            continue
        parts = line.split()
        if len(parts) >= 6:
            try:
                results.append({
                    "bs": int(parts[0]),
                    "gemm_us": float(parts[1]),
                    "attn_us": float(parts[2]),
                    "memory_us": float(parts[3]),
                    "step_us": int(parts[4]),
                    "regime": parts[5],
                })
            except (ValueError, IndexError):
                pass
    return results


def parse_verdict(lines):
    """Extract the overall verdict."""
    for line in lines:
        if "H34_VERDICT=" in line:
            return line.split("=", 1)[1].strip()
    return "UNKNOWN"


def mean(values):
    """Compute mean of a list of numbers."""
    if not values:
        return 0
    return sum(values) / len(values)


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze.py <output_dir>", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    raw_path = output_dir / "raw_output.txt"

    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found", file=sys.stderr)
        sys.exit(1)

    lines = raw_path.read_text().splitlines()

    key_ratios = parse_key_ratios(lines)
    summary = parse_summary_block(lines)
    clamping = parse_clamping_block(lines)
    regime = parse_regime_block(lines)
    verdict = parse_verdict(lines)

    # ================================================================
    # Report
    # ================================================================
    print("=" * 90)
    print("  H34: Decode GEMM Time Underestimate Below MFU Grid Minimum -- Analysis")
    print("=" * 90)
    print()

    # --- Table 1: Multi-Model Summary ---
    if summary:
        print("--- Table 1: GEMM Time Ratio bs=1/bs=8 Across Models ---")
        print()
        print(f"{'Model':<18} {'GEMM bs=1(us)':>14} {'GEMM bs=8(us)':>14} {'Ratio':>10} {'Underest':>10} {'Verdict':>10}")
        print("-" * 80)
        for r in summary:
            print(f"{r['model']:<18} {r['gemm_bs1_us']:>14.1f} {r['gemm_bs8_us']:>14.1f} {r['ratio']:>10.4f} {r['underest_x']:>9.1f}x {r['verdict']:>10}")
        print()

        ratios = [r["ratio"] for r in summary]
        print(f"  Mean ratio across models: {mean(ratios):.6f}")
        print(f"  Min ratio:  {min(ratios):.6f}")
        print(f"  Max ratio:  {max(ratios):.6f}")
        print(f"  Expected if perfectly linear: 0.125000 (1/8)")
        print()

    # --- Table 2: Key Ratio Check ---
    if key_ratios:
        print("--- Table 2: Refutation Check (threshold = 0.25) ---")
        print()
        confirmed = [r for r in key_ratios if not r["refuted"]]
        refuted = [r for r in key_ratios if r["refuted"]]
        print(f"  Models confirming hypothesis (ratio <= 0.25): {len(confirmed)}")
        print(f"  Models refuting hypothesis (ratio > 0.25):    {len(refuted)}")
        for r in key_ratios:
            status = "REFUTED" if r["refuted"] else "CONFIRMED"
            print(f"    {r['model']:<18} ratio={r['ratio']:.4f}  {status}")
        print()

    # --- Table 3: MFU Clamping ---
    if clamping:
        print("--- Table 3: MFU Clamping Verification ---")
        print()
        # Group by GEMM
        gemm_names = []
        seen = set()
        for c in clamping:
            if c["gemm"] not in seen:
                gemm_names.append(c["gemm"])
                seen.add(c["gemm"])

        for gname in gemm_names:
            entries = [c for c in clamping if c["gemm"] == gname]
            sub_grid = [e for e in entries if e["bs"] < 8]
            all_same = all(e["status"] == "SAME" for e in sub_grid)
            print(f"  {gname}: sub-grid bs<8 all clamped = {all_same}")
            if not all_same:
                for e in sub_grid:
                    if e["status"] != "SAME":
                        print(f"    bs={e['bs']} MFU={e['mfu']:.6f} ({e['status']})")
        print()

    # --- Table 4: Regime Analysis ---
    if regime:
        print("--- Table 4: Compute vs Memory Regime at Small Batch Sizes ---")
        print()
        print(f"{'BS':>6} {'GEMM(us)':>12} {'Attn(us)':>12} {'Memory(us)':>12} {'Step(us)':>12} {'Regime':>8}")
        print("-" * 66)
        for r in regime:
            print(f"{r['bs']:>6} {r['gemm_us']:>12.1f} {r['attn_us']:>12.1f} {r['memory_us']:>12.1f} {r['step_us']:>12} {r['regime']:>8}")
        print()

        mem_bound = [r for r in regime if r["regime"] == "MEM"]
        comp_bound = [r for r in regime if r["regime"] == "COMP"]
        print(f"  Memory-bound batch sizes: {[r['bs'] for r in mem_bound]}")
        print(f"  Compute-bound batch sizes: {[r['bs'] for r in comp_bound]}")

        if mem_bound:
            print()
            print("  Note: For memory-bound batch sizes, the GEMM underestimate is")
            print("  absorbed by max(compute, memory), so it does NOT affect step time.")
            print("  The underestimate only matters when the step is compute-bound.")
        print()

    # ================================================================
    # Hypothesis Mechanism Analysis
    # ================================================================
    print("=" * 90)
    print("  Mechanism Analysis")
    print("=" * 90)
    print()
    print("  The MFU clamping mechanism works as follows:")
    print("  1. GetGEMMmfu looks up M in the GEMM benchmark grid")
    print("  2. For m <= grid minimum (typically M=8), MFU is clamped to grid_min MFU")
    print("  3. GEMM time = 2*m*k*n / (peakFlops * mfu)")
    print("  4. Since mfu is constant for all m <= 8, time is proportional to m")
    print("  5. Therefore gemmTime(bs=1) / gemmTime(bs=8) = 1/8 = 0.125")
    print()
    print("  In reality, small-M GEMM kernels are memory-bandwidth bound:")
    print("  - The full weight matrix must be loaded regardless of M")
    print("  - Kernel launch overhead is amortized over fewer operations")
    print("  - Actual latency has a near-constant floor for M=1..8")
    print()

    # ================================================================
    # Verdict
    # ================================================================
    print("=" * 90)
    print("  Hypothesis Verdict")
    print("=" * 90)
    print()

    if verdict == "CONFIRMED":
        print("  OVERALL: HYPOTHESIS CONFIRMED")
        print()
        print("  The predicted GEMM time ratio bs=1/bs=8 is approximately 0.125 (1/8)")
        print("  across all models, confirming that computeTransformerGEMMTimes scales")
        print("  linearly with batch size below the MFU grid minimum. This represents")
        print("  up to 8x underestimate of actual GEMM time at bs=1.")
        print()
        print("  The root cause is MFU clamping in GetGEMMmfu: for m below the grid")
        print("  minimum, the function returns a constant MFU, so GEMM time depends")
        print("  only on the FLOPs (which scale linearly with batch size).")
    elif verdict == "REFUTED":
        print("  OVERALL: HYPOTHESIS REFUTED")
        print()
        print("  The predicted GEMM time ratio bs=1/bs=8 exceeds 0.25, indicating")
        print("  that the MFU database already compensates for the memory-bound")
        print("  floor at small batch sizes.")
    else:
        print(f"  OVERALL: {verdict}")
        print()
        print("  Mixed results across models. See per-model breakdown above.")

    print()


if __name__ == "__main__":
    main()
