#!/usr/bin/env python3
"""H35: Decode Activation Memory Factor Is Inconsequential -- Analysis.

Analyzes the impact of changing the decode activation memory factor (0.75)
on predicted step time across the evaluation grid (bs=1..256, kvLen=128..8192).

Usage:
    python3 analyze.py <output_dir>

Expected files in output_dir:
    raw_output.txt  -- Raw test output from Go test
"""

import re
import sys
from pathlib import Path


def parse_section(lines, start_marker, end_marker):
    """Extract lines between start and end markers (exclusive)."""
    in_section = False
    result = []
    for line in lines:
        if start_marker in line:
            in_section = True
            continue
        if end_marker in line:
            break
        if in_section and line.strip() and not line.startswith("---"):
            result.append(line.strip())
    return result


def parse_key_value(lines, key):
    """Extract a numeric value from KEY=VALUE lines."""
    for line in lines:
        if line.startswith(key + "="):
            val = line.split("=", 1)[1].strip().rstrip("%")
            try:
                return float(val)
            except ValueError:
                return val
    return None


def parse_verdict(lines):
    """Extract verdict fields from H35_VERDICT section."""
    verdict = {}
    section = parse_section(lines, "H35_VERDICT_START", "H35_VERDICT_END")
    for line in section:
        if "=" in line:
            k, v = line.split("=", 1)
            verdict[k.strip()] = v.strip()
    return verdict


def parse_table_rows(section_lines, header_line_idx=0):
    """Parse pipe-delimited table rows into list of dicts."""
    if len(section_lines) <= header_line_idx:
        return []

    headers = [h.strip() for h in section_lines[header_line_idx].split("|") if h.strip()]
    rows = []
    for line in section_lines[header_line_idx + 1:]:
        if "|" not in line:
            continue
        vals = [v.strip() for v in line.split("|") if v.strip()]
        if len(vals) >= len(headers):
            row = {}
            for i, h in enumerate(headers):
                row[h] = vals[i]
            rows.append(row)
    return rows


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze.py <output_dir>", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    raw_output_path = output_dir / "raw_output.txt"

    if not raw_output_path.exists():
        print(f"ERROR: {raw_output_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(raw_output_path) as f:
        lines = f.readlines()
    lines = [l.rstrip("\n") for l in lines]

    # ================================================================
    # Extract key metrics from output
    # ================================================================
    max_activ_frac = parse_key_value(lines, "H35_MAX_ACTIVATION_FRACTION")
    max_step_delta = parse_key_value(lines, "H35_MAX_STEP_DELTA_075_VS_100")
    max_step_bs = parse_key_value(lines, "H35_MAX_STEP_DELTA_BS")
    max_step_kv = parse_key_value(lines, "H35_MAX_STEP_DELTA_KV")
    global_max_delta = parse_key_value(lines, "H35_GLOBAL_MAX_DELTA")
    verdict = parse_verdict(lines)

    # ================================================================
    # Extract activation fraction data
    # ================================================================
    activ_section = parse_section(lines, "H35_ACTIVATION_FRACTION_START", "H35_ACTIVATION_FRACTION_END")

    # ================================================================
    # Extract step time data
    # ================================================================
    step_section = parse_section(lines, "H35_STEPTIME_START", "H35_STEPTIME_END")

    # ================================================================
    # Extract factor sweep data
    # ================================================================
    sweep_section = parse_section(lines, "H35_FACTOR_SWEEP_START", "H35_FACTOR_SWEEP_END")

    # ================================================================
    # Report
    # ================================================================
    print("=" * 90)
    print("  H35: Decode Activation Memory Factor Is Inconsequential -- Analysis")
    print("=" * 90)
    print()

    # --- Summary metrics ---
    print("--- Key Metrics ---")
    print()
    if max_activ_frac is not None:
        print(f"  Max activation fraction of dynamic bytes: {max_activ_frac:.4f}%")
    if max_step_delta is not None:
        print(f"  Max step time delta (0.75 -> 1.00):       {max_step_delta:.4f}%")
    if max_step_bs is not None and max_step_kv is not None:
        print(f"    at operating point:                      bs={int(max_step_bs)}, kvLen={int(max_step_kv)}")
    if global_max_delta is not None:
        print(f"  Max delta across all factors [0.5..1.5]:   {global_max_delta:.4f}%")
    print()

    # --- Activation fraction table ---
    if activ_section:
        print("--- Activation Fraction of Dynamic Memory Bytes ---")
        print()
        # Print header
        print(f"  {'BS':>6} {'KVLen':>8} {'Activ Bytes':>14} {'Dynamic Bytes':>14} {'ActFrac%':>10}")
        print(f"  {'-'*6} {'-'*8} {'-'*14} {'-'*14} {'-'*10}")
        for line in activ_section:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 6:
                try:
                    bs = parts[0]
                    kv = parts[1]
                    ab = parts[2]
                    db = parts[3]
                    tb = parts[4]
                    frac = parts[5]
                    print(f"  {bs:>6} {kv:>8} {ab:>14} {db:>14} {frac:>10}")
                except (ValueError, IndexError):
                    pass
        print()

    # --- Step time comparison table (0.75 vs 1.00) ---
    if step_section:
        print("--- Step Time Comparison: Factor 0.75 vs 1.00 ---")
        print()
        # Count memory-bound vs compute-bound
        mem_bound_count = 0
        cmp_bound_count = 0
        mem_bound_deltas = []
        cmp_bound_deltas = []
        for line in step_section:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 8:
                try:
                    regime = parts[7]
                    delta_str = parts[6].rstrip("%")
                    delta = float(delta_str)
                    if "MEM" in regime:
                        mem_bound_count += 1
                        mem_bound_deltas.append(delta)
                    elif "CMP" in regime:
                        cmp_bound_count += 1
                        cmp_bound_deltas.append(delta)
                except (ValueError, IndexError):
                    pass

        print(f"  Memory-bound operating points: {mem_bound_count}")
        print(f"  Compute-bound operating points: {cmp_bound_count}")
        if mem_bound_deltas:
            print(f"  Memory-bound delta range: [{min(mem_bound_deltas):.4f}%, {max(mem_bound_deltas):.4f}%]")
            print(f"  Memory-bound mean delta:  {sum(mem_bound_deltas)/len(mem_bound_deltas):.4f}%")
        if cmp_bound_deltas:
            print(f"  Compute-bound delta range: [{min(cmp_bound_deltas):.4f}%, {max(cmp_bound_deltas):.4f}%]")
            print(f"  Compute-bound mean delta:  {sum(cmp_bound_deltas)/len(cmp_bound_deltas):.4f}%")
        print()

    # --- Factor sweep summary ---
    if sweep_section:
        print("--- Factor Sweep Summary ---")
        print()
        # Group by factor, compute max delta for each
        factor_deltas = {}
        for line in sweep_section:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 6:
                try:
                    factor = float(parts[2])
                    delta_str = parts[5].rstrip("%")
                    delta = float(delta_str)
                    if factor not in factor_deltas:
                        factor_deltas[factor] = []
                    factor_deltas[factor].append(delta)
                except (ValueError, IndexError):
                    pass

        print(f"  {'Factor':>8} {'Max |Delta|':>12} {'Mean |Delta|':>12} {'Max Delta':>12}")
        print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
        for factor in sorted(factor_deltas.keys()):
            deltas = factor_deltas[factor]
            abs_deltas = [abs(d) for d in deltas]
            print(f"  {factor:>8.2f} {max(abs_deltas):>11.4f}% {sum(abs_deltas)/len(abs_deltas):>11.4f}% {max(deltas):>11.4f}%")
        print()

    # ================================================================
    # Hypothesis Verdict
    # ================================================================
    print("=" * 90)
    print("  Hypothesis Verdict")
    print("=" * 90)
    print()

    if max_step_delta is not None:
        threshold = 0.1
        if abs(max_step_delta) < threshold:
            print(f"  CONFIRMED: The decode activation memory factor is inconsequential.")
            print(f"  Changing the factor from 0.75 to 1.00 shifts step time by at most")
            print(f"  {abs(max_step_delta):.4f}%, which is below the {threshold}% refutation threshold.")
        else:
            print(f"  REFUTED: The decode activation memory factor is NOT inconsequential.")
            print(f"  Changing the factor from 0.75 to 1.00 shifts step time by up to")
            print(f"  {abs(max_step_delta):.4f}% at bs={int(max_step_bs)}, kvLen={int(max_step_kv)},")
            print(f"  which exceeds the {threshold}% refutation threshold.")
    else:
        print("  INCONCLUSIVE: Could not extract max step delta from test output.")

    print()
    if max_activ_frac is not None:
        if max_activ_frac < 0.5:
            print(f"  Activation bytes are at most {max_activ_frac:.4f}% of dynamic memory bytes,")
            print(f"  confirming they are a negligible fraction of total memory traffic.")
        else:
            print(f"  Activation bytes reach up to {max_activ_frac:.4f}% of dynamic memory bytes.")

    if global_max_delta is not None:
        print(f"  Across all factors [0.50, 0.75, 1.00, 1.50], the maximum step time delta")
        print(f"  is {abs(global_max_delta):.4f}%.")

    print()


if __name__ == "__main__":
    main()
