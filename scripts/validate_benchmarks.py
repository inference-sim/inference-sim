#!/usr/bin/env python3
"""Validate H100 benchmark data"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd


SHAPES = [
    "28-4-128",
    "32-32-128",
    "32-8-128",
    "40-40-128",
    "56-8-128",
    "64-8-128",
]


def load_config() -> Dict:
    """Load benchmark configuration"""
    config_path = Path("config/benchmark_config.json")
    if not config_path.exists():
        print(f"✗ Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        return json.load(f)


def validate_csv_structure(path: Path, expected_columns: List[str]) -> List[str]:
    """Validate CSV has required columns"""
    errors = []

    if not path.exists():
        return [f"Missing file: {path}"]

    try:
        df = pd.read_csv(path)

        missing = set(expected_columns) - set(df.columns)
        if missing:
            errors.append(f"{path.name}: Missing columns {missing}")

        if len(df) == 0:
            errors.append(f"{path.name}: Empty dataframe")

        return errors

    except Exception as e:
        return [f"{path.name}: Failed to read - {e}"]


def validate_mfu_range(path: Path, min_mfu: float, max_mfu: float, phase: str) -> List[str]:
    """Validate MFU values are within expected range"""
    errors = []

    try:
        df = pd.read_csv(path)

        if 'mfu' not in df.columns:
            return [f"{path.name}: Missing 'mfu' column"]

        actual_min = df['mfu'].min()
        actual_max = df['mfu'].max()

        if actual_min < min_mfu or actual_max > max_mfu:
            errors.append(
                f"{path.name}: MFU range [{actual_min:.3f}, {actual_max:.3f}] "
                f"outside expected [{min_mfu:.3f}, {max_mfu:.3f}] for {phase}"
            )

        return errors

    except Exception as e:
        return [f"{path.name}: Validation error - {e}"]


def validate_prefill_data(base_path: Path, config: Dict) -> tuple[List[str], int]:
    """Validate prefill benchmark data"""
    errors = []
    success_count = 0

    mfu_range = config["mfu_validation"]["H100"]["prefill"]

    print("\n" + "="*60)
    print("Prefill Validation")
    print("="*60)

    for shape in SHAPES:
        path = base_path / "h100" / "mha" / "prefill" / f"{shape}.csv"

        # Check structure
        errs = validate_csv_structure(path, ["seq_len", "mfu"])
        if errs:
            errors.extend(errs)
            print(f"✗ {shape:12s}: {errs[0]}")
            continue

        # Check MFU range
        errs = validate_mfu_range(path, mfu_range["min"], mfu_range["max"], "prefill")
        if errs:
            errors.extend(errs)
            print(f"✗ {shape:12s}: {errs[0]}")
            continue

        # Success
        df = pd.read_csv(path)
        print(f"✓ {shape:12s}: {len(df):3d} rows, MFU [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")
        success_count += 1

    return errors, success_count


def validate_decode_data(base_path: Path, config: Dict) -> tuple[List[str], int]:
    """Validate decode benchmark data"""
    errors = []
    success_count = 0

    mfu_range = config["mfu_validation"]["H100"]["decode"]

    print("\n" + "="*60)
    print("Decode Validation")
    print("="*60)

    for shape in SHAPES:
        for tp in [1, 2, 4]:
            path = base_path / "h100" / "mha" / "decode" / f"{shape}-tp{tp}.csv"

            # Check structure
            errs = validate_csv_structure(path, ["batch_size", "kv_len", "mfu"])
            if errs:
                errors.extend(errs)
                print(f"✗ {shape:12s} TP={tp}: {errs[0]}")
                continue

            # Check MFU range
            errs = validate_mfu_range(path, mfu_range["min"], mfu_range["max"], "decode")
            if errs:
                errors.extend(errs)
                print(f"✗ {shape:12s} TP={tp}: {errs[0]}")
                continue

            # Success
            df = pd.read_csv(path)
            print(f"✓ {shape:12s} TP={tp}: {len(df):3d} rows, MFU [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")
            success_count += 1

    return errors, success_count


def validate_gemm_data(base_path: Path, config: Dict) -> tuple[List[str], int]:
    """Validate GEMM data"""
    errors = []

    mfu_range = config["mfu_validation"]["H100"]["gemm"]

    print("\n" + "="*60)
    print("GEMM Validation")
    print("="*60)

    path = base_path / "h100" / "gemm" / "data.csv"

    # Check structure
    errs = validate_csv_structure(path, ["m", "k", "n", "mfu"])
    if errs:
        errors.extend(errs)
        print(f"✗ GEMM: {errs[0]}")
        return errors, 0

    # Check MFU range
    errs = validate_mfu_range(path, mfu_range["min"], mfu_range["max"], "gemm")
    if errs:
        errors.extend(errs)
        print(f"✗ GEMM: {errs[0]}")
        return errors, 0

    # Success
    df = pd.read_csv(path)
    print(f"✓ GEMM: {len(df):3d} rows, MFU [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")

    return errors, 1


def main():
    parser = argparse.ArgumentParser(
        description="Validate H100 benchmark data"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("InferSim/bench_data"),
        help="Benchmark data directory (default: InferSim/bench_data)"
    )

    args = parser.parse_args()

    print("="*60)
    print("H100 Benchmark Data Validation")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print()

    # Load config for MFU validation ranges
    config = load_config()

    # Validate each phase
    all_errors = []

    prefill_errors, prefill_count = validate_prefill_data(args.data_dir, config)
    all_errors.extend(prefill_errors)

    decode_errors, decode_count = validate_decode_data(args.data_dir, config)
    all_errors.extend(decode_errors)

    gemm_errors, gemm_count = validate_gemm_data(args.data_dir, config)
    all_errors.extend(gemm_errors)

    # Summary
    total_expected = 6 + 18 + 1  # 6 prefill + 18 decode + 1 gemm
    total_valid = prefill_count + decode_count + gemm_count

    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    print(f"Valid files: {total_valid}/{total_expected}")
    print(f"  Prefill: {prefill_count}/6")
    print(f"  Decode:  {decode_count}/18")
    print(f"  GEMM:    {gemm_count}/1")

    if all_errors:
        print(f"\n❌ FAILED: {len(all_errors)} errors")
        print("="*60)
        for err in all_errors:
            print(f"  • {err}")
        sys.exit(1)
    else:
        print("\n✅ PASSED: All benchmark data valid")
        print("="*60)
        sys.exit(0)


if __name__ == "__main__":
    main()
