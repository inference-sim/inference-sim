"""
Validate InferSim benchmark data for any GPU type
Checks MFU ranges, data completeness, and format consistency
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import csv


def load_config() -> Dict:
    """Load benchmark configuration"""
    config_path = Path("config/benchmark_config.json")
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        return json.load(f)


def validate_mfu_range(mfu: float, op_type: str, mfu_ranges: Dict) -> Tuple[bool, str]:
    """Validate MFU is within expected range for operation type"""
    if op_type not in mfu_ranges:
        return False, f"Unknown operation type: {op_type}"

    range_config = mfu_ranges[op_type]
    min_mfu = range_config["min"]
    max_mfu = range_config["max"]

    if mfu < min_mfu or mfu > max_mfu:
        return False, f"MFU {mfu:.4f} outside valid range [{min_mfu}, {max_mfu}] - {range_config['description']}"

    return True, "OK"


def validate_csv_file(filepath: Path, required_columns: List[str]) -> Tuple[bool, str, int]:
    """Validate CSV file exists and has required columns"""
    if not filepath.exists():
        return False, f"File not found: {filepath}", 0

    try:
        with open(filepath) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            if not headers:
                return False, "Empty or invalid CSV file", 0

            missing = set(required_columns) - set(headers)
            if missing:
                return False, f"Missing required columns: {missing}", 0

            row_count = sum(1 for _ in reader)
            return True, "OK", row_count

    except Exception as e:
        return False, f"Error reading CSV: {str(e)}", 0


def validate_mha_prefill(data_dir: Path, model_configs: List[Dict], mfu_ranges: Dict) -> Tuple[int, int]:
    """Validate MHA prefill benchmark data"""
    print("\n" + "="*60)
    print("Validating MHA Prefill Data")
    print("="*60)

    passed = 0
    failed = 0
    # Actual columns from fa3_mha_prefill.py
    required_cols = ["dtype", "seq_len", "latency_us", "mfu"]

    for model in model_configs:
        config_key = f"{model['num_attention_heads']}-{model['num_key_value_heads']}-{model['head_dim']}"
        filepath = data_dir / f"{config_key}.csv"

        print(f"\n{model['name']} ({config_key})")

        valid, msg, row_count = validate_csv_file(filepath, required_cols)
        if not valid:
            print(f"  FAIL: {msg}")
            failed += 1
            continue

        print(f"  OK: {row_count} rows")

        # Validate MFU ranges
        try:
            with open(filepath) as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader, 1):
                    mfu = float(row["mfu"])
                    valid, msg = validate_mfu_range(mfu, "prefill", mfu_ranges)
                    if not valid:
                        print(f"  WARNING row {i}: {msg}")
        except Exception as e:
            print(f"  WARNING: Could not validate MFU values: {e}")

        passed += 1

    return passed, failed


def validate_mha_decode(data_dir: Path, model_configs: List[Dict], mfu_ranges: Dict, tp_values: List[int]) -> Tuple[int, int]:
    """Validate MHA decode benchmark data"""
    print("\n" + "="*60)
    print("Validating MHA Decode Data")
    print("="*60)

    passed = 0
    failed = 0
    # Actual columns from flashinfer_mha_decode.py
    required_cols = ["dtype", "kv_dtype", "batch_size", "kv_len", "latency_us", "mfu"]

    for model in model_configs:
        config_key = f"{model['num_attention_heads']}-{model['num_key_value_heads']}-{model['head_dim']}"

        print(f"\n{model['name']} ({config_key})")

        # Validate each TP value
        for tp in tp_values:
            tp_config_key = f"{config_key}-tp{tp}"
            filepath = data_dir / f"{tp_config_key}.csv"

            valid, msg, row_count = validate_csv_file(filepath, required_cols)
            if not valid:
                print(f"  FAIL TP={tp}: {msg}")
                failed += 1
                continue

            print(f"  OK TP={tp}: {row_count} rows")

            # Validate MFU ranges
            try:
                with open(filepath) as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader, 1):
                        mfu = float(row["mfu"])
                        valid, msg = validate_mfu_range(mfu, "decode", mfu_ranges)
                        if not valid:
                            print(f"  WARNING TP={tp} row {i}: {msg}")
            except Exception as e:
                print(f"  WARNING TP={tp}: Could not validate MFU values: {e}")

            passed += 1

    return passed, failed


def validate_gemm_data(data_dir: Path, gemm_sweep: Dict, mfu_ranges: Dict) -> Tuple[bool, str]:
    """Validate GEMM benchmark data"""
    print("\n" + "="*60)
    print("Validating GEMM Data")
    print("="*60)

    filepath = data_dir / "data.csv"
    # Actual columns from deepgemm_gemm.py
    required_cols = ["m", "k", "n", "latency_us", "mfu"]

    valid, msg, row_count = validate_csv_file(filepath, required_cols)
    if not valid:
        print(f"FAIL: {msg}")
        return False, msg

    print(f"OK: {row_count} rows")

    # Note: M values are swept internally by deepgemm_gemm.py, not from config
    # Expected: M sweep (internal) * K values * N values
    # We just check that we have reasonable number of rows
    expected_k_n_combinations = (
        len(gemm_sweep["k_values"]) *
        len(gemm_sweep["n_values"])
    )

    print(f"Expected K×N combinations: {expected_k_n_combinations}")
    print(f"Found: {row_count} rows (M sweep is internal)")

    if row_count < expected_k_n_combinations:
        print(f"WARNING: Too few rows, expected at least {expected_k_n_combinations}")

    # Validate MFU ranges
    try:
        with open(filepath) as f:
            reader = csv.DictReader(f)
            low_mfu_count = 0
            for row in reader:
                mfu = float(row["mfu"])
                valid, msg = validate_mfu_range(mfu, "gemm", mfu_ranges)
                if not valid:
                    low_mfu_count += 1

            if low_mfu_count > 0:
                print(f"WARNING: {low_mfu_count} rows with MFU outside expected range")
    except Exception as e:
        print(f"WARNING: Could not validate MFU values: {e}")

    return True, "OK"


def main():
    """Validate all benchmark data for specified GPU"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Validate InferSim benchmark data")
    parser.add_argument("--gpu", type=str, default="H100",
                       help="GPU type to validate (default: H100)")
    parser.add_argument("--model-filter", type=str,
                       help="Only validate specific model")
    parser.add_argument("--phase-filter", type=str, choices=["prefill", "decode", "gemm"],
                       help="Only validate specific phase")
    parser.add_argument("--tp-filter", type=int, choices=[1, 2, 4],
                       help="Only validate specific TP value (decode only)")
    parser.add_argument("--data-dir", type=Path, default=Path("bench_data"),
                       help="Benchmark data directory (default: bench_data)")
    args = parser.parse_args()

    gpu_name = args.gpu

    print("="*60)
    print(f"{gpu_name} Benchmark Data Validation")
    print("="*60)

    # Load configuration
    config = load_config()

    # Validate GPU exists in config
    if gpu_name not in config["gpu_specs"]:
        print(f"Error: GPU '{gpu_name}' not found in config")
        print(f"Available GPUs: {', '.join(config['gpu_specs'].keys())}")
        sys.exit(1)

    if gpu_name not in config["mfu_validation"]:
        print(f"Error: MFU validation ranges not defined for '{gpu_name}'")
        sys.exit(1)

    if gpu_name not in config["gemm_sweep"]:
        print(f"Error: GEMM sweep not defined for '{gpu_name}'")
        sys.exit(1)

    gpu_lower = gpu_name.lower()
    base_dir = args.data_dir

    if not base_dir.exists():
        print(f"Error: Data directory not found: {base_dir.absolute()}")
        sys.exit(1)

    model_configs = config["benchmark_configs"]
    mfu_ranges = config["mfu_validation"][gpu_name]
    gemm_sweep = config["gemm_sweep"][gpu_name]
    tp_values = config.get("validation_configs", {}).get("tp_values", [1])

    # Apply filters
    if args.model_filter:
        model_configs = [m for m in model_configs if m["name"] == args.model_filter]
        if not model_configs:
            print(f"Error: Model '{args.model_filter}' not found in config")
            sys.exit(1)
        print(f"Filtering to model: {args.model_filter}")

    if args.tp_filter is not None:
        tp_values = [args.tp_filter]
        print(f"Filtering to TP={args.tp_filter}")

    # Determine which phases to validate
    run_prefill = args.phase_filter in [None, "prefill"]
    run_decode = args.phase_filter in [None, "decode"]
    run_gemm = args.phase_filter in [None, "gemm"]

    print(f"Data directory: {base_dir}")
    print(f"GPU: {gpu_name}")
    print(f"Models: {len(model_configs)}")
    print(f"TP values: {tp_values}")
    if args.phase_filter:
        print(f"Phase filter: {args.phase_filter}")

    # Validate all data (directory structure: bench_data/mha/{prefill,decode}/{gpu}/)
    prefill_dir = base_dir / "mha" / "prefill" / gpu_lower
    decode_dir = base_dir / "mha" / "decode" / gpu_lower
    gemm_dir = base_dir / "gemm" / gpu_lower

    prefill_passed = prefill_failed = 0
    decode_passed = decode_failed = 0
    gemm_valid = True

    if run_prefill:
        prefill_passed, prefill_failed = validate_mha_prefill(prefill_dir, model_configs, mfu_ranges)
    else:
        print("\n(Skipping prefill validation - not in filter)")

    if run_decode:
        decode_passed, decode_failed = validate_mha_decode(decode_dir, model_configs, mfu_ranges, tp_values)
    else:
        print("\n(Skipping decode validation - not in filter)")

    if run_gemm:
        gemm_valid, gemm_msg = validate_gemm_data(gemm_dir, gemm_sweep, mfu_ranges)
    else:
        print("\n(Skipping GEMM validation - not in filter)")

    # Summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)

    if run_prefill:
        print(f"MHA Prefill: {prefill_passed} passed, {prefill_failed} failed")
    if run_decode:
        print(f"MHA Decode:  {decode_passed} passed, {decode_failed} failed")
    if run_gemm:
        print(f"GEMM:        {'PASS' if gemm_valid else 'FAIL'}")

    total_tests = 0
    total_passed = 0

    if run_prefill:
        total_tests += prefill_passed + prefill_failed
        total_passed += prefill_passed
    if run_decode:
        total_tests += decode_passed + decode_failed
        total_passed += decode_passed
    if run_gemm:
        total_tests += 1
        total_passed += (1 if gemm_valid else 0)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    has_failures = (run_prefill and prefill_failed > 0) or \
                   (run_decode and decode_failed > 0) or \
                   (run_gemm and not gemm_valid)

    if has_failures:
        print("\nValidation FAILED")
        sys.exit(1)
    else:
        print("\nAll validations PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
