#!/usr/bin/env python3
"""
Run InferSim benchmarks for specific shape/phase/TP combinations.
Called by OpenShift jobs with filtered parameters.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


def parse_shape(shape_str: str) -> tuple[int, int, int]:
    """Parse shape string like '32-8-128' into (nh, nkv, dh)"""
    try:
        parts = shape_str.split("-")
        if len(parts) != 3:
            raise ValueError("Shape must have 3 components")
        nh, nkv, dh = map(int, parts)
        return nh, nkv, dh
    except (ValueError, AttributeError) as e:
        print(f"Error: Invalid shape format '{shape_str}': {e}")
        print("Expected format: nh-nkv-dh (e.g., 32-8-128)")
        sys.exit(1)


def load_config() -> dict:
    """Load benchmark configuration for GPU specs and validation"""
    config_path = Path("config/benchmark_config.json")
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        return json.load(f)


def create_temp_model_config(nh: int, nkv: int, dh: int) -> Path:
    """Create temporary HuggingFace-style config for InferSim scripts"""
    hidden_size = nh * dh

    config = {
        "hidden_size": hidden_size,
        "num_attention_heads": nh,
        "num_key_value_heads": nkv,
        "head_dim": dh,
        "torch_dtype": "bfloat16"
    }

    shape_str = f"{nh}-{nkv}-{dh}"
    config_path = Path(f"/tmp/model_config_{shape_str}.json")
    config_path.write_text(json.dumps(config, indent=2))

    print(f"Created temp config: {config_path}")
    return config_path


def run_prefill_benchmark(nh: int, nkv: int, dh: int, gpu_type: str, gpu_specs: dict):
    """Run prefill benchmark using fa3_mha_prefill.py"""
    print(f"\n{'='*60}")
    print(f"Running Prefill Benchmark: {nh}-{nkv}-{dh}")
    print(f"{'='*60}")

    config_path = create_temp_model_config(nh, nkv, dh)

    cmd = [
        sys.executable,
        "kernel_benchmark/fa3_mha_prefill.py",
        "--config-path", str(config_path.absolute()),
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="InferSim", capture_output=True, text=True)

    if result.returncode != 0:
        print(f"✗ Prefill benchmark failed:")
        print(result.stderr)
        sys.exit(1)

    print(result.stdout)

    # Move output to final location
    shape_str = f"{nh}-{nkv}-{dh}"
    output_dir = Path(f"InferSim/bench_data/{gpu_type.lower()}/mha/prefill")
    output_dir.mkdir(parents=True, exist_ok=True)

    src = Path("InferSim/attention_benchmark.csv")
    dst = output_dir / f"{shape_str}.csv"

    if src.exists():
        src.rename(dst)
        print(f"✓ Saved: {dst}")
    else:
        print(f"✗ Output file not found: {src}")
        sys.exit(1)


def run_decode_benchmark(nh: int, nkv: int, dh: int, tp: int, gpu_type: str, gpu_specs: dict):
    """Run decode benchmark using flashinfer_mha_decode.py"""
    print(f"\n{'='*60}")
    print(f"Running Decode Benchmark: {nh}-{nkv}-{dh} TP={tp}")
    print(f"{'='*60}")

    config_path = create_temp_model_config(nh, nkv, dh)
    peak_tflops = gpu_specs["peak_tflops_fp16"]

    cmd = [
        sys.executable,
        "kernel_benchmark/flashinfer_mha_decode.py",
        "--config-path", str(config_path.absolute()),
        "--fp16-tflops", str(peak_tflops),
        "--kv-cache-dtype", "bf16",
        "--tp-size", str(tp),
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="InferSim", capture_output=True, text=True)

    if result.returncode != 0:
        print(f"✗ Decode benchmark failed:")
        print(result.stderr)
        sys.exit(1)

    print(result.stdout)

    # Move output to final location
    shape_str = f"{nh}-{nkv}-{dh}"
    output_dir = Path(f"InferSim/bench_data/{gpu_type.lower()}/mha/decode")
    output_dir.mkdir(parents=True, exist_ok=True)

    src = Path("InferSim/attention_benchmark.csv")
    dst = output_dir / f"{shape_str}-tp{tp}.csv"

    if src.exists():
        src.rename(dst)
        print(f"✓ Saved: {dst}")
    else:
        print(f"✗ Output file not found: {src}")
        sys.exit(1)


def run_gemm_benchmark(gpu_type: str, gpu_specs: dict):
    """Run GEMM benchmark using deepgemm_gemm.py"""
    print(f"\n{'='*60}")
    print(f"Running GEMM Benchmark")
    print(f"{'='*60}")

    config = load_config()
    gemm_sweep = config["gemm_sweep"]["H100"]
    k_values = gemm_sweep["k_values"]
    n_values = gemm_sweep["n_values"]
    peak_tflops = gpu_specs["peak_tflops_fp16"]

    output_dir = Path(f"InferSim/bench_data/{gpu_type.lower()}/gemm")
    output_dir.mkdir(parents=True, exist_ok=True)
    gemm_output = output_dir / "data.csv"

    # Run GEMM sweeps
    gemm_tmp = Path("InferSim/gemm.csv")
    first = True

    for k in k_values:
        for n in n_values:
            print(f"  GEMM: K={k}, N={n}")

            cmd = [
                sys.executable,
                "kernel_benchmark/deepgemm_gemm.py",
                "-k", str(k),
                "-n", str(n),
                "--gpu-tflops", str(peak_tflops),
            ]

            result = subprocess.run(cmd, cwd="InferSim", capture_output=True, text=True)

            if result.returncode != 0:
                print(f"  ✗ GEMM K={k} N={n} failed: {result.stderr}")
                continue

            # Append to output file
            if gemm_tmp.exists():
                if first:
                    # Copy with header
                    gemm_output.write_text(gemm_tmp.read_text())
                    first = False
                else:
                    # Append without header
                    with open(gemm_tmp) as f:
                        lines = f.readlines()[1:]  # Skip header
                    with open(gemm_output, 'a') as f:
                        f.writelines(lines)
                gemm_tmp.unlink()

    if gemm_output.exists():
        print(f"✓ Saved: {gemm_output}")
    else:
        print(f"✗ No GEMM data generated")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run InferSim benchmarks for specific shape/phase/TP"
    )
    parser.add_argument("--gpu", type=str, required=True, help="GPU type (e.g., H100)")
    parser.add_argument("--shape", type=str, help="Attention shape: nh-nkv-dh")
    parser.add_argument("--phase-filter", type=str, choices=["prefill", "decode", "gemm"], help="Phase to run")
    parser.add_argument("--tp-filter", type=int, choices=[1, 2, 4], help="TP value (decode only)")

    args = parser.parse_args()

    # Load GPU specs
    config = load_config()
    gpu_specs = config["gpu_specs"].get(args.gpu)
    if not gpu_specs:
        print(f"Error: GPU type {args.gpu} not found in config")
        sys.exit(1)

    print(f"GPU: {args.gpu} ({gpu_specs['peak_tflops_fp16']} TFLOPs)")

    # Route to appropriate benchmark
    if args.phase_filter == "gemm":
        run_gemm_benchmark(args.gpu, gpu_specs)

    elif args.phase_filter == "prefill":
        if not args.shape:
            print("Error: --shape required for prefill")
            sys.exit(1)
        nh, nkv, dh = parse_shape(args.shape)
        run_prefill_benchmark(nh, nkv, dh, args.gpu, gpu_specs)

    elif args.phase_filter == "decode":
        if not args.shape or args.tp_filter is None:
            print("Error: --shape and --tp-filter required for decode")
            sys.exit(1)
        nh, nkv, dh = parse_shape(args.shape)
        run_decode_benchmark(nh, nkv, dh, args.tp_filter, args.gpu, gpu_specs)

    else:
        print("Error: --phase-filter required (prefill, decode, or gemm)")
        sys.exit(1)

    print("\n✅ Benchmark complete")


if __name__ == "__main__":
    main()
