#!/usr/bin/env python3
"""
Run InferSim benchmarks for specific shape/phase/TP combinations.
Called by OpenShift jobs with filtered parameters.
"""
import argparse
import json
import subprocess
import sys
import tempfile
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


def create_temp_model_config(nh: int, nkv: int, dh: int) -> tempfile.NamedTemporaryFile:
    """Create temporary HuggingFace-style config for InferSim scripts

    Returns NamedTemporaryFile object that caller must manage.
    File will be auto-deleted when object is closed/garbage collected.
    """
    hidden_size = nh * dh

    # Note: ModelConfig requires num_hidden_layers and intermediate_size even though
    # kernel benchmarks only test attention (not FFN). These are dummy values.
    # intermediate_size uses standard 4x expansion from transformer architecture
    # (GPT/Llama use hidden → 4*hidden → hidden in FFN layers)
    config = {
        "hidden_size": hidden_size,
        "num_hidden_layers": 32,  # Dummy value, not used by kernel benchmarks
        "num_attention_heads": nh,
        "num_key_value_heads": nkv,
        "head_dim": dh,
        "intermediate_size": hidden_size * 4,  # Standard 4x FFN expansion (not used by attention benchmarks)
        "torch_dtype": "bfloat16"
    }

    # Create temp file with delete=False so we control cleanup
    # suffix ensures it's recognizable
    temp_file = tempfile.NamedTemporaryFile(
        mode='w',
        suffix=f'_model_config_{nh}-{nkv}-{dh}.json',
        delete=False
    )
    temp_file.write(json.dumps(config, indent=2))
    temp_file.flush()

    print(f"Created temp config: {temp_file.name}")
    return temp_file


def run_prefill_benchmark(nh: int, nkv: int, dh: int, gpu_type: str, gpu_specs: dict):
    """Run prefill benchmark using fa3_mha_prefill.py"""
    print(f"\n{'='*60}")
    print(f"Running Prefill Benchmark: {nh}-{nkv}-{dh}")
    print(f"{'='*60}")

    temp_config = create_temp_model_config(nh, nkv, dh)

    try:
        cmd = [
            sys.executable,
            "kernel_benchmark/fa3_mha_prefill.py",
            "--config-path", temp_config.name,
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
        output_dir = Path(f"InferSim/bench_data/mha/prefill/{gpu_type.lower()}")
        output_dir.mkdir(parents=True, exist_ok=True)

        src = Path("InferSim/attention_benchmark.csv")
        dst = output_dir / f"{shape_str}.csv"

        if src.exists():
            src.rename(dst)
            print(f"✓ Saved: {dst}")
        else:
            print(f"✗ Output file not found: {src}")
            sys.exit(1)
    finally:
        # Clean up temp config file
        temp_config.close()
        Path(temp_config.name).unlink(missing_ok=True)


def run_decode_benchmark(nh: int, nkv: int, dh: int, tp: int, gpu_type: str, gpu_specs: dict):
    """Run decode benchmark using flashinfer_mha_decode.py"""
    print(f"\n{'='*60}")
    print(f"Running Decode Benchmark: {nh}-{nkv}-{dh} TP={tp}")
    print(f"{'='*60}")

    temp_config = create_temp_model_config(nh, nkv, dh)
    peak_tflops = gpu_specs["peak_tflops_fp16"]

    try:
        cmd = [
            sys.executable,
            "kernel_benchmark/flashinfer_mha_decode.py",
            "--config-path", temp_config.name,
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
        output_dir = Path(f"InferSim/bench_data/mha/decode/{gpu_type.lower()}")
        output_dir.mkdir(parents=True, exist_ok=True)

        src = Path("InferSim/attention_benchmark.csv")
        dst = output_dir / f"{shape_str}-tp{tp}.csv"

        if src.exists():
            src.rename(dst)
            print(f"✓ Saved: {dst}")
        else:
            print(f"✗ Output file not found: {src}")
            sys.exit(1)
    finally:
        # Clean up temp config file
        temp_config.close()
        Path(temp_config.name).unlink(missing_ok=True)


def run_gemm_benchmark(gpu_type: str, gpu_specs: dict):
    """Run GEMM benchmark using deepgemm_gemm.py"""
    print(f"\n{'='*60}")
    print(f"Running GEMM Benchmark")
    print(f"{'='*60}")

    config = load_config()

    # Validate gpu_type exists in gemm_sweep config
    if gpu_type not in config["gemm_sweep"]:
        print(f"Error: GPU type '{gpu_type}' not found in gemm_sweep config")
        print(f"Available GPU types: {list(config['gemm_sweep'].keys())}")
        sys.exit(1)

    gemm_sweep = config["gemm_sweep"][gpu_type]
    k_values = gemm_sweep["k_values"]
    n_values = gemm_sweep["n_values"]
    peak_tflops = gpu_specs["peak_tflops_fp16"]

    output_dir = Path(f"InferSim/bench_data/gemm/{gpu_type.lower()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    gemm_output = output_dir / "data.csv"

    # Run GEMM sweeps
    gemm_tmp = Path("InferSim/gemm.csv")
    first_success = True  # Track first successful write, not first attempt

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
                if first_success:
                    # Copy with header on first successful write
                    gemm_output.write_text(gemm_tmp.read_text())
                    first_success = False
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

    # Validate InferSim directory exists
    infersim_dir = Path("InferSim")
    if not infersim_dir.exists() or not infersim_dir.is_dir():
        print(f"Error: InferSim directory not found at {infersim_dir.absolute()}")
        print("Please run this script from the inference-sim root directory")
        sys.exit(1)

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
