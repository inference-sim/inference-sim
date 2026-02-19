#!/usr/bin/env python3
"""Collect benchmark results from PVC (via pvc-debugger pod)"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_oc_command(args, check=True):
    """Run oc command"""
    cmd = ["oc"] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def check_pvc_debugger_exists(namespace: str = "diya") -> bool:
    """Check if pvc-debugger pod exists and is running"""
    result = run_oc_command([
        "get", "pod", "pvc-debugger", "-n", namespace,
        "-o", "jsonpath={.status.phase}"
    ], check=False)

    if result.returncode != 0:
        return False

    return result.stdout.strip() == "Running"


def check_pvc_directory_exists(namespace: str = "diya") -> bool:
    """Check if bench_data directory exists on PVC"""
    result = run_oc_command([
        "exec", "pvc-debugger", "-n", namespace, "--",
        "test", "-d", "/mnt/bench_data"
    ], check=False)
    return result.returncode == 0


def copy_results_from_pvc(local_dir: Path, namespace: str = "diya") -> bool:
    """Copy benchmark data from PVC via pvc-debugger pod"""
    print(f"  Copying from PVC via pvc-debugger...")

    # Check if bench_data directory exists on PVC
    if not check_pvc_directory_exists(namespace):
        print(f"    ✗ /mnt/bench_data/ does not exist on PVC")
        print(f"    Run jobs first to generate benchmark data")
        return False

    # Create local directory
    local_dir.mkdir(parents=True, exist_ok=True)

    # Copy data directory from PVC (/mnt/bench_data/)
    # Using oc cp since rsync may not be available in container
    result = subprocess.run([
        "oc", "cp",
        f"{namespace}/pvc-debugger:/mnt/bench_data/",
        str(local_dir) + "/",
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    ✗ Failed: {result.stderr}")
        return False

    print(f"    ✓ Success")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Collect benchmark results from OpenShift pods"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="diya",
        help="OpenShift namespace (default: diya)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bench_data"),
        help="Local output directory (default: bench_data)"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Collecting Benchmark Results from PVC")
    print("="*60)
    print(f"Namespace: {args.namespace}")
    print(f"Output: {args.output_dir.absolute()}")
    print()

    # Check pvc-debugger pod exists
    print("Checking pvc-debugger pod...")
    if not check_pvc_debugger_exists(args.namespace):
        print("✗ pvc-debugger pod not found or not running")
        print("\nTo create the pvc-debugger pod, run:")
        print("  oc apply -f debug-pod.yaml")
        sys.exit(1)

    print("✓ pvc-debugger pod is running")
    print()

    # Collect from PVC
    print("Copying results from PVC...")
    if not copy_results_from_pvc(args.output_dir, args.namespace):
        print("\n✗ Failed to collect results from PVC")
        sys.exit(1)

    # Summary
    print()
    print("="*60)
    print(f"✓ Results collected from PVC")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")
    print()
    print("Next step: python scripts/validate_benchmark_data.py --gpu H100")


if __name__ == "__main__":
    main()
