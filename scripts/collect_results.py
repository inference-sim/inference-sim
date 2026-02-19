#!/usr/bin/env python3
"""Collect benchmark results from completed OpenShift jobs"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_oc_command(args, check=True):
    """Run oc command"""
    cmd = ["oc"] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def find_benchmark_pods(namespace: str = "diya") -> list[str]:
    """Find all infersim benchmark pods"""
    result = run_oc_command([
        "get", "pods", "-n", namespace,
        "-l", "app=infersim-benchmark",
        "-o", "jsonpath={.items[*].metadata.name}"
    ], check=False)

    if result.returncode != 0:
        print(f"✗ Failed to list pods: {result.stderr}")
        return []

    pods = result.stdout.strip().strip("'").split()
    return [p for p in pods if p]  # Filter empty strings


def get_pod_status(pod_name: str, namespace: str = "diya") -> str:
    """Get pod phase (Succeeded, Failed, Running, etc.)"""
    result = run_oc_command([
        "get", "pod", pod_name, "-n", namespace,
        "-o", "jsonpath={.status.phase}"
    ], check=False)

    if result.returncode != 0:
        return "Unknown"

    return result.stdout.strip()


def copy_results_from_pod(pod_name: str, local_dir: Path, namespace: str = "diya") -> bool:
    """Copy benchmark data from pod to local directory"""
    print(f"  Copying from {pod_name}...")

    # Create local directory
    local_dir.mkdir(parents=True, exist_ok=True)

    # Copy data directory from pod-local storage (jobs clone to /workspace/inference-sim)
    result = subprocess.run([
        "oc", "rsync", "-n", namespace,
        f"{pod_name}:/workspace/inference-sim/InferSim/bench_data/",
        str(local_dir) + "/"
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
    print("Collecting Benchmark Results")
    print("="*60)
    print(f"Namespace: {args.namespace}")
    print(f"Output: {args.output_dir.absolute()}")
    print()

    # Find pods
    print("Finding benchmark pods...")
    pods = find_benchmark_pods(args.namespace)

    if not pods:
        print("✗ No benchmark pods found")
        sys.exit(1)

    print(f"✓ Found {len(pods)} pods")

    # Filter to completed pods
    completed_pods = []
    for pod in pods:
        status = get_pod_status(pod, args.namespace)
        if status == "Succeeded":
            completed_pods.append(pod)
        else:
            print(f"  Skipping {pod}: {status}")

    if not completed_pods:
        print("\n✗ No completed pods to collect from")
        sys.exit(1)

    print(f"\n✓ {len(completed_pods)} completed pods")
    print()

    # Collect from each pod
    print("Copying results...")
    success_count = 0
    for pod in completed_pods:
        if copy_results_from_pod(pod, args.output_dir, args.namespace):
            success_count += 1

    # Summary
    print()
    print("="*60)
    print(f"✓ Collected from {success_count}/{len(completed_pods)} pods")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")
    print()
    print("Next step: python scripts/validate_benchmark_data.py --gpu H100")


if __name__ == "__main__":
    main()
