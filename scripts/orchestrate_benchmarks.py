#!/usr/bin/env python3
"""
Orchestrate H100 benchmarks across 6 attention shapes.

Wave-based execution:
- Wave 0: GEMM (background, independent)
- Waves 1-6: Each shape runs 4 jobs in parallel (prefill + decode TP=1/2/4)
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional


# Attention shapes to benchmark
SHAPES = [
    (28, 4, 128),   # Qwen2-7B, Qwen2.5-7B
    (32, 32, 128),  # Llama-2-7B, Llama-1-7B (MHA)
    (32, 8, 128),   # Mistral-7B, Mixtral-8x7B
    (40, 40, 128),  # Llama-2-13B, Qwen-14B (MHA)
    (56, 8, 128),   # CodeLlama-34B
    (64, 8, 128),   # Llama-2-70B, Qwen2-72B
]


def run_oc_command(args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run oc command and return result"""
    cmd = ["oc"] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def check_prerequisites():
    """Verify oc is installed and logged in"""
    print("Checking prerequisites...")

    # Check oc available
    result = run_oc_command(["version", "--client"], check=False)
    if result.returncode != 0:
        print("✗ oc CLI not found. Install: https://docs.openshift.com/container-platform/4.12/cli_reference/openshift_cli/getting-started-cli.html")
        sys.exit(1)
    print("✓ oc CLI available")

    # Check logged in
    result = run_oc_command(["whoami"], check=False)
    if result.returncode != 0:
        print("✗ Not logged in to OpenShift. Run: oc login <cluster-url>")
        sys.exit(1)
    username = result.stdout.strip()
    print(f"✓ Logged in as: {username}")

    # Check namespace access
    result = run_oc_command(["get", "namespace", "diya"], check=False)
    if result.returncode != 0:
        print("✗ Cannot access namespace 'diya'. Check permissions.")
        sys.exit(1)
    print("✓ Namespace 'diya' accessible")

    # Switch to diya namespace
    run_oc_command(["project", "diya"])
    print("✓ Using namespace: diya\n")


def generate_job_yaml(
    gpu_type: str,
    shape: Optional[tuple] = None,
    phase: Optional[str] = None,
    tp: Optional[int] = None,
    timestamp: str = None
) -> tuple[Path, str]:
    """
    Generate job YAML using generate_job.py.
    Returns (yaml_path, job_name)
    """
    timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")

    cmd = [
        sys.executable,
        "scripts/openshift/generate_job.py",
        "--gpu", gpu_type,
        "--suffix", timestamp,
    ]

    # Add shape or handle GEMM
    if shape:
        shape_str = f"{shape[0]}-{shape[1]}-{shape[2]}"
        cmd.extend(["--shape", shape_str])

    if phase:
        cmd.extend(["--phase", phase])

    if tp is not None:
        cmd.extend(["--tp", str(tp)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        context = f"shape={shape}, phase={phase}, tp={tp}" if shape else "GEMM"
        print(f"✗ Job generation failed ({context}): {result.stderr}")
        sys.exit(1)

    # Parse output to find generated YAML path
    for line in result.stdout.split('\n'):
        if "Output:" in line:
            yaml_path = Path(line.split("Output:")[1].strip())
            break
    else:
        print("✗ Could not find output YAML path")
        sys.exit(1)

    # Validate YAML path exists
    if not yaml_path.exists():
        print(f"✗ Generated YAML file not found: {yaml_path}")
        sys.exit(1)

    # Extract job name from YAML
    yaml_content = yaml_path.read_text()
    for line in yaml_content.split('\n'):
        if line.strip().startswith("name:"):
            job_name = line.split("name:")[1].strip()
            break
    else:
        print(f"✗ Could not extract job name from {yaml_path}")
        sys.exit(1)

    return yaml_path, job_name


def submit_job(yaml_path: Path, job_name: str, namespace: str = "diya") -> None:
    """Submit job to OpenShift."""
    run_oc_command(["apply", "-f", str(yaml_path), "-n", namespace])


def get_job_status(job_name: str, namespace: str = "diya") -> str:
    """
    Get job status. Returns one of: Running, Complete, Failed, Unknown
    """
    # Check for completion
    result = run_oc_command([
        "get", "job", job_name, "-n", namespace,
        "-o", "jsonpath={.status.conditions[?(@.type=='Complete')].status}"
    ], check=False)

    if result.stdout.strip() == "True":
        return "Complete"

    # Check for failure
    result = run_oc_command([
        "get", "job", job_name, "-n", namespace,
        "-o", "jsonpath={.status.conditions[?(@.type=='Failed')].status}"
    ], check=False)

    if result.stdout.strip() == "True":
        return "Failed"

    # Check if job exists
    result = run_oc_command([
        "get", "job", job_name, "-n", namespace
    ], check=False)

    if result.returncode != 0:
        return "Unknown"

    return "Running"


def wait_for_jobs(job_names: List[str], timeout_minutes: int = 30, namespace: str = "diya") -> dict:
    """
    Wait for all jobs to reach terminal state (Complete or Failed).
    Returns dict mapping job_name -> status.
    """
    deadline = time.time() + timeout_minutes * 60
    poll_interval = 10  # seconds

    print(f"Waiting for {len(job_names)} jobs (timeout: {timeout_minutes} min)...")

    while time.time() < deadline:
        statuses = {job: get_job_status(job, namespace) for job in job_names}

        # Print status summary
        complete = sum(1 for s in statuses.values() if s == "Complete")
        failed = sum(1 for s in statuses.values() if s == "Failed")
        running = sum(1 for s in statuses.values() if s == "Running")

        elapsed = int(time.time() - (deadline - timeout_minutes * 60))
        print(f"  [{elapsed//60:02d}:{elapsed%60:02d}] Complete: {complete}, Failed: {failed}, Running: {running}")

        # Check if all done
        if all(s in ["Complete", "Failed"] for s in statuses.values()):
            print(f"✓ All jobs finished")
            return statuses

        time.sleep(poll_interval)

    print(f"✗ Timeout after {timeout_minutes} minutes")
    return {job: get_job_status(job, namespace) for job in job_names}


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate H100 benchmarks across all attention shapes"
    )
    parser.add_argument("--gpu", type=str, default="H100", help="GPU type (default: H100)")
    parser.add_argument("--dry-run", action="store_true", help="Generate YAMLs but don't submit")
    parser.add_argument("--skip-gemm", action="store_true", help="Skip GEMM benchmark")

    args = parser.parse_args()

    print("="*60)
    print("H100 Comprehensive Benchmark Orchestration")
    print("="*60)
    print(f"GPU Type: {args.gpu}")
    print(f"Shapes: {len(SHAPES)}")
    print(f"Total jobs: {1 + len(SHAPES) * 4} (1 GEMM + 6 shapes × 4 phases)")
    print(f"Peak GPU usage: 5 (1 GEMM + 4 per shape)")
    print(f"Namespace: diya")
    print("="*60)

    if not args.dry_run:
        check_prerequisites()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_jobs = []

    # Wave 0: GEMM (background)
    if not args.skip_gemm:
        print("\n" + "="*60)
        print("Wave 0: GEMM (Background)")
        print("="*60)

        yaml_path, job_name = generate_job_yaml(args.gpu, phase="gemm", timestamp=timestamp)
        print(f"Generated: {yaml_path}")
        print(f"Job name: {job_name}")

        if not args.dry_run:
            try:
                submit_job(yaml_path, job_name)
                print(f"✓ GEMM job submitted: {job_name}")
                gemm_job = job_name
                all_jobs.append(job_name)
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to submit GEMM job: {e.stderr}")
                sys.exit(1)
        else:
            print("(dry-run: not submitted)")
            gemm_job = None
    else:
        gemm_job = None

    # Waves 1-6: Process each shape
    for wave_idx, shape in enumerate(SHAPES, 1):
        nh, nkv, dh = shape
        shape_str = f"{nh}-{nkv}-{dh}"

        print(f"\n{'='*60}")
        print(f"Wave {wave_idx}/6: {shape_str}")
        print(f"{'='*60}")

        wave_jobs = []

        # Generate and submit 4 jobs in parallel
        phases_configs = [
            ("prefill", None),
            ("decode", 1),
            ("decode", 2),
            ("decode", 4),
        ]

        for phase, tp in phases_configs:
            yaml_path, job_name = generate_job_yaml(
                args.gpu,
                shape=shape,
                phase=phase,
                tp=tp,
                timestamp=timestamp
            )

            phase_label = f"{phase}" + (f"-TP{tp}" if tp else "")
            print(f"  [{phase_label:15s}] {yaml_path.name}")

            if not args.dry_run:
                try:
                    submit_job(yaml_path, job_name)
                    wave_jobs.append(job_name)
                    all_jobs.append(job_name)
                except subprocess.CalledProcessError as e:
                    print(f"✗ Failed to submit {phase_label} job for shape {shape_str}: {e.stderr}")
                    sys.exit(1)

        if args.dry_run:
            print("(dry-run: jobs not submitted)")
            continue

        # Wait for wave to complete
        print(f"\n⏳ Waiting for wave {wave_idx}/6 to complete...")
        statuses = wait_for_jobs(wave_jobs, timeout_minutes=30)

        # Report wave results
        complete = sum(1 for s in statuses.values() if s == "Complete")
        failed = sum(1 for s in statuses.values() if s == "Failed")

        if failed > 0:
            print(f"⚠ Wave {wave_idx}/6: {complete}/4 succeeded, {failed}/4 failed")
            for job, status in statuses.items():
                if status == "Failed":
                    print(f"  ✗ {job}")
        else:
            print(f"✓ Wave {wave_idx}/6: All 4 jobs succeeded")

    # Wait for GEMM if still running
    if gemm_job and not args.dry_run:
        print(f"\n{'='*60}")
        print("Waiting for GEMM job...")
        print(f"{'='*60}")

        status = get_job_status(gemm_job)
        if status == "Running":
            statuses = wait_for_jobs([gemm_job], timeout_minutes=60)
            if statuses[gemm_job] == "Complete":
                print("✓ GEMM complete")
            else:
                print(f"✗ GEMM failed")
        else:
            print(f"✓ GEMM already finished: {status}")

    # Final summary
    print(f"\n{'='*60}")
    print("Orchestration Complete")
    print(f"{'='*60}")

    if args.dry_run:
        print("(dry-run mode: no jobs submitted)")
        print(f"\nGenerated YAML files in: scripts/openshift/")
        print("To submit: oc apply -f scripts/openshift/job-*.yaml -n diya")
    else:
        print(f"Total jobs submitted: {len(all_jobs) + (1 if gemm_job else 0)}")
        print(f"\nNext steps:")
        print("1. python scripts/collect_results.py")
        print("2. python scripts/validate_benchmark_data.py --gpu H100")


if __name__ == "__main__":
    main()
