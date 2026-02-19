# Comprehensive H100 Benchmarking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build wave-based orchestration system to benchmark 6 attention shapes across prefill, decode (TP=1/2/4), and GEMM phases on H100 cluster

**Architecture:** Python orchestration script generates OpenShift Job YAMLs by shape (not model), submits to diya namespace, monitors completion with wave-based sequencing (4 jobs parallel per shape, 6 shapes sequential)

**Tech Stack:** Python (subprocess, time, argparse), OpenShift CLI (oc), InferSim benchmark scripts (existing), YAML generation

**Key Constraint:** Benchmarking requires H100 cluster access via OpenShift in `diya` namespace

---

## Task 1: Add --shape Argument to generate_job.py

**Purpose:** Support attention shape as direct input instead of model lookup

**Files:**
- Modify: `scripts/openshift/generate_job.py:148-188`

**Step 1: Add --shape argument to parser**

In `scripts/openshift/generate_job.py`, add after line 157:

```python
    parser.add_argument(
        "--shape",
        type=str,
        help="Attention shape in format nh-nkv-dh (e.g., 32-8-128). Alternative to --model."
    )
```

**Step 2: Add validation for shape format**

After line 188 (after existing validation), add:

```python
    # Validate shape format if provided
    if args.shape:
        parts = args.shape.split("-")
        if len(parts) != 3:
            print("Error: --shape must be in format nh-nkv-dh (e.g., 32-8-128)")
            sys.exit(1)
        try:
            nh, nkv, dh = map(int, parts)
        except ValueError:
            print("Error: --shape components must be integers")
            sys.exit(1)

        if args.model:
            print("Error: Cannot specify both --shape and --model")
            sys.exit(1)
```

**Step 3: Test shape parsing**

Run:
```bash
python scripts/openshift/generate_job.py --gpu H100 --shape 32-8-128 --phase prefill --suffix test --output-dir /tmp
```

Expected: Error about template not generating properly (we'll fix this next), but should parse shape successfully

**Step 4: Commit shape argument parsing**

```bash
git add scripts/openshift/generate_job.py
git commit -m "feat(generate_job): add --shape argument for attention config

Supports shape format nh-nkv-dh (e.g., 32-8-128) as alternative to
--model lookup. Validates format and mutual exclusivity.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

Expected: Commit succeeds

---

## Task 2: Update generate_job_yaml() to Handle Shapes

**Purpose:** Generate job names and benchmark args from shape instead of model

**Files:**
- Modify: `scripts/openshift/generate_job.py:50-146`

**Step 1: Update function signature**

Replace line 50-58:

```python
def generate_job_yaml(
    gpu_type: str,
    job_suffix: str,
    config: dict,
    template_file: Path,
    output_file: Path,
    model: str = None,
    phase: str = None,
    tp: int = None,
    shape: str = None
):
```

**Step 2: Update job name generation**

Replace lines 68-84 with:

```python
    # Generate job name with shape/model/phase/tp info
    gpu_type_lower = gpu_type.lower()
    name_parts = [job_prefix, gpu_type_lower]

    if shape:
        # Use shape directly for job name (e.g., 32-8-128)
        name_parts.append(shape)
    elif model:
        # Convert model name to job-friendly format (llama-2-7b)
        model_slug = model.replace("_", "-")
        name_parts.append(model_slug)

    if phase:
        name_parts.append(phase)

    if tp is not None:
        name_parts.append(f"tp{tp}")

    name_parts.append(job_suffix)
    job_name = "-".join(name_parts)
```

**Step 3: Update benchmark args generation**

Replace lines 86-95 with:

```python
    # Build benchmark script arguments
    bench_args = [f"--gpu {gpu_type}"]

    if shape:
        # Pass shape parameters to benchmark scripts
        bench_args.append(f"--shape {shape}")
    elif model:
        bench_args.append(f"--model-filter {model}")

    if phase:
        bench_args.append(f"--phase-filter {phase}")
    if tp is not None:
        bench_args.append(f"--tp-filter {tp}")

    bench_args_str = " ".join(bench_args)
```

**Step 4: Update main() to pass shape**

Replace line 227-236 with:

```python
    # Generate YAML
    generate_job_yaml(
        args.gpu,
        job_suffix,
        config,
        template_file,
        output_file,
        model=args.model,
        phase=args.phase,
        tp=args.tp,
        shape=args.shape
    )
```

**Step 5: Test shape-based job generation**

Run:
```bash
python scripts/openshift/generate_job.py \
    --gpu H100 \
    --shape 32-8-128 \
    --phase prefill \
    --suffix test123 \
    --output-dir /tmp
```

Expected: Generates `/tmp/job-h100-32-8-128-prefill-test123.yaml` with `--shape 32-8-128` in BENCH_ARGS

**Step 6: Verify job name format**

Run:
```bash
grep "name:" /tmp/job-h100-32-8-128-prefill-test123.yaml | head -1
```

Expected: `  name: infersim-h100-32-8-128-prefill-test123`

**Step 7: Commit shape-based generation**

```bash
git add scripts/openshift/generate_job.py
git commit -m "feat(generate_job): support shape-based job generation

Pass attention shape directly to benchmark scripts instead of
model lookup. Simplifies benchmarking by kernel dimensions.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Create Shape-Aware Benchmark Runner

**Purpose:** Wrapper script that receives --shape and calls appropriate InferSim scripts

**Files:**
- Create: `scripts/run_benchmarks.py`

**Step 1: Create benchmark runner script**

Create file: `scripts/run_benchmarks.py`

```python
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
```

**Step 5: Test locally (dry-run without GPU)**

Run:
```bash
python -c "from scripts.run_benchmarks import parse_shape; assert parse_shape('32-8-128') == (32, 8, 128); print('✓ Shape parsing works')"
```

Expected: "✓ Shape parsing works"

**Step 6: Commit benchmark runner**

```bash
git add scripts/run_benchmarks.py
git commit -m "feat(benchmarks): add shape-based benchmark runner

Wraps InferSim scripts with shape/phase/TP filtering. Creates
temp model configs from shape parameters. Routes to appropriate
benchmark script based on phase.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Create Orchestration Script

**Purpose:** Wave-based job submission and monitoring

**Files:**
- Create: `scripts/orchestrate_benchmarks.py`

**Step 1: Create orchestration script skeleton**

Create file: `scripts/orchestrate_benchmarks.py`

```python
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
        print(f"✗ Job generation failed: {result.stderr}")
        sys.exit(1)

    # Parse output to find generated YAML path
    for line in result.stdout.split('\n'):
        if "Output:" in line:
            yaml_path = Path(line.split("Output:")[1].strip())
            break
    else:
        print("✗ Could not find output YAML path")
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


def submit_job(yaml_path: Path, namespace: str = "diya") -> str:
    """Submit job to OpenShift. Returns job name."""
    result = run_oc_command(["apply", "-f", str(yaml_path), "-n", namespace])

    # Extract job name from YAML
    yaml_content = yaml_path.read_text()
    for line in yaml_content.split('\n'):
        if line.strip().startswith("name:"):
            job_name = line.split("name:")[1].strip()
            return job_name

    print(f"✗ Could not extract job name from {yaml_path}")
    sys.exit(1)


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
            submit_job(yaml_path)
            print(f"✓ GEMM job submitted: {job_name}")
            gemm_job = job_name
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
                submit_job(yaml_path)
                wave_jobs.append(job_name)
                all_jobs.append(job_name)

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
        print("2. python scripts/validate_benchmarks.py")


if __name__ == "__main__":
    main()
```

**Step 2: Test dry-run mode**

Run:
```bash
python scripts/orchestrate_benchmarks.py --gpu H100 --dry-run --skip-gemm
```

Expected: Generates 24 YAML files (6 shapes × 4 phases), prints wave structure

**Step 3: Test prerequisites check**

Run:
```bash
python scripts/orchestrate_benchmarks.py --gpu H100 --dry-run 2>&1 | grep "✓"
```

Expected: Should show checkmarks for oc CLI, login status, namespace access

**Step 4: Commit orchestration script**

```bash
git add scripts/orchestrate_benchmarks.py
git commit -m "feat(orchestration): add wave-based benchmark orchestrator

Submits 25 jobs (1 GEMM + 6 shapes × 4 phases) with wave-based
sequencing. Parallel within shape (4 jobs), sequential across
shapes (6 waves). Auto-monitoring with status polling.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Create Result Collection Script

**Purpose:** Copy CSV files from all completed pods

**Files:**
- Create: `scripts/collect_results.py`

**Step 1: Create collection script**

Create file: `scripts/collect_results.py`

```python
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

    # Copy data directory
    result = subprocess.run([
        "oc", "rsync", "-n", namespace,
        f"{pod_name}:/mnt/inference-sim/InferSim/bench_data/",
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
        default=Path("InferSim/bench_data"),
        help="Local output directory (default: InferSim/bench_data)"
    )

    args = parser.parse_args()

    print("="*60)
    print("Collecting Benchmark Results")
    print("="*60)
    print(f"Namespace: {args.namespace}")
    print(f"Output: {args.output_dir}")
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
    print("Next step: python scripts/validate_benchmarks.py")


if __name__ == "__main__":
    main()
```

**Step 2: Test pod discovery (without actual pods)**

Run:
```bash
python -c "from scripts.collect_results import find_benchmark_pods; print('✓ Import works')"
```

Expected: "✓ Import works"

**Step 3: Commit collection script**

```bash
git add scripts/collect_results.py
git commit -m "feat(collection): add result collection from OpenShift pods

Discovers completed benchmark pods, uses oc rsync to copy CSV
files to local directory. Filters by pod status, reports summary.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Create Validation Script

**Purpose:** Verify all 25 CSV files exist with valid MFU ranges

**Files:**
- Create: `scripts/validate_benchmarks.py`

**Step 1: Create validation script**

Create file: `scripts/validate_benchmarks.py`

```python
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

    mfu_range = config["mfu_validation"]["prefill"]

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

    mfu_range = config["mfu_validation"]["decode"]

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

    mfu_range = config["mfu_validation"]["gemm"]

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
```

**Step 2: Test validation with missing data**

Run:
```bash
python scripts/validate_benchmarks.py --data-dir /tmp/empty_test
```

Expected: Shows missing files, exits with code 1

**Step 3: Commit validation script**

```bash
git add scripts/validate_benchmarks.py
git commit -m "feat(validation): add benchmark data validator

Checks all 25 CSV files exist, validates structure and MFU ranges
against config thresholds. Reports detailed summary.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Update Job Template to Call run_benchmarks.py

**Purpose:** Template should call our new shape-aware runner script

**Files:**
- Modify: `scripts/openshift/job-benchmarks-template.yaml:123`

**Step 1: Update command to call run_benchmarks.py**

Replace line 123:

```yaml
          python scripts/run_benchmarks.py ${BENCH_ARGS}
```

This assumes the template currently calls a different script. The `${BENCH_ARGS}` will contain `--shape 32-8-128 --phase-filter prefill` etc.

**Step 2: Verify template uses correct path**

Run:
```bash
grep "python scripts/run_benchmarks.py" scripts/openshift/job-benchmarks-template.yaml
```

Expected: Line found with BENCH_ARGS variable

**Step 3: Commit template update**

```bash
git add scripts/openshift/job-benchmarks-template.yaml
git commit -m "fix(template): use shape-aware run_benchmarks.py script

Template now calls run_benchmarks.py which handles shape-based
filtering and routes to appropriate InferSim scripts.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Integration Test (Dry Run)

**Purpose:** Test full orchestration in dry-run mode

**Files:**
- None (testing only)

**Step 1: Run full dry-run**

Run:
```bash
python scripts/orchestrate_benchmarks.py --gpu H100 --dry-run
```

Expected:
- Generates 25 YAML files
- Shows wave structure
- Reports job names
- No actual submission

**Step 2: Verify YAML files generated**

Run:
```bash
ls scripts/openshift/job-h100-*.yaml | wc -l
```

Expected: 25 files

**Step 3: Check sample YAML content**

Run:
```bash
grep "name:" scripts/openshift/job-h100-32-8-128-prefill-*.yaml | head -1
```

Expected: Job name includes shape like `infersim-h100-32-8-128-prefill-20260219-123456`

**Step 4: Verify BENCH_ARGS in YAML**

Run:
```bash
grep "BENCH_ARGS" scripts/openshift/job-h100-32-8-128-decode-*.yaml | head -1
```

Expected: Contains `--shape 32-8-128 --phase-filter decode --tp-filter`

**Step 5: Clean up test YAMLs**

Run:
```bash
rm scripts/openshift/job-h100-*-202*.yaml
```

Expected: Test YAML files removed

---

## Task 9: Documentation

**Purpose:** Document usage and architecture

**Files:**
- Create: `scripts/README.md`

**Step 1: Create scripts README**

Create file: `scripts/README.md`

```markdown
# Benchmarking Scripts

## Overview

Automated orchestration system for running InferSim benchmarks on OpenShift H100 cluster.

## Architecture

**Shape-Based Benchmarking:** All benchmarks organized by attention shape `{nh}-{nkv}-{dh}`, not by model name. A 32-8-128 benchmark applies to Mistral-7B, Mixtral-8x7B, and any model with matching dimensions.

**Wave-Based Execution:**
- Wave 0: GEMM (background, independent)
- Waves 1-6: Each shape runs 4 jobs in parallel (prefill + decode TP=1/2/4)

## Usage

### Full Benchmark Suite

```bash
# Generate and submit all 25 jobs
python scripts/orchestrate_benchmarks.py --gpu H100

# Collect results after completion
python scripts/collect_results.py

# Validate data
python scripts/validate_benchmarks.py
```

### Dry Run (No Submission)

```bash
# Generate YAMLs without submitting
python scripts/orchestrate_benchmarks.py --gpu H100 --dry-run

# Check generated files
ls scripts/openshift/job-h100-*.yaml
```

### Single Job Generation

```bash
# Generate single job for specific shape/phase/TP
python scripts/openshift/generate_job.py \
    --gpu H100 \
    --shape 32-8-128 \
    --phase decode \
    --tp 2

# Submit manually
oc apply -f scripts/openshift/job-h100-32-8-128-decode-tp2-*.yaml -n diya
```

## Prerequisites

- OpenShift CLI (`oc`) installed and logged in
- Access to `diya` namespace
- H100 GPU nodes available in cluster

## Output Structure

```
InferSim/bench_data/h100/
├── gemm/
│   └── data.csv                 # GEMM MFU data
└── mha/
    ├── prefill/
    │   ├── 28-4-128.csv        # 6 shapes
    │   └── ...
    └── decode/
        ├── 28-4-128-tp1.csv    # 6 shapes × 3 TPs = 18 files
        └── ...
```

## Troubleshooting

**Jobs stuck in pending:**
```bash
oc get pods -n diya -l app=infersim-benchmark
oc describe pod <pod-name> -n diya
```

**View job logs:**
```bash
oc logs -f job/<job-name> -n diya
```

**Re-run failed shape:**
```bash
# Generate jobs for single shape
python scripts/orchestrate_benchmarks.py --gpu H100 --dry-run | grep "32-8-128"
# Then manually submit the 4 YAMLs for that shape
```

## Scripts Reference

- `generate_job.py` - Generate single job YAML from template
- `orchestrate_benchmarks.py` - Wave-based job orchestration
- `run_benchmarks.py` - Called by jobs, routes to InferSim scripts
- `collect_results.py` - Copy CSV files from completed pods
- `validate_benchmarks.py` - Verify data structure and ranges
```

**Step 2: Commit documentation**

```bash
git add scripts/README.md
git commit -m "docs(scripts): add benchmarking orchestration guide

Documents wave-based orchestration, shape-based approach, and
usage patterns for comprehensive H100 benchmarking.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Update ATTENTION_CONFIGS.md

**Purpose:** Document that benchmark data now covers all 6 shapes

**Files:**
- Modify: `scripts/ATTENTION_CONFIGS.md:6-12`

**Step 1: Update benchmark data listing**

Replace lines 6-12:

```markdown
## Current Benchmark Configs

After comprehensive benchmarking, all 6 common shapes are available:

```bash
# View available benchmark data
ls InferSim/bench_data/h100/mha/prefill/
# 28-4-128.csv   ← Qwen2-7B, Qwen2.5-7B
# 32-32-128.csv  ← Llama-2-7B, Llama-1-7B (MHA)
# 32-8-128.csv   ← Mistral-7B, Mixtral-8x7B
# 40-40-128.csv  ← Llama-2-13B, Qwen-14B (MHA)
# 56-8-128.csv   ← CodeLlama-34B
# 64-8-128.csv   ← Llama-2-70B, Qwen2-72B

ls InferSim/bench_data/h100/mha/decode/
# Each shape has 3 TP variants: {shape}-tp1.csv, {shape}-tp2.csv, {shape}-tp4.csv
```
```

**Step 2: Commit documentation update**

```bash
git add scripts/ATTENTION_CONFIGS.md
git commit -m "docs(configs): update for comprehensive benchmark coverage

All 6 common attention shapes now have prefill + decode (TP=1/2/4)
benchmark data available.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Execution Instructions

### Development Phase (Tasks 1-10)

Complete tasks 1-10 on local machine (no GPU needed). This builds the orchestration infrastructure.

### Cluster Execution Phase

**Prerequisites:**
- OpenShift access configured
- Logged in: `oc login <cluster-url>`
- Namespace access: `oc project diya`
- H100 nodes available

**Run benchmarks:**

```bash
# Submit all 25 jobs with wave-based orchestration
python scripts/orchestrate_benchmarks.py --gpu H100

# Script will:
# 1. Launch GEMM job (background)
# 2. For each of 6 shapes:
#    a. Launch 4 jobs in parallel (prefill + 3 decode TPs)
#    b. Wait for all 4 to complete (~15-20 min)
#    c. Report wave status
# 3. Wait for GEMM to complete
# 4. Report final summary

# Expected runtime: 90-120 minutes
```

**Collect results:**

```bash
# After orchestration completes
python scripts/collect_results.py

# Expected: Copies CSV files from 25 pods to InferSim/bench_data/h100/
```

**Validate data:**

```bash
python scripts/validate_benchmarks.py

# Expected: ✅ PASSED with all 25 files validated
```

---

## Success Criteria

- [ ] All scripts run without errors
- [ ] Dry-run generates 25 valid YAML files
- [ ] Live run completes in < 2 hours
- [ ] 25 CSV files collected with correct naming
- [ ] All MFU values within expected ranges
- [ ] No manual intervention needed after launch

---

## Rollback Plan

If cluster execution fails:

1. **Check job logs:**
   ```bash
   oc get jobs -n diya | grep infersim
   oc logs job/<failed-job-name> -n diya
   ```

2. **Re-run single shape:**
   ```bash
   # Generate 4 jobs for single shape
   python scripts/openshift/generate_job.py --gpu H100 --shape 32-8-128 --phase prefill
   python scripts/openshift/generate_job.py --gpu H100 --shape 32-8-128 --phase decode --tp 1
   python scripts/openshift/generate_job.py --gpu H100 --shape 32-8-128 --phase decode --tp 2
   python scripts/openshift/generate_job.py --gpu H100 --shape 32-8-128 --phase decode --tp 4

   # Submit manually
   oc apply -f scripts/openshift/job-h100-32-8-128-*.yaml -n diya
   ```

3. **Partial collection:**
   ```bash
   # Collect from succeeded pods only
   python scripts/collect_results.py

   # Check what's missing
   python scripts/validate_benchmarks.py
   ```

---

## Notes

- **No model names:** Benchmark scripts receive shape parameters directly (nh, nkv, dh)
- **TP scaling:** Same attention kernel benchmarked at 3 TP values (1, 2, 4)
- **GEMM reuse:** Single GEMM benchmark applies to all models
- **Wave isolation:** Each shape wave is independent, failures don't cascade
- **Resumable:** Can restart from any wave if orchestration interrupted
