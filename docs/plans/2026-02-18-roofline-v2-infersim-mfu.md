# Roofline V2: InferSim MFU-Based Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace calibrated roofline model with MFU-based InferSim approach for 4-10% prediction accuracy

**Architecture:** Python scripts benchmark H100 kernels → generate CSV files with pre-computed MFU values → Go loads CSVs at startup → performs table lookups during simulation. NO GPU code in Go, NO runtime benchmarking.

**Tech Stack:** Python (InferSim existing scripts), Go (CSV parsing, nearest-neighbor lookup), OpenShift (cluster access via `oc`)

**Key Constraint:** Python phase REQUIRES H100 cluster access via OpenShift. Go phase works on any machine (no GPU needed).

**OpenShift Namespace:** All cluster experiments run in namespace `diya`. Job YAMLs specify `namespace: diya`.

**Approach:** **REUSE InferSim scripts directly** - no custom orchestration needed! InferSim already has production-tested `fa3_mha_prefill.py`, `flashinfer_mha_decode.py`, and `deepgemm_gemm.py`.

---

## ★ Insight ─────────────────────────────────────
**Critical Understanding of Architecture:**
1. **InferSim Scripts Ready** - `kernel_benchmark/*.py` already exist and work!
2. **Minimal Wrapper Needed** - Just one Python script to modify TFLOPs and call InferSim scripts
3. **Go = Runtime CSV Reader** - Loads pre-computed MFU data, does nearest-neighbor lookups
4. **NO GPU CODE IN GO** - Go never executes kernels, only reads static CSV files
5. **90% Reuse** - Use InferSim's existing infrastructure
─────────────────────────────────────────────────

---

## Phase 1: Python Benchmarking on H100 (Days 1-5)

### Task 1: Simple Runner Script for InferSim Benchmarks

**Prerequisites:** InferSim directory exists with kernel_benchmark/ scripts

**Files:**
- Create: `scripts/run_h100_benchmarks.py` (minimal wrapper)
- Create: `scripts/validate_h100_data.py` (validation only)

**Step 1: Verify InferSim scripts exist**

```bash
ls InferSim/kernel_benchmark/{fa3_mha_prefill,flashinfer_mha_decode,deepgemm_gemm}.py
```

Expected: All 3 scripts exist

**Step 2: Create minimal runner script**

Create file: `scripts/run_h100_benchmarks.py`

```python
"""
Simple runner for InferSim H100 benchmarks
Modifies TFLOPs for H100 and runs existing InferSim scripts
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


# High-priority configs: (nh, nkv, dh, model_name)
CONFIGS = [
    (28, 4, 128, "qwen2.5-7b"),
    (32, 8, 128, "llama-3-8b"),
    (32, 4, 128, "llama-3.1-8b"),
]

H100_TFLOPS = 989.5


def modify_prefill_script():
    """Modify fa3_mha_prefill.py to use H100 TFLOPs"""
    script_path = Path("InferSim/kernel_benchmark/fa3_mha_prefill.py")
    content = script_path.read_text()

    # Replace TFLOPs value
    modified = content.replace("fp16_tflops = 148", f"fp16_tflops = {H100_TFLOPS}")

    # Backup original
    backup_path = script_path.with_suffix(".py.bak")
    if not backup_path.exists():
        script_path.with_suffix(".py.bak").write_text(script_path.read_text())

    script_path.write_text(modified)
    print(f"✓ Modified fa3_mha_prefill.py for H100 ({H100_TFLOPS} TFLOPs)")


def create_config(nh: int, nkv: int, dh: int) -> Path:
    """Create temporary HuggingFace config"""
    config = {
        "hidden_size": nh * dh,
        "num_attention_heads": nh,
        "num_key_value_heads": nkv,
        "head_dim": dh,
        "num_hidden_layers": 32,
        "intermediate_size": 14336,
        "torch_dtype": "bfloat16"
    }

    path = Path(f"/tmp/h100_config_{nh}_{nkv}_{dh}.json")
    path.write_text(json.dumps(config, indent=2))
    return path


def run_command(cmd: List[str], cwd: Path, desc: str) -> bool:
    """Run command and return success"""
    print(f"  Running: {desc}...")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ✗ Failed: {result.stderr}")
        return False

    return True


def main():
    """Run all H100 benchmarks using InferSim scripts"""
    print("="*60)
    print("H100 Benchmark Runner (Using InferSim Scripts)")
    print("="*60)

    # Modify prefill script once
    modify_prefill_script()

    # Create output directories
    output_base = Path("InferSim/bench_data/h100")
    (output_base / "mha" / "prefill").mkdir(parents=True, exist_ok=True)
    (output_base / "mha" / "decode").mkdir(parents=True, exist_ok=True)
    (output_base / "gemm").mkdir(parents=True, exist_ok=True)

    infersim_dir = Path("InferSim")

    # Run MHA benchmarks
    for nh, nkv, dh, model_name in CONFIGS:
        config_key = f"{nh}-{nkv}-{dh}"
        print(f"\n{'='*60}")
        print(f"Config: {model_name} ({config_key})")
        print(f"{'='*60}")

        # Create temp config
        config_path = create_config(nh, nkv, dh)

        # Run prefill
        cmd = [
            sys.executable,
            "kernel_benchmark/fa3_mha_prefill.py",
            "--config-path", str(config_path.absolute())
        ]
        if run_command(cmd, infersim_dir, "MHA prefill"):
            src = infersim_dir / "attention_benchmark.csv"
            dst = output_base / "mha" / "prefill" / f"{config_key}.csv"
            if src.exists():
                src.rename(dst)
                print(f"  ✓ Prefill: {dst}")

        # Run decode
        cmd = [
            sys.executable,
            "kernel_benchmark/flashinfer_mha_decode.py",
            "--config-path", str(config_path.absolute()),
            "--fp16-tflops", str(H100_TFLOPS),
            "--kv-cache-dtype", "bf16",
            "--tp-size", "1"
        ]
        if run_command(cmd, infersim_dir, "MHA decode"):
            src = infersim_dir / "attention_benchmark.csv"
            dst = output_base / "mha" / "decode" / f"{config_key}.csv"
            if src.exists():
                src.rename(dst)
                print(f"  ✓ Decode: {dst}")

    # Run GEMM benchmarks
    print(f"\n{'='*60}")
    print("GEMM Benchmarks")
    print(f"{'='*60}")

    M_values = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    K_values = [2048, 3584, 4096, 8192]
    N_values = [6144, 11008, 14336, 18944]

    gemm_output = output_base / "gemm" / "data.csv"
    gemm_tmp = infersim_dir / "gemm.csv"

    # Run first to create header
    first = True
    for k in K_values:
        for n in N_values:
            cmd = [
                sys.executable,
                "kernel_benchmark/deepgemm_gemm.py",
                "-k", str(k),
                "-n", str(n),
                "--gpu-tflops", str(H100_TFLOPS)
            ]
            print(f"  Running: GEMM K={k}, N={n}")
            if run_command(cmd, infersim_dir, f"GEMM K={k} N={n}"):
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

    print(f"\n✓ GEMM: {gemm_output}")

    print("\n" + "="*60)
    print("✅ All benchmarks complete")
    print(f"Output: {output_base}")
    print("="*60)


if __name__ == "__main__":
    main()
```

**Step 3: Create validation script**

Create file: `scripts/validate_h100_data.py`

```python
"""Validate H100 benchmark data"""
import pandas as pd
from pathlib import Path
import sys


def validate_mha_data(base_path: Path):
    """Validate MHA benchmark data"""
    configs = ["28-4-128", "32-8-128", "32-4-128"]
    stages = ["decode", "prefill"]
    errors = []

    print("="*60)
    print("MHA Validation")
    print("="*60)

    for stage in stages:
        for config in configs:
            path = base_path / "mha" / stage / f"{config}.csv"

            if not path.exists():
                errors.append(f"Missing: {path}")
                print(f"✗ {stage:7s} {config:10s}: MISSING")
                continue

            try:
                df = pd.read_csv(path)

                # Check columns
                if stage == "decode":
                    required = ["batch_size", "kv_len", "mfu"]
                else:
                    required = ["seq_len", "mfu"]

                missing = set(required) - set(df.columns)
                if missing:
                    errors.append(f"{path}: Missing columns {missing}")
                    continue

                # Validate MFU range
                if stage == "decode":
                    if df['mfu'].min() < 0.005 or df['mfu'].max() > 0.30:
                        errors.append(f"{path}: MFU range [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")
                else:
                    if df['mfu'].min() < 0.30 or df['mfu'].max() > 0.90:
                        errors.append(f"{path}: MFU range [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")

                print(f"✓ {stage:7s} {config:10s}: {len(df):3d} rows, MFU [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")

            except Exception as e:
                errors.append(f"{path}: {e}")

    return errors


def validate_gemm_data(base_path: Path):
    """Validate GEMM data"""
    path = base_path / "gemm" / "data.csv"
    errors = []

    print("\n" + "="*60)
    print("GEMM Validation")
    print("="*60)

    if not path.exists():
        errors.append(f"Missing: {path}")
        print("✗ GEMM: MISSING")
        return errors

    try:
        df = pd.read_csv(path)

        required = ["m", "k", "n", "mfu"]
        missing = set(required) - set(df.columns)
        if missing:
            errors.append(f"{path}: Missing columns {missing}")
            return errors

        print(f"✓ GEMM: {len(df):3d} rows, MFU [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")

    except Exception as e:
        errors.append(f"{path}: {e}")

    return errors


def main():
    base_path = Path("InferSim/bench_data/h100")

    print("\n" + "="*60)
    print("H100 Data Validation")
    print("="*60)
    print(f"Path: {base_path}\n")

    errors = []
    errors.extend(validate_mha_data(base_path))
    errors.extend(validate_gemm_data(base_path))

    print("\n" + "="*60)
    if errors:
        print(f"❌ FAILED: {len(errors)} errors")
        print("="*60)
        for err in errors:
            print(f"  • {err}")
        sys.exit(1)
    else:
        print("✅ PASSED")
        print("="*60)
        sys.exit(0)


if __name__ == "__main__":
    main()
```

**Step 4: Test locally (without GPU)**

```bash
python -c "from scripts.run_h100_benchmarks import create_config; print(create_config(28, 4, 128))"
```

Expected: Path to temp config file

**Step 5: Commit**

```bash
git add scripts/run_h100_benchmarks.py scripts/validate_h100_data.py
git commit -m "feat(roofline-v2): add minimal runner for InferSim benchmarks

- Reuse InferSim's existing fa3_mha_prefill.py, flashinfer_mha_decode.py
- Reuse InferSim's deepgemm_gemm.py
- Minimal wrapper just modifies TFLOPs and runs scripts
- Validation script for data quality
- No custom orchestration needed!

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2: OpenShift Job Definition

**Files:**
- Create: `scripts/openshift/job-h100-benchmarks.yaml`
- Create: `scripts/submit_benchmark_job.py`

**Step 1: Create single OpenShift Job**

Create file: `scripts/openshift/job-h100-benchmarks.yaml`

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: infersim-h100-benchmarks
  namespace: diya
  labels:
    app: infersim-benchmark
spec:
  backoffLimit: 2
  template:
    metadata:
      labels:
        app: infersim-benchmark
    spec:
      restartPolicy: Never
      containers:
      - name: benchmark
        image: nvcr.io/nvidia/pytorch:24.01-py3
        command:
          - /bin/bash
          - -c
          - |
            cd /workspace
            pip install pandas sgl-kernel deep-gemm --quiet
            python scripts/run_h100_benchmarks.py
        workingDir: /workspace
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "64Gi"
            cpu: "8"
          requests:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "4"
        volumeMounts:
        - name: workspace
          mountPath: /workspace
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
      volumes:
      - name: workspace
        persistentVolumeClaim:
          claimName: inference-sim-workspace
      nodeSelector:
        accelerator: nvidia-h100
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

**Step 2: Create job submission script**

Create file: `scripts/submit_benchmark_job.py`

```python
"""Submit H100 benchmark job to OpenShift"""
import subprocess
import sys
import time
from pathlib import Path


def run_oc(args, check=True):
    """Run oc command"""
    cmd = ["oc"] + args
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def main():
    print("="*60)
    print("OpenShift H100 Benchmark Submission")
    print("="*60)

    # Check oc available
    result = run_oc(["version"], check=False)
    if result.returncode != 0:
        print("✗ oc not found")
        sys.exit(1)

    # Check logged in
    result = run_oc(["whoami"], check=False)
    if result.returncode != 0:
        print("✗ Not logged in. Run: oc login")
        sys.exit(1)

    print(f"✓ Logged in as: {result.stdout.strip()}")

    # Switch to diya namespace
    result = run_oc(["project", "diya"], check=False)
    if result.returncode != 0:
        print("✗ Failed to switch to namespace 'diya'")
        sys.exit(1)

    print("✓ Using namespace: diya")

    # Submit job
    job_yaml = Path("scripts/openshift/job-h100-benchmarks.yaml")
    result = run_oc(["apply", "-f", str(job_yaml)])
    print("✓ Job submitted: infersim-h100-benchmarks")

    # Wait for completion
    print("\n→ Monitoring job (Ctrl+C to stop monitoring)...")
    print("  Logs: oc logs -n diya -f job/infersim-h100-benchmarks")

    while True:
        time.sleep(10)
        result = run_oc([
            "get", "job", "infersim-h100-benchmarks",
            "-o", "jsonpath={.status.conditions[?(@.type=='Complete')].status}"
        ], check=False)

        if result.stdout.strip() == "True":
            print("\n✅ Job completed successfully")
            print("\nNext steps:")
            print("1. Get pod: POD=$(oc get pods -n diya -l app=infersim-benchmark -o jsonpath='{.items[0].metadata.name}')")
            print("2. Copy data: oc rsync -n diya ${POD}:/workspace/InferSim/bench_data/h100 ./InferSim/bench_data/h100")
            print("3. Validate: python scripts/validate_h100_data.py")
            sys.exit(0)

        result = run_oc([
            "get", "job", "infersim-h100-benchmarks",
            "-o", "jsonpath={.status.conditions[?(@.type=='Failed')].status}"
        ], check=False)

        if result.stdout.strip() == "True":
            print("\n✗ Job failed")
            print("View logs: oc logs -n diya job/infersim-h100-benchmarks")
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Job continues running in cluster.")
        sys.exit(0)
```

**Step 3: Commit**

```bash
git add scripts/openshift/ scripts/submit_benchmark_job.py
git commit -m "feat(roofline-v2): add OpenShift job for H100 benchmarks

- Single job runs InferSim scripts via wrapper
- Installs dependencies in container
- Uses diya namespace
- Python submission script with monitoring
- Automatic job status checking

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Execute Benchmarks (REQUIRES H100 + OpenShift)

**Prerequisites:** OpenShift access, logged in, diya namespace exists

**Step 1: Verify setup**

```bash
oc whoami
oc project diya
ls InferSim/kernel_benchmark/{fa3_mha_prefill,flashinfer_mha_decode,deepgemm_gemm}.py
```

Expected: All checks pass

**Step 2: Submit job**

```bash
python scripts/submit_benchmark_job.py
```

Expected: Job submitted and monitoring starts

**Step 3: Alternative - Monitor manually**

```bash
# Watch job status
oc get jobs -n diya -w

# View logs
oc logs -n diya -f job/infersim-h100-benchmarks
```

**Step 4: Retrieve data after completion**

```bash
# Find pod
POD=$(oc get pods -n diya -l app=infersim-benchmark -o jsonpath='{.items[0].metadata.name}')

# Copy data
oc rsync -n diya ${POD}:/workspace/InferSim/bench_data/h100 ./InferSim/bench_data/
```

Expected: CSV files in `InferSim/bench_data/h100/`

**Step 5: Validate data**

```bash
python scripts/validate_h100_data.py
```

Expected: "✅ PASSED"

**Step 6: Copy to main repository**

```bash
mkdir -p bench_data/h100
cp -r InferSim/bench_data/h100/* bench_data/h100/
find bench_data/h100 -name "*.csv" | wc -l
```

Expected: 7 files (6 MHA + 1 GEMM)

**Step 7: Commit benchmark data**

```bash
git add bench_data/h100/
git commit -m "data(roofline-v2): add H100 benchmark MFU data

- MHA prefill/decode for 3 configs (Qwen, Llama-3, Llama-3.1)
- GEMM data for 160 configurations
- Pre-computed MFU values from InferSim scripts
- Benchmarked on OpenShift H100 cluster (diya namespace)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: Go Implementation (Days 6-9)

### Task 4-9: [Same as before - Go CSV loader, lookup, roofline, integration]

*These tasks remain unchanged from previous version - see below for full details*

---

## Phase 2 Details (Go Implementation)

### Task 4: MFU Database Loader

**Files:**
- Create: `sim/mfu_database.go`
- Create: `sim/mfu_database_test.go`

*[Full implementation from previous plan - unchanged]*

### Task 5: MFU Lookup Logic

**Files:**
- Create: `sim/mfu_lookup.go`
- Modify: `sim/mfu_database_test.go`

*[Full implementation from previous plan - unchanged]*

### Task 6: InferSim Roofline Implementation

**Files:**
- Create: `sim/roofline_infersim.go`
- Create: `sim/roofline_infersim_test.go`

*[Full implementation from previous plan - unchanged]*

### Task 7: Simulator Integration

**Files:**
- Modify: `sim/simulator.go`
- Modify: `sim/roofline_step.go`

*[Full implementation from previous plan - unchanged]*

### Task 8: Cross-Validation

**Files:**
- Create: `scripts/compare_predictions.py`

*[Full implementation from previous plan - unchanged]*

### Task 9: Documentation

**Files:**
- Create: `docs/roofline_v2_implementation_summary.md`
- Modify: `README.md`

*[Full implementation from previous plan - unchanged]*

---

## Summary

### What Changed

**Before (Complex):**
- Custom Python orchestration with 500+ lines
- Reimplemented benchmark logic
- Multiple test files
- Complex error handling

**After (Simple):**
- ✅ **REUSE InferSim scripts** (production-tested!)
- ✅ Minimal 100-line wrapper
- ✅ Single OpenShift Job
- ✅ Much simpler to maintain

### File Structure

```
scripts/
├── run_h100_benchmarks.py      # 100-line wrapper for InferSim scripts
├── validate_h100_data.py       # Validation only
├── submit_benchmark_job.py     # OpenShift submission
└── openshift/
    └── job-h100-benchmarks.yaml  # Single job definition

InferSim/
└── kernel_benchmark/           # Use these directly!
    ├── fa3_mha_prefill.py      # ← Already exists
    ├── flashinfer_mha_decode.py # ← Already exists
    └── deepgemm_gemm.py         # ← Already exists
```

### Execution

```bash
# Submit to cluster
python scripts/submit_benchmark_job.py

# Validate
python scripts/validate_h100_data.py
```

**Much simpler!** 🎯
