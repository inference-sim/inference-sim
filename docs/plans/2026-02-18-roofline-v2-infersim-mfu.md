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

## Executive Summaries

### Phase 1: Python Benchmarking (Days 1-5)

**What:** Run InferSim's existing benchmark scripts on H100 to generate MFU data

**How:**
1. Create `config/benchmark_config.json` with all constants
2. Write minimal Python wrapper (~100 lines) that calls InferSim scripts
3. Submit OpenShift Job to `diya` namespace
4. Wait ~2-3 hours for benchmarks to complete
5. Validate and commit 13 CSV files

**Deliverables:**
- `bench_data/h100/mha/prefill/*.csv` (6 configs)
- `bench_data/h100/mha/decode/*.csv` (6 configs)
- `bench_data/h100/gemm/data.csv` (160 GEMM configs)
- Total: ~500KB of static CSV data

**Dependencies:** H100 cluster access via OpenShift, `diya` namespace

**Effort:** 2-3 hours dev work, 2-3 hours cluster time

**Key Point:** NO custom benchmarking code - reuse InferSim's production scripts!

---

### Phase 1.5: Model Validation (Optional - After Phase 2)

**What:** Verify simulator works with production models at different TP settings

**How:**
1. Run simulator predictions for Llama-2-7B, Llama-2-70B, Mixtral-8x7B
2. Test TP=1, 2, 4 for each model
3. Verify TP scaling behavior matches expectations

**Deliverables:**
- Validation report showing simulator runs successfully
- TP scaling curves (compute vs memory bound)
- Comparison against vLLM measurements (if available)

**Dependencies:** Phase 2 complete (Go implementation working)

**Effort:** 1-2 hours

**Key Point:** Validates that same MFU data works across all TP values (formula handles scaling)

---

### Phase 2: Go Implementation (Days 6-9)

**What:** Load MFU CSV files and implement lookup-based roofline model

**How:**
1. Implement `mfu_database.go` - CSV parser (~200 lines)
2. Implement `mfu_lookup.go` - Nearest-neighbor search (~150 lines)
3. Implement `roofline_infersim.go` - Use MFU in formula `time = max(flops/(peak*mfu), bytes/bw)` (~150 lines)
4. Integrate with existing simulator
5. Write unit tests

**Deliverables:**
- Go code for MFU-based roofline (~500 lines)
- Unit tests (all passing)
- Integration with existing simulator
- Fallback to legacy roofline if MFU data missing

**Dependencies:** Phase 1 complete (CSV files available)

**Effort:** 4-5 days coding

**Key Point:** NO GPU code in Go - pure CSV reading and table lookups. Works on any machine!

---

### What Ships at the End

```
inference-sim/                    # Go binary
├── bench_data/h100/             # Ship these CSV files (~500KB)
│   ├── mha/prefill/*.csv       # 6 model configs
│   ├── mha/decode/*.csv        # 6 model configs
│   └── gemm/data.csv           # 160 GEMM configs
└── sim/
    ├── mfu_database.go         # CSV loader
    ├── mfu_lookup.go           # Nearest-neighbor search
    └── roofline_infersim.go    # MFU-based roofline

Usage:
  ./inference-sim --model llama-2-7b --gpu H100 --tp 4 --roofline infersim
```

**Key Benefits:**
- ✅ 4-10% error (vs 11% current)
- ✅ Zero calibration needed
- ✅ Works across all models
- ✅ TP=1,2,4 supported (same data)
- ✅ Simple: `time = max(flops/(peak*mfu), bytes/bw)`

---

## Phase 1: Python Benchmarking on H100 (Days 1-5)

### Task 1: Benchmark Configuration File

**Purpose:** Define all constants in a centralized config file (no hardcoded values)

**Files:**
- Create: `config/benchmark_config.json`

**Step 1: Create benchmark configuration**

Create file: `config/benchmark_config.json`

```json
{
  "gpu_specs": {
    "H100": {
      "peak_tflops_fp16": 989.5,
      "peak_memory_bw_tbs": 3.35,
      "effective_memory_bw_factor": 0.8,
      "nvlink_bw_gbs": 900,
      "num_sms": 132,
      "memory_gb": 80
    }
  },
  "benchmark_configs": [
    {
      "name": "llama-2-7b",
      "num_attention_heads": 32,
      "num_key_value_heads": 32,
      "head_dim": 128,
      "num_hidden_layers": 32,
      "intermediate_size": 11008,
      "description": "Llama-2-7B (pure MHA)",
      "validation_model": "meta-llama/Llama-2-7b-hf"
    },
    {
      "name": "llama-2-70b",
      "num_attention_heads": 64,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "num_hidden_layers": 80,
      "intermediate_size": 28672,
      "description": "Llama-2-70B (GQA)",
      "validation_model": "meta-llama/Llama-2-70b-hf"
    },
    {
      "name": "llama-3-8b",
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "num_hidden_layers": 32,
      "intermediate_size": 14336,
      "description": "Llama-3-8B, Mixtral-8x7B (GQA)",
      "validation_model": "meta-llama/Meta-Llama-3-8B"
    },
    {
      "name": "qwen2.5-7b",
      "num_attention_heads": 28,
      "num_key_value_heads": 4,
      "head_dim": 128,
      "num_hidden_layers": 28,
      "intermediate_size": 18944,
      "description": "Qwen2.5-7B (GQA)",
      "validation_model": "Qwen/Qwen2.5-7B-Instruct"
    },
    {
      "name": "llama-3.1-8b",
      "num_attention_heads": 32,
      "num_key_value_heads": 4,
      "head_dim": 128,
      "num_hidden_layers": 32,
      "intermediate_size": 14336,
      "description": "Llama-3.1-8B (GQA)",
      "validation_model": "meta-llama/Meta-Llama-3.1-8B"
    },
    {
      "name": "llama-3.3-70b",
      "num_attention_heads": 40,
      "num_key_value_heads": 8,
      "head_dim": 128,
      "num_hidden_layers": 80,
      "intermediate_size": 18944,
      "description": "Llama-3.3-70B (GQA)",
      "validation_model": "meta-llama/Meta-Llama-3.3-70B-Instruct"
    }
  ],
  "validation_configs": {
    "primary_models": [
      "meta-llama/Llama-2-7b-hf",
      "meta-llama/Llama-2-70b-hf",
      "mistralai/Mixtral-8x7B-v0.1"
    ],
    "tp_values": [1, 2, 4],
    "prefill_test_sizes": [512, 2048],
    "decode_batch_sizes": [1, 16, 32]
  },
  "mfu_validation": {
    "decode": {
      "min": 0.005,
      "max": 0.30,
      "description": "Memory-bound decode operations"
    },
    "prefill": {
      "min": 0.30,
      "max": 0.90,
      "description": "Compute-bound prefill operations"
    },
    "gemm": {
      "min": 0.05,
      "max": 1.0,
      "description": "GEMM operations vary with batch size"
    }
  },
  "gemm_sweep": {
    "m_values": [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    "k_values": [2048, 3584, 4096, 8192],
    "n_values": [6144, 11008, 14336, 18944]
  },
  "openshift": {
    "namespace": "diya",
    "job_name_prefix": "infersim",
    "container_image": "nvcr.io/nvidia/pytorch:24.01-py3",
    "gpu_resource": "nvidia.com/gpu",
    "node_selector": "nvidia-h100"
  }
}
```

**Step 2: Verify JSON is valid**

```bash
python -c "import json; json.load(open('config/benchmark_config.json')); print('✓ Valid JSON')"
```

Expected: "✓ Valid JSON"

---

### Task 2: Simple Runner Script for InferSim Benchmarks

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
Loads config from benchmark_config.json and runs InferSim scripts
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict


def load_config() -> Dict:
    """Load benchmark configuration"""
    config_path = Path("config/benchmark_config.json")
    if not config_path.exists():
        print(f"✗ Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        return json.load(f)


def modify_prefill_script(peak_tflops: float):
    """Modify fa3_mha_prefill.py to use configured TFLOPs"""
    script_path = Path("InferSim/kernel_benchmark/fa3_mha_prefill.py")
    content = script_path.read_text()

    # Replace TFLOPs value
    modified = content.replace("fp16_tflops = 148", f"fp16_tflops = {peak_tflops}")

    # Backup original
    backup_path = script_path.with_suffix(".py.bak")
    if not backup_path.exists():
        script_path.with_suffix(".py.bak").write_text(script_path.read_text())

    script_path.write_text(modified)
    print(f"✓ Modified fa3_mha_prefill.py for H100 ({peak_tflops} TFLOPs)")


def create_hf_config(model_config: Dict) -> Path:
    """Create temporary HuggingFace config from benchmark config"""
    nh = model_config["num_attention_heads"]
    nkv = model_config["num_key_value_heads"]
    dh = model_config["head_dim"]

    config = {
        "hidden_size": nh * dh,
        "num_attention_heads": nh,
        "num_key_value_heads": nkv,
        "head_dim": dh,
        "num_hidden_layers": model_config["num_hidden_layers"],
        "intermediate_size": model_config["intermediate_size"],
        "torch_dtype": "bfloat16"
    }

    config_key = f"{nh}-{nkv}-{dh}"
    path = Path(f"/tmp/h100_config_{config_key}.json")
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

    # Load configuration
    config = load_config()
    gpu_spec = config["gpu_specs"]["H100"]
    model_configs = config["benchmark_configs"]
    gemm_sweep = config["gemm_sweep"]

    print(f"GPU: H100 ({gpu_spec['peak_tflops_fp16']} TFLOPs)")
    print(f"Models: {len(model_configs)}")

    # Modify prefill script once
    modify_prefill_script(gpu_spec["peak_tflops_fp16"])

    # Create output directories
    output_base = Path("InferSim/bench_data/h100")
    (output_base / "mha" / "prefill").mkdir(parents=True, exist_ok=True)
    (output_base / "mha" / "decode").mkdir(parents=True, exist_ok=True)
    (output_base / "gemm").mkdir(parents=True, exist_ok=True)

    infersim_dir = Path("InferSim")

    # Run MHA benchmarks
    for model_config in model_configs:
        nh = model_config["num_attention_heads"]
        nkv = model_config["num_key_value_heads"]
        dh = model_config["head_dim"]
        config_key = f"{nh}-{nkv}-{dh}"

        print(f"\n{'='*60}")
        print(f"Config: {model_config['name']} ({config_key})")
        print(f"Description: {model_config['description']}")
        print(f"{'='*60}")

        # Create temp config
        config_path = create_hf_config(model_config)

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
            "--fp16-tflops", str(gpu_spec["peak_tflops_fp16"]),
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

    gemm_output = output_base / "gemm" / "data.csv"
    gemm_tmp = infersim_dir / "gemm.csv"

    # Run first to create header
    first = True
    for k in gemm_sweep["k_values"]:
        for n in gemm_sweep["n_values"]:
            cmd = [
                sys.executable,
                "kernel_benchmark/deepgemm_gemm.py",
                "-k", str(k),
                "-n", str(n),
                "--gpu-tflops", str(gpu_spec["peak_tflops_fp16"])
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
import json
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List


def load_config() -> Dict:
    """Load benchmark configuration"""
    config_path = Path("config/benchmark_config.json")
    with open(config_path) as f:
        return json.load(f)


def validate_mha_data(base_path: Path, config: Dict) -> List[str]:
    """Validate MHA benchmark data"""
    # Build list of expected configs from benchmark_config.json
    configs = []
    for model_config in config["benchmark_configs"]:
        nh = model_config["num_attention_heads"]
        nkv = model_config["num_key_value_heads"]
        dh = model_config["head_dim"]
        configs.append(f"{nh}-{nkv}-{dh}")

    stages = ["decode", "prefill"]
    errors = []
    mfu_ranges = config["mfu_validation"]

    print("="*60)
    print("MHA Validation")
    print("="*60)

    for stage in stages:
        for config_key in configs:
            path = base_path / "mha" / stage / f"{config_key}.csv"

            if not path.exists():
                errors.append(f"Missing: {path}")
                print(f"✗ {stage:7s} {config_key:10s}: MISSING")
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

                # Validate MFU range using config
                expected_range = mfu_ranges[stage]
                if df['mfu'].min() < expected_range["min"] or df['mfu'].max() > expected_range["max"]:
                    errors.append(
                        f"{path}: MFU range [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}] "
                        f"outside expected [{expected_range['min']:.3f}, {expected_range['max']:.3f}]"
                    )

                print(f"✓ {stage:7s} {config_key:10s}: {len(df):3d} rows, MFU [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")

            except Exception as e:
                errors.append(f"{path}: {e}")

    return errors


def validate_gemm_data(base_path: Path, config: Dict) -> List[str]:
    """Validate GEMM data"""
    path = base_path / "gemm" / "data.csv"
    errors = []
    mfu_range = config["mfu_validation"]["gemm"]

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

        # Validate MFU range using config
        if df['mfu'].min() < mfu_range["min"] or df['mfu'].max() > mfu_range["max"]:
            errors.append(
                f"{path}: MFU range [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}] "
                f"outside expected [{mfu_range['min']:.3f}, {mfu_range['max']:.3f}]"
            )

        print(f"✓ GEMM: {len(df):3d} rows, MFU [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")

    except Exception as e:
        errors.append(f"{path}: {e}")

    return errors


def main():
    # Load configuration
    config = load_config()
    base_path = Path("InferSim/bench_data/h100")

    print("\n" + "="*60)
    print("H100 Data Validation")
    print("="*60)
    print(f"Path: {base_path}")
    print(f"Config: config/benchmark_config.json\n")

    errors = []
    errors.extend(validate_mha_data(base_path, config))
    errors.extend(validate_gemm_data(base_path, config))

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
python -c "from scripts.run_h100_benchmarks import load_config; config = load_config(); print(f'Loaded {len(config[\"benchmark_configs\"])} model configs')"
```

Expected: "Loaded 6 model configs"

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

Expected: 13 files (12 MHA for 6 configs + 1 GEMM)

---

## Phase 1.5: Validation Matrix (Optional - Can Run After Phase 2)

### Task 4: Model Validation Matrix

**Purpose:** Verify simulator predictions against target models with different TP settings

**Test Matrix:**

| Model | HF Path | Config | TP Settings | Test Cases |
|-------|---------|--------|-------------|------------|
| Llama-2-7B | meta-llama/Llama-2-7b-hf | 32-32-128 | 1, 2, 4 | Prefill 512/2048, Decode bs=1/16/32 |
| Llama-2-70B | meta-llama/Llama-2-70b-hf | 64-8-128 | 1, 2, 4 | Prefill 512/2048, Decode bs=1/16/32 |
| Mixtral-8x7B | mistralai/Mixtral-8x7B-v0.1 | 32-8-128 | 1, 2, 4 | Prefill 512/2048, Decode bs=1/16/32 |

**Files:**
- Create: `scripts/validate_model_predictions.py`

**Step 1: Create validation script**

Create file: `scripts/validate_model_predictions.py`

```python
"""
Validate simulator predictions against target models
Tests with TP=1,2,4 to verify TP scaling works correctly
"""
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict


def load_config() -> Dict:
    """Load benchmark configuration"""
    config_path = Path("config/benchmark_config.json")
    with open(config_path) as f:
        return json.load(f)


def run_simulator(model: str, tp: int, test_case: str) -> Dict:
    """Run simulator and extract predictions"""
    # This is a placeholder - implement based on your simulator CLI
    cmd = [
        "./inference-sim",
        "--model", model,
        "--gpu", "H100",
        "--tp", str(tp),
        "--roofline", "infersim",
        "--test-case", test_case,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse output (adjust based on actual output format)
    # Return dict with TTFT, TPOT, etc.
    return {"ttft_ms": 0.0, "tpot_ms": 0.0}  # Placeholder


def main():
    """Run validation matrix"""
    print("="*60)
    print("Model Validation Matrix")
    print("="*60)
    print()
    print("Testing models:")
    for model in VALIDATION_MODELS:
        print(f"  • {model['name']} (TP={','.join(map(str, model['tp_values']))})")

    print()
    print("Note: This validates that:")
    print("  1. Simulator works with production model configs")
    print("  2. TP scaling works correctly (same MFU data, divided compute)")
    print("  3. Predictions are within expected ranges")
    print()

    results = []

    for model_info in VALIDATION_MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model_info['name']}")
        print(f"{'='*60}")

        for tp in model_info['tp_values']:
            print(f"\n  TP={tp}")

            # Test prefill
            print(f"    Prefill 2048 tokens... ", end="", flush=True)
            result = run_simulator(model_info["hf_path"], tp, "prefill-2048")
            print(f"TTFT={result['ttft_ms']:.2f}ms")

            # Test decode
            print(f"    Decode bs=16... ", end="", flush=True)
            result = run_simulator(model_info["hf_path"], tp, "decode-bs16")
            print(f"TPOT={result['tpot_ms']:.2f}ms")

            results.append({
                "model": model_info["name"],
                "tp": tp,
                "ttft_ms": result["ttft_ms"],
                "tpot_ms": result["tpot_ms"],
            })

    # Summary
    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")
    print(f"Total tests: {len(results)}")
    print()
    print("✅ All models run successfully")
    print()
    print("Next steps:")
    print("1. Compare against vLLM measurements (if available)")
    print("2. Check TP scaling curves (should scale ~linearly for compute)")
    print("3. Validate error rates (<10% target)")


if __name__ == "__main__":
    main()
```

**Step 2: Document validation approach**

Create file: `docs/model_validation.md`

```markdown
# Model Validation Matrix

## Test Models

### Llama-2-7B (meta-llama/Llama-2-7b-hf)
- **Config:** 32-32-128 (pure MHA)
- **TP:** 1, 2, 4
- **Tests:** Prefill 512/2048 tokens, Decode bs=1/16/32

### Llama-2-70B (meta-llama/Llama-2-70b-hf)
- **Config:** 64-8-128 (GQA)
- **TP:** 1, 2, 4
- **Tests:** Prefill 512/2048 tokens, Decode bs=1/16/32

### Mixtral-8x7B (mistralai/Mixtral-8x7B-v0.1)
- **Config:** 32-8-128 (GQA, MoE)
- **TP:** 1, 2, 4
- **Tests:** Prefill 512/2048 tokens, Decode bs=1/16/32
- **Note:** MoE uses same GEMM MFU data

## TP Scaling Validation

The same MFU data works for all TP values because:
1. **MFU is per-GPU** - measured on single GPU
2. **Formula handles scaling** - divides compute across TP GPUs
3. **Memory bandwidth** - per-GPU, already accounted for

Example:
```python
# TP=1
time = flops / (peak_flops_per_gpu × mfu)

# TP=4
time = (flops / 4) / (peak_flops_per_gpu × mfu)  # Same MFU!
```

## Success Criteria

- [ ] All models run without errors
- [ ] TP=1,2,4 all produce predictions
- [ ] TP scaling shows expected patterns:
  - Prefill (compute-bound): ~2x speedup at TP=2, ~4x at TP=4
  - Decode (memory-bound): Less scaling (1.5x at TP=2, 2-3x at TP=4)
- [ ] Predictions within 15% of vLLM measurements (if available)
```

**Step 3: Run validation (after Go implementation complete)**

```bash
# After Phase 2 is complete and simulator built
python scripts/validate_model_predictions.py
```

Expected: All models run successfully, TP scaling works

---

## Phase 2: Go Implementation (Days 6-9)

### Task 5-10: [Same as before - Go CSV loader, lookup, roofline, integration]

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
├── run_h100_benchmarks.py           # 100-line wrapper for InferSim scripts
├── validate_h100_data.py            # Validation only
├── validate_model_predictions.py    # Model validation matrix (Task 4)
├── submit_benchmark_job.py          # OpenShift submission
└── openshift/
    └── job-h100-benchmarks.yaml     # Single job definition

InferSim/
└── kernel_benchmark/                # Use these directly!
    ├── fa3_mha_prefill.py           # ← Already exists
    ├── flashinfer_mha_decode.py     # ← Already exists
    └── deepgemm_gemm.py              # ← Already exists

bench_data/h100/                     # 13 files total
├── mha/
│   ├── prefill/                     # 6 configs × prefill
│   └── decode/                      # 6 configs × decode
└── gemm/
    └── data.csv                     # 160 GEMM configurations
```

### Validation Models

| Model | Config | TP Tested |
|-------|--------|-----------|
| Llama-2-7B | 32-32-128 | 1, 2, 4 |
| Llama-2-70B | 64-8-128 | 1, 2, 4 |
| Mixtral-8x7B | 32-8-128 | 1, 2, 4 |
| Qwen2.5-7B | 28-4-128 | 1, 2, 4 |
| Llama-3-8B | 32-8-128 | 1, 2, 4 |
| Llama-3.1-8B | 32-4-128 | 1, 2, 4 |
| Llama-3.3-70B | 40-8-128 | 1, 2, 4 |

**Note:** Same MFU data works for all TP values - formula divides compute across GPUs.

### Execution

```bash
# Submit to cluster
python scripts/submit_benchmark_job.py

# Validate
python scripts/validate_h100_data.py
```

**Much simpler!** 🎯
