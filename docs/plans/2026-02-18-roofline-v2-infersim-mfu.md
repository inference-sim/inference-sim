# Roofline V2: InferSim MFU-Based Implementation Plan (Revised)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace calibrated roofline model with MFU-based InferSim approach for 4-10% prediction accuracy

**Architecture:** Python scripts benchmark H100 kernels → generate CSV files with pre-computed MFU values → Go loads CSVs at startup → performs table lookups during simulation. NO GPU code in Go, NO runtime benchmarking.

**Tech Stack:** Python (FlashAttention-3, FlashInfer, DeepGEMM, orchestration), Go (CSV parsing, nearest-neighbor lookup), OpenShift (cluster access via `oc`)

**Key Constraint:** Python phase REQUIRES H100 cluster access via OpenShift. Go phase works on any machine (no GPU needed).

**Revision:** Uses Python orchestration instead of bash, OpenShift Jobs instead of SLURM.

---

## ★ Insight ─────────────────────────────────────
**Critical Understanding of Architecture:**
1. **Python = One-Time Benchmarking** - Runs FlashAttention/GEMM kernels, measures latency, calculates MFU, outputs CSVs (~17 files, <1MB)
2. **Go = Runtime CSV Reader** - Loads pre-computed MFU data, does nearest-neighbor lookups, uses formula: `time = max(flops/(peak*mfu), bytes/bw)`
3. **NO GPU CODE IN GO** - Go never executes kernels, only reads static CSV files and performs table lookups
4. **Python for Everything** - Orchestration, benchmarking, validation all in Python
5. **OpenShift Deployment** - Use `oc` to submit jobs, no SLURM scripts
─────────────────────────────────────────────────

---

## Phase 1: Python Benchmarking on H100 (Days 1-5)

### Task 1: Python Orchestrator for MHA Benchmarks

**Prerequisites:** H100 GPU cluster access via `oc`, InferSim repo cloned

**Files:**
- Create: `scripts/benchmark_mha_h100.py`
- Create: `scripts/openshift_job_mha.yaml`

**Step 1: Write test for benchmark orchestrator**

Create file: `scripts/test_benchmark_orchestrator.py`

```python
"""Test benchmark orchestrator"""
import tempfile
from pathlib import Path
import pytest
from benchmark_mha_h100 import BenchmarkConfig, create_hf_config, run_benchmark_suite

def test_create_hf_config():
    """Test HuggingFace config generation"""
    config = create_hf_config(nh=28, nkv=4, dh=128)

    assert config["num_attention_heads"] == 28
    assert config["num_key_value_heads"] == 4
    assert config["head_dim"] == 128
    assert config["hidden_size"] == 28 * 128
    assert config["torch_dtype"] == "bfloat16"

def test_benchmark_config():
    """Test benchmark configuration object"""
    config = BenchmarkConfig(
        name="qwen2.5-7b",
        nh=28,
        nkv=4,
        dh=128
    )

    assert config.name == "qwen2.5-7b"
    assert config.config_key == "28-4-128"
    assert config.hidden_size == 3584

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Step 2: Run test to verify it fails**

```bash
python scripts/test_benchmark_orchestrator.py
```

Expected: FAIL with "ModuleNotFoundError: No module named 'benchmark_mha_h100'"

**Step 3: Implement benchmark orchestrator**

Create file: `scripts/benchmark_mha_h100.py`

```python
"""
H100 MHA Benchmark Orchestrator
Runs FlashAttention-3 and FlashInfer benchmarks for multiple configs
"""
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import shutil


@dataclass
class BenchmarkConfig:
    """Configuration for a single attention benchmark"""
    name: str          # Model name (e.g., "qwen2.5-7b")
    nh: int           # num_attention_heads
    nkv: int          # num_key_value_heads
    dh: int           # head_dim

    @property
    def config_key(self) -> str:
        """Config identifier: 'nh-nkv-dh'"""
        return f"{self.nh}-{self.nkv}-{self.dh}"

    @property
    def hidden_size(self) -> int:
        """Hidden dimension: nh × dh"""
        return self.nh * self.dh


def create_hf_config(nh: int, nkv: int, dh: int) -> dict:
    """Create HuggingFace-style config for benchmarking"""
    return {
        "hidden_size": nh * dh,
        "num_attention_heads": nh,
        "num_key_value_heads": nkv,
        "head_dim": dh,
        "num_hidden_layers": 32,
        "intermediate_size": 14336,
        "torch_dtype": "bfloat16"
    }


def modify_prefill_script_for_h100(infersim_path: Path):
    """Modify fa3_mha_prefill.py to use H100 TFLOPs (989.5)"""
    script_path = infersim_path / "kernel_benchmark" / "fa3_mha_prefill.py"

    # Read file
    content = script_path.read_text()

    # Replace fp16_tflops value
    modified = content.replace("fp16_tflops = 148", "fp16_tflops = 989.5")

    # Backup original
    backup_path = script_path.with_suffix(".py.bak")
    if not backup_path.exists():
        shutil.copy(script_path, backup_path)

    # Write modified
    script_path.write_text(modified)
    print(f"✓ Modified {script_path} for H100 (989.5 TFLOPs)")


def run_mha_prefill(
    config: BenchmarkConfig,
    infersim_path: Path,
    output_dir: Path
) -> Optional[Path]:
    """Run FlashAttention-3 prefill benchmark"""
    print(f"\n→ Running prefill benchmark: {config.name} ({config.config_key})")

    # Create temp config file
    temp_config = Path(f"/tmp/h100_config_{config.config_key}.json")
    temp_config.write_text(json.dumps(create_hf_config(config.nh, config.nkv, config.dh), indent=2))

    # Run benchmark
    script = infersim_path / "kernel_benchmark" / "fa3_mha_prefill.py"
    cmd = [
        sys.executable,
        str(script),
        "--config-path", str(temp_config)
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=infersim_path,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )

        if result.returncode != 0:
            print(f"✗ Prefill failed: {result.stderr}")
            return None

        # Move output to destination
        output_file = output_dir / "mha" / "prefill" / f"{config.config_key}.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        source = infersim_path / "attention_benchmark.csv"
        if source.exists():
            shutil.move(source, output_file)
            print(f"  ✓ Prefill: {output_file} ({count_csv_rows(output_file)} benchmarks)")
            return output_file
        else:
            print(f"✗ Output file not found: {source}")
            return None

    except subprocess.TimeoutExpired:
        print("✗ Prefill benchmark timeout")
        return None
    except Exception as e:
        print(f"✗ Prefill error: {e}")
        return None


def run_mha_decode(
    config: BenchmarkConfig,
    infersim_path: Path,
    output_dir: Path
) -> Optional[Path]:
    """Run FlashInfer decode benchmark"""
    print(f"\n→ Running decode benchmark: {config.name} ({config.config_key})")

    # Create temp config file
    temp_config = Path(f"/tmp/h100_config_{config.config_key}.json")
    temp_config.write_text(json.dumps(create_hf_config(config.nh, config.nkv, config.dh), indent=2))

    # Run benchmark
    script = infersim_path / "kernel_benchmark" / "flashinfer_mha_decode.py"
    cmd = [
        sys.executable,
        str(script),
        "--config-path", str(temp_config),
        "--fp16-tflops", "989.5",
        "--kv-cache-dtype", "bf16",
        "--tp-size", "1"
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=infersim_path,
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            print(f"✗ Decode failed: {result.stderr}")
            return None

        # Move output
        output_file = output_dir / "mha" / "decode" / f"{config.config_key}.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        source = infersim_path / "attention_benchmark.csv"
        if source.exists():
            shutil.move(source, output_file)
            print(f"  ✓ Decode: {output_file} ({count_csv_rows(output_file)} benchmarks)")
            return output_file
        else:
            print(f"✗ Output file not found: {source}")
            return None

    except subprocess.TimeoutExpired:
        print("✗ Decode benchmark timeout")
        return None
    except Exception as e:
        print(f"✗ Decode error: {e}")
        return None


def count_csv_rows(csv_path: Path) -> int:
    """Count data rows in CSV (excluding header)"""
    try:
        return sum(1 for _ in open(csv_path)) - 1
    except:
        return 0


def run_benchmark_suite(
    configs: List[BenchmarkConfig],
    infersim_path: Path,
    output_dir: Path
) -> dict:
    """Run complete benchmark suite for all configs"""
    results = {
        "prefill": [],
        "decode": [],
        "errors": []
    }

    # Modify prefill script once
    modify_prefill_script_for_h100(infersim_path)

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config.name} ({config.config_key})")
        print(f"{'='*60}")

        # Run prefill
        prefill_output = run_mha_prefill(config, infersim_path, output_dir)
        if prefill_output:
            results["prefill"].append(str(prefill_output))
        else:
            results["errors"].append(f"Prefill failed: {config.config_key}")

        # Run decode
        decode_output = run_mha_decode(config, infersim_path, output_dir)
        if decode_output:
            results["decode"].append(str(decode_output))
        else:
            results["errors"].append(f"Decode failed: {config.config_key}")

    return results


def main():
    """Main entry point"""
    # High priority configs
    configs = [
        BenchmarkConfig(name="qwen2.5-7b", nh=28, nkv=4, dh=128),
        BenchmarkConfig(name="llama-3-8b", nh=32, nkv=8, dh=128),
        BenchmarkConfig(name="llama-3.1-8b", nh=32, nkv=4, dh=128),
    ]

    infersim_path = Path("InferSim")
    output_dir = Path("InferSim/bench_data/h100")

    if not infersim_path.exists():
        print(f"✗ InferSim directory not found: {infersim_path}")
        sys.exit(1)

    print("="*60)
    print("H100 MHA Benchmark Suite")
    print("="*60)
    print(f"Configs: {len(configs)}")
    print(f"InferSim: {infersim_path}")
    print(f"Output: {output_dir}")

    results = run_benchmark_suite(configs, infersim_path, output_dir)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Prefill benchmarks: {len(results['prefill'])}")
    print(f"✓ Decode benchmarks: {len(results['decode'])}")

    if results["errors"]:
        print(f"\n✗ Errors: {len(results['errors'])}")
        for err in results["errors"]:
            print(f"  • {err}")
        sys.exit(1)
    else:
        print("\n✅ All benchmarks completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

```bash
python scripts/test_benchmark_orchestrator.py
```

Expected: PASS

**Step 5: Test orchestrator locally (dry run)**

```bash
# Verify script imports work
python -c "from scripts.benchmark_mha_h100 import BenchmarkConfig; print('OK')"
```

Expected: "OK"

**Step 6: Commit**

```bash
git add scripts/benchmark_mha_h100.py scripts/test_benchmark_orchestrator.py
git commit -m "feat(roofline-v2): add Python MHA benchmark orchestrator

- Replace bash scripts with Python orchestration
- BenchmarkConfig dataclass for type safety
- Subprocess management for kernel benchmarks
- Error handling and timeout protection
- Progress reporting

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 2: GEMM Benchmark Script (Pure Python)

**Files:**
- Create: `scripts/benchmark_gemm_h100.py`

**Step 1: Write test for GEMM benchmark**

Create file: `scripts/test_gemm_benchmark.py`

```python
"""Test GEMM benchmark"""
import torch
from benchmark_gemm_h100 import benchmark_single_gemm, generate_gemm_configs

def test_generate_gemm_configs():
    """Test GEMM config generation"""
    configs = generate_gemm_configs()

    # Should have M × K × N combinations
    assert len(configs) > 100

    # Check structure
    for m, k, n in configs[:5]:
        assert m > 0
        assert k > 0
        assert n > 0

def test_benchmark_single_gemm():
    """Test single GEMM benchmark (CPU only for unit test)"""
    # Small test on CPU
    result = benchmark_single_gemm(
        m=16,
        k=128,
        n=256,
        device="cpu",
        peak_tflops=1.0,  # Dummy value
        num_trials=10
    )

    assert "latency_us" in result
    assert "mfu" in result
    assert result["latency_us"] > 0
```

**Step 2: Run test to verify it fails**

```bash
python scripts/test_gemm_benchmark.py
```

Expected: FAIL with "ModuleNotFoundError: No module named 'benchmark_gemm_h100'"

**Step 3: Implement GEMM benchmark**

Create file: `scripts/benchmark_gemm_h100.py`

```python
"""
H100 GEMM Benchmark
Measures MFU for various (M, K, N) configurations
"""
import torch
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import sys


def generate_gemm_configs() -> List[Tuple[int, int, int]]:
    """Generate comprehensive GEMM configurations for LLM workloads"""
    M_values = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    K_values = [2048, 3584, 4096, 8192]
    N_values = [6144, 11008, 14336, 18944]

    configs = []
    for m in M_values:
        for k in K_values:
            for n in N_values:
                configs.append((m, k, n))

    return configs


def benchmark_single_gemm(
    m: int,
    k: int,
    n: int,
    device: str = "cuda",
    peak_tflops: float = 989.5,
    num_trials: int = 100
) -> Dict[str, float]:
    """Benchmark a single GEMM: C = A @ B where A is (m, k), B is (k, n)"""

    dtype = torch.bfloat16

    # Create matrices
    A = torch.randn(m, k, device=device, dtype=dtype)
    B = torch.randn(k, n, device=device, dtype=dtype)

    # Warmup
    for _ in range(10):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_trials):
        C = torch.matmul(A, B)
    end.record()
    torch.cuda.synchronize()

    # Calculate metrics
    elapsed_ms = start.elapsed_time(end) / num_trials
    latency_us = elapsed_ms * 1000

    # FLOPs: 2 ops per multiply-add
    flops = 2 * m * k * n
    achieved_tflops = flops / (elapsed_ms / 1000) / 1e12
    mfu = achieved_tflops / peak_tflops

    return {
        "m": m,
        "k": k,
        "n": n,
        "latency_us": round(latency_us, 3),
        "mfu": round(mfu, 4)
    }


def run_gemm_suite(
    output_path: Path,
    device: str = "cuda",
    peak_tflops: float = 989.5
) -> pd.DataFrame:
    """Run complete GEMM benchmark suite"""

    configs = generate_gemm_configs()
    total = len(configs)

    print("="*60)
    print(f"H100 GEMM Benchmark Suite")
    print("="*60)
    print(f"Total configurations: {total}")
    print(f"Device: {device}")
    print(f"Peak TFLOPs: {peak_tflops}")
    print()

    results = []

    for idx, (m, k, n) in enumerate(configs, 1):
        print(f"[{idx}/{total}] Benchmarking M={m:4d}, K={k:4d}, N={n:5d}... ", end="", flush=True)

        try:
            result = benchmark_single_gemm(m, k, n, device, peak_tflops)
            results.append(result)
            print(f"MFU={result['mfu']:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                "m": m,
                "k": k,
                "n": n,
                "latency_us": 0,
                "mfu": 0
            })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary
    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Total benchmarks: {len(df)}")
    print(f"✓ MFU range: [{df['mfu'].min():.4f}, {df['mfu'].max():.4f}]")
    print(f"✓ Output: {output_path}")
    print()

    return df


def main():
    """Main entry point"""
    output_path = Path("InferSim/bench_data/h100/gemm/data.csv")

    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        sys.exit(1)

    # Verify GPU
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")

    if "H100" not in device_name:
        print(f"⚠ Warning: Expected H100, got {device_name}")

    # Run benchmark suite
    df = run_gemm_suite(output_path, device="cuda", peak_tflops=989.5)

    print("✅ GEMM benchmarks complete")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

**Step 4: Run unit tests**

```bash
python scripts/test_gemm_benchmark.py
```

Expected: PASS

**Step 5: Commit**

```bash
git add scripts/benchmark_gemm_h100.py scripts/test_gemm_benchmark.py
git commit -m "feat(roofline-v2): add Python GEMM benchmark suite

- Pure Python/PyTorch GEMM benchmarking
- 160 configurations (M × K × N sweep)
- MFU calculation with peak TFLOPs
- Progress reporting and error handling

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 3: Validation Script (Pure Python)

**Files:**
- Create: `scripts/validate_h100_data.py`

**Step 1: Implement validation script**

Create file: `scripts/validate_h100_data.py`

```python
"""
Validate H100 benchmark data
Checks MFU ranges, file existence, CSV format
"""
import pandas as pd
from pathlib import Path
import sys
from typing import List, Tuple


def validate_mha_data(base_path: Path) -> List[str]:
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

                # Check required columns
                if stage == "decode":
                    required = ["dtype", "kv_dtype", "batch_size", "kv_len", "latency_us", "mfu"]
                else:  # prefill
                    required = ["dtype", "seq_len", "latency_us", "mfu"]

                missing_cols = set(required) - set(df.columns)
                if missing_cols:
                    errors.append(f"{path}: Missing columns {missing_cols}")
                    print(f"✗ {stage:7s} {config:10s}: Missing columns {missing_cols}")
                    continue

                # Validate MFU range
                if stage == "decode":
                    # Decode is memory-bound: 0.01-0.20
                    if df['mfu'].min() < 0.005 or df['mfu'].max() > 0.30:
                        errors.append(f"{path}: Unexpected MFU range [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")
                        print(f"⚠ {stage:7s} {config:10s}: Unexpected MFU range")
                else:  # prefill
                    # Prefill is compute-bound: 0.40-0.80
                    if df['mfu'].min() < 0.30 or df['mfu'].max() > 0.90:
                        errors.append(f"{path}: Unexpected MFU range [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")
                        print(f"⚠ {stage:7s} {config:10s}: Unexpected MFU range")

                print(f"✓ {stage:7s} {config:10s}: {len(df):3d} benchmarks, MFU [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")

            except Exception as e:
                errors.append(f"{path}: Error reading CSV - {e}")
                print(f"✗ {stage:7s} {config:10s}: Error - {e}")

    return errors


def validate_gemm_data(base_path: Path) -> List[str]:
    """Validate GEMM benchmark data"""
    path = base_path / "gemm" / "data.csv"
    errors = []

    print()
    print("="*60)
    print("GEMM Validation")
    print("="*60)

    if not path.exists():
        errors.append(f"Missing: {path}")
        print(f"✗ GEMM: MISSING")
        return errors

    try:
        df = pd.read_csv(path)

        # Check required columns
        required = ["m", "k", "n", "latency_us", "mfu"]
        missing_cols = set(required) - set(df.columns)
        if missing_cols:
            errors.append(f"{path}: Missing columns {missing_cols}")
            print(f"✗ GEMM: Missing columns {missing_cols}")
            return errors

        # Validate MFU range (GEMMs vary widely: 0.10-0.90)
        if df['mfu'].min() < 0.05 or df['mfu'].max() > 1.0:
            errors.append(f"{path}: Unexpected MFU range [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")
            print(f"⚠ GEMM: Unexpected MFU range")

        print(f"✓ GEMM: {len(df):3d} benchmarks, MFU [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")

    except Exception as e:
        errors.append(f"{path}: Error reading CSV - {e}")
        print(f"✗ GEMM: Error - {e}")

    return errors


def main():
    """Main validation entry point"""
    base_path = Path("InferSim/bench_data/h100")

    print()
    print("="*60)
    print("H100 Benchmark Data Validation")
    print("="*60)
    print(f"Base path: {base_path}")
    print()

    errors = []
    errors.extend(validate_mha_data(base_path))
    errors.extend(validate_gemm_data(base_path))

    print()
    print("="*60)
    if errors:
        print(f"❌ Validation FAILED with {len(errors)} errors:")
        print("="*60)
        for err in errors:
            print(f"  • {err}")
        sys.exit(1)
    else:
        print("✅ All validations PASSED")
        print("="*60)
        sys.exit(0)


if __name__ == "__main__":
    main()
```

**Step 2: Test validation (will fail without data)**

```bash
python scripts/validate_h100_data.py || echo "Expected failure - no H100 data yet"
```

Expected: Exits with error (no data yet)

**Step 3: Commit**

```bash
git add scripts/validate_h100_data.py
git commit -m "feat(roofline-v2): add Python data validation

- Pure Python validation (no bash)
- Validates MHA prefill/decode CSVs
- Validates GEMM CSV
- Checks MFU ranges and column formats
- Clear error reporting

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 4: OpenShift Job Definitions

**Files:**
- Create: `scripts/openshift/job-mha-h100.yaml`
- Create: `scripts/openshift/job-gemm-h100.yaml`
- Create: `scripts/submit_benchmarks.py`

**Step 1: Create OpenShift Job for MHA benchmarks**

Create file: `scripts/openshift/job-mha-h100.yaml`

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: infersim-mha-h100
  labels:
    app: infersim-benchmark
    stage: mha
spec:
  backoffLimit: 2
  template:
    metadata:
      labels:
        app: infersim-benchmark
        stage: mha
    spec:
      restartPolicy: Never
      containers:
      - name: benchmark
        image: nvcr.io/nvidia/pytorch:24.01-py3
        command: ["python", "/workspace/scripts/benchmark_mha_h100.py"]
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
        - name: output
          mountPath: /workspace/InferSim/bench_data
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
      volumes:
      - name: workspace
        persistentVolumeClaim:
          claimName: inference-sim-workspace
      - name: output
        persistentVolumeClaim:
          claimName: inference-sim-output
      nodeSelector:
        accelerator: nvidia-h100
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

**Step 2: Create OpenShift Job for GEMM benchmarks**

Create file: `scripts/openshift/job-gemm-h100.yaml`

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: infersim-gemm-h100
  labels:
    app: infersim-benchmark
    stage: gemm
spec:
  backoffLimit: 2
  template:
    metadata:
      labels:
        app: infersim-benchmark
        stage: gemm
    spec:
      restartPolicy: Never
      containers:
      - name: benchmark
        image: nvcr.io/nvidia/pytorch:24.01-py3
        command: ["python", "/workspace/scripts/benchmark_gemm_h100.py"]
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
        - name: output
          mountPath: /workspace/InferSim/bench_data
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
      volumes:
      - name: workspace
        persistentVolumeClaim:
          claimName: inference-sim-workspace
      - name: output
        persistentVolumeClaim:
          claimName: inference-sim-output
      nodeSelector:
        accelerator: nvidia-h100
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

**Step 3: Create Python job submission script**

Create file: `scripts/submit_benchmarks.py`

```python
"""
Submit H100 benchmark jobs to OpenShift cluster
"""
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def run_oc_command(args: List[str], check=True) -> subprocess.CompletedProcess:
    """Run oc command"""
    cmd = ["oc"] + args
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def submit_job(yaml_path: Path) -> str:
    """Submit a job to OpenShift"""
    result = run_oc_command(["apply", "-f", str(yaml_path)])

    if result.returncode != 0:
        print(f"✗ Failed to submit {yaml_path}")
        print(result.stderr)
        sys.exit(1)

    # Extract job name
    job_name = yaml_path.stem.replace("job-", "infersim-")
    print(f"✓ Submitted job: {job_name}")
    return job_name


def wait_for_job(job_name: str, timeout: int = 7200):
    """Wait for job to complete"""
    print(f"\n→ Waiting for {job_name} to complete (timeout: {timeout}s)...")

    start = time.time()
    while time.time() - start < timeout:
        result = run_oc_command(
            ["get", "job", job_name, "-o", "jsonpath={.status.conditions[?(@.type==\"Complete\")].status}"],
            check=False
        )

        if result.stdout.strip() == "True":
            print(f"✓ Job {job_name} completed successfully")
            return True

        # Check for failure
        result = run_oc_command(
            ["get", "job", job_name, "-o", "jsonpath={.status.conditions[?(@.type==\"Failed\")].status}"],
            check=False
        )

        if result.stdout.strip() == "True":
            print(f"✗ Job {job_name} failed")
            show_job_logs(job_name)
            return False

        time.sleep(10)

    print(f"✗ Job {job_name} timeout")
    return False


def show_job_logs(job_name: str):
    """Show logs for a job"""
    print(f"\n→ Logs for {job_name}:")
    result = run_oc_command(["logs", f"job/{job_name}"], check=False)
    print(result.stdout)


def main():
    """Main entry point"""
    print("="*60)
    print("OpenShift H100 Benchmark Submission")
    print("="*60)

    # Check oc is available
    result = run_oc_command(["version"], check=False)
    if result.returncode != 0:
        print("✗ oc command not found. Install OpenShift CLI.")
        sys.exit(1)

    # Check logged in
    result = run_oc_command(["whoami"], check=False)
    if result.returncode != 0:
        print("✗ Not logged in to OpenShift. Run: oc login")
        sys.exit(1)

    print(f"✓ Logged in as: {result.stdout.strip()}")

    # Submit jobs
    job_dir = Path("scripts/openshift")

    print("\n" + "="*60)
    print("Submitting Jobs")
    print("="*60)

    mha_job = submit_job(job_dir / "job-mha-h100.yaml")
    gemm_job = submit_job(job_dir / "job-gemm-h100.yaml")

    # Wait for completion
    print("\n" + "="*60)
    print("Waiting for Completion")
    print("="*60)

    mha_success = wait_for_job(mha_job, timeout=3600)  # 1 hour
    gemm_success = wait_for_job(gemm_job, timeout=7200)  # 2 hours

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if mha_success and gemm_success:
        print("✅ All benchmarks completed successfully")
        print("\nNext steps:")
        print("1. Copy output: oc rsync <pod>:/workspace/InferSim/bench_data ./InferSim/")
        print("2. Validate: python scripts/validate_h100_data.py")
        sys.exit(0)
    else:
        print("❌ Some benchmarks failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Step 4: Test oc connectivity (local only)**

```bash
# This will fail if not logged in, which is expected
python -c "from scripts.submit_benchmarks import run_oc_command; run_oc_command(['version'])" || echo "Expected: requires oc login"
```

**Step 5: Commit**

```bash
git add scripts/openshift/ scripts/submit_benchmarks.py
git commit -m "feat(roofline-v2): add OpenShift job definitions

- Replace SLURM with OpenShift Jobs
- Python submission script using oc CLI
- H100 GPU resource requests
- Job monitoring and log viewing
- Automatic retry on failure

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

### Task 5: Execute Benchmarks (REQUIRES H100 + OpenShift)

**Prerequisites:** OpenShift access, H100 nodes available, logged in via `oc login`

**Step 1: Verify OpenShift login**

```bash
oc whoami
```

Expected: Your username

**Step 2: Submit benchmark jobs**

```bash
python scripts/submit_benchmarks.py
```

Expected: Jobs submitted, monitoring output

**Step 3: Monitor job progress (alternative)**

```bash
# Watch job status
oc get jobs -w

# View logs
oc logs -f job/infersim-mha-h100
```

**Step 4: Retrieve output data**

```bash
# Find the pod name
POD=$(oc get pods -l app=infersim-benchmark -o jsonpath='{.items[0].metadata.name}')

# Copy output data
oc rsync ${POD}:/workspace/InferSim/bench_data/h100 ./InferSim/bench_data/h100
```

Expected: CSV files copied locally

**Step 5: Validate data**

```bash
python scripts/validate_h100_data.py
```

Expected: "✅ All validations PASSED"

**Step 6: Copy to main repository**

```bash
# Create directory
mkdir -p bench_data/h100

# Copy validated data
cp -r InferSim/bench_data/h100/* bench_data/h100/

# Verify
find bench_data/h100 -name "*.csv" | wc -l
```

Expected: 7 files (6 MHA + 1 GEMM)

**Step 7: Commit benchmark data**

```bash
git add bench_data/h100/
git commit -m "data(roofline-v2): add H100 benchmark MFU data

- MHA prefill/decode for 3 configs (Qwen, Llama-3, Llama-3.1)
- GEMM data for 160 configurations
- Pre-computed MFU values for roofline model
- Total size: ~500KB

Benchmarked on OpenShift H100 cluster

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: Go Implementation (Days 6-9)

*[Phase 2 and 3 remain the same as original plan - they don't involve bash/SLURM]*

### Task 6-12: [Same as original plan]

See original plan for:
- Task 6: MFU Database Loader (Go)
- Task 7: MFU Lookup Logic (Go)
- Task 8: InferSim Roofline Implementation (Go)
- Task 9: Simulator Integration (Go)
- Task 10: Cross-Validation
- Task 11: End-to-End Validation
- Task 12: Documentation

---

## Summary of Changes

### Replaced:
- ❌ Bash scripts (`*.sh`) → ✅ Python scripts (`*.py`)
- ❌ SLURM submission → ✅ OpenShift Jobs (via `oc`)
- ❌ Shell scripting → ✅ Python subprocess management
- ❌ sed/awk/grep → ✅ Python string manipulation

### Benefits:
- ✅ **Type safety** - Python dataclasses for configs
- ✅ **Error handling** - Proper exception handling
- ✅ **Testing** - Unit tests for orchestration logic
- ✅ **Cross-platform** - Python works everywhere
- ✅ **Maintainable** - More readable than bash
- ✅ **OpenShift native** - Use `oc` CLI directly

### Key Scripts:
1. `benchmark_mha_h100.py` - MHA orchestrator (replaces bash loop)
2. `benchmark_gemm_h100.py` - GEMM suite (pure PyTorch)
3. `validate_h100_data.py` - Data validation (pure Python)
4. `submit_benchmarks.py` - OpenShift job submission (uses `oc`)

### Execution:
```bash
# Submit to cluster
python scripts/submit_benchmarks.py

# Monitor
oc get jobs -w

# Retrieve data
oc rsync <pod>:/workspace/InferSim/bench_data ./InferSim/

# Validate
python scripts/validate_h100_data.py
```

---

Plan complete and saved to `docs/plans/2026-02-18-roofline-v2-infersim-mfu-revised.md`.
