# Comprehensive H100 Benchmarking Design

**Date:** 2026-02-19
**Status:** Design Approved
**Goal:** Benchmark all 6 attention shapes for prefill, decode (TP=1/2/4), and GEMM operations

---

## Executive Summary

Generate complete MFU benchmark data for 6 attention shapes across all phases using wave-based parallel orchestration on OpenShift H100 cluster.

**Key Insight:** Benchmarks are organized by **attention shape** (nh-nkv-dh), NOT by model name. A 32-8-128 benchmark works for Mistral-7B, Mixtral-8x7B, and any model with that shape.

**Total Jobs:** 25 (1 GEMM + 6 shapes × 4 phases)
**Peak GPU Usage:** 5 H100s (1 GEMM + 4 for current shape)
**Total Wall Time:** ~90-120 minutes
**Namespace:** `diya`

---

## Current State Analysis

### What Already Exists

1. **Infrastructure:**
   - `scripts/openshift/generate_job.py` - Job YAML generator
   - `scripts/openshift/job-benchmarks-template.yaml` - Job template
   - `config/benchmark_config.json` - Has 2 configs but validation_model only for later testing

2. **InferSim Scripts:** (Production-ready)
   - `fa3_mha_prefill.py` - Takes attention shape params via config
   - `flashinfer_mha_decode.py` - Takes shape + --tp-size
   - `deepgemm_gemm.py` - Takes -k, -n dimensions

3. **OpenShift Config:**
   - Namespace: `diya`
   - Container: `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel`
   - Node selector for H100

### What Needs to Be Built

1. ~~Expand config~~ - Config is just for reference, not needed for benchmarking!
2. **Orchestration script** - Wave-based job submission with waiting
3. **Job generator** - Pass attention shape (nh, nkv, dh) to InferSim scripts
4. **Result collector** - Gather CSVs from 25 completed pods

---

## Design

### Attention Shapes to Benchmark

From `scripts/ATTENTION_CONFIGS.md`:

| Shape | Architecture | Used By |
|-------|--------------|---------|
| **28-4-128** | GQA (7:1) | Qwen2-7B, Qwen2.5-7B |
| **32-32-128** | MHA | Llama-2-7B, Llama-1-7B |
| **32-8-128** | GQA (4:1) | Mistral-7B, Mixtral-8x7B |
| **40-40-128** | MHA | Llama-2-13B, Qwen-14B |
| **56-8-128** | GQA (7:1) | CodeLlama-34B |
| **64-8-128** | GQA (8:1) | Llama-2-70B, Qwen2-72B |

**Benchmark by shape only** - no model names needed during benchmarking!

---

### Wave-Based Orchestration

#### Wave 0: GEMM (Independent)

```
Job: infersim-h100-gemm-{timestamp}
├─ Runs: deepgemm_gemm.py with K×N sweep
├─ Output: InferSim/bench_data/h100/gemm/data.csv
├─ Sweep: 4 K values × 4 N values × 15 M values = 240 configs
└─ Runtime: ~30-45 minutes (runs in background)
```

#### Waves 1-6: Attention Shapes (Sequential waves, parallel within)

For each shape (28-4-128, 32-32-128, ..., 64-8-128):

```
Shape Wave N (4 parallel jobs):
├─ Job 1: infersim-h100-{nh}-{nkv}-{dh}-prefill-{ts}
│  ├─ Args: --num-attention-heads {nh} --num-kv-heads {nkv} --head-dim {dh}
│  ├─ Output: InferSim/bench_data/h100/mha/prefill/{nh}-{nkv}-{dh}.csv
│  └─ Runtime: ~15-20 min
│
├─ Job 2: infersim-h100-{nh}-{nkv}-{dh}-decode-tp1-{ts}
│  ├─ Args: --num-attention-heads {nh} --num-kv-heads {nkv} --head-dim {dh} --tp-size 1
│  ├─ Output: InferSim/bench_data/h100/mha/decode/{nh}-{nkv}-{dh}-tp1.csv
│  └─ Runtime: ~15-20 min
│
├─ Job 3: infersim-h100-{nh}-{nkv}-{dh}-decode-tp2-{ts}
│  ├─ Args: same with --tp-size 2
│  └─ Output: .../decode/{nh}-{nkv}-{dh}-tp2.csv
│
└─ Job 4: infersim-h100-{nh}-{nkv}-{dh}-decode-tp4-{ts}
   ├─ Args: same with --tp-size 4
   └─ Output: .../decode/{nh}-{nkv}-{dh}-tp4.csv

Wait for all 4 jobs to complete, then launch next shape wave
```

---

### Orchestration Script Logic

**Script:** `scripts/orchestrate_benchmarks.py`

```python
# Attention shapes (no model names!)
SHAPES = [
    {"nh": 28, "nkv": 4, "dh": 128},   # Qwen2
    {"nh": 32, "nkv": 32, "dh": 128},  # Llama-2-7B
    {"nh": 32, "nkv": 8, "dh": 128},   # Mistral
    {"nh": 40, "nkv": 40, "dh": 128},  # Llama-2-13B
    {"nh": 56, "nkv": 8, "dh": 128},   # CodeLlama-34B
    {"nh": 64, "nkv": 8, "dh": 128},   # Llama-2-70B
]

# Wave 0: GEMM (background)
gemm_job = submit_gemm_job()

# Waves 1-6: Process each shape
for i, shape in enumerate(SHAPES):
    config_key = f"{shape['nh']}-{shape['nkv']}-{shape['dh']}"
    print(f"\n=== Wave {i+1}/6: {config_key} ===")

    # Launch 4 parallel jobs
    jobs = [
        submit_shape_job(shape, phase="prefill", tp=None),
        submit_shape_job(shape, phase="decode", tp=1),
        submit_shape_job(shape, phase="decode", tp=2),
        submit_shape_job(shape, phase="decode", tp=4),
    ]

    # Wait for all 4 to complete (15-20 minutes)
    wait_for_jobs(jobs, timeout_minutes=30)

    print(f"✓ Wave {i+1}/6 complete")

# Wait for GEMM if still running
wait_for_job(gemm_job, timeout_minutes=60)

print("\n✅ All benchmarks complete!")
```

**Key Functions:**

```python
def submit_shape_job(shape: dict, phase: str, tp: int) -> str:
    """
    Submit job for a specific shape+phase+tp combination.

    Calls generate_job.py with shape parameters:
    - Phase prefill: calls fa3_mha_prefill.py --num-attention-heads {nh} ...
    - Phase decode: calls flashinfer_mha_decode.py --num-attention-heads {nh} --tp-size {tp} ...

    Returns job_name for monitoring
    """
    pass

def wait_for_jobs(job_names: list, timeout_minutes: int):
    """
    Poll oc get job until all reach Complete or Failed.
    Returns when all are done (doesn't fail on job failures).
    """
    pass
```

---

### Output Structure

```
InferSim/bench_data/h100/
├── gemm/
│   └── data.csv                    # 240 rows (K×N×M sweep)
│
└── mha/
    ├── prefill/
    │   ├── 28-4-128.csv           # 6 files
    │   ├── 32-32-128.csv
    │   ├── 32-8-128.csv
    │   ├── 40-40-128.csv
    │   ├── 56-8-128.csv
    │   └── 64-8-128.csv
    │
    └── decode/
        ├── 28-4-128-tp1.csv       # 18 files (6 shapes × 3 TPs)
        ├── 28-4-128-tp2.csv
        ├── 28-4-128-tp4.csv
        ├── 32-32-128-tp1.csv
        ├── 32-32-128-tp2.csv
        ├── 32-32-128-tp4.csv
        ├── 32-8-128-tp1.csv
        ├── 32-8-128-tp2.csv
        ├── 32-8-128-tp4.csv
        ├── 40-40-128-tp1.csv
        ├── 40-40-128-tp2.csv
        ├── 40-40-128-tp4.csv
        ├── 56-8-128-tp1.csv
        ├── 56-8-128-tp2.csv
        ├── 56-8-128-tp4.csv
        ├── 64-8-128-tp1.csv
        ├── 64-8-128-tp2.csv
        └── 64-8-128-tp4.csv

Total: 25 CSV files
```

---

## Implementation Tasks

### Task 1: Modify generate_job.py

**Current state:** Takes `--model`, `--phase`, `--tp` and looks up model in benchmark_config.json

**Changes needed:**
- Add `--shape` option that takes "nh-nkv-dh" format (e.g., "32-8-128")
- Extract nh, nkv, dh from shape string
- Pass shape parameters directly to InferSim scripts
- **No config lookup needed** - shape is self-contained

**Example:**
```bash
python scripts/openshift/generate_job.py \
    --gpu H100 \
    --shape 32-8-128 \
    --phase prefill \
    --suffix 20260219-140000

# Generates job that runs:
# python InferSim/kernel_benchmark/fa3_mha_prefill.py \
#   --num-attention-heads 32 \
#   --num-kv-heads 8 \
#   --head-dim 128
```

### Task 2: Create orchestrate_benchmarks.py

**Purpose:** Automated wave-based job submission and monitoring

**Core logic:**
```python
SHAPES = [
    (28, 4, 128),   # Qwen2
    (32, 32, 128),  # Llama-2-7B
    (32, 8, 128),   # Mistral
    (40, 40, 128),  # Llama-2-13B
    (56, 8, 128),   # CodeLlama-34B
    (64, 8, 128),   # Llama-2-70B
]

def submit_job(shape_tuple, phase, tp):
    """Generate job YAML and submit to cluster"""
    nh, nkv, dh = shape_tuple
    shape_str = f"{nh}-{nkv}-{dh}"

    # Generate YAML
    subprocess.run([
        "python", "scripts/openshift/generate_job.py",
        "--gpu", "H100",
        "--shape", shape_str,
        "--phase", phase,
        "--tp", str(tp) if tp else "",
        "--suffix", timestamp
    ])

    # Submit to OpenShift
    job_yaml = f"scripts/openshift/job-h100-{shape_str}-{phase}-{tp or ''}.yaml"
    subprocess.run(["oc", "apply", "-f", job_yaml, "-n", "diya"])

    return extract_job_name_from_yaml(job_yaml)

def wait_for_jobs(job_names, timeout_minutes):
    """Poll until all jobs reach Complete or Failed"""
    deadline = time.time() + timeout_minutes * 60

    while time.time() < deadline:
        statuses = [get_job_status(j) for j in job_names]

        if all(s in ["Complete", "Failed"] for s in statuses):
            return statuses

        time.sleep(10)  # Poll every 10 seconds

    raise TimeoutError(f"Jobs timed out after {timeout_minutes} minutes")
```

### Task 3: Create collect_results.py

**Purpose:** Copy CSV files from all 25 completed pods

```python
def collect_results():
    """Collect benchmark results from all completed jobs"""

    # Find all infersim job pods
    result = subprocess.run([
        "oc", "get", "pods", "-n", "diya",
        "-l", "app=infersim-benchmark",
        "-o", "jsonpath='{.items[*].metadata.name}'"
    ], capture_output=True, text=True)

    pods = result.stdout.strip().split()

    for pod in pods:
        # Copy data from pod
        subprocess.run([
            "oc", "rsync", "-n", "diya",
            f"{pod}:/workspace/InferSim/bench_data/h100/",
            "./InferSim/bench_data/h100/"
        ])

    print(f"✓ Collected data from {len(pods)} pods")
```

### Task 4: Validation Script

**Purpose:** Verify all 25 CSV files exist and have valid MFU ranges

```python
def validate_data():
    """Validate all benchmark data"""
    expected_files = [
        "gemm/data.csv",
        # 6 prefill files
        *[f"mha/prefill/{s}.csv" for s in SHAPES],
        # 18 decode files (6 shapes × 3 TPs)
        *[f"mha/decode/{s}-tp{tp}.csv" for s in SHAPES for tp in [1,2,4]]
    ]

    for file in expected_files:
        path = f"InferSim/bench_data/h100/{file}"
        assert os.path.exists(path), f"Missing: {path}"

        df = pd.read_csv(path)
        validate_mfu_range(df, get_expected_range(file))
```

---

## Key Design Principles

### 1. Shape-Only Benchmarking

**Current (wrong):**
```python
# ❌ Don't do this
submit_job(model="llama-2-7b", phase="prefill")
```

**Correct:**
```python
# ✅ Do this
submit_job(shape=(32, 32, 128), phase="prefill")
```

**Why:** MFU depends only on kernel shape, not model size/layers. A 32-32-128 kernel has the same MFU whether it's in a 7B or 70B model.

### 2. Wave Progression

**Execution Timeline:**
```
T=0:    GEMM job starts (background)
        Wave 1 starts (28-4-128): 4 jobs parallel

T=15:   Wave 1 complete
        Wave 2 starts (32-32-128): 4 jobs parallel

T=30:   Wave 2 complete
        Wave 3 starts (32-8-128): 4 jobs parallel

T=45:   Wave 3 complete (GEMM finishes around now)
        Wave 4 starts (40-40-128): 4 jobs parallel

T=60:   Wave 4 complete
        Wave 5 starts (56-8-128): 4 jobs parallel

T=75:   Wave 5 complete
        Wave 6 starts (64-8-128): 4 jobs parallel

T=90:   Wave 6 complete
        ✅ All done!
```

### 3. Parallelism Within Shape

**Why 4 parallel jobs per shape:**
- Prefill and decode are independent (different kernels)
- TP=1/2/4 are independent (different hardware configs)
- All produce different output files (no conflicts)
- 4 GPUs per wave is reasonable cluster ask

### 4. GEMM Independence

**Why GEMM runs separately:**
- Architecture-agnostic (doesn't depend on attention shape)
- Reused by all models
- Long-running (~45 min)
- Can overlap with config waves

---

## Implementation Summary

### Files to Modify

1. **scripts/openshift/generate_job.py**
   - Add `--shape` argument (format: "nh-nkv-dh")
   - Parse shape and pass to InferSim scripts
   - Remove dependency on benchmark_config.json for shape lookup

### Files to Create

2. **scripts/orchestrate_benchmarks.py** (~300 lines)
   - Wave-based orchestration
   - Job submission + monitoring
   - Progress reporting

3. **scripts/collect_results.py** (~100 lines)
   - Pod discovery
   - Data collection via oc rsync
   - File organization

4. **scripts/validate_benchmarks.py** (~150 lines)
   - Check all 25 files exist
   - Validate MFU ranges
   - Generate summary report

---

## Usage

```bash
# 1. Run orchestration (fully automated)
python scripts/orchestrate_benchmarks.py --gpu H100

# Output:
# [00:00] Launching GEMM job (background)...
# [00:00] === Wave 1/6: 28-4-128 ===
# [00:00]   Launching 4 jobs...
# [00:15]   ✓ All 4 jobs complete
# [00:15] === Wave 2/6: 32-32-128 ===
# ...
# [01:30] ✅ All waves complete!
# [01:30] Collecting results...
# [01:35] ✓ 25 CSV files collected

# 2. Validate data
python scripts/validate_benchmarks.py

# 3. Use in simulator
./inference-sim --model llama-2-7b --gpu H100 --tp 2 --roofline infersim
# Simulator looks up 32-32-128 shape, loads MFU data
```

---

## Success Criteria

- [ ] 25 CSV files generated with correct naming
- [ ] All MFU values within expected ranges:
  - Prefill: 0.30-0.85
  - Decode: 0.005-0.30
  - GEMM: 0.05-0.85
- [ ] Execution time < 2 hours
- [ ] No manual intervention after initial launch
- [ ] Failed jobs tracked but don't block completion
- [ ] Easy to re-run individual shapes if needed

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Wave timeout | 30min timeout per wave (should take 15-20min) |
| Job failure | Report but continue to next wave |
| Cluster capacity | Only 5 GPUs needed (GEMM + 1 wave) |
| Result collection | Automated rsync from all pods |
| Invalid MFU | Validation script checks ranges |
| GEMM timeout | 60min timeout (separate from waves) |

---

## Timeline Estimate

- **Script development:** 4-5 hours
  - Modify generate_job.py: 1 hour
  - Create orchestrate_benchmarks.py: 2 hours
  - Create collect_results.py: 30 minutes
  - Create validate_benchmarks.py: 1 hour

- **Cluster execution:** 90-120 minutes (automated)
- **Validation:** 15 minutes

**Total:** ~6 hours development + 2 hours cluster time

---

## Next Steps After Completion

Once all 25 CSV files are validated:

1. **Go Implementation** - Load CSV files in `sim/mfu_database.go`
2. **Lookup Logic** - Nearest-neighbor search for arbitrary shapes
3. **Integration** - Use in roofline: `time = max(flops/(peak×mfu), bytes/bw)`
4. **Validation** - Test against real models with TP=1/2/4

See `docs/plans/2026-02-18-roofline-v2-infersim-mfu.md` for Phase 2.
