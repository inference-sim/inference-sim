# OpenShift Benchmark Jobs

This directory contains templates and scripts for running InferSim GPU kernel benchmarks on OpenShift clusters.

## Overview

InferSim benchmarks measure real GPU kernel performance (MFU - Model FLOPs Utilization) for three key operations:

1. **Prefill** - FlashAttention-3 for prompt processing (compute-bound, high MFU)
2. **Decode** - FlashInfer for token generation (memory-bound, low MFU)
3. **GEMM** - FP8 matrix multiplication for linear layers (varies by batch size)

These measurements are used to build accurate performance models for LLM inference.

## Key Concepts

**Shape-Based Benchmarking:** Benchmarks are organized by attention shape `{nh}-{nkv}-{dh}` (num_attention_heads, num_key_value_heads, head_dim), not by model name. Multiple models with identical attention configs share the same benchmark data.

Example: Mistral-7B and Mixtral-8x7B both use 32-8-128, so we benchmark once.

**Wave-Based Execution:** Orchestration runs in waves to manage GPU resources:
- Wave 0: GEMM job (runs in background throughout)
- Waves 1-6: Each attention shape runs 4 parallel jobs (prefill + decode TP=1/2/4)
- Each wave waits for completion before proceeding to next shape

**Total Jobs:** 25 (1 GEMM + 6 shapes × 4 phases)

## Files

- `job-benchmarks-template.yaml` - OpenShift Job template with variable substitution
- `generate_job.py` - Generate GPU-specific job YAML from template

## Prerequisites

### 1. OpenShift Access

```bash
# Login to cluster
oc login <cluster-url>

# Verify access to diya namespace
oc get namespace diya
oc project diya
```

### 2. Copy Code to PVC

The job expects code at `/mnt/inference-sim` on the `data-pvc` PersistentVolumeClaim.

```bash
# Using the pvc-debugger pod
oc exec -it pvc-debugger -n diya -- bash
cd /mnt
git clone <your-repo-url> inference-sim
exit

# Or copy from local machine
oc cp . pvc-debugger:/mnt/inference-sim -n diya
```

### 3. Clean Old Data (Optional)

```bash
oc exec -it pvc-debugger -n diya -- bash -c "
  rm -rf /mnt/inference-sim/InferSim/bench_data/mha/prefill/h100
  rm -rf /mnt/inference-sim/InferSim/bench_data/mha/decode/h100
  rm -rf /mnt/inference-sim/InferSim/bench_data/gemm/h100
  echo 'H100 data cleaned'
"
```

## Usage

### Full Benchmark Suite (Recommended)

Run all benchmarks using automated orchestration:

```bash
# Submit all 25 jobs with wave-based execution
python scripts/orchestrate_benchmarks.py --gpu H100

# The script will:
# - Check prerequisites (oc login, namespace access)
# - Submit GEMM job (Wave 0, background)
# - For each shape (6 waves):
#   - Submit 4 jobs in parallel (prefill + decode TP=1/2/4)
#   - Wait for wave to complete before next shape
# - Wait for GEMM to finish
# - Report summary

# After all jobs complete, collect results
python scripts/collect_results.py

# Validate collected data
python scripts/validate_benchmark_data.py --gpu H100
```

### Dry Run (Test Without Submitting)

```bash
# Generate all YAMLs without submitting to cluster
python scripts/orchestrate_benchmarks.py --gpu H100 --dry-run

# Check generated files (should be 25)
ls scripts/openshift/job-h100-*.yaml | wc -l

# Inspect a specific job
cat scripts/openshift/job-h100-32-8-128-prefill-*.yaml
```

### Single Job (Manual)

Generate and submit individual jobs for testing:

```bash
# Generate job for specific shape/phase/TP
python scripts/openshift/generate_job.py \
    --gpu H100 \
    --shape 32-8-128 \
    --phase prefill \
    --suffix test

# Submit manually
oc apply -f scripts/openshift/job-h100-32-8-128-prefill-test.yaml -n diya

# Monitor logs
oc logs -f job/infersim-h100-32-8-128-prefill-test -n diya
```

### Skip GEMM

```bash
# Run only attention benchmarks (no GEMM)
python scripts/orchestrate_benchmarks.py --gpu H100 --skip-gemm
```

## Output Structure

Benchmarks write CSV files to the PVC:

```
/mnt/inference-sim/InferSim/bench_data/
├── gemm/
│   └── h100/
│       └── data.csv                    # GEMM: ~168 rows (M×K×N sweep)
└── mha/
    ├── prefill/
    │   └── h100/
    │       ├── 28-4-128.csv           # Qwen2-7B config
    │       ├── 32-32-128.csv          # Llama-2-7B (MHA)
    │       ├── 32-8-128.csv           # Mistral-7B, Mixtral-8x7B
    │       ├── 40-40-128.csv          # Llama-2-13B (MHA)
    │       ├── 56-8-128.csv           # CodeLlama-34B
    │       └── 64-8-128.csv           # Llama-2-70B
    └── decode/
        └── h100/
            ├── 28-4-128-tp1.csv       # 6 shapes × 3 TPs = 18 files
            ├── 28-4-128-tp2.csv
            ├── 28-4-128-tp4.csv
            └── ...
```

Each file contains:
- **Prefill**: ~5-10 rows (seq_len sweep: 512, 1024, 2048, 4096, 8192)
- **Decode**: ~30-40 rows (batch_size × kv_len sweep, some OOM skipped)
- **GEMM**: ~168 rows (M×K×N sweep, large matrices OOM)

## Monitoring

### Check Job Status

```bash
# List all InferSim jobs
oc get jobs -n diya -l app=infersim-benchmark

# Watch job progress
oc get jobs -n diya -l app=infersim-benchmark -w

# Check pods
oc get pods -n diya -l app=infersim-benchmark
```

### View Logs

```bash
# Follow logs for specific job
oc logs -f job/<job-name> -n diya

# Example
oc logs -f job/infersim-h100-32-8-128-prefill-20260219-123456 -n diya
```

### Check Results on PVC

```bash
# Exec into debugger pod
oc exec -it pvc-debugger -n diya -- bash

# Navigate to results
cd /mnt/inference-sim/InferSim/bench_data

# Check file counts
find mha/prefill/h100 -name "*.csv" | wc -l  # Should be 6
find mha/decode/h100 -name "*.csv" | wc -l   # Should be 18
find gemm/h100 -name "*.csv" | wc -l         # Should be 1

# Check file sizes
find . -name "*.csv" -exec wc -l {} \;
```

## Troubleshooting

### Jobs Stuck in Pending

Check GPU availability:

```bash
# Check if H100 nodes are available
oc get nodes -l nvidia.com/gpu.product=NVIDIA-H100-80GB-HBM3

# Check GPU resource availability
oc describe nodes | grep -A 10 "nvidia.com/gpu"

# Check pod events
oc get pods -n diya -l app=infersim-benchmark
oc describe pod <pod-name> -n diya
```

### Job Failures

View logs to identify the issue:

```bash
# Get failed job logs
oc logs job/<job-name> -n diya

# Common issues:
# - sgl-kernel build failure → Check CUDA version (need 12.8+)
# - Import ABI mismatch → PyTorch version mismatch
# - deep-gemm build failure → Check CUDA 12.8+ for FP4 support
```

### Dependency Build Failures

**sgl-kernel ABI mismatch:**
```
ImportError: undefined symbol: _ZN3c108ListType3get...
```
Solution: Job template uses PyTorch 2.9.1 with proper ABI compatibility. Verify container image in config.

**deep-gemm missing FP4 support:**
```
error: 'CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B' was not declared
```
Solution: Requires CUDA 12.8+. Job template uses cuda12.8-cudnn9-devel image.

### Out of Memory Errors

GEMM benchmarks may hit OOM for large matrices:
```
torch.OutOfMemoryError: Tried to allocate 18.50 GiB
```

This is **expected behavior** - the benchmark skips configs that don't fit and continues. You'll get ~168 measurements instead of 224.

### Re-run Failed Shape

If a specific shape wave fails, re-run just that shape:

```bash
# Generate jobs for single shape
python scripts/openshift/generate_job.py --gpu H100 --shape 32-8-128 --phase prefill
python scripts/openshift/generate_job.py --gpu H100 --shape 32-8-128 --phase decode --tp 1
python scripts/openshift/generate_job.py --gpu H100 --shape 32-8-128 --phase decode --tp 2
python scripts/openshift/generate_job.py --gpu H100 --shape 32-8-128 --phase decode --tp 4

# Submit all 4 jobs
oc apply -f scripts/openshift/job-h100-32-8-128-*.yaml -n diya
```

## Clean Up

### Delete Jobs

```bash
# Delete specific job
oc delete job <job-name> -n diya

# Delete all InferSim jobs
oc delete jobs -l app=infersim-benchmark -n diya

# Delete completed jobs (after collecting results)
oc delete jobs -n diya --field-selector status.successful=1
```

### Delete Generated YAMLs

```bash
# Clean up generated job files (they're gitignored)
rm scripts/openshift/job-h100-*.yaml
```

## Adding New GPU Types

To benchmark a new GPU type (A100, H20, B100):

### 1. Update Configuration

Edit `config/benchmark_config.json`:

```json
{
  "gpu_specs": {
    "A100": {
      "peak_tflops_fp16": 312.0,
      "peak_memory_bw_tbs": 2.0,
      "effective_memory_bw_factor": 0.8,
      "nvlink_bw_gbs": 600,
      "num_sms": 108,
      "memory_gb": 80
    }
  },
  "mfu_validation": {
    "A100": {
      "decode": { "min": 0.005, "max": 0.30, "description": "Memory-bound" },
      "prefill": { "min": 0.30, "max": 0.90, "description": "Compute-bound" },
      "gemm": { "min": 0.05, "max": 1.0, "description": "Varies by batch" }
    }
  },
  "gemm_sweep": {
    "A100": {
      "k_values": [2048, 3584, 4096, 8192],
      "n_values": [6144, 11008, 14336, 18944]
    }
  }
}
```

### 2. Update Node Selector

Edit `scripts/openshift/generate_job.py` line 35-40 to add GPU type mapping:

```python
gpu_to_selector = {
    "H100": ("nvidia.com/gpu.product", "NVIDIA-H100-80GB-HBM3"),
    "A100": ("nvidia.com/gpu.product", "NVIDIA-A100-SXM4-80GB"),
    # Add your GPU here
}
```

### 3. Run Benchmarks

```bash
python scripts/orchestrate_benchmarks.py --gpu A100
python scripts/collect_results.py
python scripts/validate_benchmark_data.py --gpu A100
```

## Advanced Usage

### Parallel Orchestration from Multiple Terminals

Launch different shapes simultaneously from different terminals:

```bash
# Terminal 1
python scripts/orchestrate_benchmarks.py --gpu H100 --dry-run
oc apply -f scripts/openshift/job-h100-28-4-128-*.yaml -n diya

# Terminal 2
oc apply -f scripts/openshift/job-h100-32-32-128-*.yaml -n diya

# Terminal 3
oc apply -f scripts/openshift/job-h100-32-8-128-*.yaml -n diya
```

Pod anti-affinity ensures each job gets a different node.

### Debugging on PVC

```bash
# Exec into debugger pod with PVC mounted
oc exec -it pvc-debugger -n diya -- bash

# Navigate to code
cd /mnt/inference-sim

# Run benchmark directly (bypasses OpenShift job)
python scripts/run_benchmarks.py --gpu H100 --shape 32-8-128 --phase prefill

# Check output
ls -lh InferSim/bench_data/mha/prefill/h100/
```

## See Also

- `../README.md` - High-level orchestration documentation
- `../../config/benchmark_config.json` - GPU specs and validation ranges
- `../../scripts/ATTENTION_CONFIGS.md` - All benchmarked attention shapes
