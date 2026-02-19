# Benchmarking Scripts

Automated orchestration system for running InferSim GPU kernel benchmarks on OpenShift.

## Overview

**Shape-Based Benchmarking:** Benchmarks organized by attention shape `{nh}-{nkv}-{dh}` (num_attention_heads, num_key_value_heads, head_dim), not by model name. Multiple models with identical attention configs share the same benchmark data.

Example: Mistral-7B and Mixtral-8x7B both use shape 32-8-128, so we benchmark once.

**Wave-Based Execution:**
- Wave 0: GEMM job (runs in background throughout)
- Waves 1-6: Each attention shape runs 4 parallel jobs (prefill + decode TP=1/2/4)
- **Auto-collection:** Results copied locally and verified after each wave completes
- Total: **25 jobs** (1 GEMM + 6 shapes × 4 phases)
- Peak GPU usage: **5 GPUs** (1 GEMM + 4 per wave)

**Git-Based Jobs:** Each job clones fresh code from GitHub to pod-local storage:
- inference-sim: `roofline_valid` branch (orchestration scripts)
- InferSim: Fork with improvements (`roofline-benchmark-improvements` branch)
  - OOM handling in decode benchmarks
  - Removed sglang dependency (uses local `flashinfer_backend_utils.py`)

This ensures reproducibility and no manual PVC sync needed.

## Quick Start

```bash
# 1. Run all 25 benchmark jobs (auto-collects results after each wave)
python scripts/orchestrate_benchmarks.py --gpu H100

# 2. Validate all collected data
python scripts/validate_benchmark_data.py --gpu H100
```

Results saved to `bench_data/` at repo root.

## Prerequisites

1. **OpenShift Access:**
   ```bash
   oc login <cluster-url>
   oc project diya
   ```

2. **GitHub Access:**
   - **inference-sim:** Branch `roofline_valid` must be pushed to GitHub
   - **InferSim fork:** Uses `github.com/inference-sim/InferSim` branch `roofline-benchmark-improvements`
   - Jobs show git commits in logs for traceability

3. **H100 GPU nodes** available in cluster

**Note:** Jobs clone both repositories at runtime, no manual PVC setup required.

## Commands

### Full Benchmark Suite (Recommended)

```bash
python scripts/orchestrate_benchmarks.py --gpu H100
```

This will:
- Check prerequisites (oc login, namespace access)
- Submit GEMM job (Wave 0, background)
- For each of 6 shapes:
  - Submit 4 jobs in parallel (prefill + decode TP=1/2/4)
  - Wait for wave to complete
  - **Collect results locally** via `oc rsync` from pod filesystems
  - **Verify expected CSV files exist** (1 prefill + 3 decode TPs)
  - Exit if collection fails or files missing (safety checkpoint)
  - Proceed to next wave only after successful verification
- Wait for GEMM to finish
- **Collect GEMM results locally** and verify `data.csv` exists
- Auto-delete generated YAMLs (use `--keep-yamls` to preserve)
- Report summary

**Key benefit:** Results are safe on local disk before proceeding, preventing data loss if jobs are cleaned up.

### Dry Run (Test Without Submitting)

```bash
# Generate all YAMLs without submitting to cluster
python scripts/orchestrate_benchmarks.py --gpu H100 --dry-run

# Check generated files (should be 25)
ls scripts/openshift/job-h100-*.yaml | wc -l

# Inspect a specific job
cat scripts/openshift/job-h100-32-8-128-prefill-*.yaml

# Clean up test YAMLs
rm scripts/openshift/job-h100-*.yaml
```

### Skip GEMM

```bash
# Run only attention benchmarks (no GEMM)
python scripts/orchestrate_benchmarks.py --gpu H100 --skip-gemm
```

### Single Job (Manual)

```bash
# Generate job for specific shape/phase/TP
python scripts/openshift/generate_job.py \
    --gpu H100 \
    --shape 32-8-128 \
    --phase prefill \
    --suffix test

# Submit manually
oc apply -f scripts/openshift/job-h100-32-8-128-prefill-test.yaml -n diya

# Monitor
oc logs -f job/infersim-h100-32-8-128-prefill-test -n diya
```

### Collect Results (Manual)

**Note:** Results are **auto-collected and verified** after each wave when using `orchestrate_benchmarks.py`. Manual collection is only needed for custom workflows.

**Auto-collection behavior:**
- After wave completes → `oc rsync` from pod filesystems → verify 4 CSV files exist
- After GEMM finishes → `oc rsync` from GEMM pod → verify `data.csv` exists
- If collection fails or files missing → orchestration exits with error
- This ensures data is safe on local disk before proceeding to next wave

**Manual collection:**
```bash
# Collect to default location (bench_data/)
python scripts/collect_results.py

# Collect to custom location
python scripts/collect_results.py --output-dir ./results_feb19
```

Copies CSV files from completed pod filesystems to local machine via `oc rsync`.

### Validate Data

**Note:** Each job validates its own data before completion. Final validation checks all collected data together.

```bash
# Validate all collected data (bench_data/)
python scripts/validate_benchmark_data.py --gpu H100

# Validate custom location
python scripts/validate_benchmark_data.py --gpu H100 --data-dir ./results_feb19

# Validate specific phase
python scripts/validate_benchmark_data.py --gpu H100 --phase-filter prefill
```

**Validation checks:**
- CSV files exist with required columns
- Row counts match expected ranges
- MFU values within valid ranges for each operation type
- All 25 expected files present (6 prefill + 18 decode + 1 GEMM)

## Output Structure

Results collected to `bench_data/` (auto-created, gitignored):

```
bench_data/
├── gemm/
│   └── h100/
│       └── data.csv                    # ~168 rows (M×K×N sweep)
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
            ├── 28-4-128-tp1.csv       # 18 files (6 shapes × 3 TPs)
            ├── 28-4-128-tp2.csv
            ├── 28-4-128-tp4.csv
            └── ...
```

Each CSV file contains:
- **Prefill**: ~5-10 rows (seq_len sweep: 512, 1024, 2048, 4096, 8192)
- **Decode**: ~30-40 rows (batch_size × kv_len sweep, OOM configs skipped)
- **GEMM**: ~168 rows (M×K×N sweep, large matrices OOM)

## Execution Flow

When running `orchestrate_benchmarks.py --gpu H100`:

1. **Prerequisites Check** - Validates oc login, namespace access
2. **Wave 0 (GEMM)** - Submits GEMM job, runs in background
3. **Waves 1-6** - For each shape (28-4-128, 32-32-128, etc.):
   - Submit 4 jobs in parallel
   - Wait for all 4 to complete (~2-5 minutes)
   - Collect results from 4 pods → `bench_data/`
   - Verify 4 CSV files exist locally
   - Proceed to next wave
4. **GEMM Collection** - Wait for GEMM to finish, collect results
5. **Summary** - Report total jobs, ready for validation

**Timeline:** ~15-30 minutes for all waves (depends on GPU availability and queue time)

**Data Safety:** Results are copied locally and verified after each wave. If collection fails, orchestration stops immediately.

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

## Troubleshooting

### Jobs Stuck in Pending

Check GPU availability:

```bash
# Check if H100 nodes are available
oc get nodes -l nvidia.com/gpu.product=NVIDIA-H100-80GB-HBM3

# Check GPU resource availability
oc describe nodes | grep -A 10 "nvidia.com/gpu"

# Check pod events
oc describe pod <pod-name> -n diya
```

### Job Failures

```bash
# Get failed job logs
oc logs job/<job-name> -n diya

# Common issues:
# - sgl-kernel build failure → Check CUDA version (need 12.8+)
# - Import ABI mismatch → PyTorch version mismatch
# - deep-gemm build failure → Check CUDA 12.8+ for FP4 support
# - Validation errors → Ensure job template passes correct args to validator
```

**Recent fixes:**
- Fixed validation to use `--phase-filter`/`--tp-filter` (not `--phase`/`--tp`)
- Fixed InferSim fork to use local `flashinfer_backend_utils.py` (no sglang dependency)
- Added OOM handling in decode benchmarks (skips configs that don't fit)

### Re-run Failed Shape

If a specific shape wave fails:

```bash
# Delete failed jobs
oc delete jobs -l app=infersim-benchmark -n diya

# Re-run entire suite (or use --dry-run to inspect)
python scripts/orchestrate_benchmarks.py --gpu H100

# Or manually re-run single shape:
python scripts/openshift/generate_job.py --gpu H100 --shape 32-8-128 --phase prefill
python scripts/openshift/generate_job.py --gpu H100 --shape 32-8-128 --phase decode --tp 1
python scripts/openshift/generate_job.py --gpu H100 --shape 32-8-128 --phase decode --tp 2
python scripts/openshift/generate_job.py --gpu H100 --shape 32-8-128 --phase decode --tp 4
oc apply -f scripts/openshift/job-h100-32-8-128-*.yaml -n diya
```

### Out of Memory Errors

GEMM and decode benchmarks may hit OOM for large configs:
```
torch.OutOfMemoryError: Tried to allocate 18.50 GiB
```

This is **expected behavior** - benchmarks skip configs that don't fit and continue.

## Clean Up

### Delete Jobs

**When using orchestration:** Jobs can be deleted after each wave completes since results are already copied locally and verified.

```bash
# Delete all InferSim jobs
oc delete jobs -l app=infersim-benchmark -n diya

# Delete only completed jobs (keeps failed jobs for debugging)
oc delete jobs -n diya --field-selector status.successful=1 -l app=infersim-benchmark

# Delete specific wave's jobs after collection
oc delete jobs -n diya -l app=infersim-benchmark | grep "32-8-128" | awk '{print $1}' | xargs oc delete job -n diya
```

### Clean Up YAMLs

```bash
# Clean up local test YAMLs (if using --keep-yamls or --dry-run)
rm scripts/openshift/job-h100-*.yaml
```

**Note:** Generated YAMLs are auto-deleted after submission by default.

## Scripts Reference

- `openshift/generate_job.py` - Generate single job YAML from template with shape-specific config
- `orchestrate_benchmarks.py` - Wave-based orchestration with auto-collection after each wave
- `run_benchmarks.py` - Called by jobs, creates temp configs and routes to InferSim kernel scripts
- `collect_results.py` - Copy CSV files from pod filesystems via oc rsync (called automatically by orchestration)
- `validate_benchmark_data.py` - Verify data structure, columns, MFU ranges (per-job + final validation)

See also:
- `openshift/README.md` - Detailed OpenShift job documentation
- `ATTENTION_CONFIGS.md` - All 6 benchmarked attention shapes
