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

**Git-Based Jobs:** Each job clones fresh code from GitHub `roofline_valid` branch to pod-local storage, ensuring reproducibility and no manual PVC sync needed.

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

2. **GitHub Access:** Jobs clone from `github.com/inference-sim/inference-sim` branch `roofline_valid`
   - Ensure branch is pushed to GitHub before running
   - Jobs show git commit in logs for traceability

3. **H100 GPU nodes** available in cluster

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
  - **Collect results locally** (frees GPU resources for next wave)
  - **Verify files exist** before proceeding
- Wait for GEMM to finish
- **Collect GEMM results locally**
- Auto-delete generated YAMLs (use `--keep-yamls` to preserve)
- Report summary

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

### Collect Results

```bash
# Collect to default location (bench_data/)
python scripts/collect_results.py

# Collect to custom location
python scripts/collect_results.py --output-dir ./results_feb19
```

Copies CSV files from completed pod filesystems to local machine.

### Validate Data

```bash
# Validate default location (bench_data/)
python scripts/validate_benchmark_data.py --gpu H100

# Validate custom location
python scripts/validate_benchmark_data.py --gpu H100 --data-dir ./results_feb19

# Validate specific phase
python scripts/validate_benchmark_data.py --gpu H100 --phase-filter prefill
```

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
# - Missing config fields → Check run_benchmarks.py temp config
```

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

```bash
# Delete all InferSim jobs
oc delete jobs -l app=infersim-benchmark -n diya

# Delete completed jobs (after collecting results)
oc delete jobs -n diya --field-selector status.successful=1

# Clean up local test YAMLs (if using --keep-yamls or --dry-run)
rm scripts/openshift/job-h100-*.yaml
```

## Scripts Reference

- `openshift/generate_job.py` - Generate single job YAML from template
- `orchestrate_benchmarks.py` - Wave-based orchestration (submits 25 jobs)
- `run_benchmarks.py` - Called by jobs, routes to InferSim kernel scripts
- `collect_results.py` - Copy CSV files from pod filesystems to local
- `validate_benchmark_data.py` - Verify data structure, columns, MFU ranges

See also:
- `openshift/README.md` - Detailed OpenShift job documentation
- `ATTENTION_CONFIGS.md` - All 6 benchmarked attention shapes
