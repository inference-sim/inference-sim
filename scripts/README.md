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
# Generate and submit all 25 jobs (auto-cleans YAMLs after submission)
python scripts/orchestrate_benchmarks.py --gpu H100

# Keep YAMLs for debugging (optional)
python scripts/orchestrate_benchmarks.py --gpu H100 --keep-yamls

# Collect results after completion
python scripts/collect_results.py

# Validate data
python scripts/validate_benchmark_data.py --gpu H100
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
InferSim/bench_data/
├── gemm/
│   └── h100/
│       └── data.csv             # GEMM MFU data
└── mha/
    ├── prefill/
    │   └── h100/
    │       ├── 28-4-128.csv     # 6 shapes
    │       └── ...
    └── decode/
        └── h100/
            ├── 28-4-128-tp1.csv # 6 shapes × 3 TPs = 18 files
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
- `validate_benchmark_data.py` - Verify data structure and ranges
