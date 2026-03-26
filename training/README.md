# BLIS Training Loss Computation

Run BLIS binary against ground-truth training/validation experiments and compute loss metrics for model training.

## Prerequisites

```bash
# Build BLIS binary
cd /path/to/inference-sim
go build -o blis main.go

# Navigate to training directory
cd training/
```

## Usage

The script runs experiments in parallel and outputs loss metrics as JSON to stdout. It operates in two modes:

### Mode 1: Overall Loss Only (Default)

Returns aggregate loss metrics across all experiments:

```bash
python run_blis_and_compute_loss.py --latency-model roofline
```

**Output:**
```json
{
  "ttft_rmse": 12.34,
  "e2e_rmse": 15.67,
  "overall_loss": 28.01,
  "num_experiments": 10,
  "num_succeeded": 10,
  "num_failed": 0
}
```

**Loss Formula:** `overall_loss = RMSE[APE(TTFT mean)] + RMSE[APE(E2E mean)]` across all experiments

### Mode 2: Per-Experiment Breakdown

Include detailed metrics for each experiment with `--evaluate-per-experiment`:

```bash
python run_blis_and_compute_loss.py --latency-model roofline --evaluate-per-experiment
```

**Output:**
```json
{
  "ttft_rmse": 12.34,
  "e2e_rmse": 15.67,
  "overall_loss": 28.01,
  "num_experiments": 10,
  "num_succeeded": 10,
  "num_failed": 0,
  "per_experiment": [
    {
      "experiment_folder": "/path/to/exp",
      "model": "qwen/qwen3-14b",
      "workload": "chatbot",
      "ttft_mean_ape": 10.5,
      "e2e_mean_ape": 12.3,
      "combined_loss": 22.8,
      "wall_clock_seconds": 45.2,
      "latency_ape": {
        "e2e": {"mean": 12.3, "p90": 15.4, "p99": 18.2},
        "ttft": {"mean": 10.5, "p90": 13.1, "p99": 16.8},
        "itl": {"mean": 8.7}
      },
      "throughput_ape": {
        "input_tokens_per_sec": 5.2,
        "output_tokens_per_sec": 7.8,
        "requests_per_sec": 6.1
      }
    }
  ]
}
```

**Per-Experiment Fields:**
- **Loss metrics:** `ttft_mean_ape`, `e2e_mean_ape`, `combined_loss` (APE = Absolute Percentage Error %)
- **Runtime:** `wall_clock_seconds` (simulation wall-clock time)
- **Latency errors:** `latency_ape` - APE (%) for:
  - E2E: mean, P90, P99
  - TTFT: mean, P90, P99
  - ITL: mean only (percentiles unavailable in ground truth)
- **Throughput errors:** `throughput_ape` - APE (%) for input/output tokens per second, requests per second

**Note:** Values may be `null` if the metric is unavailable in ground truth or simulation.

Experiments are sorted by `combined_loss` (descending), so worst predictions appear first.

## Options

- `--latency-model` — **Required**. Latency model backend: `roofline`, `blackbox`, `crossmodel`, or `trained-roofline`
- `--data-dir` — Ground-truth experiments directory (default: `trainval_data`)
- `--blis-binary` — Path to BLIS binary (default: `../blis`)
- `--output-dir` — Output directory (default: `validation_results`) - **Note:** Currently unused, no files are created
- `--max-workers` — Maximum parallel experiment runs (default: `4`)
- `--evaluate-per-experiment` — Include per-experiment breakdown in output

## Examples

```bash
# Basic usage - overall loss only
python run_blis_and_compute_loss.py --latency-model roofline

# Save JSON to file
python run_blis_and_compute_loss.py --latency-model roofline > loss.json

# Per-experiment breakdown with 8 parallel workers
python run_blis_and_compute_loss.py \
  --latency-model trained-roofline \
  --max-workers 8 \
  --evaluate-per-experiment

# Extract just the overall loss value
python run_blis_and_compute_loss.py --latency-model roofline | jq '.overall_loss'
```

## Output Behavior

- **Stdout:** JSON output only (clean for automation/parsing)
- **Stderr:** Silent (no progress messages)
- **Exit codes:** `0` on success, `1` on failure

## Requirements

Each experiment directory must contain:
- `exp-config.yaml` — Experiment configuration
- `profile.yaml` — Workload profile
- `vllm.log` — Ground-truth logs
- `results/` — Ground-truth metrics folder