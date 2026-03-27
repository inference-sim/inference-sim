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

- `--latency-model` — **Required**. Latency model backend: `roofline`, `blackbox`, `crossmodel`, `trained-roofline`, or custom backend name
- `--alpha-coeffs` — Comma-separated alpha coefficients (e.g., `"0.00032,0.000045,0.000038"`)
- `--beta-coeffs` — Comma-separated beta coefficients (e.g., `"0.00087,0.00124,0.000021"`)
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

# With custom coefficients (for training)
python run_blis_and_compute_loss.py \
  --latency-model evolved \
  --alpha-coeffs "0.00032,0.000045,0.000038" \
  --beta-coeffs "0.00087,0.00124,0.000021"

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

---

# Inner Loop Optimizer

Bayesian optimization for latency model coefficients. Part of the agentic two-loop training system.

## Prerequisites

```bash
# Install Python dependencies
pip install -r training/requirements.txt

# Ensure outer loop has provided deliverables:
# - iteration_manifest.yaml
# - coefficient_bounds.yaml
# - sim/latency/<backend>.go
```

## Usage

```bash
cd training/

# Run inner loop optimization (default 50 trials)
python inner_loop_optimize.py

# Custom number of trials
python inner_loop_optimize.py --n-trials 100

# Custom timeout per trial (default 120s)
python inner_loop_optimize.py --timeout 180

# Skip detailed post-convergence evaluation
python inner_loop_optimize.py --no-detailed-eval
```

## What It Does

**Phase 1: Setup**
1. Reads `iteration_manifest.yaml` from outer loop
2. Verifies all declared Go source files exist
3. Compiles BLIS binary with new latency backend (~5-10s)
4. Loads coefficient bounds from `coefficient_bounds.yaml`

**Phase 2: Bayesian Optimization**
1. Runs up to 50-100 trials sampling coefficient space
2. Each trial injects coefficients via `--alpha-coeffs` and `--beta-coeffs`
3. Evaluates loss: `RMSE[APE(TTFT)] + RMSE[APE(E2E)]`
4. Updates Gaussian process surrogate model
5. **Early stopping**: Stops if best loss hasn't improved >1% in last 50 trials

**Phase 3: Post-Convergence Evaluation**
1. Runs detailed evaluation with optimal coefficients
2. Generates per-experiment diagnostics using `--evaluate-per-experiment`

## Output

Results saved to `inner_loop_results.json`:

```json
{
  "best_alpha": [0.00032, 0.000045, 0.000038],
  "best_beta": [0.00087, 0.00124, 0.000021],
  "best_loss": 8.234,
  "n_trials": 50,
  "optimization_time": 245.3,
  "converged_early": false,
  "detailed_diagnostics": { ... },
  "timestamp": "2026-03-27T14:30:00Z",
  "iteration": 3,
  "backend_name": "evolved"
}
```

## Architecture

**For understanding the system**:
- [outer-inner-loop-contract.md](docs/outer-inner-loop-contract.md) - Interface contract between outer and inner loops
- [agentic-latency-training-problem-statement.md](docs/agentic-latency-training-problem-statement.md) - Complete problem definition

**For implementing the outer loop**:
- [outer-loop-specs.md](docs/outer-loop-specs.md) - **Agent prompt specification** (what the outer loop must generate)

**For running validation**:
- [generalization-validation-protocol.md](docs/generalization-validation-protocol.md) - Cross-validation and physics checks

**Key principles:**
- Inner loop is a pre-implemented script (no agent implementation needed)
- Outer loop agent generates 3 files: manifest, Go code, bounds
- Inner loop compiles BLIS and runs Bayesian optimization automatically
- Coefficients injected at runtime (no recompilation per trial)
- One compilation per outer loop iteration (~5-10s overhead)
- 50-100 fast evaluations per iteration