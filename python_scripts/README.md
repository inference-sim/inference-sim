# InferSim Evaluation Scripts

This directory contains scripts for evaluating InferSim against other LLM inference simulators and ground truth data from real vLLM deployments.

## Contents

### Evaluation Scripts

- **`aggregate_ground_truth.py`** - Aggregates ground truth data from vLLM experiments
  - Reads experiment configurations (vLLM config, workload config)
  - Extracts server-side metrics from traces (TTFT, ITL, E2E latencies)
  - Outputs `combined_ground_truth.json` for use by evaluators

- **`blis_evaluator.py`** - Evaluates BLIS (inference-sim) against ground truth
  - Runs BLIS simulations for each QPS point in ground truth
  - Compares predicted vs actual metrics
  - Reports mean percentage errors

- **`vidur_evaluator.py`** - Evaluates Vidur simulator against ground truth
  - Runs Vidur simulations with same configs as ground truth
  - Compares predicted vs actual metrics
  - Reports mean percentage errors

### Vidur Simulator

To use `vidur_evaluator.py`, you need to clone and set up the Vidur simulator:

```bash
# Clone Vidur into evaluation directory
cd evaluation
git clone https://github.com/vidur-ai/vidur.git
cd vidur
# Follow Vidur installation instructions
```

- Used for comparative benchmarking against InferSim
- Discrete event simulator for LLM inference
- Includes vLLM scheduler implementation

## Usage

### 1. Aggregate Ground Truth Data

```bash
# Assumes ground truth experiments in eval/ground_truth/
python aggregate_ground_truth.py
# Outputs: eval/combined_ground_truth.json
```

### 2. Evaluate BLIS

```bash
python blis_evaluator.py \
    --ground-truth eval/combined_ground_truth.json \
    --blis-binary path/to/simulation_worker
```

### 3. Evaluate Vidur

```bash
python vidur_evaluator.py \
    --ground-truth eval/combined_ground_truth.json
```

## Ground Truth Data Format

The `combined_ground_truth.json` file contains:

```json
{
  "experiments": [
    {
      "experiment_name": "...",
      "model": "meta-llama/Llama-2-7b-hf",
      "vllm_config": {
        "tensor_parallelism": 1,
        "max_model_len": 4096,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "app": "chat"
      },
      "workload_config": { ... },
      "total_kv_blocks": 12345,
      "qps_sweeps": [
        {
          "qps": 1.5,
          "ttft_mean_ms": 45.2,
          "ttft_p90_ms": 67.8,
          "itl_mean_ms": 12.3,
          "e2e_mean_ms": 234.5,
          "e2e_p90_ms": 345.6
        }
      ]
    }
  ]
}
```

## Metrics

All evaluators compare these key metrics:
- **TTFT** (Time to First Token) - mean and p90
- **ITL** (Inter-Token Latency) - mean
- **E2E** (End-to-End Latency) - mean and p90

Errors are reported as mean percentage error across all QPS points.

## Requirements

- Python 3.8+
- pandas
- numpy (for Vidur)
- PyYAML

See `vidur/` directory for Vidur-specific requirements.
