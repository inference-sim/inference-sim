# Experiment Adapters

This directory contains simulator adapters that wrap different LLM serving simulators behind a common interface (`SimulatorAdapter`). Each adapter translates ground-truth experiment configurations into simulator-specific inputs, runs the simulator, and parses its output into standardised `SimulatorResult` objects.

## LLMServingSim Adapter

### Overview

The LLMServingSim adapter validates LLMServingSim's prediction accuracy against real vLLM experiments.

### Supported Configurations

- **Hardware**: H100 only
- **Models**:
  - Llama-3.1-8B (tp1, tp2)
  - Mixtral-8x7B-v0.1 (tp1, tp2, tp4)
- **Precision**: FP16 only
- **Features**: CPU offloading, multi-instance (dp>1)

### Usage

```bash
python -m experiment.run \
  --data-dir vllm_data/ground_truth \
  --output-dir results \
  --adapters llmservingsim \
  --llmservingsim-dir LLMServingSim
```

### Requirements

- LLMServingSim installed at `LLMServingSim/` (or custom path via `--llmservingsim-dir`)
- Performance models at `llm_profile/perf_models/H100/{model}/tp{N}/`
- Ground-truth data with `per_request_lifecycle_metrics.json`

### How It Works

1. **Eligibility**: Checks H100 hardware, model support, perf model existence, FP16 precision
2. **Config Generation**: Creates cluster_config JSON with TP, memory, instance settings
3. **Workload Generation**: Reads ground-truth token counts, generates constant-rate arrivals
4. **Execution**: Runs LLMServingSim via subprocess with temp configs
5. **Parsing**: Converts CSV output (ns) to SimulatorResult (ms) with nested LatencyDistribution/ThroughputMetrics
