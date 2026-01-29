
# Blackbox Inference Simulator (BLIS)

A discrete-event simulator for LLM inference platforms (e.g., vLLM).
This tool models request arrival, KV-cache dynamics, scheduling, token generation, and latency using trained performance coefficients (α/β) and configurable workload distributions. It uses trained performance coefficients (**α/β**) and configurable workload distributions to predict system behavior.

The simulator is CPU-only, extremely fast, and designed for capacity planning, saturation analysis, and performance prediction across model/GPU/TP variations without requiring real GPUs.

---

## Features

- Discrete-event simulation for **prefill**, **decode**, and **request scheduling**
- KV-cache modeling (blocks, prefix caching, prefill chunking)
- CPU-only inference cost model via learned **α/β coefficients**
- Supports multiple **LLMs**, **TP values**, **GPU types**, and **vLLM versions**
- Generates detailed **performance metrics** and **latency breakdowns**
- Two powerful latency estimation techniques - blackbox optimization and heuristic roofline approaches.

---

## Installation

**Requirements:**  
- Go ≥ **1.21**

**Build the binary:**

```bash
git clone git@github.com:inference-sim/inference-sim.git
cd inference-sim
go build -o simulation_worker main.go
```

## QuickStart

Run BLIS for `meta-llama/llama-3.1-8b-instruct` with default configs:

```bash
   ./simulation_worker run --model meta-llama/llama-3.1-8b-instruct 
```

## Usage

**Preset application workloads**

Run a preset workload (`chatbot`, `summarization`, `contentgen`, `multidoc`):

```bash
   ./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --workload chatbot
```

**Custom GPU, TP, vllm versions**

Override GPU, TP, and vLLM version:

```bash
   ./simulation_worker run --model meta-llama/llama-3.1-8b-instruct \
   --hardware H100 --tp 1 --vllm-version vllm/vllm-openai:v0.8.4
```

**Custom Workload Distribution**

Define custom workload distribution to sample input/output lengths from:

```bash
  ./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --workload distribution \
  --rate 10 \
  --max-prompts 300 \
  --prompt-tokens 800 \
  --prompt-tokens-stdev 300 \
  --output-tokens 400 \
  --output-tokens-stdev 200 
```

**Custom vLLM Configs**

```bash
  ./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --max-num-running-reqs 256 \
  --max-num-scheduled-tokens 2048
```

**Replay workload traces and save results**

```bash
   ./simulation_worker run \
   --model meta-llama/llama-3.1-8b-instruct \
   --workload traces --workload-traces-filepath traces.csv \
   --results-path results.json
```

Simulation results will be saved to `results.json`. If `--results-path` is not provided, the results are only printed, but not saved.


# Toggling Latency Estimation Approaches 

BLIS uses two powerful estimation techniques:

- Blackbox Optimization (data-driven, requires pretraining). See detailed approach in [Blackbox Approach](./docs/approach.md)
- Heuristic Roofline Approach (no pretraining needed). See detailed approach in [Roofline Approach](./docs/roofline.md)

The instructions above use the blackbox optimization approach to estimate latencies. Since the blackbox optimization approach requires pretraining, it supports only a limited set of LLM, GPU, TP and vllm versions. Refer to [coefficients.yaml](!https://github.com/inference-sim/inference-sim/blob/main/coefficients.yaml) for details on supported models, hardware, TP and vllm versions. 

To simulate a wider range of LLM, GPU, TP and vllm version combinations (regardless of pretraining support), please use the Heuristic Roofline Approach. You can toggle between blackbox optimization and roofline approaches by specifying the flags `model-config-folder` and `hardware-config` to trigger the roofline model. You should also download the HuggingFace `config.json` for the LLM to be simulated under the `model-config-folder` path. For example,

```bash
 ./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --hardware H100 \
  --tp 1 \
  --vllm-version vllm/vllm-openai:v0.8.4 \
  --model-config-folder model_configs/llama-3.1-8b-instruct \
  --hardware-config hardware_config.json
```

This example assumes that you have the HuggingFace [`config.json`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/config.json) for the LLM saved as `model_configs/llama-3.1-8b-instruct/config.json`. We have provided `config.json` files for a few commonly-used LLMs already under `model_configs/`.

> Note: Currently, we only support H100 and A100-80 GPUs.

## Example Output

After running a simulation, you will see simulated metrics printed out as follows:

```
=== Simulation Metrics ===
{
  "sim_start_timestamp": "2026-01-14 19:07:19",
  "sim_end_timestamp": "2026-01-14 19:07:19",
  "completed_requests": 40,
  "total_input_tokens": 195567,
  "total_output_tokens": 21450,
  "vllm_estimated_duration_s": 25.882896,
  "simulation_duration_s": 0.386482042,
  "responses_per_sec": 1.545422119688616,
  "tokens_per_sec": 828.7326116830203,
  "e2e_mean_ms": 5384.433599999999,
  "e2e_p90_ms": 6933.9587,
  "e2e_p95_ms": 7338.8573499999975,
  "e2e_p99_ms": 8418.08552,
  "ttft_mean_ms": 131.05245,
  "ttft_p90_ms": 144.60440000000003,
  "ttft_p95_ms": 152.2315,
  "ttft_p99_ms": 153.43732,
  "itl_mean_ms": 9.778280409492787,
  "itl_p90_ms": 8.743,
  "itl_p95_ms": 8.743,
  "itl_p99_ms": 44.8,
  "scheduling_delay_p99_ms": 7.08047
}
```



