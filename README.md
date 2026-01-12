
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

**Custom Workload Parameters**

Define custom workload characteristics:

```bash
  ./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
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

**Roofline**

```bash
 ./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --hardware H100 \
  --tp 1 \
  --vllm-version vllm/vllm-openai:v0.8.4 \
  --model-config-folder model_configs/llama-3.1-8b-instruct \
  --hardware-config hardware_config.json
```

## Supported LLMs

- ibm-granite/granite-3.1-8b-instruct
- meta-llama/llama-3.1-8b-instruct
- qwen/qwen2.5-7b-instruct
- microsoft/phi-4
- meta-llama/llama-3.3-70b-instruct
- openai/gpt-oss-120b
- openai/gpt-oss-20b
- mistralai/mistral-small-24b-instruct-2501
- mistralai/mistral-small-3.1-24b-instruct-2503
- mistralai/mixtral-8x7b-instruct-v0.1

> Note: Currently, BLIS supports only a limited set of GPU, TP and vllm versions for each of the above LLMs. Refer to [coefficients.yaml](!https://github.com/inference-sim/inference-sim/blob/main/coefficients.yaml) for details on supported models, hardware, TP and vllm versions.

## Example Output

After running a simulation, you will see simulated metrics printed out as follows:

```
=== Simulation Metrics ===
Completed Requests   : 100
Request Rate(req/s)  : 1
Total Input Tokens   : 53074
Total Output Tokens  : 51331
Simulation Duration(s): 0.121
vLLM estimated Duration(s): 103.736
e2e_mean_ms  : 3724.786
e2e_p90_ms   : 5976.007
e2e_p95_ms   : 6648.597
e2e_p99_ms   : 8006.079
ttft_mean_ms : 19.313
ttft_p90_ms  : 23.159
ttft_p95_ms  : 23.623
ttft_p99_ms  : 25.015
itl_mean_ms  : 7.205
itl_p90_ms   : 7.185
itl_p95_ms   : 7.191
itl_p99_ms   : 7.191
responses_per_sec   : 0.964
tokens_per_sec   : 494.826
```



