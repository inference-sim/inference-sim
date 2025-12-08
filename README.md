# inference-sim

A discrete-event simulator for LLM inference platforms (e.g., vLLM).
This tool models request arrival, KV-cache dynamics, scheduling, token generation, and latency using trained performance coefficients (α/β) and configurable workload distributions.

The simulator is CPU-only, extremely fast, and designed for capacity planning, saturation analysis, and performance prediction across model/GPU/TP variations without requiring real GPUs.

## Features

- Discrete-event inference simulation (prefill, decode, scheduling)
- KV cache modeling (blocks, block size, long-prefill threshold)
- CPU-only inference cost model via learned alpha/beta coefficients
- Supports multiple LLMs, TP values, GPU types, and vLLM versions
- Produces detailed performance metrics and latency breakdowns

## Getting started

- > Go >= 1.21 is required.
- build inference-sim binary
```shell
git clone git@github.com:inference-sim/inference-sim.git
cd inference-sim
go build -o simulation_worker main.go
```
- simple run
```bash
inference-sim run \
    --model meta-llama/llama-3.1-8b-instruct \
    --workload chat
```



- Optional: You can override hardware, TP, request rate, prompt/output distributions, and more:

```bash
  inference-sim run \
  --model ibm-granite/granite-3.1-8b-instruct \
  --hardware H100 \
  --tp 2 \
  --rate 10 \
  --vllm-version v8.4 \
  --max-prompts 300 \
  --prompt-tokens 800 \
  --prompt-tokens-stdev 300 \
  --output-tokens 400 \
```


