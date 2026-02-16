# Blackbox Inference Simulator (BLIS)

A discrete-event simulator for LLM inference platforms (e.g., vLLM, SGLang).
This tool models request arrival, KV-cache dynamics, scheduling, token generation, and latency using trained performance coefficients (α/β) and configurable workload distributions.

The simulator is CPU-only, extremely fast, and designed for capacity planning, saturation analysis, and performance prediction across model/GPU/TP variations without requiring real GPUs.

---

## Features

- **Discrete-event simulation** for prefill, decode, and request scheduling
- **KV-cache modeling** (blocks, prefix caching, prefill chunking)
- **CPU-only inference cost model** via learned α/β coefficients
- **HuggingFace config.json support** for model architecture
- **Dense and MoE model support** (Mixtral, DeepSeek-MoE, etc.)
- **vLLM deployment configuration** (TP, PP, EP, batch limits)
- **Two latency estimation modes**: blackbox (data-driven) and roofline (analytical)
- **Multiple workload types**: preset (chatbot, summarization) or custom distributions
- **Trace replay**: replay recorded request traces for deterministic testing
- **Multi-instance cluster simulation** with shared-clock event loop
- **Pluggable routing policies**: round-robin, least-loaded, weighted-scoring, prefix-affinity
- **Priority policies**: constant, slo-based (request prioritization)
- **Instance schedulers**: fcfs, priority-fcfs, sjf (batch formation policies)
- **Admission control**: always-admit or token-bucket rate limiting
- **YAML policy configuration**: define all policies in a single config file (`--policy-config`)

---

## Supported Models

### Dense Models
- LLaMA 3.x (8B, 70B variants)
- Qwen 2.5 (1.5B - 72B)
- Mistral (7B, Small 24B)
- Phi-4
- CodeLlama
- Granite

### MoE Models
- Mixtral 8x7B
- (Additional MoE models supported via HuggingFace config.json)

See [`defaults.yaml`](./defaults.yaml) for the full list of pre-trained model configurations.

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

---

## Quick Start

Run BLIS for `meta-llama/llama-3.1-8b-instruct` with default configs:

```bash
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct
```

---

## Usage

### Preset Workloads

Run a preset workload (`chatbot`, `summarization`, `contentgen`, `multidoc`):

```bash
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --workload chatbot
```

### Custom GPU, TP, vLLM Versions

Override GPU, TP, and vLLM version:

```bash
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct \
  --hardware H100 --tp 1 --vllm-version vllm/vllm-openai:v0.8.4
```

### Custom Workload Distribution

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

### Custom vLLM Configs

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --max-num-running-reqs 256 \
  --max-num-scheduled-tokens 2048
```

### Replay Workload Traces

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --workload traces --workload-traces-filepath traces.csv \
  --results-path results.json
```

Simulation results will be saved to `results.json`. If `--results-path` is not provided, the results are only printed.

### Multi-Instance Cluster Simulation

Run multiple instances with a routing policy:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --routing-policy weighted \
  --routing-cache-weight 0.6 --routing-load-weight 0.4
```

Available routing policies:
- `round-robin` (default) — even distribution across instances
- `least-loaded` — routes to instance with minimum queue + batch size
- `weighted` — composite score combining cache affinity and load balance
- `prefix-affinity` — routes matching prefixes to the same instance, falls back to least-loaded

### Priority and Scheduling Policies

Control request prioritization and batch formation order:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --priority-policy slo-based \
  --scheduler priority-fcfs
```

Available priority policies:
- `constant` (default) — assigns fixed priority (0.0) to all requests
- `slo-based` — higher priority for older requests (age-based urgency)

Available schedulers:
- `fcfs` (default) — first-come-first-served (existing behavior)
- `priority-fcfs` — orders by priority descending, then arrival time
- `sjf` — shortest job first by input token count

### Policy Configuration Files (YAML)

Define all policies in a single YAML file for easier management:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 \
  --policy-config examples/policy-config.yaml
```

YAML values serve as defaults; CLI flags override YAML settings. See `examples/policy-config.yaml` for format and available options.

---

## Latency Estimation Approaches

BLIS uses two estimation techniques:

### Blackbox Optimization (Data-Driven)
- Uses pre-trained linear regression coefficients (α/β)
- Requires pre-training for each (model, GPU, TP, vLLM version) combination
- High accuracy for supported configurations
- See [Blackbox Approach](./docs/approach.md)

### Roofline Approach (Analytical)
- No pre-training required
- Works with any model via HuggingFace config.json
- Based on FLOPs/memory bandwidth analysis
- See [Roofline Approach](./docs/roofline.md)

### Using Roofline Mode

To simulate models without pre-trained coefficients, use the roofline model by providing model and hardware configs:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --hardware H100 \
  --tp 1 \
  --vllm-version vllm/vllm-openai:v0.8.4 \
  --model-config-folder model_configs/llama-3.1-8b-instruct \
  --hardware-config hardware_config.json
```

This requires the HuggingFace `config.json` for the model saved under the `model-config-folder` path. Pre-configured configs for common models are provided in `model_configs/`.

> **Note:** Currently supports H100 and A100-80 GPUs.

---

## Example Output

```json
{
  "sim_start_timestamp": "2026-01-14 19:07:19",
  "sim_end_timestamp": "2026-01-14 19:07:19",
  "completed_requests": 40,
  "total_input_tokens": 195567,
  "total_output_tokens": 21450,
  "vllm_estimated_duration_s": 25.882896,
  "simulation_duration_s": 0.386482042,
  "responses_per_sec": 1.545,
  "tokens_per_sec": 828.73,
  "e2e_mean_ms": 5384.43,
  "e2e_p90_ms": 6933.96,
  "e2e_p95_ms": 7338.86,
  "e2e_p99_ms": 8418.09,
  "ttft_mean_ms": 131.05,
  "ttft_p90_ms": 144.60,
  "ttft_p95_ms": 152.23,
  "ttft_p99_ms": 153.44,
  "itl_mean_ms": 9.78,
  "itl_p90_ms": 8.74,
  "itl_p95_ms": 8.74,
  "itl_p99_ms": 44.80,
  "scheduling_delay_p99_ms": 7.08
}
```

---

## Evolutionary Policy Optimization (In Progress)

BLIS supports multi-replica cluster simulation with pluggable control policies for evolutionary optimization research. Currently implemented:

- **Multi-replica simulation** with shared-clock event loop and online routing pipeline
- **Admission policies**: always-admit, token-bucket rate limiting
- **Routing policies**: round-robin, least-loaded, weighted-scoring, prefix-affinity
- **Priority policies**: constant, slo-based (request prioritization)
- **Instance schedulers**: fcfs, priority-fcfs, sjf (batch formation order)
- **Instance observability**: snapshot-based monitoring with configurable staleness
- **Policy bundles** with YAML configuration (`--policy-config`)
- **Interface freeze**: policy interfaces are stable (additive changes only)

Upcoming:

- **Raw metrics and anomaly detection** (PR 9) → Research-Ready Checkpoint
- **Auto-scaling, tiered KV cache, decision traces** (PRs 10-13)
- **Framework integration** with OpenEvolve and GEPA for policy evolution (PR 15)

See [design documentation](./docs/plans/) for details.

---

## Project Structure

```
inference-sim/
├── main.go                 # CLI entry point
├── sim/                    # Core simulation engine
│   ├── simulator.go        # Discrete-event simulation loop
│   ├── admission.go        # Admission policy interface and templates
│   ├── routing.go          # Routing policy interface and templates
│   ├── priority.go         # Priority policy interface and templates
│   ├── scheduler.go        # Instance scheduler interface and templates
│   ├── router_state.go     # RouterState bridge type for cluster-level policies
│   ├── bundle.go           # PolicyBundle YAML configuration
│   ├── kvcache.go          # KV cache modeling
│   ├── batch.go            # Batch formation
│   ├── request.go          # Request lifecycle
│   └── model_hardware_config.go  # HuggingFace/hardware config
├── sim/cluster/            # Multi-replica cluster simulation
│   ├── cluster.go          # Shared-clock event loop, online routing
│   ├── instance.go         # Per-instance simulator wrapper
│   ├── cluster_event.go    # Cluster-level event types
│   └── snapshot.go         # Instance observability snapshots
├── cmd/                    # CLI commands (--policy-config, --routing-policy, etc.)
├── examples/               # Example configuration files
├── model_configs/          # HuggingFace config.json files
├── defaults.yaml           # Pre-trained coefficients, model defaults
├── hardware_config.json    # GPU hardware specifications
└── docs/                   # Documentation and design plans
```

---

## Contributing

Contributions are welcome! Please see the design documents in `docs/plans/` for ongoing work and architectural decisions.

---

## License

[Add license information here]
