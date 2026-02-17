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
- **ServeGen-informed workload generation**: multi-client specs with Poisson/Gamma/Weibull arrivals (`--workload-spec`)
- **Decision tracing and counterfactual analysis**: record routing decisions and evaluate alternative choices (`--trace-level`, `--counterfactual-k`)
- **Fitness evaluation**: weighted multi-objective scoring with configurable metric weights (`--fitness-weights`)
- **Real-mode HTTP client**: observe-predict-calibrate loop against live inference endpoints (`observe` subcommand)
- **Per-SLO-class metrics**: breakdown by SLO class with Jain fairness index
- **Calibration framework**: MAPE and Pearson r for simulator-vs-real accuracy assessment

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
- `weighted` — composite score combining cache availability (FreeKVBlocks) and queue pressure (QueueDepth). Weights must sum to 1.0. Most effective when instances have different cache states (e.g., prefix caching, heterogeneous capacity). For symmetric clusters with balanced load, `least-loaded` is simpler and equally effective.
- `prefix-affinity` — routes matching prefixes to the same instance, falls back to least-loaded
- `always-busiest` — pathological: routes to most-loaded instance (for anomaly detection testing)

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
- `inverted-slo` — pathological: higher priority for newer requests (causes starvation)

Available schedulers:
- `fcfs` (default) — first-come-first-served (existing behavior)
- `priority-fcfs` — orders by priority descending, then arrival time
- `sjf` — shortest job first by input token count
- `reverse-priority` — pathological: schedules lowest priority first (causes inversions)

### Fitness Evaluation and Anomaly Detection

Evaluate policy fitness using a weighted combination of metrics:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 \
  --fitness-weights "throughput:0.5,p99_ttft:0.3,mean_e2e:0.2"
```

Available fitness metric keys:
- `throughput`, `tokens_per_sec` — higher is better
- `p99_ttft`, `p50_ttft`, `mean_ttft` — lower is better (TTFT latency)
- `p99_e2e`, `p50_e2e`, `mean_e2e` — lower is better (end-to-end latency)

BLIS also detects anomalies automatically:
- **Priority Inversions** — older requests receiving worse latencies than newer ones
- **HOL Blocking** — instances with queue depth significantly exceeding cluster average
- **Rejected Requests** — admission control rejection count

### Policy Configuration Files (YAML)

Define all policies in a single YAML file for easier management:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 \
  --policy-config examples/policy-config.yaml
```

YAML values serve as defaults; CLI flags override YAML settings. See `examples/policy-config.yaml` for format and available options.

### ServeGen-Informed Workload Generation

Generate realistic workloads from a ServeGen-style YAML specification with multi-client traffic classes, configurable arrival processes, and length distributions:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 \
  --workload-spec examples/servegen-language.yaml
```

See `examples/servegen-language.yaml` for the full specification format including client decomposition, arrival processes (Poisson, Gamma, Weibull), and length distributions (Gaussian, Exponential, ParetoLogNormal, EmpiricalPDF).

### Decision Tracing and Counterfactual Analysis

Record routing decisions and evaluate what would have happened with alternative choices:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --routing-policy weighted \
  --trace-level decisions --counterfactual-k 5 --summarize-trace
```

This records each routing decision with candidate scores and computes regret (how much better the best alternative would have been). The `--summarize-trace` flag prints aggregated statistics at the end of the simulation.

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

**Key metrics:**
- **TTFT** (Time to First Token): Latency from request arrival to first output token
- **ITL** (Inter-Token Latency): Average time between consecutive output tokens
- **E2E** (End-to-End): Total latency from request arrival to completion
- **Scheduling Delay**: Time spent waiting in queue before batch formation
- **Tokens/sec**: Aggregate throughput across all completed requests
- `_p90`, `_p95`, `_p99` suffixes indicate percentile values

---

## Evolutionary Policy Optimization (In Progress)

BLIS supports multi-replica cluster simulation with pluggable control policies for evolutionary optimization research. Currently implemented:

- **Multi-replica simulation** with shared-clock event loop and online routing pipeline
- **Admission policies**: always-admit, token-bucket rate limiting, reject-all (pathological)
- **Routing policies**: round-robin, least-loaded, weighted-scoring, prefix-affinity, always-busiest (pathological)
- **Priority policies**: constant, slo-based, inverted-slo (pathological)
- **Instance schedulers**: fcfs, priority-fcfs, sjf, reverse-priority (pathological)
- **Fitness evaluation**: weighted multi-objective scoring with configurable metric weights
- **Anomaly detection**: priority inversion, HOL blocking, rejection rate counters
- **Instance observability**: snapshot-based monitoring with configurable staleness
- **Policy bundles** with YAML configuration (`--policy-config`)
- **Interface freeze**: policy interfaces are stable (additive changes only)

Completed:

- **Raw metrics and anomaly detection** (PR9) -- Research-Ready Checkpoint
- **ServeGen-informed workload generator** with observe-predict-calibrate loop (PR10)
- **Decision tracing and counterfactual analysis** with top-k regret computation (PR13)

Upcoming:

- **Auto-scaling** (PR11) and **tiered KV cache** (PR12)
- **Framework adapters** for OpenEvolve and GEPA policy evolution (PR15)
- **Integration tests** (PR16)

See [design documentation](./docs/plans/) for details.

---

## Project Structure

```
inference-sim/
├── main.go                 # CLI entry point
├── cmd/                    # CLI commands
│   ├── root.go             # CLI flags (--policy-config, --routing-policy, --workload-spec, etc.)
│   ├── observe.go          # Real-mode HTTP client for observe-predict-calibrate
│   └── default_config.go   # defaults.yaml loading
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
│   ├── snapshot.go         # Instance observability snapshots
│   ├── metrics.go          # RawMetrics, FitnessResult, anomaly detection, per-SLO-class metrics
│   ├── counterfactual.go   # Top-k candidate ranking and regret computation
│   └── evaluation.go       # EvaluationResult wrapper (metrics + trace + summary)
├── sim/workload/           # ServeGen-informed workload generation
│   ├── spec.go             # WorkloadSpec, ClientSpec, ArrivalSpec, DistSpec, YAML loading
│   ├── arrival.go          # ArrivalSampler: Poisson, Gamma, Weibull
│   ├── distribution.go     # LengthSampler: Gaussian, Exponential, ParetoLogNormal, EmpiricalPDF
│   ├── generator.go        # GenerateRequests pipeline with client decomposition
│   ├── servegen.go         # Native ServeGen data file loading
│   ├── calibrate.go        # CalibrationReport, MAPE, Pearson r
│   └── replay.go           # Trace v2 replay
├── sim/trace/              # Decision trace recording
│   ├── trace.go            # TraceLevel, TraceConfig, SimulationTrace
│   ├── record.go           # AdmissionRecord, RoutingRecord, CandidateScore
│   └── summary.go          # TraceSummary, Summarize()
├── examples/               # Example configuration files
│   ├── policy-config.yaml  # Policy bundle example
│   ├── weighted-routing.yaml  # Weighted routing example
│   └── servegen-language.yaml # ServeGen workload spec example
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
