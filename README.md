# Blackbox Inference Simulator (BLIS)

A discrete-event simulator for LLM inference serving systems. BLIS models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics (including tiered GPU+CPU offloading), scheduling policies, and token generation — all driven by trained performance coefficients or analytical roofline estimates.

The simulator is CPU-only, deterministic, and designed for capacity planning, policy optimization research, and performance prediction across model/GPU/TP configurations without requiring real GPUs.

## Table of Contents

- [Features](#features)
- [Supported Models](#supported-models)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Preset Workloads](#preset-workloads)
  - [Custom GPU, TP, vLLM Versions](#custom-gpu-tp-vllm-versions)
  - [Custom Workload Distribution](#custom-workload-distribution)
  - [Custom vLLM Configs](#custom-vllm-configs)
  - [Replay Workload Traces](#replay-workload-traces)
  - [Multi-Instance Cluster Simulation](#multi-instance-cluster-simulation)
  - [Tiered KV Cache](#tiered-kv-cache-gpu--cpu-offloading)
  - [Priority and Scheduling Policies](#priority-and-scheduling-policies)
  - [Fitness Evaluation and Anomaly Detection](#fitness-evaluation-and-anomaly-detection)
  - [Policy Configuration Files (YAML)](#policy-configuration-files-yaml)
  - [ServeGen-Informed Workload Generation](#servegen-informed-workload-generation)
  - [Decision Tracing and Counterfactual Analysis](#decision-tracing-and-counterfactual-analysis)
- [Latency Estimation Approaches](#latency-estimation-approaches)
- [Example Output](#example-output)
- [Debugging and Observability](#debugging-and-observability)
- [Evolutionary Policy Optimization](#evolutionary-policy-optimization-in-progress)
- [Project Structure](#project-structure)
- [CLI Reference](#cli-reference)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Discrete-event simulation** for prefill, decode, and request scheduling
- **KV-cache modeling** (blocks, prefix caching, prefill chunking, tiered GPU+CPU offload)
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
- **Real-mode HTTP client**: observe-predict-calibrate loop against live inference endpoints (library in `cmd/observe.go`; CLI subcommand planned)
- **Per-SLO-class metrics**: breakdown by SLO class with Jain fairness index (printed to stdout when multiple SLO classes present)
- **Calibration framework**: MAPE and Pearson r for simulator-vs-real accuracy assessment (library in `sim/workload/calibrate.go`; CLI subcommand planned)

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

You should see JSON output like:

```json
{
  "completed_requests": 100,
  "tokens_per_sec": 492.02,
  "ttft_mean_ms": 25.08,
  "e2e_mean_ms": 4541.01,
  ...
}
```

**Key metrics:** TTFT (Time to First Token) measures how quickly the first output token arrives. E2E (End-to-End) is the total request latency. `tokens_per_sec` is output token throughput (decode tokens per simulated second). See [Example Output](#example-output) for the full schema.

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
  --num-requests 300 \
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

Replay a CSV file of recorded requests for deterministic, reproducible simulation:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --workload traces --workload-traces-filepath traces.csv \
  --results-path results.json
```

Simulation results will be saved to `results.json`. If `--results-path` is not provided, the results are only printed.

**CSV trace format** (5 columns, header row required):

```csv
arrival_time,request_id,model,prefill_tokens,decode_tokens
0.0,req_0,llama,"[1,2,3,4,5]","[101,102,103]"
0.05,req_1,llama,"[10,20,30]","[201,202,203,204]"
```

| Column | Type | Description |
|--------|------|-------------|
| `arrival_time` | float | Request arrival time in **seconds** (converted to microseconds internally) |
| `request_id` | string | Identifier (ignored; BLIS generates `request_0`, `request_1`, ...) |
| `model` | string | Model name (ignored; uses `--model` flag) |
| `prefill_tokens` | JSON array | Input token IDs as JSON (e.g., `"[1,2,3]"`) |
| `decode_tokens` | JSON array | Output token IDs as JSON (e.g., `"[101,102]"`) |

Token arrays must be valid JSON integers. The length of each array determines the request's input/output token count.

### Multi-Instance Cluster Simulation

Run multiple instances with a routing policy:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --routing-policy weighted \
  --routing-scorers "queue-depth:2,kv-utilization:2,load-balance:1"
```

#### Routing Policies

| Policy | Best For | How It Works |
|--------|----------|-------------|
| `round-robin` (default) | Even distribution, no tuning needed | Cycles through instances in order |
| `least-loaded` | Low tail latency under variable load | Routes to instance with minimum effective load (queue + batch + pending) |
| `weighted` | Tunable multi-objective routing | Composable scorer pipeline — combines multiple scoring dimensions with configurable weights (see below) |
| `prefix-affinity` | Exact-duplicate request sequences | Routes requests with identical full token sequences to the same instance; falls back to least-loaded on miss. **Limitation:** hashes the entire input sequence, not just the prefix — for real prefix-aware routing, use `weighted` with the `prefix-affinity` scorer (see below). |
| `always-busiest` | Testing only | Pathological: routes to busiest instance (for anomaly detection testing) |

#### Weighted Routing: Composable Scorer Pipeline

The `weighted` policy evaluates each instance using independent scoring dimensions, combines them with configurable weights, and routes to the highest-scoring instance:

```
score(instance) = Σ weight_i × scorer_i(instance)    →    route to argmax
```

**Available scorers:**

| Scorer | Formula | What It Measures | llm-d Equivalent |
|--------|---------|------------------|-------------------|
| `prefix-affinity` | Proportional block match via router-side cache | Prefix cache locality (stateful) | `prefix-cache-scorer` |
| `queue-depth` | Min-max normalization of effective load | Queue pressure (immediate signal) | `queue-scorer` |
| `kv-utilization` | `1 − KVUtilization` | Memory headroom (lagging signal) | `kv-cache-utilization-scorer` |
| `load-balance` | `1 / (1 + effectiveLoad)` | Load balance preserving absolute differences | — |

**Configuration:**

```bash
# Via CLI (comma-separated name:weight pairs)
--routing-scorers "prefix-affinity:3,queue-depth:2,kv-utilization:2"

# Via YAML (--policy-config weighted-routing.yaml)
routing:
  policy: weighted
  scorers:
    - name: prefix-affinity
      weight: 3.0
    - name: queue-depth
      weight: 2.0
    - name: kv-utilization
      weight: 2.0
```

Weights are relative — only ratios matter. `[3,2,2]` behaves identically to `[6,4,4]`. If `--routing-scorers` is omitted, the default profile is `prefix-affinity:3,queue-depth:2,kv-utilization:2` (llm-d parity).

#### Why Scorer Choice Matters: A Concrete Example

Different scorers react to load at different speeds. At high request rates, this creates dramatic performance differences:

```
Model: llama-3.1-8b-instruct | 4 instances | 1000 requests | rate=5000 req/s

Configuration               Distribution          TTFT p99 (ms)
─────────────────────────── ───────────────────── ─────────────
queue-depth:1               251 / 250 / 250 / 249       2,598
kv-utilization:1            333 / 423 /  47 / 197       7,870  ← 3x worse
default (qd:2,kv:2,lb:1)   252 / 250 / 249 / 249       2,634
```

**Why the 3x difference?** `queue-depth` uses `PendingRequests` which updates *instantly* when a request is routed — the next routing decision immediately sees the load increase. `kv-utilization` uses KV block allocation which only updates during batch formation (~9ms for 8B models). At 5000 req/s, ~45 routing decisions happen between KV updates, all seeing the same stale utilization → requests pile on fewer instances → 3x worse tail latency.

**Reproduce this yourself:**

```bash
# Even distribution, good tail latency (immediate signal)
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --routing-policy weighted \
  --routing-scorers "queue-depth:1" \
  --num-requests 1000 --rate 5000 \
  --trace-level decisions --summarize-trace

# Skewed distribution, 3x worse tail latency (lagging signal)
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --routing-policy weighted \
  --routing-scorers "kv-utilization:1" \
  --num-requests 1000 --rate 5000 \
  --trace-level decisions --summarize-trace
```

See `examples/routing-comparison.sh` for a full automated comparison, and `examples/weighted-routing.yaml` for YAML configuration details.

### Tiered KV Cache (GPU + CPU Offloading)

Enable a two-tier KV cache where blocks are offloaded from GPU to CPU memory when GPU utilization exceeds a threshold, and reloaded on demand with modeled transfer latency:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 \
  --kv-cpu-blocks 500000 \
  --kv-offload-threshold 0.8 \
  --kv-transfer-bandwidth 50.0
```

| Flag | Default | Description |
|------|---------|-------------|
| `--kv-cpu-blocks` | 0 | CPU-tier KV cache blocks (0 = single-tier GPU only) |
| `--kv-offload-threshold` | 0.9 | GPU utilization threshold that triggers offloading to CPU |
| `--kv-transfer-bandwidth` | 100.0 | GPU↔CPU transfer bandwidth in blocks/tick |
| `--kv-transfer-base-latency` | 0 | Fixed per-transfer latency in ticks (0 = no fixed cost) |

When `--kv-cpu-blocks` is 0 (default), BLIS uses the single-tier GPU-only KV cache. Setting it to any positive value activates the tiered cache.

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

> **Warning:** Do not combine `inverted-slo` with `reverse-priority` — the two inversions cancel out, producing scheduling mathematically identical to normal `slo-based` + `priority-fcfs`. For actual pathological scheduling, use a single inversion: either `inverted-slo` with `priority-fcfs`, or `slo-based` with `reverse-priority`. See issue [#295](https://github.com/inference-sim/inference-sim/issues/295) for details.

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

The workload generation module is informed by the ServeGen characterization framework:

> Xiang et al., "ServeGen: Workload Characterization and Generation of Large Language Model Serving in Production," arXiv:2505.09999, 2025. [[paper](https://arxiv.org/abs/2505.09999)] [[code](https://github.com/alibaba/ServeGen)]

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

BLIS uses two estimation techniques. Choose based on your model support:

| | Blackbox (Data-Driven) | Roofline (Analytical) |
|---|---|---|
| **Accuracy** | High (trained on real measurements) | Moderate (first-principles estimate) |
| **Setup** | Requires pre-trained coefficients | Requires HuggingFace `config.json` + hardware spec |
| **When to use** | Supported model/GPU/TP combos in `defaults.yaml` | New models, unsupported configurations |
| **Required flags** | `--model` (coefficients loaded automatically) | `--model-config-folder` + `--hardware-config` |

### Blackbox Optimization (Data-Driven)
- Uses pre-trained linear regression coefficients (α/β) from `defaults.yaml`
- **Alpha coefficients**: model queueing time as a function of batch state
- **Beta coefficients**: model step execution time from batch features (running requests, new tokens, cached tokens)
- Automatically selected when `defaults.yaml` contains coefficients for the requested (model, GPU, TP, vLLM version) combination
- See [Blackbox Approach](./docs/approach.md)

### Roofline Approach (Analytical)
- No pre-training required — estimates latency from FLOPs and memory bandwidth
- Requires a HuggingFace `config.json` for the model (architecture parameters) and `hardware_config.json` (GPU specifications)
- Automatically activated when `--model-config-folder` is provided and no matching coefficients exist
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
  "instance_id": "cluster",
  "completed_requests": 100,
  "still_queued": 0,
  "still_running": 0,
  "injected_requests": 100,
  "total_input_tokens": 53074,
  "total_output_tokens": 51331,
  "vllm_estimated_duration_s": 104.33,
  "responses_per_sec": 0.96,
  "tokens_per_sec": 492.02,
  "e2e_mean_ms": 4541.01,
  "e2e_p90_ms": 7280.50,
  "e2e_p95_ms": 8102.28,
  "e2e_p99_ms": 9760.46,
  "ttft_mean_ms": 25.08,
  "ttft_p90_ms": 31.74,
  "ttft_p95_ms": 34.70,
  "ttft_p99_ms": 37.65,
  "itl_mean_ms": 8.78,
  "itl_p90_ms": 8.73,
  "itl_p95_ms": 8.73,
  "itl_p99_ms": 8.73,
  "scheduling_delay_p99_ms": 11.27,
  "preemption_count": 0,
  "dropped_unservable": 0
}
```

**Key metrics:**
- **TTFT** (Time to First Token): Latency from request arrival to first output token
- **ITL** (Inter-Token Latency): Average time between consecutive output tokens
- **E2E** (End-to-End): Total latency from request arrival to completion
- **Scheduling Delay**: Time spent waiting in queue before batch formation
- **Tokens/sec**: Aggregate throughput across all completed requests
- `_p90`, `_p95`, `_p99` suffixes indicate percentile values
- **Conservation fields**: `still_queued`, `still_running`, `dropped_unservable`, and `injected_requests` verify request conservation (`injected == completed + still_queued + still_running + dropped_unservable`). See [INV-1](docs/standards/invariants.md).

When using `--results-path`, the JSON output also includes a `requests` array with per-request details:

| Field | Description |
|-------|-------------|
| `requestID` | Unique request identifier |
| `arrived_at` | Arrival time (seconds) |
| `num_prefill_tokens` | Input token count |
| `num_decode_tokens` | Output token count |
| `ttft_ms` | Time to first token (ms) |
| `itl_ms` | Mean inter-token latency (ms) |
| `e2e_ms` | End-to-end latency (ms) |
| `scheduling_delay_ms` | Queue wait time (ms) |
| `handled_by` | Instance ID that processed this request (meaningful when `--num-instances` > 1) |
| `slo_class` | SLO class label (if workload-spec provides one) |
| `tenant_id` | Tenant identifier (if workload-spec provides one) |

---

## Debugging and Observability

### Output Model

Simulation results (metrics JSON, fitness scores, anomaly counters, trace summaries) are always printed to **stdout** regardless of log level. Diagnostic messages go to **stderr** and are controlled by `--log`. This means you can pipe results cleanly:

```bash
# Capture only simulation results
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct > results.txt

# Suppress diagnostics, show only results
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct 2>/dev/null
```

### Log Levels

Control verbosity of diagnostic messages with `--log` (default: `warn`):

```bash
# See policy configuration and workload generation details
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --log info

# Full event-level tracing (very verbose)
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --log debug
```

Available levels: `trace`, `debug`, `info`, `warn`, `error`, `fatal`, `panic`

### Decision Tracing

Record every routing and admission decision for post-hoc analysis:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --routing-policy weighted \
  --trace-level decisions --summarize-trace
```

The trace summary shows:
- Total admission decisions (admitted vs rejected)
- Target distribution across instances (routing balance)
- Unique targets used
- Mean and max regret (when `--counterfactual-k` > 0)

### Counterfactual Analysis

Evaluate "what if" scenarios — how much better would alternative routing choices have been:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --routing-policy weighted \
  --trace-level decisions --counterfactual-k 5 --summarize-trace
```

The `--counterfactual-k 5` flag computes regret for the top 5 alternative candidates at each routing decision. Mean and max regret indicate how often the routing policy made suboptimal choices.

### Fitness Evaluation

Compare policy configurations using a single composite score:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 \
  --fitness-weights "throughput:0.5,p99_ttft:0.3,mean_e2e:0.2"
```

Available fitness metric keys:

| Key | Direction | Description |
|-----|-----------|-------------|
| `throughput` | higher is better | Completed requests per second |
| `tokens_per_sec` | higher is better | Aggregate token throughput |
| `p99_ttft`, `p50_ttft`, `mean_ttft` | lower is better | Time to first token |
| `p99_e2e`, `p50_e2e`, `mean_e2e` | lower is better | End-to-end latency |

### Anomaly Detection

BLIS automatically detects and reports anomalies at the end of each simulation:

- **Priority Inversions**: older requests receiving worse latencies than newer ones (indicates scheduling issues). Suppressed when `--priority-policy constant` (the default), since all requests share the same priority and E2E differences reflect workload variance, not unfairness.
- **HOL Blocking**: instances with queue depth significantly exceeding cluster average (indicates routing imbalance)
- **Rejected Requests**: admission control rejection count (indicates capacity pressure)

Anomaly counters are printed when non-zero. Priority inversion detection requires a non-constant priority policy to be meaningful. Use pathological policies (`inverted-slo`, `always-busiest`, `reverse-priority`) to verify anomaly detection works. Note: `inverted-slo` and `reverse-priority` must NOT be used together — their inversions cancel out. Use `inverted-slo` with `priority-fcfs` instead.

### Stdout Metrics Sections

Beyond the primary JSON metrics, BLIS prints additional sections to stdout when relevant:

**KV Cache Metrics** — printed when any KV cache metric is non-zero:
```
=== KV Cache Metrics ===
Preemption Rate: 0.0000
Cache Hit Rate: 0.0594
KV Thrashing Rate: 0.0000
```

**Per-SLO Metrics** — printed when the workload contains 2+ SLO classes (via `--workload-spec`):
```
=== Per-SLO Metrics ===
  batch:
    TTFT: mean=45.20 p99=231.33 (n=350)
    E2E:  mean=3200.50 p99=12351.62 (n=350)
  realtime:
    TTFT: mean=42.10 p99=138.01 (n=150)
    E2E:  mean=3083.47 p99=12813.41 (n=150)
```

These sections are deterministic (same seed = same output) and appear after the JSON metrics block.

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

- **Tiered KV cache** with GPU+CPU offload/reload and transfer latency modeling (PR12)

Upcoming:

- **Auto-scaling** with actuation policies (PR11)
- **Prefill/Decode disaggregation** with cross-instance KV transfer (PR14)
- **Framework adapters** for OpenEvolve and GEPA policy evolution (PR15)
- **Integration tests** (PR16)

See [design documentation](./docs/plans/) for details.

---

## Project Structure

> For the authoritative file-level architecture documentation with interface names, method signatures, and module descriptions, see [`CLAUDE.md`](./CLAUDE.md).

```
inference-sim/
├── main.go                 # CLI entry point
├── cmd/                    # CLI commands
│   ├── root.go             # CLI flags (--policy-config, --routing-policy, --workload-spec, etc.)
│   ├── observe.go          # Real-mode HTTP client for observe-predict-calibrate
│   └── default_config.go   # defaults.yaml loading
├── sim/                    # Core simulation engine
│   ├── config.go           # Module-scoped sub-config types (R16)
│   ├── doc.go              # Package reading guide
│   ├── simulator.go        # Discrete-event simulation loop
│   ├── admission.go        # Admission policy interface and templates
│   ├── routing.go          # Routing policy interface and templates
│   ├── routing_scorers.go  # ScorerConfig, stateless scorers, ParseScorerConfigs
│   ├── routing_prefix_scorer.go # Prefix-affinity scorer + observer
│   ├── prefix_cache_index.go # PrefixCacheIndex: per-instance LRU of block hashes
│   ├── priority.go         # Priority policy interface and templates
│   ├── scheduler.go        # Instance scheduler interface and templates
│   ├── latency_model.go    # LatencyModel interface and registration
│   ├── router_state.go     # RouterState bridge type for cluster-level policies
│   ├── bundle.go           # PolicyBundle YAML configuration
│   ├── event.go            # Event types (Arrival, Queued, Step, Scheduled, Preemption, RequestLeft)
│   ├── kv_store.go         # KVStore interface and registration variables
│   ├── batch.go            # Batch struct
│   ├── batch_formation.go  # BatchFormation interface, VLLMBatchFormation
│   ├── queue.go            # FIFO wait queue
│   ├── request.go          # Request lifecycle
│   ├── metrics.go          # TTFT, TPOT, E2E collection
│   ├── metrics_utils.go    # MetricsOutput JSON struct, percentile calculations
│   ├── rng.go              # PartitionedRNG for deterministic simulation
│   ├── model_hardware_config.go  # ModelConfig, HardwareCalib structs
│   └── workload_config.go  # CSV trace loading and distribution-based workload
├── sim/kv/                 # KV cache implementations
│   ├── cache.go            # KVCacheState (single-tier GPU)
│   ├── tiered.go           # TieredKVCache (GPU+CPU)
│   └── register.go         # NewKVStore factory + init()-based registration into sim/
├── sim/latency/            # Latency model implementations
│   ├── latency.go          # BlackboxLatencyModel, RooflineLatencyModel, NewLatencyModel factory
│   ├── roofline.go         # Analytical FLOPs/bandwidth latency estimation
│   ├── config.go           # HFConfig, GetHWConfig, GetModelConfig, ValidateRooflineConfig
│   └── register.go         # init()-based registration into sim/
├── sim/cluster/            # Multi-replica cluster simulation
│   ├── cluster.go          # Shared-clock event loop, online routing
│   ├── instance.go         # Per-instance simulator wrapper
│   ├── cluster_event.go    # Cluster-level event types
│   ├── snapshot.go         # Instance observability snapshots
│   ├── metrics.go          # RawMetrics, FitnessResult, anomaly detection, per-SLO-class metrics
│   ├── counterfactual.go   # Top-k candidate ranking and regret computation
│   ├── deployment.go       # DeploymentConfig (embeds SimConfig + cluster fields)
│   ├── evaluation.go       # EvaluationResult wrapper (metrics + trace + summary)
│   └── workload.go         # Centralized request generation for cluster dispatch
├── sim/workload/           # ServeGen-informed workload generation
│   ├── spec.go             # WorkloadSpec, ClientSpec, ArrivalSpec, DistSpec, YAML loading
│   ├── arrival.go          # ArrivalSampler: Poisson, Gamma, Weibull
│   ├── distribution.go     # LengthSampler: Gaussian, Exponential, ParetoLogNormal, EmpiricalPDF, Constant
│   ├── client.go           # Rate normalization, prefix group management
│   ├── generator.go        # GenerateRequests pipeline with client decomposition
│   ├── servegen.go         # Native ServeGen data file loading
│   ├── tracev2.go          # Trace v2 format (YAML header + CSV data)
│   ├── replay.go           # Trace v2 → sim.Request with synthetic token IDs
│   ├── calibrate.go        # CalibrationReport, MAPE, Pearson r
│   ├── multimodal.go       # Multimodal token generation (text+image+audio+video)
│   ├── reasoning.go        # Reasoning multi-turn with context accumulation
│   ├── network.go          # Client-perspective latency (RTT + bandwidth)
│   ├── inference_perf.go   # inference-perf format loading and validation
│   └── scenarios.go        # Built-in presets (bursty, unfair, prefix-heavy, mixed-slo)
├── sim/trace/              # Decision trace recording
│   ├── trace.go            # TraceLevel, TraceConfig, SimulationTrace
│   ├── record.go           # AdmissionRecord, RoutingRecord, CandidateScore
│   └── summary.go          # TraceSummary, Summarize()
├── examples/               # Example configuration files
│   ├── policy-config.yaml  # Policy bundle example
│   ├── weighted-routing.yaml  # Weighted routing scorer pipeline config
│   ├── routing-comparison.sh  # Automated routing policy comparison (run to reproduce performance table)
│   ├── servegen-language.yaml # ServeGen workload spec example
│   ├── prefix-affinity-demo.yaml # Prefix-affinity routing demo (long shared prefix)
│   └── multiturn-chat-demo.yaml  # Multi-turn chat session demo
├── model_configs/          # HuggingFace config.json files
├── defaults.yaml           # Pre-trained coefficients, model defaults
├── hardware_config.json    # GPU hardware specifications
└── docs/                   # Documentation and design plans
```

---

## CLI Reference

### Core Simulation

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | LLM model name (e.g., `meta-llama/llama-3.1-8b-instruct`) |
| `--hardware` | (auto-detected) | GPU type (`H100`, `A100-80`). Auto-detected from `defaults.yaml` if omitted |
| `--tp` | (auto-detected) | Tensor parallelism degree. Auto-detected from `defaults.yaml` if omitted |
| `--vllm-version` | (auto-detected) | vLLM version string. Auto-detected from `defaults.yaml` if omitted |
| `--horizon` | max int64 | Simulation horizon in ticks (microseconds). When using --workload-spec, the spec's horizon value is used unless this flag is explicitly set. |
| `--seed` | 42 | RNG seed for deterministic simulation |
| `--results-path` | (none) | Save JSON results to file |
| `--log` | warn | Log level: trace, debug, info, warn, error, fatal, panic |
| `--defaults-filepath` | defaults.yaml | Path to trained coefficients file |
| `--model-config-folder` | (none) | Path to folder with HuggingFace `config.json` (enables roofline mode) |
| `--hardware-config` | (none) | Path to GPU hardware specifications file (for roofline mode) |

### Workload Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--workload` | distribution | Workload type: `chatbot`, `summarization`, `contentgen`, `multidoc`, `distribution`, `traces` |
| `--workload-spec` | (none) | YAML workload spec file (overrides `--workload`). See `examples/servegen-language.yaml` |
| `--workload-traces-filepath` | (none) | CSV trace file (required when `--workload traces`) |
| `--rate` | 1.0 | Requests per second (for distribution workloads) |
| `--num-requests` | 100 | Number of requests to generate. When using --workload-spec, the spec's num_requests value is used unless this flag is explicitly set. |
| `--prompt-tokens` | 512 | Mean input token count |
| `--prompt-tokens-stdev` | 256 | Input token count standard deviation |
| `--prefix-tokens` | 0 | Shared prefix token count |
| `--output-tokens` | 512 | Mean output token count |
| `--output-tokens-stdev` | 256 | Output token count standard deviation |
| `--prompt-tokens-min` | 2 | Min input token count |
| `--prompt-tokens-max` | 7000 | Max input token count |
| `--output-tokens-min` | 2 | Min output token count |
| `--output-tokens-max` | 7000 | Max output token count |

### Cluster and Routing

| Flag | Default | Description |
|------|---------|-------------|
| `--num-instances` | 1 | Number of instances in the cluster |
| `--routing-policy` | round-robin | Routing: `round-robin`, `least-loaded`, `weighted`, `prefix-affinity`, `always-busiest` |
| `--routing-scorers` | (defaults) | Scorer weights for weighted routing. Valid scorers: `prefix-affinity`, `queue-depth`, `kv-utilization`, `load-balance`. Format: `name:weight,...` |
| `--admission-policy` | always-admit | Admission: `always-admit`, `token-bucket`, `reject-all` |
| `--token-bucket-capacity` | 10000 | Token bucket max tokens |
| `--token-bucket-refill-rate` | 1000 | Token bucket refill rate (tokens/sec) |
| `--priority-policy` | constant | Priority: `constant`, `slo-based`, `inverted-slo` |
| `--scheduler` | fcfs | Scheduler: `fcfs`, `priority-fcfs`, `sjf`, `reverse-priority` |
| `--admission-latency` | 0 | Admission processing latency in microseconds |
| `--routing-latency` | 0 | Routing processing latency in microseconds |
| `--policy-config` | (none) | YAML policy bundle file. See `examples/policy-config.yaml` |
| `--snapshot-refresh-interval` | 0 | Snapshot refresh interval for KV utilization in microseconds (0 = immediate refresh every call). Controls staleness of KV-related routing signals. |

### Observability

| Flag | Default | Description |
|------|---------|-------------|
| `--trace-level` | none | Trace verbosity: `none`, `decisions` |
| `--counterfactual-k` | 0 | Number of counterfactual candidates per routing decision |
| `--summarize-trace` | false | Print trace summary after simulation |
| `--fitness-weights` | (none) | Fitness weights as `key:val,key:val` (e.g., `throughput:0.5,p99_ttft:0.3`) |

### vLLM Server Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--total-kv-blocks` | 1000000 | Total KV cache blocks. When a model match exists in defaults.yaml, the model-specific value is used unless this flag is explicitly set. |
| `--max-num-running-reqs` | 256 | Max concurrent requests in running batch |
| `--max-num-scheduled-tokens` | 2048 | Max new tokens per step across all running requests |
| `--block-size-in-tokens` | 16 | Tokens per KV cache block |
| `--max-model-len` | 2048 | Max request length (input + output tokens) |
| `--long-prefill-token-threshold` | 0 | Chunked prefill trigger threshold (0 = disabled) |
| `--alpha-coeffs` | 0.0,0.0,0.0 | Alpha coefficients for processing delay estimation |
| `--beta-coeffs` | 0.0,0.0,0.0 | Beta coefficients for step time estimation |
| `--kv-cpu-blocks` | 0 | CPU-tier KV cache blocks (0 = single-tier GPU only) |
| `--kv-offload-threshold` | 0.9 | GPU utilization threshold to trigger offloading to CPU |
| `--kv-transfer-bandwidth` | 100.0 | GPU↔CPU transfer bandwidth in blocks/tick |
| `--kv-transfer-base-latency` | 0 | Fixed per-transfer latency in ticks for CPU↔GPU KV transfers (0 = no fixed cost) |

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for the engineering standards, development workflow, and step-by-step guides for adding new components. For ongoing work and architectural decisions, see `docs/plans/`.

---

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](./LICENSE) for details.
