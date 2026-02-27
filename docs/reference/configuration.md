# Configuration Reference

This page documents all CLI flags, configuration files, and their interactions. For architectural context on what these settings control, see [Cluster Architecture](../concepts/architecture.md) and [Core Engine](../concepts/core-engine.md).

## Configuration Precedence

BLIS uses a layered configuration system where more specific sources override more general ones:

```
CLI flags (highest priority — explicit user input)
    ↓ overrides
YAML files (policy-config, workload-spec, defaults.yaml)
    ↓ overrides
Hardcoded defaults (lowest priority)
```

CLI flags only override YAML values when explicitly set. BLIS checks whether each flag was provided by the user (not just whether it has a non-default value), so default flag values do not accidentally override YAML configuration.

## Simulation Control

Top-level settings that control the simulation run.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--seed` | int64 | 42 | Random seed for deterministic simulation. Same seed produces byte-identical stdout. |
| `--horizon` | int64 | MaxInt64 | Simulation time limit in ticks (microseconds). Simulation stops when clock exceeds horizon or all requests complete. |
| `--log` | string | "warn" | Log verbosity: trace, debug, info, warn, error, fatal, panic. Logs go to stderr. |
| `--results-path` | string | "" | File path to save per-request results JSON. Empty = stdout only. |

## KV Cache Configuration

Controls GPU and CPU memory simulation for key-value cache blocks. Maps to `KVCacheConfig`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--total-kv-blocks` | int64 | 1000000\* | Total GPU-tier KV blocks. |
| `--block-size-in-tokens` | int64 | 16 | Tokens per KV block. |
| `--kv-cpu-blocks` | int64 | 0 | CPU-tier blocks. 0 disables tiered caching. |
| `--kv-offload-threshold` | float64 | 0.9 | GPU utilization fraction above which blocks are offloaded to CPU. Range [0, 1]. |
| `--kv-transfer-bandwidth` | float64 | 100.0 | GPU-CPU transfer rate in blocks/tick. Required > 0 when CPU blocks > 0. |
| `--kv-transfer-base-latency` | int64 | 0 | Fixed per-transfer latency in ticks. |

\* The CLI default is 1,000,000 but `defaults.yaml` overrides this per model when coefficients are loaded. For example, `llama-3.1-8b/H100/TP=2` uses 132,139 blocks. The override only applies if the user did not explicitly set `--total-kv-blocks`.

## Batch Formation

Controls how requests are selected for the running batch. Maps to `BatchConfig`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-num-running-reqs` | int64 | 256 | Maximum requests in the running batch simultaneously. |
| `--max-num-scheduled-tokens` | int64 | 2048 | Maximum total new tokens across all running requests per step (token budget). |
| `--long-prefill-token-threshold` | int64 | 0 | Prefill length threshold for chunked prefill. 0 = disabled (all prefill in one step). |

## Latency Model

### Regression Coefficients

Trained coefficients for the blackbox latency model. Maps to `LatencyCoeffs`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--alpha-coeffs` | float64 slice | [0, 0, 0] | Alpha coefficients [alpha0, alpha1, alpha2]. Models non-GPU overhead. |
| `--beta-coeffs` | float64 slice | [0, 0, 0] | Beta coefficients [beta0, beta1, beta2]. Models GPU step time. |

When both alpha and beta coefficients are all zeros, BLIS automatically loads pre-trained coefficients from `defaults.yaml` based on the model, GPU, and TP configuration.

### Model and Hardware Selection

Maps to `ModelHardwareConfig`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | string | (required) | LLM model name (e.g., `meta-llama/llama-3.1-8b-instruct`). |
| `--hardware` | string | "" | GPU type (e.g., `H100`, `A100`). If empty, loaded from `defaults.yaml`. |
| `--tp` | int | 0 | Tensor parallelism degree. If 0, loaded from `defaults.yaml`. |
| `--vllm-version` | string | "" | vLLM version string. If empty, loaded from `defaults.yaml`. |

### Roofline Mode

For analytical step time estimation without trained coefficients.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--roofline` | bool | false | Enable roofline mode with auto-fetch. Requires `--hardware` and `--tp`. Auto-resolves model config from `model_configs/` or HuggingFace, and hardware config from bundled `hardware_config.json`. Set `HF_TOKEN` env var for gated models. |
| `--model-config-folder` | string | "" | Path to folder containing HuggingFace `config.json`. Overrides `--roofline` auto-resolution. |
| `--hardware-config` | string | "" | Path to `hardware_config.json` with GPU specifications. Overrides `--roofline` auto-resolution. |

See [Roofline Estimation](../concepts/roofline.md) for details on the analytical model.

### Latency Mode Selection

The latency model mode is selected based on available configuration:

1. **Blackbox mode** (default): If coefficients are provided via CLI flags or loaded from `defaults.yaml`
2. **Explicit roofline mode**: If `--roofline` is set with `--hardware` and `--tp`. Model config is auto-resolved: `model_configs/` (local) → HuggingFace fetch → error. Alpha coefficients and `total_kv_blocks` are loaded from `defaults.yaml` when available. Beta coefficients are replaced by analytical roofline computation.
3. **Implicit roofline mode**: If all coefficients are zero and all four of `--model-config-folder`, `--hardware-config`, `--hardware`, and `--tp` are provided
4. **Error**: If no coefficients can be resolved and roofline inputs are incomplete

## Cluster Configuration

With `--num-instances 1` (the default), BLIS runs a single-instance simulation — requests go directly to the wait queue with no admission or routing layer. With `--num-instances N` (N > 1), the cluster simulation activates: requests pass through the admission and routing pipeline before reaching per-instance wait queues. See [Cluster Architecture](../concepts/architecture.md) for the multi-instance pipeline and [Core Engine](../concepts/core-engine.md) for single-instance internals.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num-instances` | int | 1 | Number of inference instances. 1 = single-instance mode; > 1 = cluster mode with admission and routing. |

## Admission Policy

Controls which requests enter the routing pipeline. See [Cluster Architecture: Admission](../concepts/architecture.md#admission-pipeline).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--admission-policy` | string | "always-admit" | Policy name: `always-admit`, `token-bucket`, `reject-all`. |
| `--admission-latency` | int64 | 0 | Admission decision latency in microseconds. |
| `--token-bucket-capacity` | float64 | 10000 | Token bucket maximum capacity. Required > 0 when using `token-bucket`. |
| `--token-bucket-refill-rate` | float64 | 1000 | Token bucket refill rate in tokens/second. Required > 0 when using `token-bucket`. |

## Routing Policy

Controls how admitted requests are assigned to instances. See [Cluster Architecture: Routing](../concepts/architecture.md#routing-pipeline).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--routing-policy` | string | "round-robin" | Policy name: `round-robin`, `least-loaded`, `weighted`, `prefix-affinity`, `always-busiest`. |
| `--routing-latency` | int64 | 0 | Routing decision latency in microseconds. |
| `--routing-scorers` | string | "" | Scorer configuration for `weighted` policy. Format: `name:weight,name:weight,...` |
| `--snapshot-refresh-interval` | int64 | 0 | KV utilization snapshot refresh interval in microseconds. 0 = immediate refresh. |

### Scorer Configuration

When using `--routing-policy weighted`, the `--routing-scorers` flag configures which scorers are used and their relative weights:

```bash
--routing-scorers "prefix-affinity:3,queue-depth:2,kv-utilization:2"
```

Available scorers: `prefix-affinity`, `queue-depth`, `kv-utilization`, `load-balance`.

Default (when `--routing-scorers` is empty): `prefix-affinity:3, queue-depth:2, kv-utilization:2` (llm-d parity).

See [Cluster Architecture: Scorer Composition](../concepts/architecture.md#scorer-composition) for details on each scorer.

## Scheduling and Priority

Per-instance policies that control request ordering within the wait queue. Maps to `PolicyConfig`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--scheduler` | string | "fcfs" | Scheduler: `fcfs`, `priority-fcfs`, `sjf`, `reverse-priority`. |
| `--priority-policy` | string | "constant" | Priority policy: `constant`, `slo-based`, `inverted-slo`. |

See [Core Engine: Scheduling](../concepts/core-engine.md#scheduling-policies) for policy details.

## Workload Configuration

### Workload Modes

BLIS supports four workload specification modes, in order of precedence:

| Mode | Trigger | Description |
|------|---------|-------------|
| **Workload-spec YAML** | `--workload-spec <path>` | Multi-client workload with per-client distributions. Highest priority. |
| **CLI distribution** | `--workload distribution` (default) | Single-client Gaussian distribution controlled by CLI flags. |
| **Preset** | `--workload <name>` | Named preset from `defaults.yaml` (chatbot, summarization, etc.). |
| **CSV traces** | `--workload traces` | Replay recorded traces from a CSV file. |

### Distribution Mode Flags

Used when `--workload distribution` (the default) and no `--workload-spec` is set.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--rate` | float64 | 1.0 | Request arrival rate in requests/second. |
| `--num-requests` | int | 100 | Total number of requests to generate. |
| `--prompt-tokens` | int | 512 | Mean prompt (input) token count. |
| `--prompt-tokens-stdev` | int | 256 | Standard deviation of prompt tokens. |
| `--prompt-tokens-min` | int | 2 | Minimum prompt token count. |
| `--prompt-tokens-max` | int | 7000 | Maximum prompt token count. |
| `--output-tokens` | int | 512 | Mean output token count. |
| `--output-tokens-stdev` | int | 256 | Standard deviation of output tokens. |
| `--output-tokens-min` | int | 2 | Minimum output token count. |
| `--output-tokens-max` | int | 7000 | Maximum output token count. |
| `--prefix-tokens` | int | 0 | Prefix token count for prefix caching simulation. Additive to prompt tokens. |

### Workload-Spec YAML

The `--workload-spec` flag loads a YAML file defining multi-client workloads:

```yaml
aggregate_rate: 100       # Total arrival rate in requests/second
num_requests: 1000
seed: 42
horizon: 1000000000       # Ticks (microseconds)

clients:
  - id: "interactive"
    rate_fraction: 0.6    # 60% of aggregate rate
    prefix_group: "chat"
    prefix_length: 512
    arrival:
      process: "poisson"
    input_distribution:
      type: "gaussian"
      params:
        mean: 256
        std_dev: 128
        min: 2
        max: 4096
    output_distribution:
      type: "exponential"
      params:
        mean: 128

  - id: "batch"
    rate_fraction: 0.4
    arrival:
      process: "gamma"
      cv: 2.0
    input_distribution:
      type: "gaussian"
      params:
        mean: 1024
        std_dev: 512
        min: 2
        max: 7000
    output_distribution:
      type: "gaussian"
      params:
        mean: 512
        std_dev: 256
        min: 2
        max: 7000
```

**Supported arrival processes:** `poisson`, `gamma` (with `cv` parameter), `weibull` (with `cv` parameter).

**Supported token distributions:** `gaussian`, `exponential`, `pareto_lognormal`, `constant`, `empirical`.

When `--workload-spec` is set, CLI `--seed`, `--horizon`, and `--num-requests` still override the YAML values if explicitly provided.

### Trace Files

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--workload-spec` | string | "" | Path to workload-spec YAML. |
| `--workload-traces-filepath` | string | "" | Path to CSV trace file (required when `--workload traces`). |
| `--defaults-filepath` | string | "defaults.yaml" | Path to `defaults.yaml`. |

## Policy Bundle

The `--policy-config` flag loads admission, routing, priority, and scheduling configuration from a single YAML file:

```yaml
admission:
  policy: "always-admit"
  token_bucket_capacity: 10000.0
  token_bucket_refill_rate: 1000.0

routing:
  policy: "weighted"
  scorers:
    - name: "prefix-affinity"
      weight: 3.0
    - name: "queue-depth"
      weight: 2.0
    - name: "kv-utilization"
      weight: 2.0

priority:
  policy: "constant"

scheduler: "fcfs"
```

CLI flags override policy bundle values when explicitly set. For example, `--routing-policy least-loaded` overrides the bundle's `routing.policy` setting.

## Decision Tracing

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--trace-level` | string | "none" | Trace verbosity: `none` or `decisions`. |
| `--counterfactual-k` | int | 0 | Number of counterfactual candidates per routing decision. Requires `--trace-level decisions`. |
| `--summarize-trace` | bool | false | Print trace summary after simulation. Requires `--trace-level decisions`. |

See [Cluster Architecture: Counterfactual Regret](../concepts/architecture.md#counterfactual-regret).

## Fitness Evaluation

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--fitness-weights` | string | "" | Fitness function weights. Format: `metric:weight,metric:weight,...` |

When configured, BLIS computes a single fitness score from aggregated metrics. Latency metrics are normalized via `1/(1 + value/1000)` where `value` is in ticks (microseconds) and 1000 = 1ms reference (lower is better); throughput metrics via `value/(value + reference)` where `referenceRPS = 100.0` and `referenceTPS = 10000.0` (higher is better). Useful for automated policy comparison across multiple simulation runs.

## defaults.yaml

The `defaults.yaml` file serves as a model registry and workload preset store:

```yaml
version: "1.0"

models:
  - id: "meta-llama/llama-3.1-8b-instruct"
    GPU: "H100"
    tensor_parallelism: 2
    vllm_version: "0.6.1"
    alpha_coeffs: [1601.35, 3.51, 1805.54]
    beta_coeffs: [6910.42, 17.67, 2.84]
    total_kv_blocks: 132139

defaults:
  "meta-llama/llama-3.1-8b-instruct":
    GPU: "H100"
    tensor_parallelism: 2
    vllm_version: "0.6.1"

workloads:
  chatbot:
    prompt_tokens: 512
    output_tokens: 512
    # ... distribution parameters
```

### Resolution Process

When BLIS starts:

1. If `--roofline` is set:
   - Auto-resolve model config: check `model_configs/` for existing `config.json`, fetch from HuggingFace on miss (set `HF_TOKEN` for gated models)
   - Auto-resolve hardware config from bundled `hardware_config.json`
   - Load alpha coefficients and `total_kv_blocks` from `defaults.yaml` (beta coefficients are replaced by roofline computation)
   - `--model-config-folder` and `--hardware-config` override auto-resolution when explicitly set
2. If `--alpha-coeffs` and `--beta-coeffs` are both all-zero and no roofline config is provided:
   - Look up the model in `defaults.yaml` using `--model`, `--hardware`, `--tp`, `--vllm-version`
   - Load alpha/beta coefficients and `total_kv_blocks` from the matching entry
   - Override `--total-kv-blocks` only if the user did not explicitly set it
3. If coefficients are still all-zero but `--model-config-folder` and `--hardware-config` are provided:
   - Enable roofline mode (implicit activation)
4. If coefficients were explicitly provided via CLI:
   - Use them directly, no `defaults.yaml` lookup

## Coefficient Calibration

BLIS uses a data-driven calibration strategy to ensure simulation accuracy. This process runs once per environment configuration (model, GPU, TP degree, vLLM version):

1. **Initialization**: Define baseline estimates for alpha and beta coefficients as starting points for optimization
2. **Profiling**: Execute training workloads on a live vLLM instance to collect ground-truth mean and P90 metrics for TTFT, ITL, and E2E
3. **Optimization**: Run BLIS iteratively using Blackbox Bayesian Optimization to minimize the multi-objective loss:

   $$\text{Loss} = \sum_{m \in \{\text{TTFT, ITL, E2E}\}} \left( |GT_{\text{mean},m} - Sim_{\text{mean},m}| + |GT_{\text{p90},m} - Sim_{\text{p90},m}| \right)$$

4. **Artifact generation**: Optimal alpha/beta coefficients are stored in `defaults.yaml` for production use

For environments where live profiling is not feasible, the [Roofline model](../concepts/roofline.md) provides analytical step time estimation without any training data.

## CLI Flag Summary by Sub-Config

| Sub-Config | Flags |
|------------|-------|
| **KVCacheConfig** | `--total-kv-blocks`, `--block-size-in-tokens`, `--kv-cpu-blocks`, `--kv-offload-threshold`, `--kv-transfer-bandwidth`, `--kv-transfer-base-latency` |
| **BatchConfig** | `--max-num-running-reqs`, `--max-num-scheduled-tokens`, `--long-prefill-token-threshold` |
| **LatencyCoeffs** | `--alpha-coeffs`, `--beta-coeffs` |
| **ModelHardwareConfig** | `--model`, `--hardware`, `--tp`, `--vllm-version`, `--roofline`, `--model-config-folder`, `--hardware-config` |
| **PolicyConfig** | `--scheduler`, `--priority-policy` |
| **WorkloadConfig** | `--workload`, `--workload-spec`, `--workload-traces-filepath`, `--defaults-filepath`, `--rate`, `--num-requests`, `--prompt-tokens*`, `--output-tokens*`, `--prefix-tokens` |
| **DeploymentConfig** | `--num-instances`, `--admission-policy`, `--admission-latency`, `--token-bucket-capacity`, `--token-bucket-refill-rate`, `--routing-policy`, `--routing-latency`, `--routing-scorers`, `--snapshot-refresh-interval`, `--trace-level`, `--counterfactual-k` |
| **Top-level** | `--seed`, `--horizon`, `--log`, `--results-path`, `--policy-config`, `--fitness-weights`, `--summarize-trace` |
