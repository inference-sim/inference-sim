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

\* The effective value of `--total-kv-blocks` depends on the latency backend — see [Resolution Process](#resolution-process) for the full priority chain. In blackbox mode, `defaults.yaml` overrides the 1,000,000 CLI default per model (e.g., `qwen3-14b/H100/TP=1` uses 17,600 blocks). In roofline or crossmodel mode, the value is auto-calculated from model architecture and GPU memory via `CalculateKVBlocks`, which supersedes the `defaults.yaml` value. Explicit `--total-kv-blocks` always takes precedence.

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
| `--alpha-coeffs` | float64 slice | [0, 0, 0] | Alpha coefficients [alpha0, alpha1, alpha2]. Models non-GPU overhead. Must be non-negative. |
| `--beta-coeffs` | float64 slice | [0, 0, 0] | Beta coefficients [beta0, beta1, beta2]. Models GPU step time. Must be non-negative. |

When `--alpha-coeffs` and `--beta-coeffs` are not explicitly provided on the CLI, BLIS automatically loads pre-trained coefficients from `defaults.yaml` based on the model, GPU, and TP configuration. Explicitly passing `--alpha-coeffs 0,0,0` preserves zero coefficients (they are not overridden by defaults).

### Model and Hardware Selection

Maps to `ModelHardwareConfig`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | string | (required) | LLM model name (e.g., `qwen/qwen3-14b`). |
| `--hardware` | string | "" | GPU type (e.g., `H100`, `A100`). If empty, loaded from `defaults.yaml`. |
| `--tp` | int | 0 | Tensor parallelism degree. If 0, loaded from `defaults.yaml`. |
| `--vllm-version` | string | "" | vLLM version string. If empty, loaded from `defaults.yaml`. |
| `--max-model-len` | int64 | 0 | Max total sequence length (input + output) in tokens. 0 = unlimited. Mirrors vLLM's `--max-model-len`. Auto-derived from `max_position_embeddings` in HuggingFace `config.json` for roofline/crossmodel backends. Applies `rope_scaling` factor for types `linear`, `dynamic`, `yarn`, `default`, `mrope`; excludes `su`, `longrope`, `llama3`; skips entirely for `gemma3` models. Capped at KV-feasible maximum. |

### Roofline Mode

For analytical step time estimation without trained coefficients.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--latency-model` | string | "" | Latency model backend: `blackbox` (default), `roofline`, `crossmodel`, `trained-roofline`. When set to `roofline`, `crossmodel`, or `trained-roofline`, auto-fetches HuggingFace config.json and resolves hardware config. Requires `--hardware` and `--tp`. Set `HF_TOKEN` for gated models. `trained-roofline` is recommended for new models (7% MAPE GPU step time). |
| `--model-config-folder` | string | "" | Path to folder containing HuggingFace `config.json`. Overrides `--latency-model` auto-resolution. |
| `--hardware-config` | string | "" | Path to `hardware_config.json` with GPU specifications. Overrides `--latency-model` auto-resolution. |

See [Roofline Estimation](../concepts/roofline.md) for details on the analytical model.

### Latency Mode Selection

The latency model mode is selected based on available configuration:

1. **Blackbox mode** (default): If coefficients are provided via CLI flags or loaded from `defaults.yaml`
2. **Explicit roofline mode**: If `--latency-model roofline` is set with `--hardware` and `--tp`. Model config is auto-resolved: `model_configs/` (local) → HuggingFace fetch → error. Alpha coefficients and `total_kv_blocks` are loaded from `defaults.yaml` when available. Beta coefficients are replaced by analytical roofline computation. Note: `--latency-model blackbox` explicitly prevents implicit roofline detection (respecting user intent).
3. **Explicit cross-model mode**: If `--latency-model crossmodel` is set with `--hardware` and `--tp`. Uses 7 globally-fitted coefficients (4 beta for step time + 3 alpha for CPU overhead) from `crossmodel_defaults` in `defaults.yaml`. Architecture features derived from HuggingFace config.json. MoE-aware.
4. **Implicit roofline mode**: If all coefficients are zero and all four of `--model-config-folder`, `--hardware-config`, `--hardware`, and `--tp` are provided
5. **Error**: If no coefficients can be resolved and roofline inputs are incomplete

## Cluster Configuration

With `--num-instances 1` (the default), BLIS runs a single-instance simulation — requests go directly to the wait queue with no admission or routing layer. With `--num-instances N` (N > 1), the cluster simulation activates: requests pass through the admission and routing pipeline before reaching per-instance wait queues. See [Cluster Architecture](../concepts/architecture.md) for the multi-instance pipeline and [Core Engine](../concepts/core-engine.md) for single-instance internals.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num-instances` | int | 1 | Number of inference instances. 1 = single-instance mode; > 1 = cluster mode with admission and routing. |

## PD Disaggregation

Prefill-Decode (PD) disaggregation splits each request into a prefill sub-request (runs on prefill instances) and a decode sub-request (runs on decode instances), connected by a simulated KV cache transfer. When both `--prefill-instances` and `--decode-instances` are > 0, the cluster activates PD mode automatically.

**Requires:** `--num-instances` must equal `--prefill-instances + --decode-instances`. All instance flags (`--prefill-instances`, `--decode-instances`, transfer flags) must be set together.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prefill-instances` | int | 0 | Number of prefill pool instances. 0 = PD disaggregation disabled. |
| `--decode-instances` | int | 0 | Number of decode pool instances. 0 = PD disaggregation disabled. |
| `--pd-decider` | string | "never" | Disaggregation decider: `never` (standard routing), `always` (all disaggregated), `prefix-threshold` (cache-aware threshold). |
| `--pd-transfer-bandwidth` | float64 | 25 | KV transfer bandwidth in GB/s. Typical values: 25 GB/s (RDMA CX7), 64 GB/s (PCIe5 p2p), 450 GB/s (NVLink4). |
| `--pd-transfer-base-latency` | float64 | 0.05 | KV transfer base latency in milliseconds (fixed overhead added to every transfer). |
| `--pd-kv-bytes-per-token` | int | 512 | KV cache bytes per token for transfer size estimation. Depends on model precision and head dimensions. |
| `--prefill-routing-scorers` | string | "" | Scorer config for routing within the prefill pool. Format: `name:weight,...` (same as `--routing-scorers`). Empty = use main routing policy. |
| `--decode-routing-scorers` | string | "" | Scorer config for routing within the decode pool. Empty = use main routing policy. |

**Transfer duration formula:** `base_latency_us + (num_kv_blocks × block_size_tokens × pd_kv_bytes_per_token) / bandwidth_bytes_per_us`

**Example — 2 prefill + 2 decode instances over RDMA:**
```bash
./blis run --model qwen/qwen3-14b \
  --num-instances 4 \
  --prefill-instances 2 --decode-instances 2 \
  --pd-decider always \
  --pd-transfer-bandwidth 25 \
  --pd-transfer-base-latency 0.05 \
  --pd-kv-bytes-per-token 512
```

## Admission Policy

Controls which requests enter the routing pipeline. See [Cluster Architecture: Admission](../concepts/architecture.md#admission-pipeline).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--admission-policy` | string | "always-admit" | Policy name: `always-admit`, `token-bucket`, `reject-all`. |
| `--admission-latency` | int64 | 0 | Admission decision latency in microseconds. Must be >= 0. |
| `--token-bucket-capacity` | float64 | 10000 | Token bucket maximum capacity. Required > 0 when using `token-bucket`. |
| `--token-bucket-refill-rate` | float64 | 1000 | Token bucket refill rate in tokens/second. Required > 0 when using `token-bucket`. |

## Routing Policy

Controls how admitted requests are assigned to instances. See [Cluster Architecture: Routing](../concepts/architecture.md#routing-pipeline).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--routing-policy` | string | "round-robin" | Policy name: `round-robin`, `least-loaded`, `weighted`, `always-busiest`. |
| `--routing-latency` | int64 | 0 | Routing decision latency in microseconds. Must be >= 0. |
| `--routing-scorers` | string | "" | Scorer configuration for `weighted` policy. Format: `name:weight,name:weight,...` |
| `--snapshot-refresh-interval` | int64 | 0 | Prometheus snapshot refresh interval for all instance metrics (QueueDepth, BatchSize, KVUtilization) in microseconds. 0 = immediate. |

### Scorer Configuration

When using `--routing-policy weighted`, the `--routing-scorers` flag configures which scorers are used and their relative weights:

```bash
--routing-scorers "prefix-affinity:3,queue-depth:2,kv-utilization:2"
```

Available scorers: `prefix-affinity`, `queue-depth`, `kv-utilization`, `load-balance`.

Default (when `--routing-scorers` is empty): `prefix-affinity:3, queue-depth:2, kv-utilization:2` (llm-d parity).

See [Cluster Architecture: Scorer Composition](../concepts/architecture.md#scorer-composition) for details on each scorer.

## PD Disaggregation

Prefill-decode (PD) disaggregation splits the prefill and decode phases across separate instance pools. Prefill instances process the initial prompt tokens; decode instances generate output tokens. KV cache state is transferred between pools after prefill completes.

PD disaggregation requires `--prefill-instances` and `--decode-instances` to be configured alongside `--num-instances`. The pool sizes must satisfy `prefill + decode ≤ num-instances`; any remaining instances receive locally-routed (non-disaggregated) requests via the standard routing path.

### Disaggregation Deciders

The `--pd-decider` flag selects the policy that decides whether each request is routed through the disaggregated PD path:

| Decider | Description |
|---------|-------------|
| `never` | Never disaggregate. All requests use the standard routing path (default). |
| `always` | Always disaggregate. All requests go through the PD prefill → transfer → decode pipeline. |
| `prefix-threshold` | Disaggregate when the number of **non-cached input tokens** exceeds `--pd-prefix-threshold`. |
| `direct-to-decode` | Disaggregate when `len(InputTokens) >= --pd-direct-decode-threshold`. Short prompts skip disaggregation and route directly to the decode pool. |

### direct-to-decode Semantics

The `direct-to-decode` decider compares total input token count against the threshold:

- Disaggregate when `len(InputTokens) >= threshold`; route directly to decode pool when `len(InputTokens) < threshold`
- Empty inputs always route directly to decode (no disaggregation)
- Non-disaggregated requests route to the decode pool only (INV-P2-4), where both prefill and decode phases run on the same instance with potential interference cost

**Default threshold: 256 tokens.** Short prompts (< 256 tokens) skip the transfer overhead by running both phases locally on a decode instance. Long prompts benefit from dedicated prefill compute. Tune based on your break-even point between transfer cost and co-location interference.

### prefix-threshold Semantics

The `prefix-threshold` decider compares non-cached token count (not total token count) against the threshold:

- **Non-cached tokens** = `len(InputTokens) − (cached_blocks × block_size)`
- Requests with a long cached prefix contribute fewer non-cached tokens even if the total prompt is large
- Disaggregate when `non-cached_tokens > threshold`; route locally when `non-cached_tokens ≤ threshold`

This reflects the real cost driver: disaggregating prefill is only beneficial when there is significant computation to offload. A request whose prefix is already in the router-side cache has less uncached compute, and disaggregating it would waste KV transfer bandwidth.

**Default threshold: 512 tokens.** This means requests with more than 512 non-cached input tokens are disaggregated. Adjust based on your workload's typical uncached prompt length and your inter-pool transfer cost.

### PD Disaggregation Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prefill-instances` | int | 0 | Number of prefill-pool instances (0 = PD disabled). |
| `--decode-instances` | int | 0 | Number of decode-pool instances (0 = PD disabled). |
| `--pd-decider` | string | "never" | Disaggregation decider: `never`, `always`, `prefix-threshold`, `direct-to-decode`. |
| `--pd-prefix-threshold` | int | 512 | Non-cached token threshold for `prefix-threshold` decider (>= 0). Ignored for other deciders. |
| `--pd-direct-decode-threshold` | int | 256 | Input token threshold for `direct-to-decode` decider (>= 0). Prompts shorter than this skip disaggregation. Ignored for other deciders. |
| `--pd-transfer-bandwidth` | float64 | 25.0 | KV cache transfer bandwidth between pools in GB/s. Models a shared global fabric — all concurrent transfers compete for this single bandwidth budget. Set to the NIXL/RDMA link speed. Default 25.0 matches a 200 Gbit/s RDMA fabric (≈ 25 GB/s). |
| `--pd-transfer-base-latency` | float64 | 0.05 | Base transfer latency per transfer in milliseconds. |
| `--pd-kv-bytes-per-token` | int | 512 | KV cache bytes per token for transfer size calculation. |
| `--pd-transfer-contention` | bool | false | Enable fair-share bandwidth contention model (INV-P2-2): when N transfers are in-flight, each gets `bandwidth/N`. Models a shared global fabric where all concurrent transfers contend for the same bandwidth pool. Disabled by default. |
| `--pd-interference-prefill` | float64 | 0 | Co-location interference factor for prefill-dominant batches (majority is prefill, minority is decode). Multiplier = `1 + factor × (minority/total)`. Factor 0.5 at 50/50 split → 1.25× step time. 0 = disabled (no co-location penalty). |
| `--pd-interference-decode` | float64 | 0 | Co-location interference factor for decode-dominant batches (majority is decode, minority is prefill). Multiplier = `1 + factor × (minority/total)`. Factor 0.5 at 50/50 split → 1.25× step time. 0 = disabled (no co-location penalty). |
| `--prefill-routing-scorers` | string | "" | Scorer config for weighted routing within the prefill pool. |
| `--decode-routing-scorers` | string | "" | Scorer config for weighted routing within the decode pool. |

### Per-Pool Hardware Overrides

When prefill and decode pools use different hardware (e.g., high-TP prefill for compute-bound prompt processing, low-TP decode for memory-bandwidth-bound token generation), per-pool hardware overrides allow each pool to have its own configuration. All per-pool flags default to the global config when unset.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prefill-tp` | int | 0 | Tensor parallelism for prefill pool (0 = use global `--tp`). |
| `--decode-tp` | int | 0 | Tensor parallelism for decode pool (0 = use global `--tp`). |
| `--prefill-hardware` | string | "" | GPU type for prefill pool (empty = use global `--hardware`). |
| `--decode-hardware` | string | "" | GPU type for decode pool (empty = use global `--hardware`). |
| `--prefill-latency-model` | string | "" | Latency backend for prefill pool (empty = use global `--latency-model`). |
| `--decode-latency-model` | string | "" | Latency backend for decode pool (empty = use global `--latency-model`). |
| `--prefill-max-model-len` | int64 | 0 | Max model length for prefill pool (0 = use global `--max-model-len`). |
| `--decode-max-model-len` | int64 | 0 | Max model length for decode pool (0 = use global `--max-model-len`). |

**KV auto-calculation:** When an analytical latency backend (`roofline`, `crossmodel`, `trained-roofline`) is active and a pool's TP or GPU differs from the global config, KV blocks are auto-calculated per-pool using the pool's effective hardware. This ensures each pool's KV capacity matches its actual GPU memory.

**Known limitation — admission vs. per-pool capacity:** Admission decisions are made at cluster level against the global config before disaggregation routing. If a pool has a lower `MaxModelLen` or `TotalKVBlocks` than the global config, a request that passes admission may be dropped at the instance level (prefill enqueue guard or decode KV allocation). This mirrors real disaggregated inference systems where the gateway admission check uses global thresholds. To avoid unexpected drops, ensure per-pool `MaxModelLen` values are not smaller than the typical request input length, and monitor `droppedAtDecodeKV` in simulation results.

**Example — heterogeneous TP (8-way prefill, 2-way decode):**
```bash
./blis run --model meta-llama/llama-3.1-70b-instruct \
  --num-instances 4 \
  --prefill-instances 2 --decode-instances 2 \
  --pd-decider always \
  --latency-model roofline --hardware H100 --tp 4 \
  --prefill-tp 8 --decode-tp 2
```

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
| **Preset** | `--workload <name>` | Named preset from `defaults.yaml`: `chatbot`, `contentgen`, `summarization`, `multidoc`. |
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

**Supported arrival processes:** `poisson`, `gamma` (with `cv` parameter), `weibull` (with `cv` parameter), `constant`.

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
# Section 1: Hardware/TP mappings (keyed by model ID)
defaults:
  qwen/qwen3-14b:
    GPU: H100
    tensor_parallelism: 1
    vllm_version: vllm/vllm-openai:v0.11.0
    hf_repo: Qwen/Qwen3-14B

# Section 2: Workload presets
workloads:
  chatbot:
    prompt_tokens: 256
    prompt_tokens_stdev: 100
    output_tokens: 256
    output_tokens_stdev: 100
    # ... min/max bounds

# Section 3: Trained coefficients (keyed by model+GPU+TP)
models:
  - id: qwen/qwen3-14b
    GPU: H100
    tensor_parallelism: 1
    vllm_version: vllm/vllm-openai:v0.11.0
    alpha_coeffs: [8888.09, 0.18, 0.0]
    beta_coeffs: [13578.19, 39.44, 27.32]
    total_kv_blocks: 17600
```

### Resolution Process

When BLIS starts, it resolves latency coefficients and KV block counts through a layered process. Explicit CLI flags always take precedence (R18).

**Latency coefficient resolution:**

1. If `--latency-model roofline` or `--latency-model crossmodel` is set:
   - Auto-resolve model config: check `model_configs/` for existing `config.json`, fetch from HuggingFace on miss (set `HF_TOKEN` for gated models)
   - Auto-resolve hardware config from bundled `hardware_config.json`
   - For roofline: load alpha coefficients and per-model KV blocks from `defaults.yaml` (beta coefficients are replaced by analytical computation). Warns if no per-model KV blocks found
   - For crossmodel: load global alpha + beta coefficients from `crossmodel_defaults` in `defaults.yaml`, and per-model KV blocks if available
   - `--model-config-folder` and `--hardware-config` override auto-resolution when explicitly set
2. If `--alpha-coeffs` and `--beta-coeffs` are not explicitly provided on the CLI and no analytical backend is selected:
   - Look up the model in `defaults.yaml` using `--model`, `--hardware`, `--tp`, `--vllm-version`
   - Load alpha/beta coefficients from the matching entry
3. If coefficients are still all-zero (no defaults found) but `--model-config-folder` and `--hardware-config` are provided:
   - Enable roofline mode (implicit activation)
4. If coefficients were explicitly provided via CLI (including explicit zeros):
   - Use them directly, no `defaults.yaml` lookup

**`--total-kv-blocks` resolution** (highest priority wins):

1. **Explicit CLI flag** — if `--total-kv-blocks` is set, that value is used regardless of backend
2. **Auto-calculation** (roofline/crossmodel only) — when `MemoryGiB > 0` in the hardware config, `CalculateKVBlocks` derives the block count from model architecture and GPU memory, superseding the `defaults.yaml` value. Three failure modes: (a) if `MemoryGiB` is missing from `hardware_config.json`, BLIS warns and falls back to the `defaults.yaml` value (layer 3) or hardcoded default (layer 4); (b) if model architecture params cannot be extracted from `config.json`, BLIS exits with an error; (c) if the calculation itself fails (e.g., unsupported activation function), BLIS exits with an error. Only the `MemoryGiB`-missing case is a graceful fallback — other failures are fatal. Auto-calculation currently requires SwiGLU-family activations (`silu`, `swiglu`, `geglu`); models with other activations (e.g., Falcon's `gelu`) should set `--total-kv-blocks` explicitly
3. **`defaults.yaml`** — per-model block count loaded for the model/GPU/TP combination (e.g., 17,600 for qwen3-14b/H100/TP=1). For roofline/crossmodel with `MemoryGiB > 0`, this value is superseded by auto-calculation (layer 2). It remains the effective value only for blackbox mode or when `MemoryGiB` is unavailable in the hardware config
4. **Hardcoded default** — 1,000,000 (CLI flag default, used only when no other source provides a value)

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
| **ModelHardwareConfig** | `--model`, `--hardware`, `--tp`, `--vllm-version`, `--latency-model`, `--model-config-folder`, `--hardware-config`, `--max-model-len` |
| **PolicyConfig** | `--scheduler`, `--priority-policy` |
| **WorkloadConfig** | `--workload`, `--workload-spec`, `--workload-traces-filepath`, `--defaults-filepath`, `--rate`, `--num-requests`, `--prompt-tokens*`, `--output-tokens*`, `--prefix-tokens` |
| **DeploymentConfig** | `--num-instances`, `--admission-policy`, `--admission-latency`, `--token-bucket-capacity`, `--token-bucket-refill-rate`, `--routing-policy`, `--routing-latency`, `--routing-scorers`, `--snapshot-refresh-interval`, `--trace-level`, `--counterfactual-k` |
| **PD Disaggregation** | `--prefill-instances`, `--decode-instances`, `--pd-decider` (`never`\|`always`\|`prefix-threshold`), `--pd-prefix-threshold`, `--pd-transfer-bandwidth`, `--pd-transfer-base-latency`, `--pd-kv-bytes-per-token`, `--prefill-routing-scorers`, `--decode-routing-scorers`, `--prefill-tp`, `--decode-tp`, `--prefill-hardware`, `--decode-hardware`, `--prefill-latency-model`, `--decode-latency-model`, `--prefill-max-model-len`, `--decode-max-model-len` |
| **Top-level** | `--seed`, `--horizon`, `--log`, `--results-path`, `--policy-config`, `--fitness-weights`, `--summarize-trace` |
