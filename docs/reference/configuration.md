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

### Parameter Resolution by Category

The general precedence (CLI → YAML → hardcoded) applies everywhere, but each parameter category has its own resolution layers. The chains below show highest-to-lowest priority for each category.

**Latency coefficients** (`--alpha-coeffs`, `--beta-coeffs`):

1. Explicit CLI flags — if passed, used directly (no `defaults.yaml` lookup)
2. Analytical computation — roofline or trained-physics backends compute from architecture + hardware specs

**Hardware and TP** (`--hardware`, `--tp`, `--vllm-version`):

1. Explicit CLI flags
2. `defaults.yaml` `defaults[]` entry — matched by `--model`
3. Error — roofline/trained-physics require these values

**KV cache blocks** (`--total-kv-blocks`): See the detailed [Resolution Process](#resolution-process) below — three layers: CLI flag, auto-calculation, or 1M default.

**Workload parameters** (`--rate`, `--num-requests`, `--prompt-tokens`, etc.):

1. `--workload-spec` YAML file — when set, all token distribution and arrival parameters come from the YAML; CLI distribution flags are ignored
2. CLI distribution flags — when `--workload distribution` (default) and no `--workload-spec`
3. Named preset from `defaults.yaml` — when `--workload <name>` (e.g., `chatbot`)
4. Hardcoded CLI flag defaults — (e.g., `--prompt-tokens 512`, `--output-tokens 512`)

!!! note
    `--seed`, `--horizon`, and `--num-requests` are exceptions — they override the workload-spec YAML values even when `--workload-spec` is set. `--rate` does NOT override `aggregate_rate` in the YAML (see [Common Pitfalls](#common-pitfalls)).

**Routing, admission, scheduling, and preemption** (`--routing-policy`, `--admission-policy`, `--scheduler`, `--preemption-policy`, etc.):

1. Explicit CLI flags
2. `--policy-config` YAML bundle — loads all policy settings from one file
3. Hardcoded defaults — `round-robin`, `always-admit`, `fcfs`

**Batch formation** (`--max-num-running-reqs`, `--max-num-scheduled-tokens`, etc.):

1. Explicit CLI flags
2. Hardcoded defaults — 256 running reqs, 2048 scheduled tokens

Batch formation has no YAML override path — `defaults.yaml` and `--policy-config` do not include batch settings.

### Known Unit Gotchas

All internal timestamps in the DES (arrival time, schedule time, completion time, clock) use **ticks**, where **1 tick = 1 microsecond (μs)**. Output metrics convert to milliseconds for human readability, but several fields and flags use different units:

| Field / Flag | Unit | Notes |
|-------------|------|-------|
| `ttft_ms`, `e2e_ms`, `itl_ms` (per-request JSON) | milliseconds | Converted from ticks by dividing by 1,000 |
| `scheduling_delay_ms` (per-request JSON) | milliseconds | Converted from ticks by dividing by 1,000. Historically was in ticks (μs) despite the `_ms` suffix — fixed by BC-14. Old hypothesis scripts (pre-fix) divide by 1,000 again unnecessarily. |
| `scheduling_delay_p99_ms` (aggregate) | milliseconds | Always was in milliseconds |
| `--horizon` | ticks (μs) | Simulation time limit. 1,000,000 = 1 second |
| `--admission-latency`, `--routing-latency` | ticks (μs) | Decision latency injected into the DES event queue |
| `think_time_us` (workload YAML) | microseconds | Inter-round delay in multi-turn sessions. 5,000,000 = 5 seconds |
| `aggregate_rate`, `--rate` | requests/second | Not ticks — real-world time unit |
| `--kv-transfer-bandwidth` | blocks/tick | Transfer rate between GPU and CPU KV tiers |
| `--kv-transfer-base-latency` | ticks (μs) | Fixed per-transfer overhead |

### Common Pitfalls

**Capacity estimate mismatch (issue #390).** CLI distribution mode defaults to `--prompt-tokens 512, --output-tokens 512`. If you estimate per-instance capacity using CLI mode, then run a workload-spec YAML with shorter sequences (e.g., mean 256/128), the YAML workload will achieve ~1.5x higher throughput than the CLI estimate predicted. Always derive capacity estimates from the actual workload you plan to run, not from CLI defaults.

**`--rate` does NOT override workload-spec YAML.** The `--rate` flag only applies in CLI distribution mode. When `--workload-spec` is set, request rate comes from `aggregate_rate` in the YAML file — the `--rate` flag is ignored. To change the rate for a YAML workload, edit the `aggregate_rate` field in the spec.

**`aggregate_rate` override for inference-perf specs.** When converting inference-perf specs via `blis convert infperf`, per-stage rates in the spec override a user-specified `aggregate_rate`. If the sum of stage rates differs from `aggregate_rate`, BLIS logs a warning and uses the stage-rate sum. This prevents silent rate scaling errors.

**`--total-kv-blocks` phantom default.** The CLI default is 1,000,000 blocks, but this value almost never takes effect. For all latency backends (roofline, trained-physics), auto-calculation from model architecture and GPU memory supersedes it when a HuggingFace `config.json` is available and the hardware config specifies `MemoryGiB`. The 1M default is a last-resort fallback — if your simulation uses it, check whether auto-calculation is failing (missing `config.json`, missing `MemoryGiB`, or unsupported model architecture).

**`enable_multi_turn_chat` semantic mismatch (issue #517).** inference-perf's `enable_multi_turn_chat` creates one persistent session per virtual user. BLIS's closest equivalent is `multi_turn.single_session: true` in the workload YAML, but the session mechanics differ. When converting inference-perf specs, verify that the converted multi-turn behavior matches your intent.

**Custom `defaults.yaml` files must remove `total_kv_blocks` entries (migration note).** BLIS uses strict YAML parsing (`KnownFields(true)`) to catch typos and invalid fields. As of issue #1035, `total_kv_blocks` is no longer a recognized field in `defaults.yaml` — KV capacity is now always auto-calculated from model architecture and GPU memory. If you maintain a custom `defaults.yaml` file, remove all `total_kv_blocks` entries or BLIS will exit with a parse error: `field total_kv_blocks not found`. To override auto-calculation, use the `--total-kv-blocks` CLI flag instead.

## Simulation Control

Top-level settings that control the simulation run.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--seed` | int64 | 42 | Random seed for deterministic simulation. Same seed produces byte-identical stdout. |
| `--horizon` | int64 | MaxInt64 | Simulation time limit in ticks (microseconds). Simulation stops when clock exceeds horizon or all requests complete. |
| `--log` | string | "warn" | Log verbosity: trace, debug, info, warn, error, fatal, panic. Logs go to stderr. |
| `--metrics-path` | string | "" | File path to write MetricsOutput JSON (aggregate P50/P95/P99 TTFT, E2E, throughput stats). blis run only — blis replay uses `--results-path` instead. Empty = no file output. |

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

\* The effective value of `--total-kv-blocks` follows a 3-layer resolution: (1) explicit `--total-kv-blocks` CLI flag, (2) auto-calculation from model architecture and GPU memory via `CalculateKVBlocks` (for all backends when `config.json` and `MemoryGiB` are available), (3) hardcoded default of 1,000,000 blocks. See [Resolution Process](#resolution-process) for details.

## Batch Formation

Controls how requests are selected for the running batch. Maps to `BatchConfig`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--max-num-running-reqs` | int64 | 256 | Maximum requests in the running batch simultaneously. |
| `--max-num-scheduled-tokens` | int64 | 2048 | Maximum total new tokens across all running requests per step (token budget). |
| `--long-prefill-token-threshold` | int64 | 0 | Prefill length threshold for chunked prefill. 0 = disabled (all prefill in one step). |
| `--preemption-policy` | string | "fcfs" | Preemption victim selection: `fcfs` (tail-of-batch, default) or `priority` (least-urgent SLO tier evicted first, matching vLLM `--scheduling-policy priority`). |

## Latency Model

### Regression Coefficients

Trained coefficients for physics-informed latency estimation. Maps to `LatencyCoeffs`.

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
| `--hardware` | string | "" | GPU type. Bundled options: `H100`, `A100-SXM`, `A100-80`. If empty, loaded from `defaults.yaml`. Add new GPUs to `hardware_config.json`. |
| `--tp` | int | 0 | Tensor parallelism degree. If 0, loaded from `defaults.yaml`. |
| `--vllm-version` | string | "" | vLLM version string. If empty, loaded from `defaults.yaml`. |
| `--max-model-len` | int64 | 0 | Max total sequence length (input + output) in tokens. 0 = unlimited. Mirrors vLLM's `--max-model-len`. Auto-derived from `max_position_embeddings` in HuggingFace `config.json` for roofline/trained-physics backends. Applies `rope_scaling` factor for types `linear`, `dynamic`, `yarn`, `default`, `mrope`; excludes `su`, `longrope`, `llama3`; skips entirely for `gemma3` models. Capped at KV-feasible maximum. |

### Roofline Mode

For analytical step time estimation without trained coefficients.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--latency-model` | string | "roofline" | Latency model backend: `roofline` (default), `trained-physics`. Both backends auto-fetch HuggingFace config.json for KV block auto-calculation (may require network access). Both require `config.json` for latency estimation and KV sizing. Requires `--hardware` and `--tp`. Set `HF_TOKEN` for gated models. `trained-physics` is recommended for new models. |
| `--model-config-folder` | string | "" | Path to folder containing HuggingFace `config.json`. Overrides `--latency-model` auto-resolution. |
| `--hardware-config` | string | "" | Path to `hardware_config.json` with GPU specifications. Overrides `--latency-model` auto-resolution. |

See [Roofline Estimation](../concepts/roofline.md) for details on the analytical model.

### Latency Mode Selection

The latency model mode is selected based on available configuration:

1. **Roofline mode** (default): Auto-resolves model config from HuggingFace and hardware config from bundled `hardware_config.json`. Requires `--hardware` and `--tp` (loaded from `defaults.yaml` when available).
2. **Trained-physics mode** (recommended for new models): If `--latency-model trained-physics` is set with `--hardware` and `--tp`. Uses 13 globally-fitted coefficients (10 beta for roofline corrections with architecture-aware MoE scaling + 3 alpha for CPU overhead) from `trained_physics_coefficients` in `defaults.yaml`. Physics-informed basis functions with learned corrections.

## Cluster Configuration

With `--num-instances 1` (the default), BLIS runs a single-instance simulation — requests go directly to the wait queue with no admission or routing layer. With `--num-instances N` (N > 1), the cluster simulation activates: requests pass through the admission and routing pipeline before reaching per-instance wait queues. See [Cluster Architecture](../concepts/architecture.md) for the multi-instance pipeline and [Core Engine](../concepts/core-engine.md) for single-instance internals.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num-instances` | int | 1 | Number of inference instances. 1 = single-instance mode; > 1 = cluster mode with admission and routing. |

## Admission Policy

Controls which requests enter the routing pipeline. See [Cluster Architecture: Admission](../concepts/architecture.md#admission-pipeline).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--admission-policy` | string | "always-admit" | Policy name: `always-admit`, `token-bucket`, `reject-all`, `tier-shed`, `gaie-legacy`. |
| `--admission-latency` | int64 | 0 | Admission decision latency in microseconds. Must be >= 0. |
| `--token-bucket-capacity` | float64 | 10000 | Token bucket maximum capacity. Required > 0 when using `token-bucket`. |
| `--token-bucket-refill-rate` | float64 | 1000 | Token bucket refill rate in tokens/second. Required > 0 when using `token-bucket`. |

**Tier-shed admission** (`--admission-policy tier-shed`): Sheds lower-priority SLO tiers under overload. Configured via `--policy-config` YAML only:

| YAML field | Type | Default | Description |
|------------|------|---------|-------------|
| `admission.tier_shed_threshold` | int | 0 | Per-instance in-flight threshold above which shedding activates. 0 = shed at any load. |
| `admission.tier_shed_min_priority` | int | 3 | Minimum SLO tier priority admitted under overload. 3 = admit Standard+Critical, shed sheddable tiers (priority < 0). No range constraint (GAIE priorities are arbitrary integers). |
| `admission.slo_priorities` | map[string]int | nil | Custom SLO class priority overrides. Merges on top of GAIE defaults. See [SLO Tier Priorities](#slo-tier-priorities) below. |

**GAIE-legacy admission** (`--admission-policy gaie-legacy`): Saturation-based shedding matching production llm-d/GAIE. Non-sheddable requests always pass; sheddable requests rejected when pool-average saturation >= 1.0. Saturation = `avg(max(qd/qdThreshold, kvUtil/kvThreshold))` across instances. Configured via `--policy-config` YAML only:

| YAML field | Type | Default | Source | Description |
|------------|------|---------|--------|-------------|
| `admission.gaie_qd_threshold` | float64 | 5 | GAIE `DefaultQueueDepthThreshold` (`config.go:31`) | Per-instance queue depth threshold. Must be > 0. |
| `admission.gaie_kv_threshold` | float64 | 0.8 | GAIE `DefaultKVCacheUtilThreshold` (`config.go:33`) | Per-instance KV cache utilization threshold. Must be in (0, 1.0]. |
| `admission.slo_priorities` | map[string]int | nil | — | Custom SLO class priority overrides (shared with tier-shed). |

### SLO Tier Priorities

Each SLO class has an integer priority that determines admission ordering, shedding decisions, and gateway queue dispatch. Priorities follow the GAIE (Gateway API Inference Extension) convention where **negative priority = sheddable**.

**Default priorities (GAIE-compatible):**

| SLO Class | Priority | Sheddable? | Description |
|-----------|----------|------------|-------------|
| `critical` | 4 | No | Highest priority. Never shed by tier-shed or tenant budgets. |
| `standard` | 3 | No | Default for empty/unknown SLO class. Protected from shedding. |
| `batch` | -1 | Yes | Offline/batch workloads. Shed under overload or tenant budget pressure. |
| `sheddable` | -2 | Yes | Explicitly sheddable workloads. |
| `background` | -3 | Yes | Lowest priority. First to be shed. |

**Key semantic:** `IsSheddable(class) = Priority(class) < 0`. This matches llm-d's `sheddable.go` contract. Classes with priority >= 0 are protected from tenant budget shedding and are shed by tier-shed only when their priority is below `tier_shed_min_priority`.

**Custom overrides:** Override specific priorities via the policy bundle YAML. Unspecified classes retain defaults.

```yaml
admission:
  policy: "tier-shed"
  slo_priorities:
    batch: 0       # make batch non-sheddable (protected like standard)
    critical: 10   # increase critical priority gap
```

**Where priorities are used in the codebase:**

| Component | File | How priorities are used |
|-----------|------|----------------------|
| Tier-shed admission | `sim/admission.go` | Rejects requests with `Priority(class) < MinAdmitPriority` under overload |
| Tenant budget enforcement | `sim/cluster/cluster_event.go` | Sheds over-budget requests where `IsSheddable(class)` is true (priority < 0) |
| Gateway queue dispatch | `sim/cluster/gateway_queue.go` | Priority-ordered dispatch: higher priority dequeued first; capacity shedding evicts lowest priority |
| Backward compatibility | `sim/admission.go` | `SLOTierPriority()` delegates to `DefaultSLOPriorityMap().Priority()` |

**Per-tenant fair-share budgets** (`tenant_budgets`): A secondary admission layer that runs *after* the admission policy. If the admission policy rejects a request, tenant budgets are not consulted. If the admission policy admits a request, tenant budgets then apply: over-budget tenants have sheddable requests (`IsSheddable(class) = priority < 0`) preferentially shed, while non-sheddable traffic (critical, standard) is always protected. Configured via `--policy-config` YAML only (no CLI flag):

| YAML field | Type | Default | Description |
|------------|------|---------|-------------|
| `tenant_budgets` | map[string]float64 | nil | Per-tenant fraction of total cluster capacity (NumInstances × MaxRunningReqs). Absent key = unlimited. 0.0 = effectively zero concurrent slots (one request may slip through per admission tick due to DES admission-before-routing event ordering; see IsOverBudget docstring). Values must be in [0, 1]. |

Example:

```yaml
admission:
  policy: "tier-shed"
  tier_shed_threshold: 0
  tier_shed_min_priority: 3  # admit standard(3) and critical(4); shed sheddable tiers (priority < 0)
  slo_priorities:             # optional: override specific priorities
    batch: 0                  # promote batch to non-sheddable

tenant_budgets:
  alice: 0.3   # alice may use at most 30% of total cluster capacity
  bob: 0.7     # bob may use at most 70% of total cluster capacity
```

## Routing Policy

Controls how admitted requests are assigned to instances. See [Cluster Architecture: Routing](../concepts/architecture.md#routing-pipeline).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--routing-policy` | string | "round-robin" | Policy name: `round-robin`, `least-loaded`, `weighted`, `always-busiest`. |
| `--routing-latency` | int64 | 0 | Routing decision latency in microseconds. Must be >= 0. |
| `--routing-scorers` | string | "" | Scorer configuration for `weighted` policy. Format: `name:weight,name:weight,...` |
| `--snapshot-refresh-interval` | int64 | 0 | Prometheus snapshot refresh interval for all instance metrics (QueueDepth, BatchSize, KVUtilization, PreemptionCount) in microseconds. 0 = immediate. |

### Scorer Configuration

When using `--routing-policy weighted`, the `--routing-scorers` flag configures which scorers are used and their relative weights:

```bash
--routing-scorers "precise-prefix-cache:2,queue-depth:1,kv-utilization:1"
```

Available scorers: `prefix-affinity`, `precise-prefix-cache`, `no-hit-lru`, `queue-depth`, `kv-utilization`, `load-balance`, `active-requests`, `running-requests`, `load-aware`.

Default (when `--routing-scorers` is empty): `precise-prefix-cache:2, queue-depth:1, kv-utilization:1` (llm-d parity).

See [Cluster Architecture: Scorer Composition](../concepts/architecture.md#scorer-composition) for details on each scorer.

## Scheduling and Priority

Per-instance policies that control request ordering within the wait queue. Maps to `PolicyConfig`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--scheduler` | string | "fcfs" | Scheduler: `fcfs`, `priority-fcfs`, `sjf`, `reverse-priority`. |
| `--priority-policy` | string | "constant" | Priority policy: `constant`, `slo-based`, `inverted-slo`. |
| `--preemption-policy` | string | "fcfs" | Preemption victim selection: `fcfs` (tail-of-batch, default) or `priority` (least-urgent SLO tier evicted first, matching vLLM `--scheduling-policy priority`). |

See [Core Engine: Scheduling](../concepts/core-engine.md#scheduling-policies) for policy details.

## Workload Configuration

### Workload Modes

BLIS supports three workload specification modes, in order of precedence:

| Mode | Trigger | Description |
|------|---------|-------------|
| **Workload-spec YAML** | `--workload-spec <path>` | Multi-client workload with per-client distributions. Highest priority. |
| **CLI distribution** | `--workload distribution` (default) | Single-client Gaussian distribution controlled by CLI flags. |
| **Preset** | `--workload <name>` | Named preset from `defaults.yaml`: `chatbot`, `contentgen`, `summarization`, `multidoc`. |

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
| `--defaults-filepath` | string | "defaults.yaml" | Path to `defaults.yaml`. |
| `--trace-output` | string | "" | Export workload as TraceV2 files (`<prefix>.yaml` + `<prefix>.csv`). |

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

preemption:
  policy: "priority"    # fcfs (default) or priority (least-urgent SLO tier evicted first)

# Node pool infrastructure (Phase 1A — optional; omit for backward-compatible single-pool mode)
node_pools:
  - name: "gpu-pool-1"
    gpu_type: "H100"      # pool-authoritative: overrides --gpu flag for GPU label (all backends) and hardware calibration (roofline/trained-physics backends); see issues #892/#893
    gpus_per_node: 8
    gpu_memory_gib: 80.0
    initial_nodes: 2
    min_nodes: 1
    max_nodes: 4
    provisioning_delay:
      mean: 30.0   # seconds
      stddev: 5.0  # 0 = constant delay

# Per-GPU hardware calibration overrides for roofline/trained-physics backends (issue #893 — optional)
# Key: GPU type string matching a pool's gpu_type. Value: HardwareCalib for that GPU.
# When a pool's gpu_type is found in this map, the matched calibration overrides the CLI
# --gpu calibration at instance construction time (both sync and deferred/NodeReadyEvent paths),
# ensuring pool-placed instances use the correct TFlopsPeak/BwPeakTBs for roofline math.
# Omitting this field (zero value) is safe: no override, backward-compatible with all callers.
# Keys must exactly match the gpu_type strings used in the node_pools entries above.
hw_config_by_gpu:
  H100:
    tflops_peak: 1979.0    # FP16 TFLOPS
    bw_peak_tbs: 3.35      # HBM bandwidth in TB/s
    mfu_prefill: 0.5
    mfu_decode: 0.5
  A100:
    tflops_peak: 1248.0
    bw_peak_tbs: 2.0
    mfu_prefill: 0.5
    mfu_decode: 0.5

# Instance lifecycle (Phase 1A — all zero/empty = backward-compatible defaults)
instance_lifecycle:
  loading_delay:
    mean: 10.0    # seconds to load model weights onto GPU
    stddev: 1.0   # 0 = constant delay
  warm_up_request_count: 5    # requests served before leaving WarmingUp state
  warm_up_ttft_factor: 2.0    # TTFT multiplier applied to warm-up requests (≥ 1.0)
  drain_policy: "WAIT"        # IMMEDIATE | WAIT | REDIRECT

# SLO priority overrides (optional; omit for GAIE defaults)
# GAIE defaults: critical=4, standard=3, batch=-1, sheddable=-2, background=-3
# Negative priority = sheddable. Override to change which classes are sheddable.
# admission:
#   slo_priorities:
#     batch: 0    # make batch non-sheddable

# Per-tenant fair-share budgets (Phase 1B — optional; omit for no tenant enforcement)
# Each value is a fraction of total cluster capacity (NumInstances × MaxRunningReqs).
# Absent key = unlimited. 0.0 = effectively zero concurrent slots (DES ordering caveat: see IsOverBudget docstring). Values must be in [0, 1].
# Non-sheddable traffic (priority >= 0: critical, standard) is always protected from budget shedding.
tenant_budgets:
  team-a: 0.4
  team-b: 0.4
```

CLI flags override policy bundle values when explicitly set. For example, `--routing-policy least-loaded` overrides the bundle's `routing.policy` setting.

!!! note "Node pools and instance lifecycle are YAML-only"
    `node_pools` and `instance_lifecycle` have no corresponding CLI flags. They must be set via `--policy-config`. Omitting them is safe — the simulator falls back to single-pool, no-lifecycle mode for full backward compatibility.

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
```

### Resolution Process

When BLIS starts, it resolves latency configuration through a layered process. Explicit CLI flags always take precedence (R18).

**Hardware and TP defaults resolution (all backends):**

Before any backend-specific logic runs, BLIS loads hardware/TP/vLLM-version defaults from `defaults.yaml` for the specified `--model` when those flags are not explicitly provided. This ensures analytical backends (roofline, trained-physics) can auto-resolve without requiring explicit `--hardware` and `--tp` for models listed in `defaults.yaml`.

**Backend-specific resolution:**

1. If `--latency-model roofline` (default) or `trained-physics`:
   - Auto-resolve model config: check `model_configs/` for existing `config.json`, fetch from HuggingFace on miss (set `HF_TOKEN` for gated models)
   - Auto-resolve hardware config from bundled `hardware_config.json`
   - For roofline: beta coefficients are computed analytically from model architecture and hardware specs
   - For trained-physics: load 13 global coefficients (10 beta for roofline corrections with architecture-aware MoE scaling + 3 alpha for CPU overhead) from `trained_physics_coefficients` in `defaults.yaml`
   - `--model-config-folder` and `--hardware-config` override auto-resolution when explicitly set
2. If `--alpha-coeffs` and `--beta-coeffs` are explicitly provided via CLI:
   - Use them directly, no `defaults.yaml` lookup

**`--total-kv-blocks` resolution** (highest priority wins):

1. **Explicit CLI flag** — if `--total-kv-blocks` is set, that value is used regardless of backend
2. **Auto-calculation** (all backends) — when `MemoryGiB > 0` in the hardware config and `config.json` is available, `CalculateKVBlocks` derives the block count from model architecture and GPU memory. BLIS attempts to resolve `config.json` by checking `--model-config-folder`, then `model_configs/` (cached/bundled), then fetching from HuggingFace (set `HF_TOKEN` for gated models). Failure modes: (a) if `MemoryGiB` is missing from `hardware_config.json`, BLIS warns and falls back to the hardcoded default (layer 3); (b) if model architecture params cannot be extracted from `config.json`, BLIS warns and falls back to the hardcoded default; (c) if the calculation itself fails (e.g., unsupported activation function), BLIS warns and falls back to the hardcoded default. Auto-calculation currently requires SwiGLU-family activations (`silu`, `swiglu`, `geglu`); models with other activations (e.g., Falcon's `gelu`) will fall back to the hardcoded default unless `--total-kv-blocks` is explicitly set
3. **Hardcoded default** — 1,000,000 (CLI flag default, used when auto-calculation is unavailable or fails)

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
| **WorkloadConfig** | `--workload`, `--workload-spec`, `--defaults-filepath`, `--rate`, `--num-requests`, `--prompt-tokens*`, `--output-tokens*`, `--prefix-tokens` |
| **DeploymentConfig** | `--num-instances`, `--admission-policy`, `--admission-latency`, `--token-bucket-capacity`, `--token-bucket-refill-rate`, `--routing-policy`, `--routing-latency`, `--routing-scorers`, `--snapshot-refresh-interval`, `--trace-level`, `--counterfactual-k` | YAML-only (no CLI flag): `node_pools`, `instance_lifecycle`, `hw_config_by_gpu` |
| **Top-level** | `--seed`, `--horizon`, `--log`, `--metrics-path` (run only), `--trace-output`, `--policy-config`, `--fitness-weights`, `--summarize-trace` |

---

## blis observe

Dispatches a workload to a real inference server and records request-level timing into TraceV2 files for later replay and calibration.

### Required

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--server-url` | string | "" | Inference server URL (required). |
| `--model` | string | "" | Model name for API requests (required). |
| `--trace-header` | string | "" | Output path for TraceV2 header YAML (required). |
| `--trace-data` | string | "" | Output path for TraceV2 data CSV (required). |

### Workload Input

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--workload-spec` | string | "" | Path to WorkloadSpec YAML (alternative to `--rate` + distribution flags). |
| `--rate` | float64 | 0 | Requests per second for distribution synthesis. |

### Optional

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--api-key` | string | "" | Bearer token for server authentication. |
| `--server-type` | string | "vllm" | Server type (`vllm`, `tgi`, etc.). |
| `--max-concurrency` | int | 256 | Maximum simultaneous in-flight requests. |
| `--warmup-requests` | int | 0 | Number of initial requests to exclude from trace. |
| `--no-streaming` | bool | false | Disable streaming (use non-streaming HTTP). |
| `--seed` | int64 | 42 | RNG seed for workload generation. |
| `--horizon` | int64 | 0 | Observation horizon in microseconds (0 = from spec or unlimited). |
| `--num-requests` | int | 0 | Maximum requests to generate (0 = from spec or unlimited). |

### Distribution Synthesis

Used when `--rate` is set instead of `--workload-spec`. Same flag names as `blis run` but with different defaults tuned for observe workloads.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prompt-tokens` | int | 512 | Average prompt token count. |
| `--prompt-tokens-stdev` | int | 50 | Prompt token standard deviation. |
| `--prompt-tokens-min` | int | 1 | Minimum prompt tokens. |
| `--prompt-tokens-max` | int | 2048 | Maximum prompt tokens. |
| `--output-tokens` | int | 512 | Average output token count. |
| `--output-tokens-stdev` | int | 50 | Output token standard deviation. |
| `--output-tokens-min` | int | 1 | Minimum output tokens. |
| `--output-tokens-max` | int | 2048 | Maximum output tokens. |
| `--prefix-tokens` | int | 0 | Shared prefix token count. |
| `--api-format` | string | "completions" | API format: `completions` (`/v1/completions`) or `chat` (`/v1/chat/completions`). |
| `--unconstrained-output` | bool | false | Do not set `max_tokens` (let server decide output length). |
| `--rtt-ms` | float64 | 0 | Measured network round-trip time in milliseconds (recorded in trace header for calibrate). |

---

## blis replay

Replays a captured TraceV2 file through the discrete-event simulator. Replay reuses the full simulation engine, so it accepts the same sim-config flags as `blis run` — see the sections above for [Simulation Control](#simulation-control), [KV Cache Configuration](#kv-cache-configuration), [Batch Formation](#batch-formation), [Latency Model](#latency-model), [Cluster Configuration](#cluster-configuration), [Admission Policy](#admission-policy), [Routing Policy](#routing-policy), [Scheduling and Priority](#scheduling-and-priority), [Decision Tracing](#decision-tracing), and [Fitness Evaluation](#fitness-evaluation).

### Replay-Specific Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--trace-header` | string | "" | Path to TraceV2 header YAML file (required). |
| `--trace-data` | string | "" | Path to TraceV2 data CSV file (required). |
| `--results-path` | string | "" | File to write `[]SimResult` JSON (fields: `request_id`, `ttft_us`, `e2e_us`, `input_tokens`, `output_tokens`) for `blis calibrate` consumption. |

---

## blis calibrate

Compares real observed latencies (from `blis observe`) against simulator predictions (from `blis replay`) and produces a calibration report with per-metric MAPE, Pearson R, and quality grades.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--trace-header` | string | "" | Path to TraceV2 header YAML file (from `blis observe`; required). |
| `--trace-data` | string | "" | Path to TraceV2 data CSV file (from `blis observe`; required). |
| `--sim-results` | string | "" | Path to SimResult JSON file (from `blis replay --results-path`; required). |
| `--report` | string | "" | Path to write calibration report JSON (required). |
| `--warmup-requests` | int | -1 | Number of initial requests to exclude. Default: from trace header `warm_up_requests`; pass 0 to include all. |
| `--network-rtt-us` | int64 | -1 | Network RTT in microseconds added to sim-side latencies. Default: from trace header `network.measured_rtt_ms`. |
| `--network-bandwidth-mbps` | float64 | 0 | Network bandwidth in Mbps for upload/download delay calculation (0 = no delay). |

---

## blis convert

Converts external workload formats into BLIS WorkloadSpec v2 YAML. Three subcommands are available.

### `blis convert preset`

Generates a WorkloadSpec from a named preset in `defaults.yaml`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--name` | string | "" | Preset name (e.g., `chatbot`, `summarization`, `contentgen`, `multidoc`). |
| `--rate` | float64 | 1.0 | Request rate in requests/second. |
| `--num-requests` | int | 100 | Number of requests. |
| `--defaults-filepath` | string | "defaults.yaml" | Path to `defaults.yaml`. |

### `blis convert servegen`

Converts a ServeGen data directory into WorkloadSpec format.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--path` | string | "" | Path to ServeGen data directory. |

### `blis convert infperf`

Converts an inference-perf YAML specification into WorkloadSpec format.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--spec` | string | "" | Path to inference-perf YAML spec. |

---

## blis compose

Merges multiple WorkloadSpec v2 YAML files into a single combined specification.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--from` | string (repeatable) | (none) | Path to v2 WorkloadSpec YAML file. Can be repeated to merge multiple specs. |
