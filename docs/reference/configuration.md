# Configuration Reference

This page documents all CLI flags, configuration files, and their interactions. For architectural context on what these settings control, see [Cluster Architecture](../concepts/architecture.md) and [Core Engine](../concepts/core-engine.md).

## Configuration Precedence

BLIS takes its configuration from four sources. They do not overlap: each setting has exactly one home.

```
Scenario file (--scenario in --scenarios)  — the deployment: model, hardware, fabric, pools,
                                             parallelism, engine settings, coefficient sets
CLI flags                                  — workload, policies, topology, seed/horizon/output
YAML files (policy-config, workload-spec,  — CLI flags override these values when explicitly set
            kv-offload-config, lora-config, defaults.yaml)
Hardcoded defaults (lowest priority)
```

CLI flags only override YAML values when explicitly set. BLIS checks whether each flag was provided by the user (not just whether it has a non-default value), so default flag values do not accidentally override YAML configuration. No CLI flag overrides a scenario setting.

### Parameter Resolution by Category

**Deployment and engine settings** — from the scenario only. See [Scenario and Deployment](#scenario-and-deployment) below for the full list. There is no CLI override and no per-model fallback.

**Latency** — priced by [blis-latency-kernel](../guide/latency-models.md), the only latency backend. The scenario names the coefficient sets; they are read from the [blis-registry](https://github.com/inference-sim/blis-registry) clone named by `--registry`. There are no coefficient flags.

**KV cache blocks** — computed by the kernel's memory methods from the scenario (model, hardware, parallelism, `gpu_memory_utilization`, `cache_dtype`, `block_size`), per data-parallel rank. There is no flag to set it.

**Workload parameters** (`--rate`, `--num-requests`, `--prompt-tokens`, etc.):

1. `--workload-spec` YAML file — when set, all token distribution and arrival parameters come from the YAML; CLI distribution flags are ignored
2. CLI distribution flags — when `--workload distribution` (default) and no `--workload-spec`
3. Named preset from the catalog (`<catalog>/workloads/<name>.yaml`, #1769) — when `--workload <name>` (e.g., `chatbot`)
4. Hardcoded CLI flag defaults — (e.g., `--prompt-tokens 512`, `--output-tokens 512`)

!!! note
    `--seed`, `--horizon`, and `--num-requests` are exceptions — they override the workload-spec YAML values even when `--workload-spec` is set. `--rate` does NOT override `aggregate_rate` in the YAML (see [Common Pitfalls](#common-pitfalls)).

**Routing, admission, scheduling, and preemption** (`--routing-policy`, `--admission-policy`, `--scheduler`, `--preemption-policy`, etc.):

1. Explicit CLI flags
2. `--policy-config` YAML bundle — loads all policy settings from one file
3. Hardcoded defaults — `round-robin`, `always-admit`, `fcfs`

**Batch formation** — `max_num_seqs` and `max_num_batched_tokens` come from the scenario's pool `engine` block. Only `--long-prefill-token-threshold` and `--preemption-policy` remain on the CLI.

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

### Common Pitfalls

**Capacity estimate mismatch (issue #390).** CLI distribution mode defaults to `--prompt-tokens 512, --output-tokens 512`. If you estimate per-instance capacity using CLI mode, then run a workload-spec YAML with shorter sequences (e.g., mean 256/128), the YAML workload will achieve ~1.5x higher throughput than the CLI estimate predicted. Always derive capacity estimates from the actual workload you plan to run, not from CLI defaults.

**`--rate` does NOT override workload-spec YAML.** The `--rate` flag only applies in CLI distribution mode. When `--workload-spec` is set, request rate comes from `aggregate_rate` in the YAML file — the `--rate` flag is ignored. To change the rate for a YAML workload, edit the `aggregate_rate` field in the spec.

**`aggregate_rate` override for inference-perf specs.** When converting inference-perf specs via `blis convert inference-perf`, per-stage rates in the spec override a user-specified `aggregate_rate`. If the sum of stage rates differs from `aggregate_rate`, BLIS logs a warning and uses the stage-rate sum. This prevents silent rate scaling errors.

**`enable_multi_turn_chat` semantic mismatch (issue #517).** inference-perf's `enable_multi_turn_chat` creates one persistent session per virtual user. BLIS's closest equivalent is `multi_turn.single_session: true` in the workload YAML, but the session mechanics differ. When converting inference-perf specs, verify that the converted multi-turn behavior matches your intent.

## Deployment Inputs (required)

`blis run` and `blis replay` both require all four of these. Omitting any is refused.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--scenario` | string | "" | **Required.** Scenario **file name** within `--scenarios`, e.g. `llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml`. The file is a [blis-schemas](https://github.com/inference-sim/blis-schemas) Scenario + Deployment (two YAML documents) — see [Scenario and Deployment](#scenario-and-deployment). |
| `--scenarios` | string | "" | **Required.** Directory holding scenario YAML files. Any directory works. The kernel module's own fixtures are at `$(go env GOMODCACHE)/github.com/inference-sim/blis-latency-kernel@v0.1.0/testdata/aisimulate`; this repository's `testdata/scenarios/` holds a P/D example (`glm-5-h200-3p1d-ib.yaml`) and an MTP example (`glm-5-h200-tp8-mtp3.yaml`). |
| `--registry` | string | "" | **Required.** [blis-registry](https://github.com/inference-sim/blis-registry) clone root, holding the fitted coefficient sets the scenario names. Pinned release: `v0.1.1`. |
| `--catalog` | string | "" | Path to the [blis-catalog](https://github.com/inference-sim/blis-catalog) **clone root** (pinned release `0.2.1`; see [Catalog compatibility](../getting-started/installation.md#catalog-compatibility)) — model graphs, chips, fabrics, storage devices and workload presets. **No default and no search path** — supply this flag or the `BLIS_CATALOG` environment variable, or the run is refused naming both (#1731). `--catalog` wins when both are set, and the override is announced on stderr. A relative value is resolved against the process working directory, an absolute value is used as given. Also registered on `convert preset` (and the deprecated `observe`) for the workload presets in the `workloads/` namespace. BLIS never fetches or writes a catalog file at run time (NS-6). |

```bash
git clone --branch 0.2.1 --depth 1 https://github.com/inference-sim/blis-catalog.git
git clone --branch v0.1.1 --depth 1 https://github.com/inference-sim/blis-registry.git
export BLIS_CATALOG=$PWD/blis-catalog
./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios <dir of scenario files> --registry $PWD/blis-registry \
  --rate 10 --num-requests 100
```

### Scenario and Deployment

A scenario file holds two YAML documents in the blis-schemas format:

- **Scenario** — `name`, `engine_version`, `model` (a catalog model), `coefficients` (the registry coefficient sets that price it), and `cluster` (`hardware`, `fabric`, `nodes`, `gpus_per_node`).
- **Deployment** — `pools`, each with a `role` (`colocated`, `prefill` or `decode`), `nodes`, a `parallel` block (`tp`, `pp`, `dp`, `enable_expert_parallel`) and an `engine` block (`block_size`, `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`, `quantization`, `cache_dtype`, `cudagraph_mode`, `gpu_memory_utilization`, `enable_prefix_caching`, `speculative: {method, num_spec_tokens}`); plus optional `offload` and `pd_transfer`.

These settings come from the scenario (and the kernel), and no CLI flag sets them:

| Setting | Source |
|---------|--------|
| Model | Scenario `model` |
| Hardware, fabric | Scenario `cluster.hardware`, `cluster.fabric` |
| TP / DP / expert parallelism | Pool `parallel.tp`, `parallel.dp`, `parallel.enable_expert_parallel` |
| KV block budget | The kernel's memory methods, per data-parallel rank |
| Block size | Pool `engine.block_size` |
| Max running requests | Pool `engine.max_num_seqs` |
| Token budget per step | Pool `engine.max_num_batched_tokens` |
| Max sequence length | Pool `engine.max_model_len` — **required** in the scenario |
| Prefix caching | Pool `engine.enable_prefix_caching` |
| KV-cache dtype | Pool `engine.cache_dtype` |
| Speculative decoding | Pool `engine.speculative.method` / `num_spec_tokens` (`--speculative-acceptance-rate` is still required on the CLI when the scenario drafts tokens) |
| Step time, memory, P/D KV transfer, offload tier transfer, host overheads | Priced by blis-latency-kernel with the scenario's registry coefficients |

**Data parallelism (MoE only).** A pool with `dp > 1` becomes one replica per data-parallel rank, each sized as one vLLM EngineCore. `dp > 1` on a dense model is refused.

**Node pools.** When `node_pools` are configured in `--policy-config`, every pool's `gpu_type` must equal the scenario's `cluster.hardware`.

## Simulation Control

Top-level settings that control the simulation run.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--seed` | int64 | 42 | Random seed for deterministic simulation. Same seed produces byte-identical stdout. |
| `--horizon` | int64 | MaxInt64 | Simulation time limit in ticks (microseconds). Simulation stops when clock exceeds horizon or all requests complete. |
| `--log` | string | "warn" | Log verbosity: trace, debug, info, warn, error, fatal, panic. Logs go to stderr. |
| `--metrics-path` | string | "" | File path to write MetricsOutput JSON (aggregate P50/P95/P99 TTFT, E2E, throughput stats, plus the `cache_hit_rate` and `catalog` provenance fields when those apply). Accepted on **both** `blis run` and `blis replay` (#1583). Distinct from replay's `--results-path`, which writes the **per-request** `[]SimResult` array and is replay-only. Empty = no file output. |

## KV Cache Configuration

The GPU-tier block budget and block size come from the scenario and the kernel (see [Scenario and Deployment](#scenario-and-deployment)). The remaining flags configure CPU / storage offload tiers. Maps to `KVCacheConfig`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--kv-cpu-blocks` | int64 | 0 | Legacy CPU-tier blocks. 0 disables tiered caching. The GPU↔CPU transfer is priced by the kernel from the catalog's `cpu_dram` storage device. |
| `--kv-offload-threshold` | float64 | 0.9 | GPU utilization fraction above which blocks are offloaded to CPU on the legacy `--kv-cpu-blocks` path. Range [0, 1]. |
| `--kv-offload-config` | string | "" | Path to a YAML file with a top-level `kv_offload:` block (multi-tier offload). Each tier must name a catalog storage `device_class`; the kernel prices every tier transfer (its `TierTime`). An explicit `read_bandwidth` / `write_bandwidth` / `base_latency` on a tier is refused. KV offload combined with P/D disaggregation is refused. See [KV Cache Management](../guide/kv-cache.md). |

## Batch Formation

`max_num_seqs` and `max_num_batched_tokens` come from the scenario's pool `engine` block. Maps to `BatchConfig`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--long-prefill-token-threshold` | int64 | 0 | Prefill length threshold for chunked prefill. 0 = disabled (all prefill in one step). |
| `--preemption-policy` | string | "fcfs" | Preemption victim selection: `fcfs` (tail-of-batch, default) or `priority` (least-urgent SLO tier evicted first, matching vLLM `--scheduling-policy priority`). Priority mode uses `slo_priorities` from the policy bundle when set (shared with admission). |

## Latency Model

BLIS prices every step with [blis-latency-kernel](https://github.com/inference-sim/blis-latency-kernel) (`v0.1.0`, adapter `sim/kernelmodel`). There is no backend selector and no coefficient flag: the scenario names the coefficient sets and `--registry` locates them. See [Latency Models](../guide/latency-models.md).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--speculative-acceptance-rate` | float64 | 0.0 | Mean fraction of draft tokens accepted, in [0, 1]. **Required** when the scenario's pool drafts tokens (`engine.speculative.num_spec_tokens > 0`). The draft length and method come from the scenario. |

## Cluster Configuration

With `--num-instances 1` (the default), BLIS runs a single-instance simulation — requests go directly to the wait queue with no admission or routing layer. With `--num-instances N` (N > 1), the cluster simulation activates: requests pass through the admission and routing pipeline before reaching per-instance wait queues. See [Cluster Architecture](../concepts/architecture.md) for the multi-instance pipeline and [Core Engine](../concepts/core-engine.md) for single-instance internals.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num-instances` | int | 1 | Number of inference instances. 1 = single-instance mode; > 1 = cluster mode with admission and routing. Must not exceed the scenario's rank capacity (pool `nodes` × `gpus_per_node` / (pp × tp × pcp)); `testdata/scenarios/llama-3.1-70b-instruct-h200-tp4-4node.yaml` allows up to 8. |
| `--prefill-instances` | int | 0 | Prefill instances for P/D disaggregation (0 = disabled). Requires a scenario with `prefill` and `decode` pools, and must not exceed the prefill pool's rank capacity (pool `nodes` × `gpus_per_node` / (pp × tp × pcp)). |
| `--decode-instances` | int | 0 | Decode instances for P/D disaggregation. Same rules, against the decode pool. |
| `--pd-decider` | string | "never" | P/D disaggregation decider: `never`, `always`, `prefix-threshold`. |

**P/D disaggregation.** A scenario with `prefill` and `decode` pools runs one kernel per pool; the KV handoff is priced by the kernel's `PDTransferTime` over the scenario's fabric between the two instances' placements. Refused: P/D topology over a colocated scenario; a disaggregated scenario without `--prefill-instances`/`--decode-instances`; pools that differ in block size, `dp` or draft configuration; KV offload combined with P/D; shared (`--prefill-decode-instances`) or encode instances.

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

Each SLO class has an integer priority that determines admission ordering, shedding decisions, gateway queue dispatch, and (with `--preemption-policy priority`) preemption victim selection. Priorities follow the GAIE (Gateway API Inference Extension) convention where **negative priority = sheddable**.

`slo_priorities` overrides affect both admission (tier-shed, GAIE-legacy) and preemption (`--preemption-policy priority`). Both subsystems share the same priority mapping.

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
| Gateway queue dispatch | `sim/cluster/gateway_queue.go` | Priority-ordered or SLO-deadline dispatch: higher priority dequeued first (priority mode), earliest SLO deadline within flow (slo-deadline mode); capacity shedding evicts lowest priority |
| Backward compatibility | `sim/admission.go` | `SLOTierPriority()` delegates to `DefaultSLOPriorityMap().Priority()` |

**Per-tenant fair-share budgets** (`tenant_budgets`): A secondary admission layer that runs *after* the admission policy. If the admission policy rejects a request, tenant budgets are not consulted. If the admission policy admits a request, tenant budgets then apply: over-budget tenants have sheddable requests (`IsSheddable(class) = priority < 0`) preferentially shed, while non-sheddable traffic (critical, standard) is always protected. Configured via `--policy-config` YAML only (no CLI flag):

| YAML field | Type | Default | Description |
|------------|------|---------|-------------|
| `tenant_budgets` | map[string]float64 | nil | Per-tenant fraction of total cluster capacity (NumInstances × MaxNumSeqs). Absent key = unlimited. 0.0 = effectively zero concurrent slots (one request may slip through per admission tick due to DES admission-before-routing event ordering; see IsOverBudget docstring). Values must be in [0, 1]. |

Example:

```yaml
admission:
  policy: "tier-shed"
  tier_shed_threshold: 0
  tier_shed_min_priority: 3  # admit standard(3) and critical(4); shed sheddable tiers (priority < 0)
  slo_priorities:             # optional: override specific priorities
    batch: 0                  # promote batch to non-sheddable
  slo_targets:                # optional: per-class TTFT targets in µs for slo-deadline ordering
    critical: 100000          # 100ms TTFT target
    standard: 500000          # 500ms TTFT target

tenant_budgets:
  alice: 0.3   # alice may use at most 30% of total cluster capacity
  bob: 0.7     # bob may use at most 70% of total cluster capacity
```

## Flow Control (Gateway Queue)

When `--flow-control` is enabled, admission IS the queue — requests are enqueued into per-priority-band, per-tenant flow queues and dispatched under saturation gating. See [Admission: Flow Control](../guide/admission.md#flow-control-mode).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--flow-control` | bool | false | Enable flow-control admission (replaces legacy admission) |
| `--saturation-detector` | string | "never" | Saturation detection: `utilization`, `concurrency`, `never` |
| `--queue-depth-threshold` | int | 5 | Queue depth threshold for utilization-based saturation |
| `--kv-cache-util-threshold` | float64 | 0.8 | KV cache utilization threshold for saturation |
| `--max-concurrency` | int | 100 | Max in-flight requests for concurrency-based saturation |
| `--dispatch-order` | string | "fifo" | Cross-band dispatch: `fifo` (globally-earliest), `priority` (highest band first), `slo-deadline` (earliest SLO deadline within flow) |
| `--slo-targets` | string | "" | Per-SLO-class TTFT targets in µs for slo-deadline ordering (e.g., `critical=100000,standard=500000`) |
| `--fairness-policy` | string | "global-strict" | Intra-band flow selection: `global-strict` (earliest seqID), `round-robin` (tenant cycling) |
| `--per-band-capacity` | int | 0 | Max requests per priority band (0=unlimited) |
| `--max-gateway-queue-depth` | int | 0 | Global queue depth limit (0=unlimited) |

## Routing Policy

Controls how admitted requests are assigned to instances. See [Cluster Architecture: Routing](../concepts/architecture.md#routing-pipeline).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--routing-policy` | string | "round-robin" | Policy name: `round-robin`, `least-loaded`, `weighted`, `always-busiest`. |
| `--routing-latency` | int64 | 0 | Routing decision latency in microseconds. Must be >= 0. |
| `--routing-scorers` | string | "" | Scorer configuration for `weighted` policy. Format: `name:weight,name:weight,...` |
| `--snapshot-refresh-interval` | int64 | 50000 | Prometheus snapshot refresh interval for all instance metrics (QueueDepth, BatchSize, KVUtilization, PreemptionCount) in microseconds. Default 50ms = llm-d parity. 0 = immediate/oracle mode. |

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
| `--preemption-policy` | string | "fcfs" | Preemption victim selection: `fcfs` (tail-of-batch, default) or `priority` (least-urgent SLO tier evicted first, matching vLLM `--scheduling-policy priority`). Priority mode evicts the running request with the highest `Request.Priority` value (vLLM convention: background=7 is evicted first). |

See [Core Engine: Scheduling](../concepts/core-engine.md#scheduling-policies) for policy details.

## Workload Configuration

### Workload Modes

BLIS supports three workload specification modes, in order of precedence:

| Mode | Trigger | Description |
|------|---------|-------------|
| **Workload-spec YAML** | `--workload-spec <path>` | Multi-client workload with per-client distributions. Highest priority. |
| **CLI distribution** | `--workload distribution` (default) | Single-client Gaussian distribution controlled by CLI flags. |
| **Preset** | `--workload <name>` | Named preset read from the catalog (`<catalog>/workloads/<name>.yaml`, #1769): `chatbot`, `contentgen`, `summarization`, `multidoc`. Needs `--catalog` / `BLIS_CATALOG`, which `blis run` already requires. |

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
| `--defaults-filepath` | string | "defaults.yaml" | Path to `defaults.yaml` (the LoRA cost block; see [defaults.yaml](#defaultsyaml)). |
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

# Node pool infrastructure (optional; omit for single-pool mode)
# Every pool's gpu_type must equal the scenario's cluster.hardware, and gpu_type must be
# unique across pools (#1537). See docs/guide/cluster.md.
node_pools:
  - name: "gpu-pool-1"
    gpu_type: "h200"      # must equal the scenario's cluster.hardware
    gpus_per_node: 8
    gpu_memory_gib: 141.0
    initial_nodes: 2
    min_nodes: 1
    max_nodes: 4
    provisioning_delay:
      mean: 30.0   # seconds
      stddev: 5.0  # 0 = constant delay

# Instance lifecycle (Phase 1A — all zero/empty = backward-compatible defaults)
instance_lifecycle:
  loading_delay:
    mean: 10.0    # seconds to load model weights onto GPU
    stddev: 1.0   # 0 = constant delay
  warm_up_request_count: 5    # requests served before leaving WarmingUp state
  warm_up_ttft_factor: 2.0    # TTFT multiplier applied to warm-up requests (≥ 1.0)
  drain_policy: "WAIT"        # IMMEDIATE | WAIT | REDIRECT
  warm_start_initial_instances: false  # true = startup instances skip loading_delay (model pre-deployed); autoscaler-added instances always pay loading_delay

# SLO priority overrides (optional; omit for GAIE defaults)
# GAIE defaults: critical=4, standard=3, batch=-1, sheddable=-2, background=-3
# Negative priority = sheddable. Override to change which classes are sheddable.
# admission:
#   slo_priorities:
#     batch: 0    # make batch non-sheddable

# Per-tenant fair-share budgets (Phase 1B — optional; omit for no tenant enforcement)
# Each value is a fraction of total cluster capacity (NumInstances × MaxNumSeqs).
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

The `defaults.yaml` file (located by `--defaults-filepath`, default `defaults.yaml`) carries only the LoRA control-plane cost terms and a `version`. It holds no latency coefficients, no per-model deployment policy and no workload presets: latency coefficients live in [blis-registry](https://github.com/inference-sim/blis-registry), named by the scenario; the deployment lives in the scenario; workload presets live in the catalog (`<catalog>/workloads/<name>.yaml`, #1769).

These are the top-level keys the file may carry, and strict parsing accepts no others (`KnownFields(true)`, R10 — the authoritative list is `cmd.Config` in `cmd/default_config.go`; the bundled `defaults.yaml` is the worked example):

```yaml
version: 0.0.1

# LoRA control-plane cost terms (#1464). Inert unless a run declares adapters.
lora:
  load_base_latency_us: 1500.0
  # ... bandwidth, per-rank footprint, per-rank step-overhead tiers
```

!!! note "Node pools and the scenario's hardware"
    When `node_pools` are configured (via `--policy-config`), every pool's `gpu_type` must equal the scenario's `cluster.hardware`. KV capacity and step time for every placed instance come from the kernel for the scenario's hardware. `gpu_type` must also be unique across pools (#1537).

---

## CLI Flag Summary by Sub-Config

| Sub-Config | Flags |
|------------|-------|
| **Deployment (required)** | `--scenario`, `--scenarios`, `--registry`, `--catalog` (or `BLIS_CATALOG`) |
| **KVCacheConfig** | `--kv-cpu-blocks`, `--kv-offload-threshold`, `--kv-offload-config` (block budget and block size: scenario + kernel) |
| **BatchConfig** | `--long-prefill-token-threshold` (`max_num_seqs`, `max_num_batched_tokens`: scenario) |
| **Latency** | `--speculative-acceptance-rate` (everything else: scenario + registry, priced by blis-latency-kernel) |
| **PolicyConfig** | `--scheduler`, `--preemption-policy` |
| **WorkloadConfig** | `--workload` (preset read from `<catalog>/workloads/<name>.yaml`, #1769), `--workload-spec`, `--rate`, `--concurrency`, `--num-requests`, `--prompt-tokens*`, `--output-tokens*`, `--prefix-tokens` |
| **DeploymentConfig** | `--num-instances`, `--prefill-instances`, `--decode-instances`, `--pd-decider`, `--admission-policy`, `--admission-latency`, `--token-bucket-capacity`, `--token-bucket-refill-rate`, `--routing-policy`, `--routing-latency`, `--routing-scorers`, `--snapshot-refresh-interval`, `--trace-level`, `--counterfactual-k`. YAML-only (no CLI flag): `node_pools`, `instance_lifecycle` |
| **LoRA** | `--lora-config`, `--lora-adapter-capacity`, `--lora-*` cost overrides, `--defaults-filepath` (the `lora:` block) |
| **Top-level** | `--seed`, `--horizon`, `--log`, `--metrics-path` (`run` and `replay`), `--trace-output`, `--policy-config`, `--fitness-weights`, `--summarize-trace` |

---

## blis observe

!!! warning "Deprecated"
    `blis observe` is deprecated ([#1901](https://github.com/inference-sim/inference-sim/issues/1901)).

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

Used when `--rate` is set instead of `--workload-spec`. Same flag names and defaults as `blis run`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prompt-tokens` | int | 512 | Average prompt token count. |
| `--prompt-tokens-stdev` | int | 256 | Prompt token standard deviation. |
| `--prompt-tokens-min` | int | 2 | Minimum prompt tokens. |
| `--prompt-tokens-max` | int | 7000 | Maximum prompt tokens. |
| `--output-tokens` | int | 512 | Average output token count. |
| `--output-tokens-stdev` | int | 256 | Output token standard deviation. |
| `--output-tokens-min` | int | 2 | Minimum output tokens. |
| `--output-tokens-max` | int | 7000 | Maximum output tokens. |
| `--prefix-tokens` | int | 0 | Shared prefix token count. |
| `--api-format` | string | "completions" | API format: `completions` (`/v1/completions`) or `chat` (`/v1/chat/completions`). |
| `--unconstrained-output` | bool | false | Do not set `max_tokens` (let server decide output length). |
| `--rtt-ms` | float64 | 0 | Measured network round-trip time in milliseconds (recorded in trace header for calibrate). |

---

## blis replay

Replays a captured TraceV2 file (any TraceV2, including one from `blis convert`) through the discrete-event simulator. Replay runs on the kernel through the same deployment path as `blis run`, so it requires the same [Deployment Inputs](#deployment-inputs-required) and accepts the same sim-config flags — see [Simulation Control](#simulation-control), [KV Cache Configuration](#kv-cache-configuration), [Batch Formation](#batch-formation), [Latency Model](#latency-model), [Cluster Configuration](#cluster-configuration), [Admission Policy](#admission-policy), [Routing Policy](#routing-policy), [Scheduling and Priority](#scheduling-and-priority), [Decision Tracing](#decision-tracing), and [Fitness Evaluation](#fitness-evaluation). A run's exported trace replayed with identical flags (including `--horizon`) yields byte-identical stdout (INV-13).

### Replay-Specific Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--trace-header` | string | "" | Path to TraceV2 header YAML file (required). |
| `--trace-data` | string | "" | Path to TraceV2 data CSV file (required). |
| `--results-path` | string | "" | File to write `[]SimResult` JSON (fields: `request_id`, `ttft_us`, `e2e_us`, `input_tokens`, `output_tokens`) for `blis calibrate` consumption. Replay-only — `blis run` does not register it. |

`blis replay` also accepts `--metrics-path` (the aggregate `MetricsOutput` JSON, documented under
[Simulation Control](#simulation-control)); it is not replay-specific, so it is not repeated in the
table above. The two are complementary rather than alternatives: `--results-path` writes per-request
rows, `--metrics-path` writes the run aggregate that `blis calibrate --sim-metrics` reads.

---

## blis calibrate

!!! warning "Deprecated"
    `blis calibrate` is deprecated ([#1901](https://github.com/inference-sim/inference-sim/issues/1901)).

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

Generates a WorkloadSpec from a named preset in the catalog
(`<catalog>/workloads/<name>.yaml`, #1769 — the same definition `blis run --workload` and
`blis observe --workload` read).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--name` | string | "" | Preset name (e.g., `chatbot`, `summarization`, `contentgen`, `multidoc`). |
| `--rate` | float64 | 1.0 | Request rate in requests/second. |
| `--num-requests` | int | 100 | Number of requests. |
| `--catalog` | string | "" | Catalog clone root holding `workloads/<name>.yaml`. No default; `BLIS_CATALOG` is the fallback (the flag wins when both are set). |

### `blis convert servegen`

Converts a ServeGen data directory into WorkloadSpec format.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--path` | string | "" | Path to ServeGen data directory. |

### `blis convert inference-perf`

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
