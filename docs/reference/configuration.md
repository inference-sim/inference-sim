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

**Scheduling and preemption** (`--scheduler`, `--preemption-policy`):

1. Explicit CLI flags
2. `--policy-config` YAML bundle
3. The scenario's engine settings, then the default `fcfs`

**Routing, admission and flow control** are the embedded llm-d router's, configured by the scenario's `router:` block and pool labels ([Router](#router)). See [Routing](../guide/routing.md).

**Batch formation** — `max_num_seqs` and `max_num_batched_tokens` come from the scenario's pool `engine` block. `long_prefill_token_threshold` comes from the same block, per pool (blis-schemas#75): each P/D pool runs its own. It has no flag: an engine setting is the scenario's (inference-sim#2021 B6). `--preemption-policy` remains on the CLI.

### Known Unit Gotchas

All internal timestamps in the DES (arrival time, schedule time, completion time, clock) use **ticks**, where **1 tick = 1 microsecond (μs)**. Output metrics convert to milliseconds for human readability, but several fields and flags use different units:

| Field / Flag | Unit | Notes |
|-------------|------|-------|
| `ttft_ms`, `e2e_ms`, `itl_ms` (per-request JSON) | milliseconds | Converted from ticks by dividing by 1,000 |
| `scheduling_delay_ms` (per-request JSON) | milliseconds | Converted from ticks by dividing by 1,000. Historically was in ticks (μs) despite the `_ms` suffix — fixed by BC-14. Old hypothesis scripts (pre-fix) divide by 1,000 again unnecessarily. |
| `scheduling_delay_p99_ms` (aggregate) | milliseconds | Always was in milliseconds |
| `--horizon` | ticks (μs) | Simulation time limit. 1,000,000 = 1 second |
| `--gateway-latency`, `--epp-scrape-jitter`, `--kv-event-delay` | ticks (μs) | Gateway hop before the router; scrape phase-offset bound; KV-event delivery delay |
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
| `--scenarios` | string | "" | **Required.** Directory holding scenario YAML files. Any directory works. The kernel module's own fixtures are at `$(go env GOMODCACHE)/github.com/inference-sim/blis-latency-kernel@v0.3.0/testdata/aisimulate`; this repository's `testdata/scenarios/` holds a P/D example (`glm-5-h200-3p1d-ib.yaml`), an MTP example (`glm-5-h200-tp8-mtp3.yaml`) and a DeepSeek-V4.1-Flash DSpark example (`deepseek-v4.1-flash-h100-tp8-dspark5.yaml`). |
| `--registry` | string | "" | **Required.** [blis-registry](https://github.com/inference-sim/blis-registry) clone root, holding the fitted coefficient sets the scenario names. Pinned release: `v0.2.0`. |
| `--catalog` | string | "" | Path to the [blis-catalog](https://github.com/inference-sim/blis-catalog) **clone root** (pinned release `0.3.0`; see [Catalog compatibility](../getting-started/installation.md#catalog-compatibility)) — model graphs, chips, fabrics, storage devices and workload presets. **No default and no search path** — supply this flag or the `BLIS_CATALOG` environment variable, or the run is refused naming both (#1731). `--catalog` wins when both are set, and the override is announced on stderr. A relative value is resolved against the process working directory, an absolute value is used as given. Also registered on `convert preset` (and the deprecated `observe`) for the workload presets in the `workloads/` namespace. BLIS never fetches or writes a catalog file at run time (NS-6). |

```bash
git clone --branch 0.3.0 --depth 1 https://github.com/inference-sim/blis-catalog.git
git clone --branch v0.2.0 --depth 1 https://github.com/inference-sim/blis-registry.git
export BLIS_CATALOG=$PWD/blis-catalog
./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios <dir of scenario files> --registry $PWD/blis-registry \
  --rate 10 --num-requests 100
```

### Scenario and Deployment

A scenario file holds two YAML documents in the blis-schemas format:

- **Scenario** — `name`, `engine_version`, `model` (a catalog model), `coefficients` (the registry coefficient sets that price it), and `cluster` (`hardware`, `fabric`, `nodes`, `gpus_per_node`). `engine_version` is required: `0.32.0` runs blis-vllm's translated vLLM v0.32.0 engine; any other is refused (see [latency models](../guide/latency-models.md)).
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
| Max sequence length | Pool `engine.max_model_len` — **required** in the scenario, as a count (`auto` is refused) |
| Prefix caching | Pool `engine.enable_prefix_caching` |
| KV-cache dtype | Pool `engine.cache_dtype` |
| Speculative decoding | Pool `engine.speculative.method` / `num_spec_tokens` (acceptance is resolved from the registry's profiles, the workload, the scenario's `acceptance` pin or the CLI; see [Latency Models](../guide/latency-models.md#speculative-acceptance-1939)) |
| Step time, memory, P/D KV transfer, offload tier transfer, host overheads | Priced by blis-latency-kernel with the scenario's registry coefficients |

**Data parallelism (MoE only).** A pool with `dp > 1` becomes one engine per data-parallel rank, each sized as one vLLM EngineCore. The ranks of one instance step in lockstep and the router sees them as one endpoint, balanced across ranks by vLLM's internal DP load balancer; `parallel.data_parallel_load_balancing: external` (blis-schemas#76, per pool; `--dp-load-balancing` is a run-wide override that must agree with every mode a pool states; under P/D every rank is already its own endpoint, so a P/D pool's mode is resolved but changes no routing) instead makes every rank its own endpoint, as vLLM's external-LB modes do ([Latency Models](../guide/latency-models.md#data-and-expert-parallelism-moe), #1909). `dp > 1` on a dense model is refused.

**Node pools.** When `node_pools` are configured in `--policy-config`, every pool's `gpu_type` must equal the scenario's `cluster.hardware`.

## Simulation Control

Top-level settings that control the simulation run.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--seed` | int64 | 42 | Random seed for deterministic simulation. Same seed produces byte-identical stdout. |
| `--horizon` | int64 | MaxInt64 | Simulation time limit in ticks (microseconds). Simulation stops when clock exceeds horizon or all requests complete. |
| `--log` | string | "warn" | Log verbosity: trace, debug, info, warn, error, fatal, panic. Logs go to stderr. |
| `--metrics-path` | string | "" | File path to write MetricsOutput JSON (aggregate P50/P95/P99 TTFT, E2E, throughput stats, plus the file-only `cache_hit_rate`, `catalog`, measured-window aggregates (#1938), `kernel_provenance` (#1942) and build-version fields; see the results guide). Accepted on **both** `blis run` and `blis replay` (#1583). Distinct from replay's `--results-path`, which writes the **per-request** `[]SimResult` array and is replay-only. Empty = no file output. |

## KV Cache Configuration

The GPU-tier block budget and block size come from the scenario and the kernel (see [Scenario and Deployment](#scenario-and-deployment)). The remaining flags configure CPU / storage offload tiers. Maps to `KVCacheConfig`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--kv-cpu-blocks` | int64 | 0 | Legacy CPU-tier blocks. 0 disables tiered caching. The GPU↔CPU transfer is priced by the kernel from the catalog's `cpu_dram` storage device. |
| `--kv-offload-threshold` | float64 | 0.9 | GPU utilization fraction above which blocks are offloaded to CPU on the legacy `--kv-cpu-blocks` path. Range [0, 1]. |
| `--kv-offload-config` | string | "" | Path to a YAML file with a top-level `kv_offload:` block (multi-tier offload). Each tier must name a catalog storage `device_class`; the kernel prices every tier transfer (its `TierTime`). An explicit `read_bandwidth` / `write_bandwidth` / `base_latency` on a tier is refused. Under P/D every pool gets its own tier of this size (a MultiConnector of NIXL and the OffloadingConnector). When the scenario states an `offload` block, that block is the configuration and this flag may only restate it: a file that resolves differently is refused (#1910). See [KV Cache Management](../guide/kv-cache.md). |

## Batch Formation

`max_num_seqs` and `max_num_batched_tokens` come from the scenario's pool `engine` block. Maps to `BatchConfig`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--renderer-num-workers` | int | 1 | Render threads per API server (vLLM `renderer_num_workers`). Each request's admission overhead queues for one, in arrival order (INV-R1). 0 = uncontended per-request delay. |
| `--api-server-count` | int | 0 | API servers per endpoint (vLLM `--api-server-count`). 0 = vLLM's default: dp behind an internal or hybrid DP replica, else 1. |
| `--preemption-policy` | string | "fcfs" | Preemption victim selection: `fcfs` (tail-of-batch, default) or `priority` (least-urgent SLO tier evicted first, matching vLLM `--scheduling-policy priority`). Priority mode uses `priority.slo_priorities` from the policy bundle when set. |

## Latency Model

BLIS prices every step with [blis-latency-kernel](https://github.com/inference-sim/blis-latency-kernel) (`v0.3.0`, adapter blis-vllm `compat/kernelmodel`). There is no backend selector and no coefficient flag: the scenario names the coefficient sets and `--registry` locates them. See [Latency Models](../guide/latency-models.md).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--speculative-acceptance-rate` | float64 | 0.0 | The accepted fraction of drafts, (AL − 1)/K, in [0, 1]. The highest acceptance layer (#1939); not with `--speculative-acceptance-length`. |
| `--speculative-acceptance-length` | float64 | 0.0 | The mean acceptance length AL (tokens per verification step, counting the bonus token), in [1, K + 1]. The highest acceptance layer; not with `--speculative-acceptance-rate`. |
| `--speculative-thinking` | string | | `on` or `off`: with `--speculative-category`, selects the registry's acceptance profile, overriding the workload's selectors. |
| `--speculative-category` | string | | The workload category (e.g. `coding`) that selects the registry's acceptance profile. |
| `--speculative-fallback-category` | string | | A category to use, recorded as a fallback, when the workload's has no profile. Without it an unmatched category is refused. A scenario pin's own `fallback_category` wins over it (blis-schemas `acceptance.Resolve`). |

## Cluster Configuration

Every run, `--num-instances 1` (the default) included, passes each request through the embedded llm-d router (admission, flow control, endpoint picking) before it reaches a per-instance wait queue. With one instance the router can only pick that instance, and with the default configuration it admits every request on arrival, so a single-instance run reports what a run without a router would (see [Routing](../guide/routing.md)). See [Cluster Architecture](../concepts/architecture.md) for the multi-instance pipeline and [Core Engine](../concepts/core-engine.md) for single-instance internals.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--num-instances` | int | 1 | Number of inference instances. 1 = single-instance mode; > 1 = cluster mode behind the llm-d router. Must not exceed the scenario's rank capacity (pool `nodes` × `gpus_per_node` / (pp × tp × pcp)); `testdata/scenarios/llama-3.1-70b-instruct-h200-tp4-4node.yaml` allows up to 8. |
| `--prefill-instances` | int | 0 | Prefill instances for P/D disaggregation (0 = disabled). Requires a scenario with `prefill` and `decode` pools, and must not exceed the prefill pool's rank capacity (pool `nodes` × `gpus_per_node` / (pp × tp × pcp)). |
| `--decode-instances` | int | 0 | Decode instances for P/D disaggregation. Same rules, against the decode pool. |
| `--pd-transfer-contention` | bool | false | Concurrent P/D KV handoffs share the fabric links they cross: each link is divided max-min fairly among the handoffs on it, from the kernel's `PDTransferDemand`. A handoff alone still takes exactly `PDTransferTime`. Run and replay alike (INV-13). |

**P/D disaggregation.** Whether a request is disaggregated is the router's decision: the EndpointPickerConfig needs a `disagg-profile-handler` with a P/D decider (`always-disagg-pd-decider` or `prefix-based-pd-decider`); prefill and decode pods carry `llm-d.ai/role` labels automatically. See [Routing](../guide/routing.md). A scenario with `prefill` and `decode` pools runs one kernel per pool; the KV handoff is priced by the kernel's `PDTransferTime` over the scenario's fabric between the two instances' placements. Refused: P/D topology over a colocated scenario; a disaggregated scenario without `--prefill-instances`/`--decode-instances`; pools that differ in block size, `dp` or draft configuration; a `pd_transfer.connector` other than NIXL's (`NixlConnector`, `NixlPullConnector`, `NixlPushConnector`, `MultiConnector`); shared (`--prefill-decode-instances`) or encode instances. The scenario's `offload` block (or `--kv-offload-config`) gives each pool its own tier; see [Latency models](../guide/latency-models.md#prefilldecode-disaggregation) for the handoff (decode-side load, prefill KV hold, contention).

## Router

Every cluster run routes with the llm-d router (the Endpoint Picker, EPP) embedded in-process. Its inputs are the deployment's, so the `--scenario` states them (#1950): the deployment's `router:` block and each pool's `labels`. Each input is parsed by the router's own code; a value the real router rejects is rejected with the router's message. See [Routing](../guide/routing.md#the-routers-inputs) for the mapping, the supported plugins, what is refused at startup, and the mapping from removed BLIS routing and admission flags to llm-d configuration.

| Scenario field | Description |
|----------------|-------------|
| `router.epp_flags` | The llm-d router's (EPP's) command-line flags, verbatim, one entry per flag. Default: a standalone router whose pool is every simulated pod (`--endpoint-selector=llm-d.ai/inference-serving=true --endpoint-target-ports=8000`). |
| `router.endpoint_picker_config` | EndpointPickerConfig file, as the EPP reads it with `--config-file`. Default: the router's built-in default configuration. Refused together with an EPP flag `--config-file` or `--config-text`. |
| `router.manifests` | Kubernetes manifest files. The router reads `InferencePool`, `InferenceObjective` and `InferenceModelRewrite` objects; other kinds are ignored. |
| `pools[].labels` | Labels on the pool's simulated model-server pods, so an `InferencePool` selector or label filter from a real deployment matches. A label every pool states with one value is on every pod; any other is on its pool's pods only. |

Paths are relative to the scenario file; a file that cannot be read is refused at startup. `--epp-flags`, `--epp-config`, `--epp-manifests` and `--pod-labels` were removed (unknown flags), as `--tp` was. The flags below model the environment around the router, not the deployment, and stay on `run` and `replay`:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--gateway-latency` | int64 | 0 | Microseconds from a request's arrival to the router's first look at it (network and proxy hops before the Endpoint Picker). Must be >= 0. |
| `--epp-scrape-jitter` | int64 | 0 | Upper bound (µs) of the seeded phase offset of each endpoint's metrics collection; 0 = every endpoint is scraped in phase. Must be >= 0. |
| `--kv-event-delay` | int64 | 0 | Microseconds from an engine publishing its KV-cache events (at the end of a step) to the router's `precise-prefix-cache-producer` ingesting them. Must be >= 0. Recorded in the trace header. |

**Recorded in the trace.** The resolved router inputs are written to the TraceV2 header (`--trace-output`); the EndpointPickerConfig (and a `--config-file` EPP flag) is inlined as `--config-text` so the header carries the configuration itself. `blis replay` rebuilds the recorded router from the header. A replay whose scenario states no router replays the recorded one; a replay whose scenario states a router (a `router:` block or pool labels) that differs from the header's, or whose router environment flags differ from it, is refused (INV-13).

**Pod labels.** BLIS sets `llm-d.ai/inference-serving=true` on every pod and `llm-d.ai/role` (`prefill`, `decode`, `prefill-decode`, `encode`) from the scenario's pools, in addition to `pools[].labels`; blis-schemas refuses a pool label `llm-d.ai/role` that contradicts the pool's role.

**Objectives.** Request priorities come from the `InferenceObjective` objects in the manifests, which the router reconciles only for an `InferencePool` (`--pool-name`); a request whose objective has none gets priority 0. `examples/llm-d/inferencepool.yaml` defines a pool and one objective per SLO class (`critical` 4, `standard` 3, `batch` -1, `sheddable` -2, `background` -3); `examples/llm-d/scenarios/llama-3.1-70b-instruct-h200-tp4-4node-inferencepool.yaml` routes with it. Each request carries its SLO class in the `x-llm-d-inference-objective` header and its tenant in `x-llm-d-inference-fairness-id`. See [Admission and Flow Control](../guide/admission.md).

### SLO Tier Priorities

Each SLO class has an integer priority. Engine-side, the mapping drives priority scheduling (`--scheduler priority-fcfs`) and, with `--preemption-policy priority`, preemption victim selection. It follows the GAIE convention where **negative priority = sheddable**. Router-side priorities come from `InferenceObjective` objects (above), not from this mapping.

**Default priorities:**

| SLO Class | Priority | Sheddable? |
|-----------|----------|------------|
| `critical` | 4 | No |
| `standard` | 3 | No (default for an empty or unknown class) |
| `batch` | -1 | Yes |
| `sheddable` | -2 | Yes |
| `background` | -3 | Yes |

**Custom overrides:** set `priority.slo_priorities` in the policy bundle. Unspecified classes keep their defaults; values must be in [-100, 100].

```yaml
priority:
  slo_priorities:
    batch: 0       # batch is no longer sheddable for engine scheduling
    critical: 10
```

The mapping lives in `sim/slo_priority.go` (`sim.SLOPriorityMap`); `InvertForVLLM` converts it to the instance convention (lower = more urgent) when a request is enqueued.

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

The `--policy-config` flag loads engine-level policy and infrastructure configuration from a single YAML file. Strict parsing rejects unknown keys. Admission, routing and flow control are not part of a bundle; they are the llm-d router's ([Router](#router)).

```yaml
priority:
  slo_priorities:       # optional: override engine-level SLO class priorities
    batch: 0            # defaults: critical=4, standard=3, batch=-1, sheddable=-2, background=-3

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
```

CLI flags override policy bundle values when explicitly set. For example, `--scheduler sjf` overrides the bundle's `scheduler` setting.

!!! note "Node pools and instance lifecycle are YAML-only"
    `node_pools` and `instance_lifecycle` have no corresponding CLI flags. They must be set via `--policy-config`. Omitting them is safe — the simulator falls back to single-pool, no-lifecycle mode for full backward compatibility.

## Decision Tracing

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--trace-level` | string | "none" | Trace verbosity: `none` or `decisions`. |
| `--counterfactual-k` | int | 0 | Record the top-k endpoints the router scored, with their state, per routing decision. Requires `--trace-level decisions`. |
| `--summarize-trace` | bool | false | Print trace summary after simulation. Requires `--trace-level decisions`. |

See [Routing: Decisions](../guide/routing.md#decisions) and [Cluster Architecture](../concepts/architecture.md).

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
| **BatchConfig** | scenario (`max_num_seqs`, `max_num_batched_tokens`, `long_prefill_token_threshold`) |
| **Latency** | `--speculative-acceptance-rate` / `--speculative-acceptance-length` and the acceptance selectors (everything else: scenario + registry, priced by blis-latency-kernel) |
| **PolicyConfig** | `--scheduler`, `--preemption-policy` |
| **WorkloadConfig** | `--workload` (preset read from `<catalog>/workloads/<name>.yaml`, #1769), `--workload-spec`, `--rate`, `--concurrency`, `--num-requests`, `--prompt-tokens*`, `--output-tokens*`, `--prefix-tokens` |
| **DeploymentConfig** | `--num-instances`, `--prefill-instances`, `--decode-instances`, `--pd-transfer-contention`, `--trace-level`, `--counterfactual-k`. YAML-only (no CLI flag): `node_pools`, `instance_lifecycle` |
| **Router** | the scenario's `router:` block and `pools[].labels`; `--gateway-latency`, `--epp-scrape-jitter`, `--kv-event-delay` |
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

Replays a captured TraceV2 file (any TraceV2, including one from `blis convert`) through the discrete-event simulator. Replay runs on the kernel through the same deployment path as `blis run`, so it requires the same [Deployment Inputs](#deployment-inputs-required) and accepts the same sim-config flags — see [Simulation Control](#simulation-control), [KV Cache Configuration](#kv-cache-configuration), [Batch Formation](#batch-formation), [Latency Model](#latency-model), [Cluster Configuration](#cluster-configuration), [Router](#router), [Scheduling and Priority](#scheduling-and-priority), [Decision Tracing](#decision-tracing), and [Fitness Evaluation](#fitness-evaluation). A run's exported trace replayed with identical flags (including `--horizon`) yields byte-identical stdout (INV-13).

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
