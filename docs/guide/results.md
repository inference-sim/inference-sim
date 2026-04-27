# Metrics & Results

This guide covers how to read BLIS output — from the primary JSON metrics to anomaly counters, KV cache diagnostics, per-SLO breakdowns, fitness scores, and trace summaries.

```bash
# Quick example: run with all diagnostic output enabled
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --rate 200 --num-requests 1000 \
  --trace-level decisions --summarize-trace \
  --fitness-weights "p99_ttft:3,mean_e2e:1,throughput:2"
```

## Primary Metrics (JSON Output)

The JSON output on stdout (and in the `--metrics-path` file when set) contains:

| Field | Unit | Description |
|-------|------|-------------|
| `instance_id` | string | Instance identifier (`"cluster"` for aggregate output) |
| `completed_requests` | count | Requests that finished before horizon |
| `still_queued` | count | Requests in the wait queue at horizon end |
| `still_running` | count | Requests in the running batch at horizon end |
| `injected_requests` | count | Total requests that entered the simulator |
| `total_input_tokens` | count | Sum of input tokens across all completed requests |
| `total_output_tokens` | count | Sum of output tokens across all completed requests |
| `vllm_estimated_duration_s` | s | Estimated vLLM wall-clock duration for the workload |
| `responses_per_sec` | req/s | Throughput (completed requests / effective duration) |
| `tokens_per_sec` | tok/s | Output token throughput |
| `e2e_mean_ms` | ms | Mean End-to-End latency |
| `e2e_p90_ms` | ms | 90th percentile E2E |
| `e2e_p95_ms` | ms | 95th percentile E2E |
| `e2e_p99_ms` | ms | 99th percentile E2E — tail latency |
| `ttft_mean_ms` | ms | Mean Time To First Token — average responsiveness |
| `ttft_p90_ms` | ms | 90th percentile TTFT |
| `ttft_p95_ms` | ms | 95th percentile TTFT |
| `ttft_p99_ms` | ms | 99th percentile TTFT — tail latency |
| `itl_mean_ms` | ms | Mean Inter-Token Latency — streaming smoothness |
| `itl_p90_ms` | ms | 90th percentile ITL |
| `itl_p95_ms` | ms | 95th percentile ITL |
| `itl_p99_ms` | ms | 99th percentile ITL |
| `scheduling_delay_p99_ms` | ms | 99th percentile scheduling delay — queue wait time |
| `kv_allocation_failures` | count | KV cache allocation failures (omitted when zero) |
| `preemption_count` | count | Total preemption events (integer; see also KV Cache Metrics for the rate) |
| `dropped_unservable` | count | Requests dropped at enqueue due to context limit violations |
| `length_capped_requests` | count | Requests force-completed at `MaxModelLen` |
| `timed_out_requests` | count | Requests that exceeded their deadline |
| `requests` | array | Per-request detail records (omitted when empty; see [Per-Request Fields](#per-request-fields)) |

### Scheduling Delay

Scheduling delay isolates the WaitQ wait time from compute time. High scheduling delay + low preemptions = **queue saturation** (add instances). Low scheduling delay + high TTFT = **compute saturation** (reduce batch size or use chunked prefill).

!!! warning "Per-request units"
    All per-request latency fields (`ttft_ms`, `e2e_ms`, `itl_ms`, `scheduling_delay_ms`) are in **milliseconds** — converted from internal ticks by dividing by 1,000. Aggregate metrics (`scheduling_delay_p99_ms`, etc.) are also in milliseconds. See [Known Unit Gotchas](../reference/configuration.md#known-unit-gotchas) for the full unit reference. Note: hypothesis scripts written before BC-14 may divide `scheduling_delay_ms` by 1,000 unnecessarily — that field is now already in ms.

### Per-Request Fields

When the `requests` array is non-empty, each entry contains:

| Field | Unit | Description |
|-------|------|-------------|
| `requestID` | string | Unique request identifier |
| `arrived_at` | ticks | Arrival timestamp in simulation ticks |
| `num_prefill_tokens` | count | Input (prompt) token count |
| `num_decode_tokens` | count | Output token count |
| `ttft_ms` | ms | Time To First Token for this request |
| `itl_ms` | ms | Mean Inter-Token Latency for this request |
| `e2e_ms` | ms | End-to-End latency for this request |
| `scheduling_delay_ms` | ms | Time spent in the wait queue before first scheduling |
| `slo_class` | string | SLO class (`critical`, `standard`, `batch`, etc.) — omitted if empty |
| `tenant_id` | string | Tenant label — omitted if empty |
| `handled_by` | string | Instance ID that processed the request — omitted if empty |
| `model` | string | Model tag — omitted if empty |
| `length_capped` | bool | `true` if request was force-completed at `MaxModelLen` — omitted when `false` |
| `gateway_queue_delay_ms` | ms | Time spent in the gateway queue — omitted when flow control is disabled |
| `session_id` | string | Multi-turn session link — omitted for single-turn requests |
| `round_index` | int | Round within session (`0` = first turn); always present, defaults to `0` for non-session requests |

## Anomaly Counters

When anomalies are detected, BLIS prints `=== Anomaly Counters ===`:

| Counter | What It Means | Action |
|---------|--------------|--------|
| **Priority Inversions** | A lower-priority request was scheduled before a higher-priority one | Check scheduler choice — use `priority-fcfs` for SLO workloads |
| **HOL Blocking Events** | A long prefill blocked shorter requests | Enable chunked prefill: `--long-prefill-token-threshold 256` |
| **Rejected Requests (Admission)** | Admission policy rejected the request at cluster ingress | Check token bucket capacity or admission policy |
| **Shed (tier)** | Per-SLO-class breakdown of admission rejections under overload — printed as indented sub-items beneath Rejected Requests (Admission) | Adjust `slo_priorities` in the policy bundle or raise admission thresholds |
| **Rejected Requests (Routing)** | No routable instances for the request's model — all instances are `Loading` or `Draining` | Increase `initial_nodes`, reduce `loading_delay.mean`, or stagger drain operations |
| **Dropped Unservable** | Request exceeds `--max-model-len` context window or needs more KV blocks than exist | Check `--max-model-len` setting; increase `--total-kv-blocks` or reduce max input tokens |
| **Timed Out Requests** | Request exceeded its client deadline before completing | Increase `--timeout` or reduce load |
| **Length-Capped Requests** | Request was force-completed when it reached `MaxModelLen` tokens during decode | Expected if workloads push against `--max-model-len`; set `--max-model-len 0` (unlimited) to disable the cap |
| **Gateway Queue Depth (horizon)** | Requests still waiting in the gateway queue when the simulation ended | Reduce arrival rate or increase cluster capacity |
| **Gateway Queue Shed** | Requests shed from the gateway queue because it was full | Increase `--max-gateway-queue-depth` or enable `--flow-control` with a saturation detector |

## KV Cache Metrics

When KV cache activity is nonzero, BLIS prints `=== KV Cache Metrics ===`:

| Metric | Meaning | Concern Threshold |
|--------|---------|-------------------|
| **Preemption Rate** | Fraction of requests that were preempted (KV evicted) | > 5% indicates KV pressure |
| **Cache Hit Rate** | Fraction of blocks served from prefix cache | Higher is better — indicates prefix reuse |
| **KV Thrashing Rate** | Repeated preemption-reallocation cycles | > 0 indicates severe memory pressure |

## Per-SLO-Class Metrics

When multiple SLO classes are present in the workload, BLIS prints per-class TTFT and E2E distributions. This lets you verify that `critical` requests meet SLOs even when `batch` traffic is heavy.

## Per-Model Metrics

When instances serve different models (multi-model deployment), BLIS prints per-model TTFT mean/p99, E2E mean/p99, and throughput (req/s). This appears automatically when requests carry model tags. Output format:

```
=== Per-Model Metrics ===
  qwen/qwen3-14b:
    TTFT: p50=1234.56 p99=5678.90 (n=250)
    E2E:  p50=9876.54 p99=12345.67 (n=250)
    Throughput: 50.00 req/s, 6400.00 tok/s
```

Per-model metrics appear on stdout only. The `--metrics-path` JSON file (see [Primary Metrics](#primary-metrics-json-output)) contains only the aggregate `MetricsOutput` fields. The same applies to per-tenant, session, and PD metrics — all are stdout-only sections.

## Per-Tenant Metrics

When requests carry `tenant_id` labels, BLIS prints per-tenant request counts, total output tokens served, and a [Jain Fairness Index](https://en.wikipedia.org/wiki/Fairness_measure) over the token distribution. This section appears automatically and is omitted when no requests carry tenant labels (backward-compatible with legacy and untenanted workloads). A workload with a single named tenant shows the section with Jain=1.0 (trivially fair).

```
=== Per-Tenant Metrics ===
  alice: requests=50, tokens=12500
  bob: requests=50, tokens=12480
  Jain Fairness Index: 0.9999
```

Tenants are listed in lexicographic order. The Jain Fairness Index ranges from `1/N` (maximally unfair — one tenant receives everything) to `1.0` (perfectly fair — all tenants receive equal tokens). A balanced two-tenant workload produces a value ≥ 0.99.

To tag requests with tenant labels, set `tenant_id` in your workload spec cohort:

```yaml
cohorts:
  - name: alice-traffic
    tenant_id: alice
    ...
  - name: bob-traffic
    tenant_id: bob
    ...
```

## Session Metrics

When a workload contains multi-turn sessions (requests with `session_id`), BLIS prints a `=== Session Metrics ===` block:

| Line | What It Means |
|------|--------------|
| **Sessions** | Number of distinct multi-turn sessions in the workload |
| **TTFT cold (round 0)** | TTFT distribution for the first round of each session — no KV cache warmth from prior context |
| **TTFT warm (round≥1)** | TTFT distribution for follow-up rounds — benefits from cached context |
| **Session duration** | End-to-end duration from session start (first round arrival) to last round completion |

Cold vs. warm TTFT split reveals prefix cache effectiveness: if `TTFT warm` ≈ `TTFT cold`, prefix caching is not activating for continuations — check `--cache-signal-delay` and scorer configuration.

## PD Disaggregation Metrics

When PD disaggregation is active (`--prefill-instances > 0`), BLIS prints a `=== PD Metrics ===` block:

| Field | What It Means |
|-------|--------------|
| **Disaggregated Requests** | Requests routed through the prefill→KV transfer→decode path |
| **Dropped at Decode KV** | Requests whose transferred KV blocks could not be accepted by the decode instance |
| **Prefill Throughput** | Sub-request completion rate on prefill instances (sub-req/s) |
| **Decode Throughput** | Sub-request completion rate on decode instances (sub-req/s) |
| **Load Imbalance Ratio** | `max(prefill_load, decode_load) / min(...)` — `1.0` = perfectly balanced; `inf` = one pool has no completions |
| **Parent TTFT** | Client-visible TTFT (prefill TTFT + KV transfer duration + first decode step); distribution in microseconds |
| **KV Transfer Duration** | Time to transfer KV blocks from prefill to decode instance; distribution in microseconds |
| **Peak Concurrent Transfers** | Maximum simultaneous in-flight KV transfers (only with `--pd-transfer-contention`) |
| **Mean Transfer Queue Depth** | Average queue depth at the transfer bandwidth bottleneck (only with `--pd-transfer-contention`) |

## Fitness Evaluation

For automated multi-configuration comparison:

```bash
--fitness-weights "p99_ttft:3,mean_e2e:1,throughput:2"
```

Valid metric keys: `throughput`, `tokens_per_sec`, `p99_ttft`, `p50_ttft`, `mean_ttft`, `p99_e2e`, `p50_e2e`, `mean_e2e`.

### How Normalization Works

- **Latency metrics:** `1 / (1 + value/1000)` — lower latency → higher score. Reference: 1000 ticks = 1ms
- **Throughput metrics:** `value / (value + reference)` — higher throughput → higher score. References: RPS=100, TPS=10,000

!!! warning "Normalization compresses large differences"
    The `1/(1+x/1000)` function compresses large raw differences into small score differences. A 38% TTFT p99 improvement (39,000→64,000 ticks) maps to only 2-8% fitness score difference. Always examine raw metrics alongside fitness scores for meaningful comparison.

## Common Patterns

### Saturation Curves

As arrival rate increases past per-instance service capacity (μ ≈ 1/step_time), TTFT p99 grows super-linearly. The queue growth rate `excess = λ/k - μ` determines how quickly latency degrades.

### Tail Latency Spikes

P99 diverges from mean sharply near saturation. A workload at 90% capacity may show 2x mean TTFT but 10x P99 TTFT.

### Snapshot Staleness Effects

With `kv-utilization` scorer alone at `--snapshot-refresh-interval 100ms`: +354% TTFT degradation. The default composite scorer mitigates ~99% of this effect.

### Policy Equivalence at Low Load

All routing policies produce equivalent results (within 5%) at low utilization. Differentiation requires moderate-to-high load where queueing dynamics dominate.

### Alpha Overhead

BLIS models non-GPU overhead (tokenization, API serialization) as `alpha` coefficients. Alpha queueing time (alpha0 + alpha1 × inputLen) delays request enqueue, creating an event gap, but does not occupy the GPU. Alpha output processing time (alpha2) adds to TTFT/E2E metrics but does not affect step scheduling. This means:

- Simulated E2E > theoretical M/M/k E2E (especially at high load)
- The divergence is 28-71% at ρ ≥ 0.5 but only 0.3-3.3% at ρ ≤ 0.3
- To compare with theoretical models, use `rho_eff = lambda × step_total` not `lambda × E2E_total`

## Per-Request Results

For detailed analysis, save per-request data:

```bash
./blis run --model qwen/qwen3-14b \
  --rate 100 --num-requests 500 --metrics-path metrics.json
```

Each request record includes TTFT, E2E, scheduling delay, and completion status.

## Further Reading

- [Configuration Reference](../reference/configuration.md#fitness-evaluation) — fitness weight syntax
- [Tutorial: Capacity Planning](../getting-started/tutorial.md) — applying results to capacity decisions
