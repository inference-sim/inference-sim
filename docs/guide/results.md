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

The JSON output on stdout contains:

| Field | Unit | Description |
|-------|------|-------------|
| `ttft_mean_ms` | ms | Mean Time To First Token — average responsiveness |
| `ttft_p90_ms` | ms | 90th percentile TTFT |
| `ttft_p99_ms` | ms | 99th percentile TTFT — tail latency |
| `e2e_mean_ms` | ms | Mean End-to-End latency — average total request time |
| `e2e_p99_ms` | ms | 99th percentile E2E |
| `itl_mean_ms` | ms | Mean Inter-Token Latency — streaming smoothness |
| `scheduling_delay_p99_ms` | ms | 99th percentile scheduling delay — queue wait time |
| `responses_per_sec` | req/s | Throughput |
| `completed_requests` | count | Requests that finished before horizon |

### Scheduling Delay

Scheduling delay isolates the WaitQ wait time from compute time. High scheduling delay + low preemptions = **queue saturation** (add instances). Low scheduling delay + high TTFT = **compute saturation** (reduce batch size or use chunked prefill).

!!! warning "Per-request units gotcha"
    With `--results-path`, per-request `scheduling_delay_ms` is in **ticks (microseconds)** despite the field name. The aggregate `scheduling_delay_p99_ms` IS in milliseconds (divided by 1000). Always check units when comparing per-request to aggregate metrics.

## Anomaly Counters

When anomalies are detected, BLIS prints `=== Anomaly Counters ===`:

| Counter | What It Means | Action |
|---------|--------------|--------|
| **Priority Inversions** | A lower-priority request was scheduled before a higher-priority one | Check scheduler choice — use `priority-fcfs` for SLO workloads |
| **HOL Blocking Events** | A long prefill blocked shorter requests | Enable chunked prefill: `--long-prefill-token-threshold 256` |
| **Rejected Requests** | Admission policy rejected the request | Check token bucket capacity or admission policy |
| **Dropped Unservable** | Request exceeds `--max-model-len` context window or needs more KV blocks than exist | Check `--max-model-len` setting; increase `--total-kv-blocks` or reduce max input tokens |
| **Dropped KV Allocations** | Decode sub-request could not allocate KV blocks at the decode instance (PD mode only) | Increase `--total-kv-blocks` on decode instances or reduce decode pool load |

## KV Cache Metrics

When KV cache activity is nonzero, BLIS prints `=== KV Cache Metrics ===`:

| Metric | Meaning | Concern Threshold |
|--------|---------|-------------------|
| **Preemption Rate** | Fraction of requests that were preempted (KV evicted) | > 5% indicates KV pressure |
| **Cache Hit Rate** | Fraction of blocks served from prefix cache | Higher is better — indicates prefix reuse |
| **KV Thrashing Rate** | Repeated preemption-reallocation cycles | > 0 indicates severe memory pressure |

## PD Disaggregation Metrics

When PD disaggregation is active (`--prefill-instances` and `--decode-instances` > 0), BLIS prints `=== PD Metrics ===`:

| Field | Unit | Description |
|-------|------|-------------|
| **Disaggregated Requests** | count | Number of parent requests that completed KV transfer (TransferCompleteTime > 0). Includes requests subsequently dropped due to decode KV exhaustion — those are also counted in `DroppedUnservable`, not separately subtracted here. |
| **Prefill Throughput** | sub-req/s | Completed prefill sub-requests per second across all prefill instances. |
| **Decode Throughput** | sub-req/s | Completed decode sub-requests per second across all decode instances. |
| **Load Imbalance Ratio** | ratio | `max(PrefillRPS, DecodeRPS) / min(PrefillRPS, DecodeRPS)`. `1.0` = balanced; `inf (one pool idle)` = one pool has zero completions. |
| **Parent TTFT (μs)** | microseconds | Distribution of prefill sub-request TTFT: `arrival_time → first token generated on prefill instance`. Includes admission delay + prefill queue wait + prefill compute time. **Does not include KV transfer time or decode queue latency.** |
| **KV Transfer Duration (μs)** | microseconds | Distribution of `TransferCompleteTime - TransferStartTime`. Captures network transfer time only. |
| **Peak Concurrent Transfers** | count | Maximum number of KV transfers in-flight simultaneously. Only printed when `--pd-transfer-contention` is enabled. Useful for diagnosing peak contention load on the inter-pool network. |
| **Mean Transfer Queue Depth** | dimensionless | Average number of transfers in-flight at each transfer start event. Computed as `sum(activeTransfers at start) / count(starts)`. **Not** a queue-length metric — it measures average concurrent occupancy at scheduling time, not how many transfers are blocked. Only printed when `--pd-transfer-contention` is enabled. |

**Unit note:** `sub-req/s` counts sub-requests, not parent requests. In a fully disaggregated steady-state simulation, `PrefillThroughput ≈ DecodeThroughput ≈ responses_per_sec` in the JSON output, since each original request generates one prefill sub-request and one decode sub-request.

**TTFT note:** In PD disaggregation, BLIS records `ParentTTFT` at prefill completion (when the prefill sub-request finishes computing the prompt). This differs from client-visible TTFT in production llm-d/vLLM deployments, where TTFT is measured at the first decode token (after KV transfer completes). `ParentTTFT` will be lower than client-visible TTFT by the sum of KV transfer time + decode queue wait + first decode step time.

**Load Imbalance Ratio note:** This metric measures throughput balance (sub-request completions/s per pool) as an **aggregate over the full simulation**, not a real-time queue depth signal. In a correctly-sized disaggregated system, both pools complete the same number of sub-requests, so the ratio approaches 1.0. To diagnose bottlenecks, compare `PrefillThroughput` vs `DecodeThroughput` directly and inspect P99 TTFT (for prefill backpressure) and P99 E2E (for decode backpressure). A ratio near 1.0 does not guarantee real-time balance — one pool may queue while the other idles if arrival patterns are bursty.

**JSON output note:** PD metrics are printed to stdout only. They do not appear in the per-request JSON results file (`--results-path`). The JSON file contains per-request latency fields (`ttft_ms`, `itl_ms`, `e2e_ms`) for every completed sub-request. In PD mode, each original request produces **two sub-requests** (one prefill, one decode), so the JSON file contains **2N rows** for N disaggregated requests. The decode sub-request row has near-zero TTFT (it enters the decode instance with input tokens already computed). The prefill sub-request row captures the prefill latency. For end-to-end analysis, join the two rows by request ID prefix (e.g., `"req-1_prefill"` and `"req-1_decode"`).

## Per-SLO-Class Metrics

When multiple SLO classes are present in the workload, BLIS prints per-class TTFT and E2E distributions. This lets you verify that `critical` requests meet SLOs even when `batch` traffic is heavy.

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
  --rate 100 --num-requests 500 --results-path results.json
```

Each request record includes TTFT, E2E, scheduling delay, and completion status.

## Further Reading

- [Configuration Reference](../reference/configuration.md#fitness-evaluation) — fitness weight syntax
- [Tutorial: Capacity Planning](../getting-started/tutorial.md) — applying results to capacity decisions
