# Metrics & Results

This guide covers how to read BLIS output — from the primary JSON metrics to anomaly counters, KV cache diagnostics, per-SLO breakdowns, fitness scores, and trace summaries.

```bash
# Quick example: run with all diagnostic output enabled
./blis run --model meta-llama/llama-3.1-8b-instruct \
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
| **Dropped Unservable** | Request needs more KV blocks than exist | Increase `--total-kv-blocks` or reduce max input tokens |

## KV Cache Metrics

When KV cache activity is nonzero, BLIS prints `=== KV Cache Metrics ===`:

| Metric | Meaning | Concern Threshold |
|--------|---------|-------------------|
| **Preemption Rate** | Fraction of requests that were preempted (KV evicted) | > 5% indicates KV pressure |
| **Cache Hit Rate** | Fraction of blocks served from prefix cache | Higher is better — indicates prefix reuse |
| **KV Thrashing Rate** | Repeated preemption-reallocation cycles | > 0 indicates severe memory pressure |

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
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --rate 100 --num-requests 500 --results-path results.json
```

Each request record includes TTFT, E2E, scheduling delay, and completion status.

## Further Reading

- [Configuration Reference](../reference/configuration.md#fitness-evaluation) — fitness weight syntax
- [Tutorial: Capacity Planning](../getting-started/tutorial.md) — applying results to capacity decisions
