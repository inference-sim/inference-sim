# Cluster Simulation

This guide covers running multi-instance BLIS simulations — the full pipeline from request arrival through admission, routing, scheduling, and metrics aggregation.

```bash
# Quick example: 4-instance cluster with tracing
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --rate 100 --num-requests 500 \
  --trace-level decisions --summarize-trace
```

## Single-Instance vs Cluster Mode

| Setting | Behavior |
|---------|----------|
| `--num-instances 1` (default) | Single-instance: requests go directly to the wait queue, no admission or routing |
| `--num-instances N` (N > 1) | Cluster mode: requests pass through admission → routing → per-instance queues |

## The Pipeline

```
Request → Admission → Routing → Instance WaitQueue → Batch Formation → Step → Completion
                                                          ↓
                                                    KV Allocation + Latency Estimation
```

Each stage is configurable:

| Stage | Controls | Key Flags |
|-------|----------|-----------|
| **Admission** | Whether to accept the request | `--admission-policy`, `--token-bucket-capacity` |
| **Routing** | Which instance receives it | `--routing-policy`, `--routing-scorers` |
| **Scheduling** | What order within the instance | `--scheduler`, `--priority-policy` |
| **Batch Formation** | Which requests form the next batch | `--max-num-running-reqs`, `--max-num-scheduled-tokens` |

## Tensor Parallelism

The `--tp` flag sets the tensor parallelism degree for all instances. TP affects both latency (FLOPs split across GPUs) and memory (KV blocks split across GPUs):

```bash
# TP=2: 2 GPUs per instance
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --tp 2 --rate 100 --num-requests 500

# TP=4: 4 GPUs per instance (lower latency, fewer KV blocks per GPU)
./blis run --model qwen/qwen3-14b \
  --num-instances 2 --tp 4 --rate 100 --num-requests 500
```

!!! note "Homogeneous instances"
    All instances share the same SimConfig (model, GPU, TP, KV blocks). BLIS does not currently model heterogeneous fleets (mixed GPU types or TP configurations).

## Scaling and Saturation

Instance scaling produces **super-linear** TTFT improvement near saturation. With the default model (Qwen3-14B / H100 / TP=1, ~17 req/s per instance at saturation), scaling from 4→12 instances at rate=200 improves TTFT p99 from ~1,500ms to ~54ms.

This happens because the per-instance queue growth rate `excess = λ/k - μ` drops faster than linearly:

```
4 instances:  excess = 200/4 - 17  = 33 req/s per instance   → rapid queue growth
8 instances:  excess = 200/8 - 17  = 8 req/s per instance    → near saturation
12 instances: excess = 200/12 - 17 = -0.3 req/s per instance → balanced (sub-saturation)
```

At sub-saturation (excess ≤ 0): TTFT converges to the baseline (~54ms) and further scaling provides diminishing returns.

## Admission Control

For rate-limiting and traffic shaping policies, see the [Admission Control](admission.md) page.

## Admission and Routing Latency

Model real network/processing overhead between gateway and backend:

```bash
--admission-latency 1000   # 1ms admission decision overhead
--routing-latency 500      # 0.5ms routing decision overhead
```

These add simulated delays to the admission and routing pipeline, modeling gRPC overhead, service mesh hops, and queue serialization in production deployments.

## Decision Tracing

Log every routing decision for offline analysis:

```bash
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --rate 100 --num-requests 500 \
  --trace-level decisions --summarize-trace --counterfactual-k 3
```

The trace summary shows:
- **Target Distribution** — how many requests went to each instance
- **Mean/Max Regret** — how much better an alternative routing decision could have been

!!! info "Counterfactual regret for weighted policies"
    For score-based policies (weighted, least-loaded), counterfactual regret is **structurally zero** — the chosen instance is always the highest-scoring one. Regret is only meaningful for non-score-based policies like round-robin.

## Event Ordering

The cluster uses `(timestamp, priority, seqID)` ordering for deterministic event processing:

- Cluster events at time T process before instance events at time T
- Same-time instance ties broken by lowest instance index
- This ensures determinism (INV-6) but means results differ from a simple M/M/k queueing model

## Work-Conserving Property

BLIS is work-conserving (INV-8): it never idles while requests wait. After every step completion, if the WaitQ has requests, a new StepEvent is immediately scheduled. Real systems may have scheduling delays not modeled here.

## Model Autoscaling

BLIS can simulate dynamic replica scaling through a four-stage WVA-equivalent pipeline:

```
ScalingTickEvent
  → Collector.Collect()       — snapshots per-replica signals from RouterState
  → Analyzer.Analyze()        — derives RequiredCapacity / SpareCapacity per model
  → Engine.Optimize()         — selects variant and exact-N replica delta
  → (HPAScrapeDelay elapses)
  → Actuator.Apply()          — places or drains replicas
```

The autoscaler is **disabled by default**. Enable it by setting `model_autoscaler_interval_us` in `--policy-config`:

```yaml
# 30-second autoscaler tick; V2SaturationAnalyzer with WVA reference defaults
model_autoscaler_interval_us: 30000000
```

### Analyzers

| Analyzer | Description | When to use |
|----------|-------------|-------------|
| `V2SaturationAnalyzer` (default) | WVA token-based KV+queue saturation model | Production parity with llm-d WVA; no latency data needed |
| `QueueingModelAnalyzer` | M/M/1/K-SD with Nelder-Mead/EKF parameter estimation | SLO-aware capacity modeling; requires latency signals (#1198/#954) |

### Stabilization Windows

The autoscaler applies HPA-style stabilization windows to prevent oscillation. A scale-up (or scale-down) decision is only forwarded to the actuator once the signal has been continuously present for the window duration. The window timer resets if the signal disappears.

```yaml
# 2-minute scale-up window; 5-minute scale-down window (matches Kubernetes HPA default)
scale_up_stabilization_window_us: 120000000
scale_down_stabilization_window_us: 300000000
```

### Actuation Delay

`hpa_scrape_delay` (mean/stddev in **seconds**) models the lag between WVA metric emission and the HPA acting on it. Zero (default) means same-tick actuation.

```yaml
hpa_scrape_delay:
  mean: 30.0     # seconds
  stddev: 5.0
```

See [Configuration Reference: Model Autoscaler](../reference/configuration.md#model-autoscaler) for the full field reference including `autoscaler_analyzer` and `qm_config`.

## Further Reading

- [Cluster Architecture](../concepts/architecture.md) — internal mechanics of the shared-clock event loop
- [Routing Policies](routing.md) — scorer composition and signal freshness
- [Metrics & Results](results.md) — understanding trace summaries and per-SLO metrics
