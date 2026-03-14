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

!!! note "Per-pool hardware configuration"
    By default, all instances share the same SimConfig (model, GPU, TP, KV blocks). When using PD disaggregation, per-pool hardware overrides (`--prefill-tp`, `--decode-tp`, `--prefill-hardware`, `--decode-hardware`, etc.) allow prefill and decode pools to use different configurations. See [Per-Pool Hardware Overrides](../reference/configuration.md#per-pool-hardware-overrides) for details.

## Scaling and Saturation

Instance scaling produces **super-linear** TTFT improvement near saturation. Doubling from 4→8 instances at near-capacity (rate=500) improves TTFT p99 by 7.4x, not 2x.

This happens because the per-instance queue growth rate `excess = λ/k - μ` drops faster than linearly:

```
4 instances: excess = 500/4 - 57.4 = 67.6 req/s per instance → rapid queue growth
8 instances: excess = 500/8 - 57.4 = 5.1 req/s per instance  → minimal queueing
```

At sub-saturation (rate=100): scaling effect vanishes (1.06x).

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

In PD (Prefill-Decode) disaggregated mode (`--prefill-instances`, `--decode-instances`), the trace additionally captures per-request PD pipeline events:

- **DisaggregationRecord** — one per admitted request: whether it was routed to the prefill pool or handled locally
- **PrefillRoutingRecord** — one per disaggregated request: which prefill instance was chosen (with optional counterfactual)
- **KVTransferRecord** — one per disaggregated request: transfer duration, block count, and instance pair
- **DecodeRoutingRecord** — one per disaggregated request: which decode instance was chosen (with optional counterfactual)

Example PD trace run with counterfactual analysis:

```bash
./blis run --model qwen/qwen3-14b \
  --prefill-instances 2 --decode-instances 2 \
  --rate 50 --num-requests 200 \
  --trace-level decisions --summarize-trace --counterfactual-k 2
```

This adds a PD disaggregation summary to stdout:

```
=== PD Disaggregation Summary ===
Disaggregation Decisions: 200
  Disaggregated: 180
KV Transfers: 180
Mean Transfer Duration (µs): 42.3
```

Interpreting the output:
- **Disaggregation Decisions** — total requests that reached the disaggregation decision point (includes both pooled and local paths)
- **Disaggregated** — how many were routed to the prefill pool (remainder used standard routing)
- **KV Transfers** — should equal `Disaggregated` for successful runs; a smaller number indicates decode-phase KV OOM drops (also shown in the Anomaly Counters section)
- **Mean Transfer Duration** — average KV cache transfer latency in microseconds; tune `--pd-transfer-bandwidth` and `--pd-transfer-base-latency` to match your interconnect
- **Peak Concurrent Transfers / Mean Transfer Queue Depth** — only printed when `--pd-transfer-contention` is enabled; see below

**When to enable `--pd-transfer-contention`:** By default, each KV transfer gets the full configured bandwidth, modeling an uncongested link. Enable this flag when you want to model inter-pool link saturation: under high concurrency (many simultaneous disaggregated requests), all in-flight transfers share the bandwidth equally (`bandwidth/N` per transfer). If `Peak Concurrent Transfers` is frequently > 1 in a non-contention run and transfer latency is a bottleneck, enabling this flag will give a more conservative (and realistic) estimate of transfer durations.

!!! note "Single shared bandwidth pool"
    The contention model uses a **single cluster-wide bandwidth pool**: all concurrent KV transfers — regardless of which prefill or decode instance pair is involved — share the same `--pd-transfer-bandwidth` budget. This matches deployments where all inter-pool traffic flows through a single congestion point (e.g., a shared RDMA fabric or one NIC per node). In deployments where each prefill instance has an independent NIC, the model will over-estimate contention: set `--pd-transfer-bandwidth` to the aggregate shared capacity, not per-NIC bandwidth, for accurate results.

### Co-Location Interference Model

When prefill and decode requests share the same instance (no disaggregation, or mixed fleets), the two phases interfere: decode steps slow down prefill iteration, and prefill increases memory pressure for decode tokens. Use `--pd-interference-prefill` and `--pd-interference-decode` to model this slowdown:

```bash
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 \
  --pd-interference-prefill 0.5 \
  --pd-interference-decode 0.3
```

The interference model applies a multiplier to every step's latency:

```
multiplier = 1.0 + factor × (minority_phase_count / total_batch_size)
```

where `minority_phase_count` is the number of requests in the less-common phase. A factor of `0.5` at a 50/50 split produces a **1.25×** step time increase (25% slowdown). A factor of `1.0` produces at most **1.5×** (50% slowdown).

- `--pd-interference-prefill` — factor applied when prefill is the majority phase (decode co-locates as minority)
- `--pd-interference-decode` — factor applied when decode is the majority phase (prefill co-locates as minority)
- When the batch is exactly 50/50, the larger of the two factors is used (conservative worst-case)
- When the batch is phase-pure (all prefill or all decode), the multiplier is always 1.0 — no penalty

!!! note "No effect in disaggregated mode"
    In PD disaggregated mode (`--prefill-instances`, `--decode-instances`), each pool receives only its own phase (INV-PD-2: Pool Exclusivity). Batches are always phase-pure, so interference factors have no effect. These flags are useful for **non-disaggregated** deployments, or for **break-even analysis**: compare the slowdown from co-location against the latency added by KV transfer in disaggregated mode.

Use `--counterfactual-k N` to record the top-N alternative routing candidates and regret for both prefill and decode routing decisions.

!!! info "Counterfactual regret for weighted policies"
    For score-based policies (weighted, least-loaded), counterfactual regret is **structurally zero** — the chosen instance is always the highest-scoring one. Regret is only meaningful for non-score-based policies like round-robin.

## Context Window Enforcement

BLIS supports a `--max-model-len` flag that enforces a maximum total sequence length (input + output tokens) per request, mirroring vLLM's `--max-model-len`.

When set (> 0):
- Requests whose input alone fills the window (`input >= max_model_len`) are dropped at enqueue time (`DroppedUnservable` counter)
- Requests with an explicit output budget (`MaxOutputLen > 0` in the workload spec) are dropped when `input + budget > max_model_len`
- A runtime cap force-completes any request whose `ProgressIndex` reaches `max_model_len` during decode (defense-in-depth)

**Auto-derivation in roofline/crossmodel/trained-roofline modes:** When `--max-model-len` is not explicitly set and an analytical latency backend is active, BLIS auto-derives the value from `max_position_embeddings` in the HuggingFace `config.json`. For models with `rope_scaling`, the scaling factor is applied (with type-specific exclusions matching vLLM). The derived value is then capped to the KV-feasible maximum (`total_kv_blocks × block_size_tokens`) to prevent over-admission when GPU memory is the constraint. Set `--max-model-len 0` (default for non-analytical modes) to disable the limit.

**Per-pool limits:** `--prefill-max-model-len` and `--decode-max-model-len` allow different limits per pool. If a pool's limit is smaller than the global admission check, requests may pass admission but be dropped at the instance level. See the warning in [Configuration Reference: Per-Pool Hardware Overrides](../reference/configuration.md#per-pool-hardware-overrides).

## Event Ordering

The cluster uses `(timestamp, priority, seqID)` ordering for deterministic event processing:

- Cluster events at time T process before instance events at time T
- Same-time instance ties broken by lowest instance index
- This ensures determinism (INV-6) but means results differ from a simple M/M/k queueing model

## Work-Conserving Property

BLIS is work-conserving (INV-8): it never idles while requests wait. After every step completion, if the WaitQ has requests, a new StepEvent is immediately scheduled. Real systems may have scheduling delays not modeled here.

## PD Disaggregation Mode

BLIS supports prefill-decode disaggregation, where prefill and decode steps run on separate instance pools:

```bash
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --prefill-instances 2 --decode-instances 4 \
  --pd-decider always \
  --pd-transfer-bandwidth 25 --pd-transfer-base-latency 0.05
```

!!! warning "Set `--pd-decider always`"
    Without `--pd-decider always`, setting `--prefill-instances` and `--decode-instances` has no effect — requests use standard routing across all instances and no PD metrics are collected.

In PD mode, the pipeline changes:

```
Request → Admission → Disaggregation Decision
  → Prefill Routing (prefill pool) → Prefill Instance → KV Transfer
    → Decode Routing (decode pool) → Decode Instance → Completion
```

PD-specific metrics appear in the `=== PD Metrics ===` output section. See [Metrics & Results](results.md#pd-disaggregation-metrics) for field descriptions, and [Configuration Reference](../reference/configuration.md#pd-disaggregation) for all flags.

### PD Troubleshooting

**No `=== PD Metrics ===` section in output?**

- Check that `--pd-decider always` is set. Without it, requests use standard routing even if pool flags are set.
- Verify `--prefill-instances + --decode-instances <= --num-instances`. BLIS exits with a fatal error if the sum exceeds total instances — check the error message for details.

**`Disaggregated Requests` count is lower than expected?**

- If decode pool KV capacity is exhausted, decode sub-requests are dropped (`DroppedUnservable` counter). Increase `--total-kv-blocks` for decode instances or reduce `--decode-instances` to fewer, larger instances.

**High `DroppedUnservable` with PD disaggregation?**

- BLIS uses hard drops when decode KV capacity is exhausted (no fallback routing or preemption). In production vLLM / llm-d, many of these scenarios are handled by preemption or fallback to another decode instance. To approximate production headroom, add at least 20% buffer to decode KV capacity:
  ```bash
  --total-kv-blocks 1200  # instead of 1000 (20% headroom)
  ```

**High `Load Imbalance Ratio`?**

- Ratio >> 1.0 means one pool is bottlenecked. Compare `PrefillThroughput` vs `DecodeThroughput`: if prefill is faster, add decode instances; if decode is faster, add prefill instances.
- Use `--snapshot-refresh-interval 0` when PD disaggregation is active for Immediate routing signal freshness. Periodic snapshots (interval > 0) can cause routing oscillation in disaggregated pipelines.

### Known Simplifications

The BLIS PD model is a Phase 1 simulation approximation. Key differences from production vLLM / llm-d:

- **Atomic KV transfer**: KV blocks are transferred atomically after prefill completes. Incremental (pipelined) transfer is not modeled.
- **No cross-instance preemption**: In vLLM, a new decode arrival may evict existing decode requests' KV blocks from GPU memory. BLIS treats each instance independently — decode arrivals do not trigger preemption of in-flight requests on the same instance.
- **No transfer retry**: If a decode instance has insufficient KV capacity, the request is dropped (counted in `DroppedUnservable`). No retry or fallback routing is attempted.
- **Fixed pool sizes**: Pool membership is static for the duration of the simulation. Dynamic autoscaling is planned for a future PR.

## Further Reading

- [Cluster Architecture](../concepts/architecture.md) — internal mechanics of the shared-clock event loop
- [Routing Policies](routing.md) — scorer composition and signal freshness
- [Metrics & Results](results.md) — understanding trace summaries and per-SLO metrics
