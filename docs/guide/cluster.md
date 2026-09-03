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
| **Batch Formation** | Which requests form the next batch | `--max-num-seqs`, `--max-num-batched-tokens` |

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

### Multi-node tensor parallelism

When node pools are configured (`--policy-config` with `node_pools`), an instance whose `--tp` exceeds a pool's `gpus_per_node` can occupy **whole nodes across the same pool** — enabling multi-node TP for models too large for a single node (e.g. TP=16 on 2×8 H100). This happens automatically and only as a fallback: BLIS first tries to fit the instance on a single node in any matching pool, and spans nodes only when no single node can host the full TP group.

Multi-node TP is modeled as **whole-node occupancy**: it engages only when `tp` is a whole multiple of `gpus_per_node` (e.g. `tp=16` on 8-GPU nodes → 2 nodes), and the instance takes complete nodes so every node carries an equal TP rank count. This is the shape vLLM's multiprocessing (`mp`) executor enforces (it asserts `world_size % nnodes == 0` and derives an equal per-node rank count); vLLM's Ray backend tolerates an asymmetric spread but warns against it, so BLIS is deliberately stricter than the most permissive backend. (vLLM's docs actually recommend avoiding multi-node TP altogether in favor of TP-within-node + PP-across-nodes — see the interconnect note below.) If `tp ≤ gpus_per_node` but the pool is merely fragmented (no single node momentarily has room), the instance is **not** spanned (the resulting asymmetric rank split is discouraged and not modeled); it stays pending, exactly as before. Single-node placement is unchanged.

A spanning instance is billed for every node it occupies (`cost_per_hour × nodes_spanned`).

!!! note "Configuration constraints"
    Two node-pool configurations are rejected at startup (with a clear panic) because they would otherwise produce silently-wrong span/cost numbers — both tracked for proper support by [#1543](https://github.com/inference-sim/inference-sim/issues/1543):

    - A per-role tensor-parallel override (`--prefill-tp` / `--decode-tp`) that **differs from the global `--tp`** is not supported with `node_pools`: placement, node-span, and cost all use the global `--tp`, so a differing per-role TP would be simulated at one degree but placed and billed at another. Use a uniform `--tp`.
    - Every pool must have a **distinct `gpu_type`**: cost (`cost_per_hour`) and capacity (`gpu_memory_gib`) are resolved by first-match on `gpu_type`, so two pools sharing a type would resolve ambiguously.

!!! warning "Cross-node interconnect: bandwidth priced, per-collective latency not calibrated"
    Cross-node collective traffic **is** priced ([#1530](https://github.com/inference-sim/inference-sim/issues/1530)): when a collective's group does not fit inside one node, the trained-physics communication terms charge it at a blended intra/inter-node bandwidth derived from the actual placement. Two things gate it. The `--latency-model` must be `trained-physics` (roofline models no communication at all — [#1663](https://github.com/inference-sim/inference-sim/issues/1663)), and the placed GPU's entry in `--hardware-config` must declare `IntraNodeBwGBps` / `InterNodeBwGBps`; BLIS warns once, on first span, when either is missing, so an unpriced span is never silent. Expect roughly +10% on step time for TP=16 on 2×8 H100 — modest because NCCL's hierarchical all-reduce sends only the reduced chunk across the fabric. What is still **optimistic** is the fixed per-collective launch and round-trip cost: it *is* modeled (`InterNodeLatencyUs`), but it ships **uncalibrated at 0**, so nothing is charged for it out of the box — and at decode message sizes it is plausibly the larger of the two effects. Supplying a measured value is [#1661](https://github.com/inference-sim/inference-sim/issues/1661). Note also that cross-node TP is itself the aggressive topology: vLLM recommends **pipeline parallelism across nodes and tensor parallelism within a node**, precisely because per-layer TP all-reduce means many small collectives over inter-node links (InfiniBand/Ethernet, roughly an order of magnitude below NVLink). Multi-node placement is `blis run` only: `blis replay` rejects `node_pools` outright, and `blis observe` cannot express them at all (it has no `--policy-config` and builds no simulator — its timing comes from a real server). Pipeline parallelism is not yet modeled (tracked by [#1535](https://github.com/inference-sim/inference-sim/issues/1535)).

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

## Further Reading

- [Cluster Architecture](../concepts/architecture.md) — internal mechanics of the shared-clock event loop
- [Routing Policies](routing.md) — scorer composition and signal freshness
- [Metrics & Results](results.md) — understanding trace summaries and per-SLO metrics
