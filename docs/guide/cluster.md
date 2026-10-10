# Cluster Simulation

This guide covers running multi-instance BLIS simulations — the full pipeline from request arrival through admission, routing, scheduling, and metrics aggregation.

```bash
# Quick example: 4-instance cluster with tracing
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
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
| **Batch Formation** | Which requests form the next batch | scenario `max_num_seqs`, `max_num_batched_tokens` |

## Deployment Shape: Parallelism, DP and P/D

Each instance's shape comes from the scenario, not from flags: the pool's `parallel` block
(`tp`, `pp`, `dp`, `enable_expert_parallel`) and its `engine` settings. blis-latency-kernel prices
the step time and sizes the KV budget for that shape. `--num-instances` sets how many instances
run; all instances of a pool are identical.

`--num-instances` must not exceed the scenario's rank capacity: pool `nodes` × `gpus_per_node` /
(pp × tp × pcp). The single-node TP4 Llama scenario fits 2 instances;
`testdata/scenarios/llama-3.1-70b-instruct-h200-tp4-4node.yaml` (4 H200 nodes over 400G
InfiniBand, TP4) fits 8. For another shape, copy a scenario, raise `cluster.nodes` and the pool's
`nodes`, and add a `cluster.fabric`.

**Data parallelism (MoE only).** A pool with `dp > 1` becomes one replica per data-parallel rank,
each sized as one vLLM EngineCore (its own KV budget and batch limits). `dp > 1` on a dense model
is refused.

**Prefill/decode disaggregation.** A scenario with `prefill` and `decode` pools runs one kernel per
pool. Choose the topology with `--prefill-instances` and `--decode-instances` (whose sum must not exceed `--num-instances`) and `--pd-decider`;
each must not exceed its pool's rank capacity, `nodes × gpus_per_node / (pp × tp × pcp)`. The KV
handoff is priced by the kernel's `PDTransferTime` over the scenario's fabric between the two
instances' placements. This repository's `testdata/scenarios/glm-5-h200-3p1d-ib.yaml` is a 3P1D
example:

```bash
./blis run --scenario glm-5-h200-3p1d-ib.yaml --scenarios testdata/scenarios \
  --registry $PWD/blis-registry \
  --num-instances 4 --prefill-instances 3 --decode-instances 1 --pd-decider always \
  --rate 10 --num-requests 100
```

Refused: a P/D topology over a colocated scenario; a disaggregated scenario without
`--prefill-instances`/`--decode-instances`; prefill and decode pools that differ in block size,
`dp` or draft configuration; KV offload combined with P/D; shared (`--prefill-decode-instances`)
or encode instances.

### Node pools and multi-node placement

When node pools are configured (`--policy-config` with `node_pools`), every pool's `gpu_type` must
equal the scenario's `cluster.hardware`, and `gpu_type` must be unique across pools.

An instance whose TP group exceeds a pool's `gpus_per_node` can occupy **whole nodes across the same
pool**. This happens only as a fallback (BLIS first tries a single node in any matching pool) and
only when `tp` is a whole multiple of `gpus_per_node` (e.g. `tp=16` on 8-GPU nodes → 2 nodes), so
every node carries an equal rank count — the shape vLLM's multiprocessing executor enforces. If
`tp ≤ gpus_per_node` but the pool is merely fragmented, the instance is not spanned; it stays
pending. A spanning instance is billed for every node it occupies (`cost_per_hour × nodes_spanned`).
Node pools are `blis run` only: `blis replay` rejects `node_pools`.

## Scaling and Saturation

Instance scaling produces **super-linear** TTFT improvement near saturation, because the per-instance queue growth rate `excess = λ/k - μ` drops faster than linearly. For example, with a per-instance saturation rate μ = 17 req/s at λ = 200 req/s:

```
4 instances:  excess = 200/4 - 17  = 33 req/s per instance   → rapid queue growth
8 instances:  excess = 200/8 - 17  = 8 req/s per instance    → near saturation
12 instances: excess = 200/12 - 17 = -0.3 req/s per instance → balanced (sub-saturation)
```

At sub-saturation (excess ≤ 0): TTFT converges to its unloaded baseline and further scaling provides diminishing returns. Measure μ for your own scenario and workload.

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
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
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
