# Cluster Simulation

This guide covers running multi-instance BLIS simulations — the full pipeline from request arrival through the llm-d router, per-instance scheduling, and metrics aggregation. Admission, routing and flow control are made by the llm-d router embedded in the simulator; see [Routing with the llm-d Router](routing.md).

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
| `--num-instances 1` (default) | One instance behind the llm-d router: the router can only pick that instance, but its admission and flow control apply. With the default configuration it admits every request at once, and results equal a run without a router |
| `--num-instances N` (N > 1) | Every request passes through the llm-d router, which admits, queues (flow control) and routes it to an instance |

## The Pipeline

```
Arrival → gateway latency → llm-d router → dispatch → Instance WaitQueue → Batch Formation → Step → Completion
                                 ↓                                              ↓
                              rejection                          KV Allocation + Latency Estimation
```

1. A request arrives at the cluster.
2. After `--gateway-latency` microseconds (default 0) the llm-d router sees it, as the
   Endpoint Picker would see a request Envoy hands it.
3. The router runs its admission, flow control (when the `flowControl` feature gate is on) and
   scheduling profiles, and either picks an endpoint or rejects the request. A rejection lands
   in exactly one INV-1 bucket; see [Admission Control](admission.md).
4. The picked instance enqueues the request; its own scheduler and batch formation take over.

| Stage | Controls | Configured by |
|-------|----------|---------------|
| **Admission, flow control, routing** | Whether the request is admitted, how long it waits, which instance receives it | The llm-d router's inputs: the scenario's `router:` block and `pools[].labels`; see [the router's inputs](routing.md#the-routers-inputs) |
| **Scheduling** | What order within the instance | `--scheduler`, the policy bundle's `priority` section |
| **Batch Formation** | Which requests form the next batch | scenario `max_num_seqs`, `max_num_batched_tokens` |

With no router inputs, the router uses its built-in default configuration over every
simulated pod.

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
pool. Choose the topology with `--prefill-instances` and `--decode-instances` (whose sum must not exceed `--num-instances`);
each must not exceed its pool's rank capacity, `nodes × gpus_per_node / (pp × tp × pcp)`. The KV
handoff is priced by the kernel's `PDTransferTime` over the scenario's fabric between the two
instances' placements.

Which requests are disaggregated is the router's decision: give it an EndpointPickerConfig with
the `disagg-profile-handler`, a P/D decider (`always-disagg-pd-decider` or
`prefix-based-pd-decider`) and `prefill` and `decode` scheduling profiles. BLIS labels prefill and
decode pods with `llm-d.ai/role`, so the `prefill-filter` and `decode-filter` plugins select them.
`examples/llm-d/pd.yaml` disaggregates every request. This repository's
`testdata/scenarios/glm-5-h200-3p1d-ib.yaml` is a 3P1D example whose `router:` block names such a
configuration:

```bash
./blis run --scenario glm-5-h200-3p1d-ib.yaml --scenarios testdata/scenarios \
  --registry $PWD/blis-registry \
  --num-instances 4 --prefill-instances 3 --decode-instances 1 \
  --rate 10 --num-requests 100
```

Refused: a P/D topology over a colocated scenario; a disaggregated scenario without
`--prefill-instances`/`--decode-instances`; prefill and decode pools that differ in block size,
`dp` or draft configuration; shared (`--prefill-decode-instances`) or encode instances.
KV offload works with P/D: each pool sizes the same tiers by its own bytes per block and prices
them with its own kernel. The handoff follows NIXL: the decode side pulls only what its prefix
cache lacks, the prefill side holds its KV until the pull completes, and with
`--pd-transfer-contention` concurrent handoffs share the NIC and NVLink links they cross, which
matters when a shared link (e.g. the decode node's NIC in xP1D) runs close to saturation. See
[Latency models](latency-models.md#prefilldecode-disaggregation).

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

Admission and flow control are the llm-d router's; see [Admission Control](admission.md).

## Gateway Latency

`--gateway-latency` (µs, default 0) models the network and proxy hops between a request's
arrival and the router's first look at it. The router itself takes zero simulated time to
decide. `--epp-scrape-jitter` (µs, default 0) bounds a seeded phase offset of each endpoint's
metrics collection.

## Decision Tracing

Log every routing decision, with the scores the router's primary scheduling profile gave every endpoint:

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
  --num-instances 4 --rate 100 --num-requests 500 \
  --trace-level decisions --summarize-trace --counterfactual-k 3
```

The trace summary shows:
- **Target Distribution** — how many requests went to each instance
- **Mean/Max Regret** — the best alternative endpoint's score minus the chosen endpoint's score

`--counterfactual-k` ranks the top-k endpoints by those scores.

!!! info "Regret under the max-score picker"
    With `max-score-picker` the chosen endpoint is always the highest-scoring one, so regret is zero. Regret is non-zero only with pickers that may choose a lower-scoring endpoint (`random-picker`, `weighted-random-picker`).

## Event Ordering

The cluster uses `(timestamp, priority, seqID)` ordering for deterministic event processing:

- Cluster events at time T process before instance events at time T
- Same-time instance ties broken by lowest instance index
- This ensures determinism (INV-6) but means results differ from a simple M/M/k queueing model

## Work-Conserving Property

BLIS is work-conserving (INV-8): it never idles while requests wait. After every step's end (`driverStepEnd`), the engine driver steps the engine again, so waiting requests are scheduled at once. Real systems may have scheduling delays not modeled here.

## Further Reading

- [Cluster Architecture](../concepts/architecture.md) — internal mechanics of the shared-clock event loop
- [Routing with the llm-d Router](routing.md) — the router's inputs, supported plugins and the signals it sees
- [Metrics & Results](results.md) — understanding trace summaries and per-SLO metrics
