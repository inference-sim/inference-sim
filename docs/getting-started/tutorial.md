# Tutorial: Capacity Planning

This tutorial walks through a complete capacity planning exercise: determining how many inference instances you need to serve a target request rate while meeting latency SLOs.

**Scenario:** You're deploying Llama-3.1-70B-Instruct on H200 GPUs (FP8, vLLM, TP4). Your SLO is TTFT p99 < 500ms. You need to find the minimum number of instances for 40 requests/second.

!!! note "Set up the catalog and registry first"
    Every command below needs the catalog (`--catalog` or `BLIS_CATALOG`) and the registry
    (`--registry`). Clone both at their pinned releases —
    `git clone --branch 0.2.1 --depth 1 https://github.com/inference-sim/blis-catalog.git`
    and `git clone --branch v0.1.1 --depth 1 https://github.com/inference-sim/blis-registry.git`
    — and run `export BLIS_CATALOG=$PWD/blis-catalog` once. See [Quick Start](quickstart.md)
    and [Catalog compatibility](installation.md#catalog-compatibility).

## Step 0: A Scenario Big Enough for a Cluster

The scenario states how many nodes the deployment has, and `--num-instances` cannot exceed
the pool's rank capacity (pool nodes × gpus_per_node / (pp × tp × pcp)). The kernel's
single-node `llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml` allows only two instances, so this
tutorial uses the committed fixture `testdata/scenarios/llama-3.1-70b-instruct-h200-tp4-4node.yaml`:
the same model, engine settings and TP4 on four 8-GPU H200 nodes over 400G InfiniBand — up to
eight instances.

For other shapes, copy any scenario into a directory of your own, raise `cluster.nodes` and the
pool's `nodes`, and add a `cluster.fabric` (a multi-node cluster must name one, e.g.
`ib-400g`); `--scenarios` accepts any directory of scenario YAMLs.

## Step 1: Estimate Instance Capacity

Before scaling up, measure the throughput of a single instance under load. Run enough requests at a high arrival rate to saturate the instance — this reveals the maximum throughput with continuous batching:

```bash
./blis run \
  --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
  --rate 500 --num-requests 2000
```

Check the `responses_per_sec` value in the output. For this deployment with the default workload (512 input / 512 output tokens), a saturated instance handles roughly **11 requests/second**.

!!! note "Why measure at high load?"
    With continuous batching, throughput depends on batch size. At low arrival rates (e.g., `--rate 2`), requests trickle in one at a time and the instance processes only ~2 req/s — not because it's slow, but because there's nothing else to batch. At saturation, many concurrent requests share each decode step, amortizing overhead. Always measure capacity under load.

This means for 40 req/s, you need at minimum `ceil(40/11) = 4` instances. Let's verify with simulation.

!!! info "`--rate` is the total arrival rate"
    The `--rate` flag specifies the **total** arrival rate across the cluster, not per-instance. With `--rate 40 --num-instances 4`, each instance receives roughly 40/4 = 10 req/s (distributed by the routing policy).

## Step 2: Baseline — Single Instance at Low Load

```bash
./blis run \
  --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
  --rate 2 --num-requests 50
```

At 2 req/s (well below capacity), TTFT p99 is around 120ms and mean ITL around 14ms. Note these values — this is your best-case baseline that won't improve further with more instances.

## Step 3: Scale Up and Find the Saturation Point

Run simulations at increasing instance counts for 40 req/s:

```bash
# 2 instances (20 req/s per instance vs ~11 saturated capacity → overloaded)
./blis run \
  --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
  --num-instances 2 --rate 40 --num-requests 4000

# 4 instances (10 req/s per instance → just under capacity)
./blis run \
  --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
  --num-instances 4 --rate 40 --num-requests 4000

# 8 instances (5 req/s per instance → comfortable headroom)
./blis run \
  --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
  --num-instances 8 --rate 40 --num-requests 4000
```

Compare the cluster-level `ttft_p99_ms` and `itl_mean_ms` across runs:

- **2 instances:** TTFT p99 around 60 seconds — the per-instance arrival rate (20 req/s) far exceeds capacity (~11 req/s), so requests queue up and wait
- **4 instances:** TTFT p99 around 130ms — the queue no longer grows, but each instance runs large batches, so mean ITL (~28ms) is double the baseline
- **8 instances:** TTFT p99 around 120ms and mean ITL around 17ms — close to baseline

!!! tip "Understanding saturation"
    **Saturation means requests arrive faster than they can be served, so the queue grows continuously.** In queueing theory terms, the per-instance excess rate is `excess = λ/k - μ`, where λ is the total arrival rate (40 req/s), k is the instance count, and μ is the per-instance service rate (~11 req/s). When excess > 0, the queue grows at that rate and TTFT degrades.

    The improvement from 2→4 instances is dramatic (60s → 130ms) because the excess rate goes from ~9 req/s per instance to below zero. Past that point, more instances buy smaller batches — lower ITL — rather than lower TTFT.

## Step 4: Identify the Bottleneck Type

When TTFT is high, there are three possible causes:

1. **Queue saturation** — arrival rate exceeds service capacity → add instances
2. **Memory saturation** — KV cache preemptions degrade throughput → add KV blocks or reduce batch size
3. **Compute saturation** — step time dominates → reduce batch size or use chunked prefill

Check the output for clues:

- High `preemption_count` → memory saturation
- High `scheduling_delay_p99_ms` → queue saturation (requests waiting in the WaitQ)
- Low `preemption_count` + low `scheduling_delay` + high TTFT → compute saturation

## Step 5: Compare Routing Policies

With 4 instances at 40 req/s (near saturation), compare routing strategies:

```bash
# Round-robin (baseline)
./blis run \
  --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
  --num-instances 4 --rate 40 --num-requests 4000 \
  --routing-policy round-robin

# Weighted (default profile)
./blis run \
  --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
  --num-instances 4 --rate 40 --num-requests 4000 \
  --routing-policy weighted

# Least-loaded
./blis run \
  --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
  --num-instances 4 --rate 40 --num-requests 4000 \
  --routing-policy least-loaded
```

With uniform workloads (same prompt/output distribution), routing policies produce similar results because all instances are roughly equally loaded. Routing differentiation becomes meaningful with **heterogeneous workloads**. Prefix-aware scorers pull same-prefix requests onto the instance that already caches the prefix — which helps when there are many prefix groups, and hurts when there is only one:

```bash
./blis run \
  --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
  --num-instances 4 --rate 40 --num-requests 4000 \
  --routing-policy weighted \
  --routing-scorers "prefix-affinity:5,queue-depth:1" \
  --prefix-tokens 512
```

Here every request shares the same 512-token prefix, so the heavy prefix weight sends nearly everything to one instance and TTFT p99 climbs to minutes, while round-robin with the same `--prefix-tokens 512` stays near 140ms. Watch for this hot-spotting whenever prefix weights dominate load-balancing weights.

## Step 6: Evaluate with Fitness Scores

For automated comparison across many configurations, use fitness evaluation:

```bash
./blis run \
  --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
  --num-instances 8 --rate 40 --num-requests 4000 \
  --routing-policy weighted \
  --fitness-weights "p99_ttft:3,mean_e2e:1,throughput:2"
```

The fitness score is a weighted sum of normalized metrics — **higher is better**. Each latency metric is normalized to a [0, 1] range using `1/(1+x/1000)`, and throughput is normalized to [0, 1] using `throughput/max_throughput`. With weights `p99_ttft:3, mean_e2e:1, throughput:2`, the score is a weighted sum out of a theoretical maximum of 6.0 (the sum of all weights).

!!! warning "Fitness score normalization"
    The `1/(1+x/1000)` normalization compresses large raw differences into small score differences. A 38% TTFT improvement may appear as only an 8% fitness score difference. Always examine raw metrics (`ttft_p99_ms`, `e2e_mean_ms`, `responses_per_sec`) alongside fitness scores when making capacity decisions.

## Step 7: Validate Against Your SLO

Your SLO: TTFT p99 < 500ms at 40 req/s.

From the simulations above, 4 instances already meet the SLO (TTFT p99 ~130ms), and 8 instances also bring ITL back near baseline. Add 20-30% headroom for traffic spikes (real deployments see bursty traffic that exceeds the Poisson assumption).

## Key Takeaways

1. **Measure capacity under load** — run at high arrival rates (e.g., `--rate 500`) to measure saturated throughput; low-load measurements underestimate capacity due to small batch sizes
2. **Saturation is non-linear** — TTFT degrades super-linearly as you approach capacity. Scaling from 2→4 instances can cut TTFT p99 by more than 400x, not just 2x, because the per-instance excess rate drops below zero
3. **Check the bottleneck type** — preemption count, scheduling delay, and raw TTFT tell you whether to add instances, add memory, or tune batch size
4. **Routing matters for heterogeneous workloads** — with uniform traffic, routing policies produce similar results; with prefix-heavy or mixed-SLO workloads, prefix-aware and load-aware routing differ sharply — in either direction
5. **Deterministic replay** — use `--seed` to get identical results for A/B comparisons

## What's Next

- **[Routing Policies](../guide/routing.md)** — deep dive into scorer composition and signal freshness
- **[KV Cache & Memory](../guide/kv-cache.md)** — tune KV blocks, prefix caching, and chunked prefill
- **[Metrics & Results](../guide/results.md)** — understand all output fields and common patterns
