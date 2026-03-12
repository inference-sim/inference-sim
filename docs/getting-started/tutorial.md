# Tutorial: Capacity Planning

This tutorial walks through a complete capacity planning exercise: determining how many inference instances you need to serve a target request rate while meeting latency SLOs.

**Scenario:** You're deploying Qwen3 14B on H100 GPUs with TP=1. Your SLO is TTFT p99 < 500ms. You need to find the minimum number of instances for 200 requests/second.

## Step 1: Estimate Instance Capacity

Before scaling up, measure the throughput of a single instance under load. Run enough requests at a high arrival rate to saturate the instance — this reveals the maximum throughput with continuous batching:

```bash
./blis run \
  --model qwen/qwen3-14b \
  --rate 500 --num-requests 2000
```

Check the `responses_per_sec` value in the output. For Qwen3 14B / H100 / TP=1 with default workload (512 input / 512 output tokens), a saturated instance handles roughly **17 requests/second**.

!!! note "Why measure at high load?"
    With continuous batching, throughput depends on batch size. At low arrival rates (e.g., `--rate 2`), requests trickle in one at a time and the instance processes only ~2 req/s — not because it's slow, but because there's nothing else to batch. At saturation, many concurrent requests share each decode step, amortizing overhead and reaching ~17 req/s. Always measure capacity under load.

This means for 200 req/s, you need at minimum `ceil(200/17) = 12` instances. Let's verify with simulation.

!!! info "`--rate` is the total arrival rate"
    The `--rate` flag specifies the **total** arrival rate across the cluster, not per-instance. With `--rate 200 --num-instances 8`, each instance receives roughly 200/8 = 25 req/s (distributed by the routing policy).

## Step 2: Baseline — Single Instance at Low Load

```bash
./blis run \
  --model qwen/qwen3-14b \
  --rate 2 --num-requests 50
```

At 2 req/s (well below capacity), TTFT p99 is around 50ms. Note this value — this is your best-case baseline that won't improve further with more instances.

## Step 3: Scale Up and Find the Saturation Point

Run simulations at increasing instance counts for 200 req/s:

```bash
# 4 instances (50 req/s per instance vs ~17 saturated capacity → heavily overloaded)
./blis run \
  --model qwen/qwen3-14b \
  --num-instances 4 --rate 200 --num-requests 1000

# 8 instances (25 req/s per instance → still above capacity but batching helps)
./blis run \
  --model qwen/qwen3-14b \
  --num-instances 8 --rate 200 --num-requests 1000

# 12 instances (~17 req/s per instance → balanced, near baseline)
./blis run \
  --model qwen/qwen3-14b \
  --num-instances 12 --rate 200 --num-requests 1000
```

Compare the cluster-level `ttft_p99_ms` across runs:

- **4 instances:** TTFT p99 around 1,500ms — the per-instance arrival rate (50 req/s) far exceeds capacity (~17 req/s), so requests queue up and wait
- **8 instances:** TTFT p99 drops to around 60ms — per-instance arrival rate (25 req/s) is above single-request capacity but manageable with continuous batching
- **12 instances:** TTFT p99 around 54ms — near baseline, with comfortable headroom

!!! tip "Understanding saturation"
    **Saturation means requests arrive faster than they can be served, so the queue grows continuously.** In queueing theory terms, the per-instance excess rate is `excess = λ/k - μ`, where λ is the total arrival rate (200 req/s), k is the instance count, and μ is the per-instance service rate (~17 req/s). When excess > 0, the queue grows at that rate and TTFT degrades.

    The improvement from 4→8 instances is dramatic (1,500ms → 60ms) because the excess rate drops from ~33 req/s to ~8 req/s per instance — a much larger relative change than the 2x instance count would suggest.

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

With 8 instances at 200 req/s (near saturation), compare routing strategies:

```bash
# Round-robin (baseline)
./blis run \
  --model qwen/qwen3-14b \
  --num-instances 8 --rate 200 --num-requests 1000 \
  --routing-policy round-robin

# Weighted (default profile)
./blis run \
  --model qwen/qwen3-14b \
  --num-instances 8 --rate 200 --num-requests 1000 \
  --routing-policy weighted

# Least-loaded
./blis run \
  --model qwen/qwen3-14b \
  --num-instances 8 --rate 200 --num-requests 1000 \
  --routing-policy least-loaded
```

With uniform workloads (same prompt/output distribution), routing policies produce similar results because all instances are roughly equally loaded. Routing differentiation becomes meaningful with **heterogeneous workloads** — for example, prefix-heavy traffic where `prefix-affinity` routing concentrates same-prefix requests on the same instance for KV cache reuse:

```bash
./blis run \
  --model qwen/qwen3-14b \
  --num-instances 8 --rate 200 --num-requests 1000 \
  --routing-policy weighted \
  --routing-scorers "prefix-affinity:5,queue-depth:1" \
  --prefix-tokens 512
```

## Step 6: Evaluate with Fitness Scores

For automated comparison across many configurations, use fitness evaluation:

```bash
./blis run \
  --model qwen/qwen3-14b \
  --num-instances 12 --rate 200 --num-requests 1000 \
  --routing-policy weighted \
  --fitness-weights "p99_ttft:3,mean_e2e:1,throughput:2"
```

The fitness score is a weighted sum of normalized metrics — **higher is better**. Each latency metric is normalized to a [0, 1] range using `1/(1+x/1000)`, and throughput is normalized to [0, 1] using `throughput/max_throughput`. With weights `p99_ttft:3, mean_e2e:1, throughput:2`, the score is a weighted sum out of a theoretical maximum of 6.0 (the sum of all weights).

!!! warning "Fitness score normalization"
    The `1/(1+x/1000)` normalization compresses large raw differences into small score differences. A 38% TTFT improvement may appear as only an 8% fitness score difference. Always examine raw metrics (`ttft_p99_ms`, `e2e_mean_ms`, `responses_per_sec`) alongside fitness scores when making capacity decisions.

## Step 7: Validate Against Your SLO

Your SLO: TTFT p99 < 500ms at 200 req/s.

From the simulations above, 8 instances already meet the SLO (TTFT p99 ~60ms), and 12 instances provide comfortable headroom (~54ms). Add 20-30% headroom for traffic spikes (real deployments see bursty traffic that exceeds the Poisson assumption).

## Key Takeaways

1. **Measure capacity under load** — run at high arrival rates (e.g., `--rate 500`) to measure saturated throughput; low-load measurements underestimate capacity due to small batch sizes
2. **Saturation is non-linear** — TTFT degrades super-linearly as you approach capacity. Scaling from 4→8 instances can produce a 25x improvement, not just 2x, because the per-instance excess rate drops dramatically
3. **Check the bottleneck type** — preemption count, scheduling delay, and raw TTFT tell you whether to add instances, add memory, or tune batch size
4. **Routing matters for heterogeneous workloads** — with uniform traffic, routing policies produce similar results; with prefix-heavy or mixed-SLO workloads, prefix-affinity and load-aware routing provide meaningful differentiation
5. **Deterministic replay** — use `--seed` to get identical results for A/B comparisons

## What's Next

- **[Routing Policies](../guide/routing.md)** — deep dive into scorer composition and signal freshness
- **[KV Cache & Memory](../guide/kv-cache.md)** — tune KV blocks, prefix caching, and chunked prefill
- **[Metrics & Results](../guide/results.md)** — understand all output fields and common patterns
- **[Hypothesis Experimentation](../guide/experimentation.md)** — run rigorous experiments with the `/hypothesis-experiment` skill
