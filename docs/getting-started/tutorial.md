# Tutorial: Capacity Planning

This tutorial walks through a complete capacity planning exercise: determining how many inference instances you need to serve a target request rate while meeting latency SLOs.

**Scenario:** You're deploying LLaMA 3.1 8B on H100 GPUs with TP=2. Your SLO is TTFT p99 < 500ms. You need to find the minimum number of instances for 500 requests/second.

## Step 1: Understand Instance Capacity

Before simulating, estimate the theoretical capacity of a single instance. BLIS uses beta coefficients to model step time:

```
step_time ≈ beta0 + beta1 × cache_miss_tokens + beta2 × decode_tokens
```

For LLaMA 3.1 8B / H100 / TP=2, the default beta coefficients are `[6910.42, 17.67, 2.84]` (from `defaults.yaml`), giving a step time of approximately 9-17ms depending on batch composition. With a typical workload (512 input / 512 output tokens), a single instance can process roughly **57 requests/second**.

This means for 500 req/s, you need at minimum `ceil(500/57) ≈ 9` instances. Let's verify with simulation.

## Step 2: Baseline — Single Instance

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --rate 50 --num-requests 200
```

At 50 req/s (well below the ~57 req/s capacity), TTFT should be low. Note the `ttft_p99_ms` value — this is your best-case baseline.

## Step 3: Scale Up and Find the Saturation Point

Run simulations at increasing instance counts for 500 req/s:

```bash
# 4 instances (~228 req/s capacity → heavily overloaded)
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --rate 500 --num-requests 2000

# 8 instances (~456 req/s capacity → near saturation)
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 8 --rate 500 --num-requests 2000

# 12 instances (~684 req/s capacity → comfortable headroom)
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 12 --rate 500 --num-requests 2000
```

Compare `ttft_p99_ms` across runs. You'll see:

- **4 instances:** TTFT p99 extremely high (queue growing without bound)
- **8 instances:** TTFT p99 elevated (close to saturation, `excess = λ/k - μ` is small but positive)
- **12 instances:** TTFT p99 near baseline (sufficient capacity)

!!! tip "Understanding DES saturation"
    In a discrete-event simulator, saturation manifests as **unbounded queue growth**: when the arrival rate exceeds the per-instance service rate, the WaitQ grows at `excess = λ/k - μ` requests per second, where λ is the total arrival rate, k is the instance count, and μ is the per-instance service rate. Unlike real systems with CPU load metrics, the DES signal for saturation is queue depth growth rate.

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

With 12 instances at 500 req/s, compare routing strategies:

```bash
# Round-robin (baseline)
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 12 --rate 500 --num-requests 2000 \
  --routing-policy round-robin

# Weighted (default profile)
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 12 --rate 500 --num-requests 2000 \
  --routing-policy weighted

# Least-loaded
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 12 --rate 500 --num-requests 2000 \
  --routing-policy least-loaded
```

For prefix-heavy workloads (like RAG with shared system prompts), try the prefix-affinity-dominant profile:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 12 --rate 500 --num-requests 2000 \
  --routing-policy weighted \
  --routing-scorers "prefix-affinity:5,queue-depth:1" \
  --prefix-tokens 512
```

## Step 6: Evaluate with Fitness Scores

For automated comparison across many configurations, use fitness evaluation:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 12 --rate 500 --num-requests 2000 \
  --routing-policy weighted \
  --fitness-weights "p99_ttft:3,mean_e2e:1,throughput:2"
```

!!! warning "Fitness score normalization"
    Fitness scores use `1/(1+x/1000)` normalization for latency metrics, which compresses large raw differences into small score differences. A 38% TTFT improvement may appear as only an 8% fitness score difference. Always examine raw metrics alongside fitness scores.

## Step 7: Validate Against Your SLO

Your SLO: TTFT p99 < 500ms at 500 req/s.

From the simulations above, find the minimum instance count where `ttft_p99_ms < 500`. That's your capacity plan. Add 20-30% headroom for traffic spikes (real deployments see bursty traffic that exceeds the Poisson assumption).

## Key Takeaways

1. **Compute capacity first** — estimate `1/step_time` per instance from beta coefficients
2. **Saturation is non-linear** — TTFT degrades super-linearly as you approach capacity. Scaling from 4→8 instances produces a 7x improvement, not 2x (queue growth rate drops faster than linear)
3. **Check the bottleneck type** — preemption count, scheduling delay, and raw TTFT tell you whether to add instances, add memory, or tune batch size
4. **Routing matters at scale** — routing policy choice can change TTFT p99 by 3x at high request rates
5. **Deterministic replay** — use `--seed` to get identical results for A/B comparisons

## What's Next

- **[Routing Policies](../guide/routing.md)** — deep dive into scorer composition and signal freshness
- **[KV Cache & Memory](../guide/kv-cache.md)** — tune KV blocks, prefix caching, and chunked prefill
- **[Interpreting Results](../guide/results.md)** — understand all output fields and common patterns
- **[Hypothesis Experimentation](../guide/experimentation.md)** — run rigorous experiments with the `/hypothesis-experiment` skill
