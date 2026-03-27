# H1: Arrival Burstiness Elevates TTFT and E2E Latency at Equal Throughput

**Status**: Confirmed
**Date**: 2026-03-24

## Hypothesis

> For workloads matched on (1) total input tokens/s, (2) total output tokens/s, and (3) prefix-hit
> patterns, a bursty arrival process (Gamma CV=3) produces significantly higher mean and p99 TTFT
> and E2E latencies than a smooth arrival process (Poisson, CV=1), due to queueing delay
> accumulation during transient bursts.

**Refuted if:** The difference in TTFT p99 between bursty and smooth conditions is less than 20%
at any utilization level ≥ 50%, OR if the difference is not monotonically increasing with
utilization.

## Theoretical Basis (Kingman's G/G/1 Approximation)

Kingman's formula for expected queue wait:

    E[W_q] ≈ ρ/(1-ρ) × (CV_a² + CV_s²)/2 × (1/μ)

Where ρ = utilization, CV_a = arrival process CV, CV_s = service time CV, μ = service rate.

| Condition | CV_a | Expected E[W_q] formula |
|-----------|------|------------------------|
| Smooth    | 1    | ρ/(1-ρ) × (1 + CV_s²)/2 × 1/μ |
| Bursty    | 3    | ρ/(1-ρ) × (9 + CV_s²)/2 × 1/μ |

**Ratio (bursty/smooth):**
- If CV_s = 1 (exponential service): ratio = (9+1)/(1+1) = **5.0×**
- If CV_s = 2 (realistic LLM variance):  ratio = (9+4)/(1+4) = **2.6×**
- If CV_s = 3 (high variance):           ratio = (9+9)/(1+9) = **1.8×**

The theory predicts that even with high service variability, bursty arrivals produce ≥1.8× higher
queueing delay. The effect size MUST grow with utilization ρ (the ρ/(1-ρ) factor amplifies both).

## Control Variables

The following are held constant between smooth and bursty conditions:
- `aggregate_rate` (req/s) — same value
- Input token distribution: Gaussian(μ=512, σ=128, min=64, max=1024)
- Output token distribution: Exponential(μ=256)
- Client mix: 70% prefixed (prefix_group=system-prompt, prefix_length=256), 30% unprefixed
- `num_requests`: 2000 per run
- Model: qwen/qwen3-14b
- Instances: 1 (single instance, maps directly to G/G/1 queue theory)
- KV cache: default (1M blocks — prefix always cached, equalizing hit rates)
- Seeds: 42, 123, 456 (3 replicates per condition × rate)

## Independent Variable

- **Smooth:** `process: poisson` (CV=1)
- **Bursty:** `process: gamma`, `cv: 3.0` (CV=3, pronounced bursts)

## Dependent Variables

Primary:
- `ttft_mean_ms`, `ttft_p99_ms`
- `e2e_mean_ms`, `e2e_p99_ms`

Secondary:
- `scheduling_delay_p99_ms` (direct measure of queue wait)
- `responses_per_sec` (to verify actual throughput equality)
- `completed_requests` (to verify request conservation)

## Experiment Design

- **Rates tested:** 50, 150, 300 req/s (spanning low to potentially-saturating utilization)
- **Seeds:** 42, 123, 456 (enables paired t-test with 3 replicates per condition)
- **Total runs:** 3 rates × 2 conditions × 3 seeds = **18 runs**
- **Horizon:** 120 seconds simulated time (3× the nominal completion time even at lowest rate)
- **Analysis:** Paired t-test per rate level; relative effect size (bursty/smooth ratio);
  comparison against Kingman prediction; verification that ratio grows with utilization
