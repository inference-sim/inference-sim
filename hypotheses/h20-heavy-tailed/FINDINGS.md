# H20: Heavy-Tailed Input Distributions (ParetoLogNormal vs Gaussian)

**Status:** Refuted
**Resolution:** Refuted -- wrong mental model
**Family:** Workload/arrival
**VV&UQ:** Validation
**Tier:** 2
**Type:** Statistical (Dominance)
**Date:** 2026-02-23
**Rounds:** 2

## Hypothesis

> Heavy-tailed input distributions (ParetoLogNormal) should produce more preemptions and HOL blocking than Gaussian at the same mean input length (~256 tokens), because occasional very long requests hold KV blocks for extended periods, starving short requests.

## Experiment Design

**Classification:** Statistical/Dominance

**Configurations compared:**
- A (Gaussian): `--workload-spec` with `input_distribution: {type: gaussian, params: {mean: 256, std_dev: 50, min: 32, max: 512}}`, `--total-kv-blocks 2000`, `--block-size-in-tokens 16`
- B (ParetoLogNormal): `--workload-spec` with `input_distribution: {type: pareto_lognormal, params: {alpha: 1.5, xm: 50, mu: 5.5, sigma: 1.2, mix_weight: 0.70}}`, same KV/block settings

**Controlled variables:** Output distribution (Gaussian mean=128), arrival process (Poisson), rate (1000 req/s), instances (4), routing (least-loaded), scheduler (FCFS), model (llama-3.1-8b-instruct), block size (16 tokens)

**Varied variable:** Input token length distribution (Gaussian vs ParetoLogNormal)

**Seeds:** 42, 123, 456

**Preconditions verified:**
- Both distributions target mean ~256 input tokens
  - Gaussian: mean=256 (symmetric, clamped [32, 512])
  - ParetoLogNormal analytical mean: `0.70 * (1.5*50/0.5) + 0.30 * exp(5.5 + 1.44/2)` = `0.70*150 + 0.30*502.7` = 255.8 tokens
- KV blocks (2000) with block_size=16: `ceil(256/16) = 16` blocks per request at mean, `2000/(16*4) = ~31` concurrent requests per instance -- moderate pressure
- preflight_kv_check advisory passed (2000 > 4 * ceil(2000/16) = 500)
- Conservation (INV-1) verified for all 18 runs: PASS (Round 2: formula corrected to include `dropped_unservable` per `docs/standards/invariants.md`: `injected == completed + queued + running + dropped_unservable`)

**Reference experiment:** `hypotheses/h16-gamma-vs-poisson/run.sh` (ED-6)
- **Differences:** Input distribution changed (gaussian -> pareto_lognormal for treatment); arrival process is poisson for both (not gamma); `--total-kv-blocks 2000` and `--block-size-in-tokens 16` added; no `--summarize-trace` or `--trace-level`

## Results

### Experiment 1: Core (rate=1000, KV=2000, 500 requests)

| Seed | Distribution | TTFT mean (ms) | TTFT p99 (ms) | E2E mean (ms) | E2E p99 (ms) | Preemptions |
|------|-------------|----------------|---------------|---------------|--------------|-------------|
| 42 | Gaussian | 742.03 | 2508.25 | 2394.29 | 3964.49 | 326 |
| 42 | ParetoLN | 656.36 | 2075.47 | 2288.87 | 3790.53 | 294 |
| 42 | Ratio (P/G) | 0.88x | 0.83x | 0.96x | 0.96x | 0.90x |
| 123 | Gaussian | 866.74 | 3722.91 | 2556.82 | 5350.89 | 392 |
| 123 | ParetoLN | 569.14 | 2065.00 | 2204.91 | 3725.82 | 230 |
| 123 | Ratio (P/G) | 0.66x | 0.55x | 0.86x | 0.70x | 0.59x |
| 456 | Gaussian | 602.25 | 2423.93 | 2228.06 | 4069.55 | 290 |
| 456 | ParetoLN | 521.64 | 2616.59 | 2049.69 | 3943.82 | 190 |
| 456 | Ratio (P/G) | 0.87x | 1.08x | 0.92x | 0.97x | 0.66x |

**Summary:** ParetoLogNormal produces **fewer** preemptions in all 3 seeds (avg 238 vs 336, 0.71x) -- the primary predicted metric is **reversed** in every seed. TTFT p99 is lower for ParetoLN in 2/3 seeds (0.55x-0.83x), but marginally higher in seed 456 (1.08x). The preemption reversal (consistent across all seeds) is the primary refutation signal; the TTFT p99 pattern is mixed.

### Experiment 2: Sub-saturation control (rate=200, KV=2000, 500 requests)

| Seed | Distribution | TTFT mean (ms) | TTFT p99 (ms) | E2E mean (ms) | E2E p99 (ms) | Preemptions |
|------|-------------|----------------|---------------|---------------|--------------|-------------|
| 42 | Gaussian | 21.24 | 33.10 | 1333.98 | 2028.94 | 0 |
| 42 | ParetoLN | 24.20 | 93.01 | 1366.75 | 2116.96 | 0 |
| 123 | Gaussian | 21.10 | 32.28 | 1366.22 | 2167.40 | 0 |
| 123 | ParetoLN | 24.18 | 125.15 | 1323.06 | 2001.92 | 0 |
| 456 | Gaussian | 21.87 | 35.77 | 1363.33 | 2082.05 | 0 |
| 456 | ParetoLN | 22.47 | 76.26 | 1327.88 | 2052.26 | 0 |

**Summary:** Zero preemptions for both distributions (no KV pressure at sub-saturation). ParetoLN shows 2.1-3.9x higher TTFT p99 -- this is the **intrinsic prefill cost** of heavy-tailed inputs (long requests take longer to prefill), NOT HOL blocking. E2E p99 is within 5% (decode dominates).

### Experiment 3: KV-abundant control (rate=1000, KV=100000, 500 requests)

| Seed | Distribution | TTFT mean (ms) | TTFT p99 (ms) | E2E mean (ms) | E2E p99 (ms) | Preemptions |
|------|-------------|----------------|---------------|---------------|--------------|-------------|
| 42 | Gaussian | 122.69 | 193.34 | 1543.65 | 2197.26 | 0 |
| 42 | ParetoLN | 153.83 | 290.32 | 1571.51 | 2224.24 | 0 |
| 123 | Gaussian | 138.59 | 230.76 | 1565.39 | 2264.53 | 0 |
| 123 | ParetoLN | 146.50 | 325.22 | 1518.50 | 2166.15 | 0 |
| 456 | Gaussian | 130.87 | 257.35 | 1546.88 | 2210.78 | 0 |
| 456 | ParetoLN | 131.52 | 350.98 | 1517.25 | 2295.30 | 0 |

**Summary:** Zero preemptions for both (KV not a bottleneck). ParetoLN TTFT p99 is 1.36-1.50x higher -- **intrinsic prefill cost** from heavy-tailed inputs persists even without KV pressure. E2E p99 is within 4% across seeds.

## Root Cause Analysis

### Why ParetoLogNormal produces FEWER preemptions (reversed prediction)

The hypothesis assumed that heavy-tailed distributions produce occasional very long requests that hog KV blocks, starving other requests. The data shows the **opposite**: ParetoLogNormal consistently produces fewer preemptions than Gaussian across all 3 seeds (190-294 vs 290-392).

**Root cause: The distribution MEDIAN drives KV pressure, not the mean or tail.**

The ParetoLogNormal sampler (`sim/workload/distribution.go:59-82`) is a mixture distribution:
- **Pareto component (70% of samples):** shape alpha=1.5, scale xm=50. Pareto median = `xm * 2^(1/alpha)` = `50 * 2^(0.667)` = ~79 tokens. Most Pareto samples are SHORT (50-150 tokens).
- **LogNormal component (30% of samples):** mu=5.5, sigma=1.2. LogNormal median = `exp(mu)` = `exp(5.5)` = ~245 tokens.
- **Net effect:** ~70% of ParetoLogNormal requests need only `ceil(79/16) = 5` blocks, cycling through KV rapidly. The heavy tail produces occasional large requests, but these are RARE (only 30% * tail_fraction). The **majority of requests are short**, creating "breathing room" by releasing blocks quickly.

The Gaussian distribution (mean=256, std_dev=50, clamped [32, 512]) has median ≈ mean ≈ 256. Nearly ALL Gaussian requests need ~16 blocks each. There is no "breathing room" -- KV occupancy is uniformly high.

**Mechanism tracing through code:**

1. **Block allocation** (`sim/kvcache.go:159-175`): Each request needs `ceil(input_tokens / block_size)` blocks. ParetoLogNormal's median ~79 tokens needs 5 blocks; Gaussian's median ~256 tokens needs 16 blocks. At any instant, ParetoLogNormal's running requests occupy ~3.2x fewer blocks on average.

2. **Preemption trigger** (`sim/batch_formation.go:145-175`): `preemptForTokens()` is called when `AllocateKVBlocks()` fails at `batch_formation.go:147`. With fewer blocks occupied, ParetoLogNormal has more free blocks available, triggering fewer allocation failures.

3. **Block release** (`sim/kvcache.go:361-376`): `ReleaseKVBlocks()` frees blocks in reverse order when a request completes. Short ParetoLogNormal requests complete prefill faster (fewer tokens) and release their (fewer) blocks sooner, maintaining higher free-block counts.

4. **The key is instantaneous KV occupancy, not peak demand.** While ParetoLogNormal occasionally produces a request needing 100+ blocks, the 70% of requests needing only 5 blocks keep total occupancy LOW. The steady-state free-block count is higher for ParetoLogNormal than Gaussian.

### Why TTFT p99 shows different patterns at different operating points

**Exp 1 (KV-constrained, overloaded):** ParetoLN has lower TTFT p99 in 2/3 seeds because fewer preemptions means less queue re-cycling. Seed 456 shows ParetoLN 1.08x higher -- the stochastic nature of the tail means some seeds produce more clustered long requests.

**Exp 2 (sub-saturation):** ParetoLN has 2.1-3.9x higher TTFT p99. With zero queue pressure, TTFT is dominated by **prefill computation time**. Step time = `beta0 + beta1 * cacheMissTokens + beta2 * decodeTokens` (`sim/latency_model.go`). ParetoLogNormal p99 input length is much higher than Gaussian p99 input length (heavy tail), so the p99 prefill step takes longer. This is **intrinsic prefill cost**, not HOL blocking.

**Exp 3 (KV-abundant, overloaded):** ParetoLN has 1.36-1.50x higher TTFT p99. Zero preemptions, but batch step time is longer when a heavy-tailed request enters the batch because `beta1 * cacheMissTokens` is larger. The 42% premium is the pure prefill cost of the tail.

### First-principles verification of median calculation (RCV-2)

Pareto median = `xm * 2^(1/alpha)` = `50 * 2^(1/1.5)` = `50 * 2^0.6667`:
- `2^0.6667 = exp(0.6667 * ln(2)) = exp(0.4621) = 1.587`
- Pareto median = 50 * 1.587 = **79.4 tokens** -> ceil(79.4/16) = **5 blocks**

Gaussian median = mean = 256 tokens -> ceil(256/16) = **16 blocks**

Block occupancy ratio at median: 16/5 = **3.2x higher for Gaussian**. With 2000 blocks / 4 instances = 500 blocks per instance:
- Gaussian: 500/16 = ~31 concurrent requests at median
- ParetoLogNormal: 500/5 = ~100 concurrent requests at median (for Pareto-drawn requests)

This ~3.2x higher occupancy for Gaussian explains why it triggers more preemptions.

## Devil's Advocate (RCV-5)

**If this is "Refuted," argue why it might be Confirmed:**
The hypothesis could hold with different ParetoLogNormal parameters. If mix_weight were 0.30 instead of 0.70 (making LogNormal the majority component with median ~245), the median would be closer to Gaussian, and the tail requests (from Pareto or LogNormal tail) might dominate KV pressure. Additionally, with much tighter KV constraints (e.g., 800 blocks), the ParetoLogNormal's occasional 1000+ token requests might cause cascading preemptions (#349) that overwhelm the short-request breathing room.

**If this is "Refuted," argue why the mental model might not be entirely wrong:**
The intuition that "long requests hog KV blocks" IS mechanistically correct -- a single 1000-token request does hold 63 blocks vs 16 for a 256-token request. The error is in **frequency accounting**: the heavy tail is RARE (perhaps 5-10% of requests exceed 500 tokens), while the short-request majority (70% < 100 tokens) dominates the instantaneous block occupancy. The mental model was correct about the per-request effect but wrong about the aggregate steady-state effect.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| ParetoLN produces FEWER preemptions than Gaussian (reversed prediction) | Refutation -- wrong mental model | Documented here |
| Distribution MEDIAN drives KV pressure, not mean or tail | Surprise | Documented here |
| ParetoLN TTFT p99 is intrinsically higher at sub-saturation (pure prefill cost) | Confirmation (partial) | Documented here |
| KV-abundant control eliminates preemptions for both distributions | Confirmation | Documented here |
| INV-1 conservation holds across all 18 runs (Round 2: formula corrected to include dropped_unservable) | Confirmation | Documented here |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? None found
- [x] Any new rules needed? None -- the median-drives-pressure insight is workload-specific, not a general rule
- [x] Any new invariants needed? None
- [x] Any existing rules/invariants confirmed? INV-1 (conservation) holds across all 18 runs. R19 (livelock protection) validated: 2000 blocks with ParetoLogNormal tail requests did not trigger livelock (DroppedUnservable=0, meaning no request exceeded total KV capacity).

## Scope and Limitations (RCV-6)

- **Operating point tested:** 4 instances, 500 requests, Poisson arrivals, FCFS scheduler, least-loaded routing, block_size=16, KV blocks in {2000, 100000}, rates in {200, 1000}
- **Parameters findings depend on:** ParetoLogNormal mix_weight=0.70 (Pareto-heavy). If mix_weight were lower (e.g., 0.30), LogNormal component dominates and the breathing-room effect weakens.
- **What was NOT tested:**
  - Different ParetoLogNormal parameter regimes (lower alpha for heavier tails, different mix_weight)
  - Very tight KV constraints (800 blocks) where tail requests might cause cascading preemptions
  - Multi-turn workloads (context accumulation would shift the effective input distribution)
  - Other routing policies (round-robin might interact differently with distribution shape)
- **Generalizability:** The finding that median (not mean) drives KV pressure generalizes to any mixture distribution where one component has much lower median than the other. It does NOT generalize to pure heavy-tailed distributions (ParetoLogNormal with mix_weight=1.0 has no LogNormal component).
- **Uncertainty quantification:** UQ not performed -- single operating point per experiment. The effect direction (ParetoLN fewer preemptions) is consistent across all 3 seeds, but the magnitude varies: preemption ratio ranges from 0.59x to 0.90x (seed-dependent). The 3-seed sample is too small for confidence intervals.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| Preemption count ratio (ParetoLN/Gaussian) | 0.71x avg (0.59x-0.90x) | High -- consistent direction across all 3 seeds |
| TTFT p99 ratio at overload (Exp 1) | 0.82x avg (0.55x-1.08x) | Medium -- 2/3 seeds favor ParetoLN, seed 456 marginal (1.08x). TTFT p99 is NOT the primary refutation metric; preemption count (reversed in ALL seeds) is. |
| TTFT p99 ratio at sub-saturation (Exp 2) | 2.94x avg (2.13x-3.88x) | High -- intrinsic prefill cost, consistent direction |
| KV-abundant control (Exp 3) preemptions | 0 for both | High -- confirms KV mechanism |
| Sample size | 3 seeds x 3 experiments x 2 distributions = 18 runs | Adequate for directionality |
| Mechanism | Distribution median drives instantaneous KV occupancy | High -- first-principles calculation confirms, KV-abundant control validates |

## Implications for Users

1. **Do not assume heavy-tailed distributions are worse for KV pressure.** ParetoLogNormal with a Pareto-heavy mix (mix_weight >= 0.5) actually produces FEWER preemptions than Gaussian at the same mean, because the majority of samples are short.

2. **The key metric for KV capacity planning is the distribution median, not the mean.** Two distributions with mean=256 can have very different KV pressure profiles if their medians differ. ParetoLogNormal (median ~79) is KV-friendlier than Gaussian (median ~256) despite having the same mean.

3. **Heavy-tailed distributions DO increase TTFT p99 at sub-saturation.** Even without KV pressure or queueing, the p99 prefill time is longer because p99 input length is longer. This is an intrinsic property of the distribution shape, not a system behavior.

4. **ParetoLogNormal parameters matter significantly.** The mix_weight parameter controls whether Pareto (short, high-frequency) or LogNormal (medium, lower-frequency) dominates. At mix_weight=0.70 (tested), Pareto dominates and KV pressure is low. At lower mix_weight, results may differ.

## Reproducing

```
cd hypotheses/h20-heavy-tailed
./run.sh
```
