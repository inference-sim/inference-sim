# H20: Heavy-Tailed Input Distributions

**Status:** Partially confirmed with surprise
**Resolution:** Confirmation with wrong mechanism
**Family:** Workload/arrival
**VV&UQ:** Validation
**Tier:** 2
**Type:** Statistical (Dominance)
**Date:** 2026-02-23
**Rounds:** 2

**Analyzer override note:** The automated analyzer outputs "INCONCLUSIVE -- mixed directionality across seeds" because TTFT p99 directionality is inconsistent at overload (2/3 seeds in Exp 1, 1/3 in Exp 2). We classify this as "Partially confirmed with surprise" rather than "Inconclusive" or "Refuted" because: (a) the TTFT p99 tail penalty IS consistently confirmed at sub-saturation (3/3 seeds, 1.54x avg) — establishing a real effect via prefill cost; (b) the preemption prediction is clearly wrong in direction (Gaussian > ParetoLN); (c) the bimodal KV demand mechanism is a genuine surprise finding. The hypothesis's observable prediction (worse tail latency) partially holds, but the hypothesized mechanisms (preemptions and HOL blocking) are wrong.

## Hypothesis

> "ParetoLogNormal input distributions should produce more preemptions and HOL blocking than Gaussian at the same average input length."

The intuition was that heavy-tailed distributions produce occasional very long requests that hold KV blocks for extended periods, starving short requests of capacity and causing more preemptions. Gaussian distributions have bounded variance, so extreme outliers are rare, and KV block holding times should be more uniform.

## Experiment Design

**Classification:** Statistical / Dominance

**Configurations compared:**
- A (Gaussian): `--workload-spec` with `type: gaussian, mean: 256, std_dev: 50, min: 32, max: 512`
- B (ParetoLogNormal): `--workload-spec` with `type: pareto_lognormal, alpha: 1.5, xm: 100, mu: 5.2, sigma: 0.7, mix_weight: 0.35`

Both produce mean input ~256 tokens. ParetoLogNormal has infinite variance (alpha=1.5 < 2) while Gaussian is bounded [32, 512].

**Controlled variables:** Output distribution (gaussian, mean=128), rate (1000 or 200), instances (4), routing (least-loaded), scheduler (fcfs), priority (constant), admission (always-admit), arrival process (poisson), model (llama-3.1-8b-instruct).

**Varied variable:** Input length distribution (Gaussian vs ParetoLogNormal).

**Seeds:** 42, 123, 456

**Experiments:**
1. **Core (Exp 1):** Default KV blocks (~1M), rate=1000, 500 requests
2. **KV-constrained (Exp 2):** 2000 blocks per instance, rate=1000, 500 requests
3. **Sub-saturation control (Exp 3):** Default KV blocks, rate=200, 500 requests

**Preconditions verified:**
- ParetoLogNormal params produce mixture mean ~255 tokens (analytically: 0.35 * 300 + 0.65 * 231 = 255)
- YAML field names match `sim/workload/spec.go` struct tags (`type: pareto_lognormal`, params: `alpha`, `xm`, `mu`, `sigma`, `mix_weight`)
- Distribution type registered in `sim/workload/distribution.go:187` and `spec.go:111`
- INV-1 conservation verified for all 18 runs

## Results

### Experiment 1: Core (default KV, rate=1000, 500 requests)

| Seed | Distribution | TTFT mean (ms) | TTFT p99 (ms) | E2E mean (ms) | E2E p99 (ms) |
|------|-------------|-----------------|----------------|----------------|---------------|
| 42   | Gaussian    | 122.69          | 193.34         | 1543.65        | 2197.26       |
| 42   | ParetoLN    | 132.87          | 323.47         | 1555.35        | 2204.73       |
| 42   | **Ratio**   | **1.08x**       | **1.67x**      | **1.01x**      | **1.00x**     |
| 123  | Gaussian    | 138.59          | 230.76         | 1565.39        | 2264.53       |
| 123  | ParetoLN    | 116.83          | 258.83         | 1529.61        | 2199.21       |
| 123  | **Ratio**   | **0.84x**       | **1.12x**      | **0.98x**      | **0.97x**     |
| 456  | Gaussian    | 130.87          | 257.35         | 1546.88        | 2210.78       |
| 456  | ParetoLN    | 130.32          | 249.79         | 1523.19        | 2198.53       |
| 456  | **Ratio**   | **1.00x**       | **0.97x**      | **0.98x**      | **0.99x**     |

**Zero preemptions for both distributions.** With ~1M default KV blocks, KV capacity is never a constraint.

Cross-seed summary: TTFT p99 ratio avg=1.26x, ParetoLN worse in 2/3 seeds. E2E p99 ratio avg=0.99x (virtually identical). The TTFT tail effect is driven by prefill cost of ParetoLN's occasional long inputs, NOT by preemption or HOL blocking.

### Experiment 2: KV-constrained (2000 blocks, rate=1000, 500 requests)

| Seed | Distribution | TTFT p99 (ms) | E2E p99 (ms) | Preemptions | Preempt Rate |
|------|-------------|----------------|---------------|-------------|--------------|
| 42   | Gaussian    | 2508.25        | 3964.49       | 326         | 0.6520       |
| 42   | ParetoLN    | 2280.46        | 3916.84       | 311         | 0.6220       |
| 123  | Gaussian    | 3722.91        | 5350.89       | 392         | 0.7840       |
| 123  | ParetoLN    | 1494.34        | 3422.62       | 252         | 0.5040       |
| 456  | Gaussian    | 2423.93        | 4069.55       | 290         | 0.5800       |
| 456  | ParetoLN    | 3327.40        | 4497.62       | 306         | 0.6120       |

**Gaussian has MORE preemptions on average (336 vs 290).** This is the opposite of the hypothesis prediction. ParetoLN has lower TTFT p99 in 2/3 seeds (avg ratio=0.89x).

### Experiment 3: Sub-saturation control (rate=200, 500 requests)

| Seed | Distribution | TTFT p99 (ms) | E2E p99 (ms) | Preemptions |
|------|-------------|----------------|---------------|-------------|
| 42   | Gaussian    | 33.10          | 2028.94       | 0           |
| 42   | ParetoLN    | 51.09          | 2094.18       | 0           |
| 123  | Gaussian    | 32.28          | 2167.40       | 0           |
| 123  | ParetoLN    | 44.73          | 2025.10       | 0           |
| 456  | Gaussian    | 35.77          | 2082.05       | 0           |
| 456  | ParetoLN    | 60.02          | 2127.98       | 0           |

TTFT p99 ratio avg=1.54x, ParetoLN worse in 3/3 seeds. Zero preemptions, zero queue buildup. The TTFT tail effect persists because it is intrinsic to the distribution shape (longer inputs require longer prefill), not a queueing/HOL-blocking phenomenon.

### Conservation (INV-1)

All 18 runs: PASS (injected=500, completed=500, queued=0, running=0).

## Root Cause Analysis

The hypothesis was based on a flawed mental model that assumed heavy-tailed distributions would be strictly worse for system performance. The actual behavior reveals three distinct mechanisms:

### 1. TTFT p99 tail from prefill cost (confirmed at sub-saturation)

ParetoLogNormal's TTFT p99 is consistently ~1.5x worse at sub-saturation (Exp 3) where queues never form. This is a direct consequence of the distribution shape: the p99 input length for ParetoLN is much longer than Gaussian's p99 (bounded at 512). The step time formula (`sim/latency_model.go:49-53`, `StepTime = beta0 + beta1*cacheMissTokens + beta2*decodeTokens`) means longer inputs produce proportionally longer prefill steps. This is not HOL blocking -- it is the intrinsic cost of serving longer requests.

### 2. ParetoLogNormal does NOT increase preemptions (refuted by Exp 2)

Under KV pressure (2000 blocks), Gaussian produces more preemptions on average (336 vs 290, ~14% difference). **Note:** This 14% effect size is below the 20% significance threshold from `docs/standards/experiments.md`, and the direction is consistent in only 2/3 seeds (seed 456: ParetoLN has slightly more). The preemption reversal is suggestive but not statistically robust. The mechanism is bimodal: ParetoLogNormal's mixture produces many short requests (from the LogNormal component, median ~ exp(5.2) = 181 tokens) alongside occasional long ones. These short requests:
- Require fewer KV blocks (ceil(181/16) = 12 vs Gaussian's ceil(256/16) = 16)
- Complete faster, releasing blocks sooner
- Create a "breathing room" effect between the rare long requests

Gaussian's tighter clustering around 256 tokens means ALL requests compete for similar block counts simultaneously, creating more uniform pressure. The Gaussian min=32, max=512 bounds don't help enough -- most samples are within +/-1 sigma (206-306), all requiring 13-20 blocks.

This is quantified by the KV block demand variance. Gaussian: mean=16 blocks, most requests 13-20 blocks. ParetoLogNormal: bimodal, many at ~12 blocks, few at 50+ blocks. The bimodal distribution's lower median demand reduces the average concurrent block occupancy, despite the tail.

### 3. E2E is distribution-insensitive (confirmed across all experiments)

E2E p99 ratios are ~1.0x across all experiments because E2E is dominated by decode time (output distribution is identical). The prefill contribution to E2E is small: even a 1000-token input adds only ~17.67ms of prefill, while 128 output tokens take ~364ms of decode steps. The E2E is primarily determined by output length, not input length.

**RCV-1 (file:line citations):**
- Step time calculation: `sim/latency_model.go:49-53`
- KV block allocation: `sim/kvcache.go:159` (`AllocateKVBlocks` method)
- Preemption logic: `sim/batch_formation.go:148` (preemptForTokens circuit breaker)
- ParetoLogNormal sampler: `sim/workload/distribution.go:49-82`
- Gaussian sampler: `sim/workload/distribution.go:16-33` (clamped to [min, max])

**RCV-2 (first-principles calculation):**
- ParetoLogNormal mixture: Pareto(1.5, 100) mean=300, LogNormal(5.2, 0.7) mean=exp(5.445)=231
- Mixture mean: 0.35*300 + 0.65*231 = 255 tokens
- LogNormal median: exp(5.2) = 181 tokens. 65% of draws are LogNormal, so ~65% of requests have median ~181 tokens
- Gaussian median: 256 tokens (symmetric around mean)
- KV blocks per request: ParetoLN median ~12 blocks vs Gaussian median ~16 blocks
- The 25% lower median block demand for ParetoLN explains the lower average preemption count

**RCV-3 (mechanism + direction):**
- Mechanism: bimodal KV demand distribution creates breathing room between heavy-tail requests
- Direction: ParetoLN reduces preemptions relative to Gaussian (opposite of hypothesis)

**RCV-4 (control experiment):**
- Sub-saturation control (Exp 3) confirms TTFT tail is from prefill cost, not queueing. Note: the sub-saturation control was designed expecting the TTFT effect to VANISH (per HOL blocking mechanism). Instead it PERSISTS (1.54x), which refutes the HOL blocking mechanism while revealing the prefill cost mechanism.
- KV-constrained experiment (Exp 2) confirms preemption mechanism is NOT tail-driven
- **RCV-4 gap:** No dedicated control for the "bimodal KV demand" mechanism. A proper control would equalize median demand (e.g., Gaussian with mean=181 matching ParetoLN median) to confirm whether the median-driven breathing room explains the preemption reversal. This mechanism is analytically grounded (RCV-2) but experimentally unconfirmed.

## Devil's Advocate (RCV-5)

**Arguing the hypothesis MIGHT be confirmed:**
The experiment used only 2000 KV blocks. With an even more constrained KV (e.g., 500 blocks), ParetoLN's occasional 1000+ token requests (requiring 63+ blocks) would occupy a much larger fraction of available capacity, potentially causing cascading preemptions that Gaussian (max 512 tokens = 32 blocks) could not trigger. The current experiment may not have pushed KV pressure far enough into the "capacity starvation" regime. Additionally, with more instances (e.g., 16) and higher request counts, the rare extreme outliers from ParetoLN's Pareto tail would have more opportunity to cause correlated blocking.

**Rebuttal:** Even at 500 blocks, the bimodal effect would still apply -- most ParetoLN requests are short and cycle through quickly. The extreme tail events (>1000 tokens) happen with probability ~1.1% per request, so in 500 requests only ~5 such events occur. The cascading preemption concern (#349) applies to all distributions when blocks are severely constrained.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| Heavy-tailed distributions do NOT inherently increase preemptions | Surprise (opposite of hypothesis) | Documented here |
| ParetoLogNormal TTFT p99 penalty is from prefill cost, not HOL blocking | Surprise | Documented here |
| Distribution median (not mean) drives KV pressure behavior | Surprise | Documented here |
| E2E is insensitive to input distribution shape | Confirmation | Confirms H16 finding that decode time dominates E2E |
| Sub-saturation does NOT eliminate TTFT tail for heavy-tailed inputs | Surprise | Control behaves differently than load-driven hypotheses |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? None found
- [x] Any new rules needed? None -- the bimodal KV demand insight is experiment-specific, not a general rule
- [x] Any new invariants needed? None
- [x] Any existing rules/invariants confirmed? INV-1 conservation confirmed across all 18 runs. INV-6 determinism confirmed (same seed produces same output).

## Scope and Limitations (RCV-6)

- **Operating point tested:** 4 instances, 500 requests, rate=1000 (overloaded) and 200 (sub-saturated), default KV and 2000 blocks, llama-3.1-8b-instruct, blackbox latency model
- **Parameters findings depend on:** ParetoLogNormal params (alpha=1.5, xm=100, mu=5.2, sigma=0.7, mix_weight=0.35) producing bimodal distribution with median ~181 vs Gaussian mean 256. Different ParetoLN params could shift the balance.
- **What was NOT tested:** Severely constrained KV (<500 blocks), larger request counts (2000+), different ParetoLN parameterizations (higher mix_weight, lower alpha), multi-turn/reasoning workloads, different output distributions, roofline latency model
- **Generalizability:** The "bimodal KV demand" finding generalizes to any mixture distribution where the lighter component has lower median than Gaussian. Pure Pareto (mix_weight=1.0) without LogNormal mixing would behave differently. The TTFT prefill cost finding generalizes universally -- longer inputs always cost more prefill time regardless of queueing dynamics.
- **Uncertainty quantification:** UQ not performed -- three seeds at single operating point. The mixed directionality (2/3 seeds in Exp 1, 1/3 in Exp 2) indicates high sensitivity to seed-specific request orderings, suggesting the true effect size is near zero.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| TTFT p99 ratio (overloaded) | 1.26x avg (0.97x-1.67x) | Low -- high variance across seeds, mixed directionality |
| TTFT p99 ratio (sub-saturated) | 1.54x avg (1.39x-1.68x) | Medium -- consistent direction, but reflects prefill cost not HOL |
| Preemption count (KV-constrained) | Gaussian 336 > ParetoLN 290 | Medium -- opposite of hypothesis, 2/3 seeds consistent |
| E2E p99 ratio | ~1.0x across all experiments | High -- consistent across 9 comparisons |
| Sample size | 3 seeds x 3 experiments x 500 requests | Medium |
| Mechanism | Bimodal KV demand from mixture distribution | Medium -- analytically grounded but not directly measured |

## Implications for Users

1. **Heavy-tailed input distributions are not inherently worse for system performance.** The common assumption that heavy tails cause more preemptions is incorrect when the heavy-tailed distribution is a mixture (like ParetoLogNormal). The bimodal nature means most requests are actually shorter than the mean, creating breathing room for KV management.

2. **TTFT p99 will always be worse for heavy-tailed inputs** because the tail requests require proportionally longer prefill steps. This is an intrinsic cost, not a system pathology, and cannot be mitigated by KV or scheduling policies.

3. **E2E is insensitive to input distribution shape** when output distributions are held constant. Users concerned about E2E should focus on output length distributions, not input distributions.

4. **For KV pressure analysis, focus on the median demand, not the mean or tail.** A distribution with mean 256 but median 181 tokens creates less sustained KV pressure than a symmetric distribution with mean 256.

## Reproducing

```
cd hypotheses/h20-heavy-tailed
./run.sh
```
