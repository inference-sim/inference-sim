# H5: Token-Bucket Admission Control Under Burst

**Status:** Confirmed with nuance
**Resolution:** Confirmation with wrong mechanism. The directional prediction (token-bucket reduces TTFT) holds across all 3 seeds with 56-69x improvement. However, the hypothesized mechanism (burst smoothing — reject some during bursts, improve latency for the rest) is not plausible: the per-input-token cost model (`admission.go:45`, cost = `len(req.InputTokens)` ≈ 512 >> capacity 500) rejects 96% of traffic, making the improvement pure load shedding. At calibrated parameters (cap=100K >> mean input), rejection drops to 0.8-5% and TTFT improvement is <5%, showing the burst-smoothing mechanism does not produce meaningful results. The conceptual hypothesis exposed a design limitation: the cost model is underdocumented and users would not expect per-token admission costs.
**Family:** Robustness/failure-mode
**VV&UQ:** Validation
**Tier:** 3 (system understanding)
**Type:** Statistical / Dominance
**Date:** 2026-02-20
**Rounds:** 3 (Round 1: wrong root cause. Round 2: corrected token-cost math. Round 3: calibrated bucket disproved burst smoothing.)

## Hypothesis

> During traffic bursts (Gamma arrivals with high CV), accepting all requests floods the queues and increases tail latency. A token bucket that rejects excess requests should cap queue depth, trading some rejected requests for much better latency for admitted ones.

## Experiment Design

**Classification:** Statistical / Dominance — token-bucket TTFT p99 < always-admit TTFT p99 across all seeds.

**Configurations compared:**
- A: `--admission-policy always-admit` (baseline — accept all requests)
- B: `--admission-policy token-bucket --token-bucket-capacity 500 --token-bucket-refill-rate 400`

**Controlled variables:** model (llama-3.1-8b), instances (4), requests (500), routing (least-loaded), scheduler (fcfs), workload (Gamma CV=3.5 arrivals, exponential input mean=512, output mean=256)
**Varied variable:** Admission policy (always-admit vs token-bucket)
**Seeds:** 42, 123, 456
**Preconditions verified:** Gamma arrivals with CV=3.5 produce measurable burstiness (verified by rate-scaling experiment showing effect amplification)

## Results

### Experiment 1: Core Hypothesis (3 seeds)

| Seed | Policy       | TTFT Mean | TTFT P99 | E2E P99   | Rejected | Completed | Effect |
|------|-------------|----------:|---------:|----------:|---------:|----------:|--------|
| 42   | always-admit |    558.2  |  1,281.5 |  12,713.0 |        0 |       500 |        |
| 42   | token-bucket |     14.2  |     18.6 |   7,398.2 |      480 |        20 | **68.9x better P99** |
| 123  | always-admit |    594.5  |  1,147.3 |  13,672.9 |        0 |       500 |        |
| 123  | token-bucket |     14.4  |     18.3 |   6,512.0 |      481 |        19 | **62.8x better P99** |
| 456  | always-admit |    606.6  |  1,186.2 |  12,239.0 |        0 |       500 |        |
| 456  | token-bucket |     14.4  |     21.1 |  12,166.5 |      483 |        17 | **56.3x better P99** |

**Token-bucket reduces TTFT p99 by 56-69x across all 3 seeds — but via 96% load shedding, not burst smoothing (see Root Cause Analysis).**

### Experiment 2: Rate Scaling

| Rate | Always-Admit P99 | Token-Bucket P99 | Rejected | P99 Ratio |
|-----:|-----------------:|-----------------:|---------:|----------:|
|  200 |            165.3 |             18.0 |      459 |    **9.2x** |
|  500 |            538.4 |             17.4 |      471 |   **30.9x** |
| 1000 |            976.5 |             19.0 |      475 |   **51.4x** |
| 2000 |          1,281.5 |             18.6 |      480 |   **68.9x** |
| 3000 |          1,352.0 |             20.4 |      484 |   **66.3x** |

The effect amplifies with rate until saturation (~rate=2000). At rate=200, the system has enough headroom to absorb some bursts (9.2x effect). At rate=2000+, queue pressure is persistent (69x effect). This confirms ED-2: the effect is rate-dependent.

### Experiment 3: Token-Bucket Parameter Sensitivity

| Configuration             | TTFT P99 | E2E P99   | Rejected | Completed |
|--------------------------|--------:|---------:|--------:|--------:|
| always-admit (baseline)   | 1,281.5 | 12,713.0 |       0 |     500 |
| bucket (cap=100, ref=100) |    16.4 |  7,521.2 |     488 |      12 |
| bucket (cap=250, ref=200) |    15.8 |  7,538.1 |     489 |      11 |
| bucket (cap=500, ref=400) |    18.6 |  7,398.2 |     480 |      20 |
| bucket (cap=1000,ref=600) |    24.2 |  7,595.4 |     475 |      25 |
| bucket (cap=2000,ref=1000)|    30.0 | 10,738.9 |     467 |      33 |

Larger bucket capacity/refill admits more requests (12→33) at the cost of modestly worse TTFT P99 (16→30ms). The tradeoff is monotonic and smooth.

### Experiment 4: Calibrated Bucket (cap >> mean_input)

Addresses reviewer feedback (Opus 4.6): cap=500 < mean_input=512 is structural rejection, not burst smoothing. This experiment uses cap=100,000 (200x the mean input) with refill=600,000 to match demand.

| Seed | Policy | TTFT P99 | E2E P99 | Rejected | Completed | Effect |
|------|--------|--------:|---------:|--------:|---------:|--------|
| 42 | always-admit | 1,281.5 | 12,713.0 | 0 | 500 | |
| 42 | calibrated | 1,222.7 | 12,708.1 | 22 | 478 | **1.05x better P99** |
| 123 | always-admit | 1,147.3 | 13,672.9 | 0 | 500 | |
| 123 | calibrated | 1,145.6 | 13,676.0 | 4 | 496 | **1.00x — no effect** |
| 456 | always-admit | 1,186.2 | 12,239.0 | 0 | 500 | |
| 456 | calibrated | 1,145.1 | 12,582.0 | 26 | 474 | **1.04x better P99** |

**INCONCLUSIVE: With properly calibrated bucket (cap=100K, refill=600K), only 0.8-5.2% rejection and <5% TTFT P99 improvement.** The token-bucket at these parameters barely smooths Gamma CV=3.5 bursts.

**This reveals a fundamental tradeoff**: either you reject massively (96%) for massive TTFT improvement (69x), or you calibrate properly and get negligible improvement (<5%). There is **no practical sweet spot** under Gamma CV=3.5 where moderate rejection gives moderate improvement.

## Root Cause Analysis

The directional outcome (TTFT reduction) holds, but the burst-smoothing mechanism is refuted. The 96% rejection rate is driven by a mechanism more fundamental than burstiness:

### Critical finding: token cost is per-input-token, not per-request

The token bucket charges `len(req.InputTokens)` per request (`sim/admission.go:45`), NOT 1 token per request. With exponential input distribution mean=512, the average request costs **512 tokens** — which exceeds the bucket capacity of 500 tokens. This means:

- **Token demand rate**: 2,000 req/s × 512 tokens/req = **1,024,000 tokens/s**
- **Token supply rate**: 400 tokens/s refill
- **Supply/demand ratio**: 400 / 1,024,000 = **0.04%**

The bucket is starved by a factor of 2,560x. The 96% rejection rate is the **mathematically inevitable** steady-state for these parameters, independent of burstiness.

### Why 17-20 requests are admitted (not 0)

Despite the extreme demand/supply mismatch:
1. **Initial bucket (500 tokens)** admits the first request with input_length ≤ 500 (P(X ≤ 500) = 62% for Exp(512))
2. **Gamma quiet gaps** allow partial refill — during a 100ms gap, 40 tokens accumulate
3. **Short requests exist** — Exp(512) produces some requests with input_length in 10-50 range that slip through on tiny refills

### The burstiness amplifies but doesn't cause the rejection

1. **Gamma CV=3.5 produces extreme burstiness**: With shape=1/CV²=0.082, inter-arrival times cluster near zero (mode=0 for shape<1), with occasional long gaps.

2. **Admitted requests see empty queues**: The few requests admitted during quiet periods arrive at near-idle instances, producing TTFT≈14ms (bare prefill time, zero queuing delay).

3. **Always-admit requests queue up**: All 500 requests enter the system, building deep queues during bursts. Queue depth drives TTFT up to 1,281ms P99.

4. **Rate scaling confirms**: Even at rate=200 (where burstiness is less impactful), 459/500 requests are rejected. This is because the token demand per request (512) still dwarfs the refill rate (400/s), regardless of arrival pattern.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| **Directional prediction holds (56-69x TTFT improvement) but via load shedding, not burst smoothing** | Confirmation with wrong mechanism | The predicted direction holds, but the mechanism is 96% load shedding, not burst smoothing. Calibrated bucket (cap=100K) shows <5% improvement, confirming the mechanism is wrong. |
| 69x TTFT improvement is load shedding, not burst smoothing | Design limitation | Token-bucket at cap=500 rejects 96% of traffic. The improvement is trivially explained by near-empty queues for the 4% that get through. |
| Token-bucket cost is per-input-token (`admission.go:45`), not per-request | Design limitation | `--token-bucket-capacity` and `--token-bucket-refill-rate` parameter names suggest per-request cost. Users must size relative to input token counts. File issue to clarify CLI help text. |
| Gamma arrival sampler (Marsaglia-Tsang) produces expected burstiness | Confirmation | Validated by rate-scaling monotonicity |
| No practical sweet spot found under Gamma CV=3.5 with mean input ~512 | Design limitation | At tested parameters: either massive load shedding (96% reject → 69x better) or negligible effect (<5% reject → <5% better). The mechanism may produce measurable effects under lower-variance or shorter-request workloads, but this was not tested. |

## Devil's Advocate (RCV-5)

**If this is "Refuted," argue why it might be Confirmed:**
The calibrated bucket (cap=100K) showed a consistent 4% TTFT improvement in 2 of 3 seeds. This could be a real but tiny burst-smoothing effect masked by workload noise. Also, the experiment only tested Gamma CV=3.5 with exponential input mean=512 — lower CV (1.5-2.0) or shorter requests (mean=64) might produce a workable sweet spot.

**If this is "Confirmed," argue why it might be Refuted (the original direction):**
The 69x improvement is trivially explained by near-empty queues when 96% of traffic is rejected. A firewall that blocks all traffic also has great latency. The calibrated experiment shows that at practical parameters, the effect is within noise.

## Scope and Limitations (RCV-6)

- **Operating point tested:** Gamma CV=3.5, rate=2000, 500 requests, 4 instances, exponential input mean=512
- **Parameters findings depend on:** Per-input-token cost model (`admission.go:45`); mean input > bucket capacity for original params
- **What was NOT tested:** Lower CV (1.5-2.0), shorter requests (mean=64-128), Weibull arrivals, queue-depth-based admission, intermediate bucket sizes between 500 and 100K
- **Generalizability:** Refutation is specific to Gamma CV=3.5 with mean_input ≈ 512. May not apply at lower CV or shorter requests.
- **Uncertainty quantification:** Calibrated experiment: 0-5% improvement across 3 seeds. No formal CI computed.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| TTFT improvement (original) | 56-69x across 3 seeds | High — but from 96% load shedding |
| TTFT improvement (calibrated) | 0-5% across 3 seeds | Medium — within noise, no formal test |
| Rejection rate (original) | 96% (480-483/500) | High — mathematically inevitable |
| Mechanism | Per-input-token cost prevents burst smoothing | High — verified at `admission.go:45` |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? **None found.** Admission pipeline works correctly: rejections counted, conservation implied (completed + rejected = total for all configs).
- [x] Any new rules needed? **Candidate: Document token-bucket cost model.** The `--token-bucket-capacity` and `--token-bucket-refill-rate` parameter names suggest per-request cost, but the actual cost is `len(req.InputTokens)`. Users must size bucket parameters relative to expected input token counts, not request counts. Consider adding a `--help` clarification or a CLI validation warning when `capacity < expected_mean_input_tokens`.
- [x] Any new invariants needed? **None** — conservation (INV-1) holds implicitly.
- [x] Any existing rules/invariants confirmed? **INV-1 implied** (completed + rejected = 500 for all token-bucket runs). **R3 (validate CLI flags)** — token-bucket capacity and refill rate are validated.

## Implications for Users

1. **Token-bucket does NOT smooth bursts under Gamma CV=3.5.** At practical parameters (cap >> mean input), the bucket admits almost everything and provides negligible latency improvement. The hypothesis that admission control can smooth bursty traffic at high CV is wrong for this system — the queues build and drain too quickly for per-request token-level admission to help.

2. **The 69x result is load shedding, not admission control.** Rejecting 96% of traffic trivially improves latency for the 4% that pass. This is useful only if you genuinely want to serve 4% of traffic (e.g., protecting a degraded system from total overload). It is not a tuning knob for normal operation.

3. **Token-bucket cost model is per-input-token**, not per-request (`admission.go:45`). Parameters must be sized relative to token distributions. With mean input=512 tokens, `--token-bucket-capacity 500` cannot admit a single average request. This is a documentation gap.

4. **For burst smoothing under high CV, look elsewhere.** Rate limiting at the arrival layer (e.g., Poisson thinning before the token-bucket) or queue-depth-based admission (reject when queue exceeds threshold) may be more effective than token-bucket for Gamma CV>3. This experiment does not test those alternatives.

5. **The experiment's value is in the refutation.** Knowing that token-bucket burst smoothing doesn't work at CV=3.5 prevents users from deploying it in that regime expecting it to help.

## Reproducing

```bash
cd hypotheses/h5-token-bucket-burst
./run.sh
```
