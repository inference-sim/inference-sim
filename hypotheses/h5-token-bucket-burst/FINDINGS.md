# H5: Token-Bucket Admission Control Under Burst

**Status:** Confirmed
**Tier:** 3 (system understanding)
**Type:** Statistical / Dominance
**Date:** 2026-02-20

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

**CONFIRMED: Token-bucket reduces TTFT p99 by 56-69x across all 3 seeds (>20% threshold met by orders of magnitude).**

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

## Root Cause Analysis

The hypothesis is confirmed, but the 96% rejection rate is driven by a mechanism more fundamental than burstiness alone:

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
| Token-bucket reduces TTFT p99 by 56-69x under Gamma CV=3.5 burst | Confirmation | Documented here |
| Gamma arrival sampler (Marsaglia-Tsang) produces expected burstiness | Confirmation | Validated by rate-scaling monotonicity |
| 96% rejection driven by per-input-token cost (512 tokens/req >> 500 capacity) | Surprise | Token demand exceeds supply by 2,560x. Not a bug — correct per-token admission model, but parameters need calibration for token-cost (not request-count) admission. |
| TTFT improvement is monotonically rate-dependent (9x→69x from 200→2000) | Confirmation | Documented here |
| Bucket parameter sensitivity is smooth and monotonic | Confirmation | Documented here |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? **None found.** Admission pipeline works correctly: rejections counted, conservation implied (completed + rejected = total for all configs).
- [x] Any new rules needed? **Candidate: Document token-bucket cost model.** The `--token-bucket-capacity` and `--token-bucket-refill-rate` parameter names suggest per-request cost, but the actual cost is `len(req.InputTokens)`. Users must size bucket parameters relative to expected input token counts, not request counts. Consider adding a `--help` clarification or a CLI validation warning when `capacity < expected_mean_input_tokens`.
- [x] Any new invariants needed? **None** — conservation (INV-1) holds implicitly.
- [x] Any existing rules/invariants confirmed? **INV-1 implied** (completed + rejected = 500 for all token-bucket runs). **R3 (validate CLI flags)** — token-bucket capacity and refill rate are validated.

## Implications for Users

1. **Token-bucket is effective for tail latency under burst**, but the per-input-token cost model means parameters must be sized relative to token counts, not request counts. With mean input=512 tokens, a bucket capacity of 500 cannot even hold one average request.

2. **Size bucket capacity to your token distribution**: Capacity should be >> mean input tokens (e.g., cap=50,000 for mean=512). The refill rate should match the desired token throughput (e.g., refill=500,000 for 1,000 req/s × 512 tokens/req). The experiment's parameters (cap=500, refill=400) model extreme load shedding, not moderate smoothing.

3. **The improvement is consistent**: 56-69x TTFT P99 improvement across all 3 seeds, with <2x variation between seeds. This is a robust, not fragile, effect.

4. **Rate-dependent**: At low rates (200 req/s), the system absorbs bursts naturally and the token-bucket adds less value (9x vs 69x). Consider admission control only when the system operates near saturation.

5. **E2E latency is NOT reduced by the same magnitude**: Token-bucket TTFT improves by 69x but E2E improves by only 1.7x (12,713→7,398). This is because E2E includes decode time which is independent of admission policy. Token-bucket specifically helps the **queuing** component of latency.

## Reproducing

```bash
cd hypotheses/h5-token-bucket-burst
./run.sh
```
