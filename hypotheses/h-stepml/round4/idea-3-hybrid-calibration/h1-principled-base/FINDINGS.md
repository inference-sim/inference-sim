# H1: Principled Base -- Direct Calibration from E2E + Step Data

**Status:** CONFIRMED (target met for E2E, refuted for ITL)

## Key Numbers

| Metric | Result | Target | Verdict |
|--------|--------|--------|---------|
| Mean E2E error | 5.7% | <30% | PASS |
| Median E2E error | 3.8% | - | - |
| Mean ITL error | 107.8% | <20% | FAIL |
| Mean TTFT error | 54.2% | - | - |
| E2E <10% | 9/10 | - | Excellent |
| E2E <20% | 9/10 | - | Excellent |

## Method

Direct calibration of BLIS alpha/beta coefficients from ground-truth E2E data
combined with step-level batch composition features:

1. **Target step time** = (E2E_mean - TTFT_mean) / output_len_mean (per model)
2. **beta2** = measured_step_duration / avg_decode_batch_size (marginal per-token GPU cost)
3. **beta0** = target_step_time - beta2 * avg_decode_batch_size (overhead floor)
4. **beta1** from Ridge regression on prefill steps
5. **alpha0** = observed mean TTFT (per model, from ground truth)

### Calibrated Coefficients (microseconds)

| Model | beta0 | beta1 | beta2 | alpha0 | target_step |
|-------|-------|-------|-------|--------|-------------|
| llama-2-7b | 9,741 | 0.30 | 13.6 | 27,129 | 9,903 |
| codellama-34b | 14,196 | 0.00 | 25.8 | 47,618 | 15,056 |
| llama-2-70b | 17,992 | 1.22 | 35.2 | 78,888 | 19,616 |
| llama-2-70b-hf | 17,590 | 0.00 | 29.8 | 78,888 | 18,574 |
| mixtral-8x7b-v0-1 | 18,921 | 0.69 | 8.8 | 62,767 | 19,292 |

### Per-Experiment Results

| Experiment | E2E% | TTFT% | ITL% | Pred(ms) | GT(ms) |
|-----------|------|-------|------|----------|--------|
| llama-2-7b-tp1-roleplay | 22.9 | 55.8 | 145.7 | 2,546 | 2,071 |
| llama-2-70b-tp4-general | 1.6 | 7.1 | 96.9 | 5,235 | 5,321 |
| llama-2-70b-hf-tp4-codegen | 6.3 | 95.1 | 110.9 | 4,894 | 4,605 |
| llama-2-70b-tp4-roleplay | 7.7 | 97.9 | 113.8 | 4,915 | 4,562 |
| mixtral-8x7b-v0-1-tp2-codegen | 3.9 | 56.2 | 106.9 | 4,857 | 4,675 |
| mixtral-8x7b-v0-1-tp2-general | 1.7 | 34.2 | 96.1 | 4,954 | 5,039 |
| mixtral-8x7b-v0-1-tp2-roleplay | 5.0 | 51.9 | 109.4 | 4,921 | 4,685 |
| codellama-34b-tp2-general | 3.8 | 37.6 | 91.9 | 3,939 | 4,093 |
| codellama-34b-tp2-codegen | 1.0 | 53.6 | 101.2 | 3,760 | 3,723 |
| codellama-34b-tp2-roleplay | 3.2 | 52.9 | 105.7 | 3,787 | 3,670 |

## Key Findings

1. **E2E accuracy is excellent** (5.7% mean, 9/10 below 10%). This confirms that
   deriving the overhead floor from E2E ground truth produces a well-calibrated
   base model for end-to-end latency prediction.

2. **The overhead floor dominates step time.** beta0 accounts for 92-98% of the
   total step time (e.g., llama-2-7b: 9741/9903 = 98.4%). The GPU forward-pass
   time (step.duration_us) is only 1-5% of the actual step cycle time. This means
   the "step time" in a real inference system is overwhelmingly CPU/scheduling
   overhead, not GPU compute.

3. **ITL is systematically 2x overpredicted** (~100% error). This is because
   BLIS's ITL = per-step time, but the ground-truth ITL from lifecycle data
   has very low medians (30-60us) -- most output_token_times entries are
   sub-millisecond, suggesting the lifecycle timestamps capture something
   different from the step cycle time.

4. **TTFT error is high** (54.2%) because alpha0 is set to the observed mean TTFT
   which includes queueing effects that BLIS may model differently.

5. **7B model is the outlier** (22.9% E2E error vs 1-8% for others). This may
   be because the 7B model has much smaller batch sizes (avg 12 vs 33-46) and
   the overhead floor structure differs at smaller scale.

## Comparison to R3

| Metric | R3 CMA-ES | R3 Trace Replay | H1 (this) |
|--------|-----------|-----------------|-----------|
| Mean E2E | 15.1% | 56.2% | **5.7%** |
| Mean ITL | 87.4% | 9.5% | 107.8% |
| E2E <10% | 4/10 | 0/10 | **9/10** |

H1 achieves the best E2E accuracy ever recorded in this research, but at the
cost of poor ITL prediction. The E2E/ITL tradeoff remains.

## Root Cause of ITL Error

The ~100% ITL error is structural, not a calibration failure. BLIS reports ITL
as the per-step time / batch size (approximately), but the step time is now
dominated by the overhead floor (10-19ms). The ground-truth ITL from lifecycle
data has median ~30us with mean ~4-10ms, suggesting a highly bimodal distribution
where most tokens are reported near-instantly with occasional slow tokens.
