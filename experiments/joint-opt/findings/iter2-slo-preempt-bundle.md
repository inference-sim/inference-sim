# Iteration 2 — SLO-Priority Preemption Ordering

**Status:** [ ] Running | [ ] Complete
**Date:**
**Seeds:** 42, 123, 456

## Hypothesis
H-main: SLO-priority preemption ordering (`slo-priority-preemption`) reduces critical
TTFT P99 compared to LIFO (`vllm`) under mixed sustained+burst workload, without
degrading aggregate goodput by more than 5%.

## Configuration
- **Treatment:** `--batch-formation slo-priority-preemption`
- **Ablation:** `--batch-formation vllm` (LIFO baseline)
- Common: `pa:4,qd:3`, `priority-fcfs`, `tier-shed`

## Treatment Results
| Seed | Critical TTFT P99 | Standard Goodput | Sheddable Goodput |
|------|------------------|-----------------|-------------------|
| 42   |                  |                 |                   |
| 123  |                  |                 |                   |
| 456  |                  |                 |                   |
| Mean |                  |                 |                   |

## Ablation Results (LIFO)
| Seed | Critical TTFT P99 | Standard Goodput | Sheddable Goodput |
|------|------------------|-----------------|-------------------|
| 42   |                  |                 |                   |
| 123  |                  |                 |                   |
| 456  |                  |                 |                   |
| Mean |                  |                 |                   |

## Comparison
| Metric               | Treatment | Ablation | Delta  | Pass? |
|----------------------|-----------|----------|--------|-------|
| Critical TTFT P99    |           |          |        |       |
| Standard Goodput     |           |          |        |       |
| Sheddable Goodput    |           |          |        |       |
| Aggregate Goodput    |           |          |        |       |

## Findings
- [ ] Critical TTFT P99 improved (treatment < ablation)
- [ ] Aggregate goodput degradation < 5%
- [ ] Sheddable tier absorbs preemption cost

## Notes
