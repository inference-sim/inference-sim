# Iteration 4 — Tier Budget Batch Formation

**Status:** [ ] Running | [ ] Complete
**Date:**
**Seeds:** 42, 123, 456

## Hypothesis
H-main: TierBudgetBatchFormation with critical fraction f_c=0.50 and standard fraction
f_s=0.70 provides better critical-tier TTFT P99 than equal-share (f_c=0.333, f_s=0.50)
under mixed sustained+burst workload.

## Configuration
- **Treatment:** `--batch-formation tier-budget --tier-budget-critical-frac 0.50 --tier-budget-standard-frac 0.70`
- **Ablation:** `--batch-formation tier-budget --tier-budget-critical-frac 0.333 --tier-budget-standard-frac 0.50`
- Common: `pa:4,qd:3`, `priority-fcfs`, `tier-shed`, tiered-LRU (structural)

## Treatment Results (f_c=0.50, f_s=0.70)
| Seed | Critical TTFT P99 | Standard Goodput | Sheddable Goodput |
|------|------------------|-----------------|-------------------|
| 42   |                  |                 |                   |
| 123  |                  |                 |                   |
| 456  |                  |                 |                   |
| Mean |                  |                 |                   |

## Ablation Results (equal-share f_c=0.333, f_s=0.50)
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
- [ ] Critical TTFT P99 improved with higher critical fraction
- [ ] Standard tier not starved (goodput maintained)
- [ ] Sheddable tier absorbs budget reduction gracefully

## Notes
