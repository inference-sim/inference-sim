# Iteration 1 — Joint Composition Validation

**Status:** [ ] Running | [ ] Complete
**Date:**
**Seeds:** 42, 123, 456

## Hypothesis
The compound strategy (SLO-aware routing + priority-fcfs + tier-shed admission)
produces reproducible results identical to the Iteration 0 baseline.

## Configuration
- Routing: `pa:4,qd:3` (weighted)
- Scheduler: `priority-fcfs`
- Admission: `tier-shed`
- Batch formation: `vllm` (LIFO)
- KV eviction: default (no tiered-LRU)

## Results
| Seed | Critical TTFT P99 | Standard Goodput | Sheddable Goodput |
|------|------------------|-----------------|-------------------|
| 42   |                  |                 |                   |
| 123  |                  |                 |                   |
| 456  |                  |                 |                   |
| Mean |                  |                 |                   |

## Comparison with Iter 0
| Metric               | Iter 0 Mean | Iter 1 Mean | Delta |
|----------------------|-------------|-------------|-------|
| Critical TTFT P99    |             |             |       |
| Standard Goodput     |             |             |       |
| Sheddable Goodput    |             |             |       |

## Findings
- [ ] Determinism confirmed (identical results across iterations for same seed)
- [ ] Joint composition behaves as expected

## Notes
