# Iteration 3 — SLO-Aware Tiered LRU KV Eviction

**Status:** [ ] Running | [ ] Complete
**Date:**
**Seeds:** 42, 123, 456

## Hypothesis
H-main: Tiered LRU eviction (critical blocks evicted last) preserves critical-tier
KV cache hit rate under pressure, reducing critical TTFT P99 vs Iteration 2 treatment
(which uses default eviction).

## Configuration
- **Treatment:** Build with tiered-LRU (structural, always active) + `slo-priority-preemption`
- **Ablation:** Iteration 2 treatment results (no tiered-LRU)
- Common: `pa:4,qd:3`, `priority-fcfs`, `tier-shed`
- Note: Tiered LRU has no CLI flag; it is structural in this build.

## Treatment Results (Tiered LRU)
| Seed | Critical TTFT P99 | Standard Goodput | Sheddable Goodput | KV Hit Rate (Critical) |
|------|------------------|-----------------|-------------------|----------------------|
| 42   |                  |                 |                   |                      |
| 123  |                  |                 |                   |                      |
| 456  |                  |                 |                   |                      |
| Mean |                  |                 |                   |                      |

## Comparison with Iter 2 Treatment
| Metric                 | Iter 3 | Iter 2 Treatment | Delta  | Pass? |
|------------------------|--------|------------------|--------|-------|
| Critical TTFT P99      |        |                  |        |       |
| Standard Goodput       |        |                  |        |       |
| Sheddable Goodput      |        |                  |        |       |
| KV Hit Rate (Critical) |        |                  |        |       |

## Findings
- [ ] Critical KV cache hit rate improved
- [ ] Critical TTFT P99 improved
- [ ] Sheddable tier KV eviction rate increased (expected)

## Notes
