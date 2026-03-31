# Iteration 0 — Baseline Measurement

**Status:** [ ] Running | [ ] Complete
**Date:**
**Seeds:** 42, 123, 456

## Configuration
- Routing: `pa:4,qd:3` (no kv-util)
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

## Notes
- Set `aggregate_rate` in workload-mixed.yaml to 2x measured saturation throughput
- Record saturation throughput here: ______ req/s
