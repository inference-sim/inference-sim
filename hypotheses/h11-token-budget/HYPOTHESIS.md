# H11: Batch Formation Token Budget Throughput-Latency Tradeoff

**Status**: Confirmed with nuance
**Date**: 2026-02-22

## Hypothesis

> Batch formation with larger token budgets (--max-num-scheduled-tokens from 512 to 8192) should improve throughput but worsen ITL (inter-token latency), because more tokens per step increases step time while allowing more concurrent request processing.

**Refuted if:** Throughput does not increase monotonically with token budget, OR ITL mean and ITL p99 both decrease with larger budgets, across all 3 seeds.
