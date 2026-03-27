# H-Phase-Structure: Latency Model Phase Linearity Validation

**Status**: Confirmed
**Date**: 2026-02-21

## Hypothesis

> Prefill cost is proportional to prompt token count and decode cost is proportional to generated token count. TTFT should be linear in input_tokens (R-squared > 0.95 for linear fit) with output held constant, and (E2E - TTFT) should be linear in output_tokens (R-squared > 0.95) with input held constant.

**Refuted if:** R-squared for either linear fit is below 0.90 across all 3 seeds under zero-queueing conditions (single instance, max-running-reqs=1, rate=0.01), indicating non-linearity in the latency model.
