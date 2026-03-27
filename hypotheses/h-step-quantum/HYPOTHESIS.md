# H-Step-Quantum: Step-Time Quantum vs DES-to-M/M/1 Wait-Time Divergence

**Status**: Refuted
**Date**: 2026-02-23

## Hypothesis

> Reducing the DES step-time quantum (by scaling beta coefficients) should proportionally reduce the DES-to-M/M/1 mean wait time divergence. Specifically: at rho=0.7, the W_q error (currently ~60% with ~6.9ms steps) should scale linearly with step_time / mean_service_time, approaching 0% as step time approaches 0.

**Refuted if:** Reducing beta coefficients by 10x and 100x does not reduce the W_q divergence proportionally, or the divergence increases instead of decreasing, indicating the step-time quantum is not the primary source of DES-to-M/M/1 divergence.
