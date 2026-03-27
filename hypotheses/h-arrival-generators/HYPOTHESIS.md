# H-Arrival-Generators: Validate Arrival Sampler Distributions

**Status**: Confirmed with design limitation
**Date**: 2026-02-21

## Hypothesis

> For each arrival sampler (Poisson, Gamma CV=1.5, Gamma CV=3.5, Weibull CV=1.5, Weibull CV=3.5), generating 10K+ inter-arrival times should yield (a) sample mean within 5% of theoretical mean, (b) sample CV within 10% of theoretical CV, and (c) KS test p > 0.05 against the theoretical CDF.

**Refuted if:** Any sampler fails all three criteria (mean, CV, KS) across all 3 seeds, indicating a sampler implementation bug rather than a boundary condition.
