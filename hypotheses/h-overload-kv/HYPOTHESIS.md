# H-Overload-KV: Combined Overload + KV Cache Pressure

**Status**: Confirmed with nuance
**Date**: 2026-02-22

## Hypothesis

> Under extreme overload (2x-10x saturation) combined with KV cache pressure, the simulator should maintain conservation (INV-1), not panic, and preemptions should increase gracefully — no livelock or silent data loss.

**Refuted if:** Any configuration produces a panic, conservation invariant violation, or silent data loss (requests disappearing from the accounting) under combined overload and KV pressure.
