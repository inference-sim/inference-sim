# H-Overload: 10x Overload Robustness

**Status**: Confirmed
**Date**: 2026-02-21

## Hypothesis

> Under stress condition of 10x the saturation rate, the system should exhibit defined overload behavior (queue growth with always-admit, or rejection with token-bucket) and NOT exhibit undefined behavior (panic, deadlock, silent data loss). Conservation (INV-1: injected == completed + queued + running + rejected) must hold at all overload levels.

**Refuted if:** Any configuration at any overload level (1x-10x) produces a panic, deadlock, non-zero exit code, or conservation invariant violation.
