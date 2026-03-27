# H12: Request Conservation Invariant

**Status**: Confirmed (with bug discovery)
**Date**: 2026-02-20

## Hypothesis

> No matter what routing, scheduling, or admission policy is used, every injected request must end up completed, queued, or running at simulation end: `injected == completed + still_queued + still_running`. With admission control: `num_requests == injected + rejected`. This is a fundamental correctness property.

**Refuted if:** Any single configuration produces a nonzero difference between `injected` and `completed + still_queued + still_running` at simulation end.
