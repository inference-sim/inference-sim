# H-Liveness: Scheduler Liveness Under Admissible Load

**Status**: Confirmed
**Date**: 2026-02-21

## Hypothesis

> For ALL scheduler configurations (FCFS, SJF, priority-FCFS) at arrival rates below saturation (rho < 0.9), every admitted request should eventually complete (zero still_queued, zero still_running at simulation end), and the queue length trace should be bounded (no monotonic growth).

**Refuted if:** Any scheduler configuration shows still_queued > 0 or still_running > 0 at simulation end for any seed at any tested rate, indicating a starvation or livelock bug.
